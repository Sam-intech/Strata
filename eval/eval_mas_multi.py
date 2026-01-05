from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

from orchestrator import build_orchestrator
from agents.data_agent import TARGET
# ====================================================================


LABELED_KINDS = {"diabetes_prediction", "pima"}  # based on your loaders

def extract_agg_and_score(out: Dict[str, Any]) -> float:
  """
  Handles multiple possible orchestrator return shapes.
  Returns risk score as float.
  """

  # 1) If orchestrator returns {"final_output": {...}}
  if isinstance(out, dict) and "final_output" in out:
    payload = out["final_output"]
  else:
    payload = out

  # 2) If payload is already {"result": {...}, "explanation": ...}
  if isinstance(payload, dict) and "result" in payload:
    agg = payload["result"]
  # 3) If it's LangGraph state style: {"aggregated": {...}}
  elif isinstance(payload, dict) and "aggregated" in payload:
    agg = payload["aggregated"]
  else:
    # last resort: assume payload itself is the aggregated dict
    agg = payload

  # Now find score in common locations
  if isinstance(agg, dict):
    # expected: agg["clinical"]["risk_T2D_now"]
    if "clinical" in agg and isinstance(agg["clinical"], dict) and "risk_T2D_now" in agg["clinical"]:
      return float(agg["clinical"]["risk_T2D_now"])

    # fallback: some versions may store clinical output differently
    if "clinical_output" in agg and isinstance(agg["clinical_output"], dict) and "risk_T2D_now" in agg["clinical_output"]:
      return float(agg["clinical_output"]["risk_T2D_now"])

  raise KeyError(f"Could not locate risk score in orchestrator output. Top-level keys: {list(out.keys()) if isinstance(out, dict) else type(out)}")



def evaluate_dataset(orch, *, dataset_path: Path, dataset_kind: str, threshold: float = 0.5,) -> Dict[str, Any]:
  # Load + map to canonical schema (includes TARGET where available)
  dset = orch.data_agent.resolve_batch_ref(
    str(dataset_path),
    dataset_kind=dataset_kind,
    include_target=True
  )

  # Drop rows without labels (important)
  dset = dset.dropna(subset=[TARGET]).copy()
  dset[TARGET] = dset[TARGET].astype(int)
  dset = dset.reset_index(drop=True)

  y_true: List[int] = []
  y_score: List[float] = []
  y_pred: List[int] = []

  # Iterate rows -> MAS inference
  feature_cols = [c for c in dset.columns if c != TARGET]

  for idx, row in dset.iterrows():
    patient_raw: Dict[str, Any] = row[feature_cols].to_dict()
    out = orch.invoke(
      run_id=f"eval_{dataset_kind}_{idx}",
      mode="evaluation",
      dset_df = dset,
      dset_row_index = idx,
      # patient_raw=patient_raw,
      # labs_raw={},
    )

    # Pull risk score from aggregated output
    # agg = out["final_output"]["result"]
    # score = float(agg["clinical"]["risk_T2D_now"])
    # pred = 1 if score >= threshold else 0
    score = extract_agg_and_score(out)
    if not isinstance(score, (int, float, np.floating)):
      raise TypeError(f"Risk score is not numeric. Got type={type(score)} value={score}")
    pred = 1 if score >= threshold else 0

    y_true.append(int(row[TARGET]))
    y_score.append(score)
    y_pred.append(pred)

  y_true_np = np.array(y_true, dtype=int)
  y_score_np = np.array(y_score, dtype=float)
  y_pred_np = np.array(y_pred, dtype=int)

  metrics = {
      "dataset_kind": dataset_kind,
      "n": len(y_true_np),
      "threshold": threshold,
      "accuracy": accuracy_score(y_true_np, y_pred_np),
      "precision": precision_score(y_true_np, y_pred_np, zero_division=0),
      "recall": recall_score(y_true_np, y_pred_np, zero_division=0),
      "f1": f1_score(y_true_np, y_pred_np, zero_division=0),
      "roc_auc": roc_auc_score(y_true_np, y_score_np) if len(np.unique(y_true_np)) == 2 else float("nan"),
  }

  return {
      "metrics": metrics,
      "y_true": y_true_np,
      "y_pred": y_pred_np,
      "y_score": y_score_np,
  }



def save_visuals(out_dir: Path, dataset_kind: str, y_true, y_pred, y_score, metrics: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart (Accuracy/Precision/Recall/F1/AUC)
    names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    vals = [metrics[k] for k in names if np.isfinite(metrics[k])]

    plt.figure()
    plt.bar([k.upper() for k in names if np.isfinite(metrics[k])], vals)
    plt.ylim(0, 1)
    plt.title(f"MAS Metrics ({dataset_kind})")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_kind}_metrics_bar.png", dpi=200)
    plt.close()

    # ROC curve (if valid)
    if np.isfinite(metrics["roc_auc"]):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (MAS) - {dataset_kind}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{dataset_kind}_roc.png", dpi=200)
        plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {dataset_kind}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_kind}_confusion.png", dpi=200)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--preprocessor_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="artifacts/eval_mas")
    p.add_argument("--threshold", type=float, default=0.5)

    # Provide paths for up to 4 datasets (only labeled ones get full metrics)
    p.add_argument("--diabetes_prediction", type=str, default=None)
    p.add_argument("--pima", type=str, default=None)
    p.add_argument("--mohammed", type=str, default=None)
    p.add_argument("--readmission", type=str, default=None)

    args = p.parse_args()

    orch = build_orchestrator(
        model_path=Path(args.model_path),
        preprocessor_path=Path(args.preprocessor_path),
        enable_explanations=False,
        use_checkpointer=False,
        sqlite_path=None,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_rows = []

    provided = {
        "diabetes_prediction": args.diabetes_prediction,
        "pima": args.pima,
        "mohammed": args.mohammed,
        "readmission": args.readmission,
    }

    for kind, path_str in provided.items():
        if not path_str:
            continue

        path = Path(path_str)
        if kind not in LABELED_KINDS:
            # Unlabeled: run inference-only summary (optional; no metrics)
            print(f"[{kind}] has no ground-truth labels in your loader -> skipping metric evaluation.")
            continue

        res = evaluate_dataset(
            orch,
            dataset_path=path,
            dataset_kind=kind,
            threshold=args.threshold,
        )
        metrics = res["metrics"]
        results_rows.append(metrics)

        # Save visuals + raw preds
        save_visuals(out_dir, kind, res["y_true"], res["y_pred"], res["y_score"], metrics)

        pd.DataFrame({
            "y_true": res["y_true"],
            "y_pred": res["y_pred"],
            "y_score": res["y_score"],
        }).to_csv(out_dir / f"{kind}_predictions.csv", index=False)

        print(f"\n=== MAS Metrics ({kind}) n={metrics['n']} threshold={metrics['threshold']} ===")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            v = metrics[k]
            print(f"{k}: {v:.4f}" if np.isfinite(v) else f"{k}: N/A")

    if results_rows:
        pd.DataFrame(results_rows).to_csv(out_dir / "mas_metrics_summary.csv", index=False)
        print(f"\nSaved summary to: {out_dir / 'mas_metrics_summary.csv'}")


if __name__ == "__main__":
    main()
