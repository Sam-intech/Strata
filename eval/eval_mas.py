from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
  roc_auc_score,
  roc_curve,
  confusion_matrix,
  ConfusionMatrixDisplay,
)

# Import your orchestrator + schema constants
from orchestrator import build_orchestrator
from agents.data_agent import FEATURES, TARGET
# ==========================================================================


def extract_score_and_label(final_output: Dict[str, Any]) -> Tuple[float, int]:
  """
  Pulls:
    - probability score: aggregated.clinical.risk_T2D_now
    - predicted label: thresholded at 0.5
  """
  agg = final_output["result"]
  score = float(agg["clinical"]["risk_T2D_now"])
  pred = 1 if score >= 0.5 else 0
  return score, pred


def plot_metrics_bar(metrics: Dict[str, float], outpath: Path) -> None:
  names = list(metrics.keys())
  values = list(metrics.values())

  plt.figure()
  plt.bar(names, values)
  plt.ylim(0, 1)
  plt.ylabel("Score")
  plt.title("MAS Diagnostic Performance Metrics")
  plt.tight_layout()
  plt.savefig(outpath, dpi=200)
  plt.close()


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, auc: float, outpath: Path) -> None:
  fpr, tpr, _ = roc_curve(y_true, y_score)

  plt.figure()
  plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
  plt.plot([0, 1], [0, 1], linestyle="--")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC Curve (MAS)")
  plt.legend()
  plt.tight_layout()
  plt.savefig(outpath, dpi=200)
  plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, outpath: Path) -> None:
  cm = confusion_matrix(y_true, y_pred)
  disp = ConfusionMatrixDisplay(cm)
  disp.plot()
  plt.title("Confusion Matrix (MAS)")
  plt.tight_layout()
  plt.savefig(outpath, dpi=200)
  plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .joblib")
    parser.add_argument("--preprocessor_path", type=str, required=True, help="Path to preprocessor .joblib")
    parser.add_argument("--out_dir", type=str, default="artifacts/eval", help="Where to save outputs")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)

    # Keep only rows with a valid label
    df = df.dropna(subset=[TARGET]).copy()

    # Ensure label is int 0/1
    df[TARGET] = df[TARGET].astype(int)

    # Split (metrics are computed on test only)
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df[TARGET] if df[TARGET].nunique() > 1 else None,
    )

    orch = build_orchestrator(
        model_path=Path(args.model_path),
        preprocessor_path=Path(args.preprocessor_path),
        enable_explanations=False,   # keep eval fast; turn on if you want
        use_checkpointer=False,
        sqlite_path=None,
    )

    y_true: List[int] = []
    y_score: List[float] = []
    y_pred: List[int] = []

    for i, row in test_df.iterrows():
        patient_raw: Dict[str, Any] = {k: row.get(k) for k in FEATURES}
        gt = int(row[TARGET])

        out = orch.invoke(
            run_id=f"eval_{i}",
            mode="evaluation",
            patient_raw=patient_raw,
            labs_raw={},  # you can pass lab dicts later if your dataset has them
        )

        score, pred = extract_score_and_label(out)

        y_true.append(gt)
        y_score.append(score)
        y_pred.append(pred)

    y_true_np = np.array(y_true, dtype=int)
    y_score_np = np.array(y_score, dtype=float)
    y_pred_np = np.array(y_pred, dtype=int)

    # Metrics
    accuracy = accuracy_score(y_true_np, y_pred_np)
    precision = precision_score(y_true_np, y_pred_np, zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, zero_division=0)
    f1 = f1_score(y_true_np, y_pred_np, zero_division=0)

    # ROC-AUC needs probability scores + both classes present
    if len(np.unique(y_true_np)) == 2:
        auc = roc_auc_score(y_true_np, y_score_np)
    else:
        auc = float("nan")

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": auc,
    }

    # Print metrics (copy into Results chapter)
    print("\n=== MAS Evaluation Metrics (Test Set) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if np.isfinite(v) else f"{k}: N/A (single-class test set)")

    # Save plots
    plot_metrics_bar(metrics={k: v for k, v in metrics.items() if np.isfinite(v)}, outpath=out_dir / "mas_metrics_bar.png")

    if np.isfinite(auc):
        plot_roc(y_true_np, y_score_np, auc, outpath=out_dir / "mas_roc_curve.png")

    plot_confusion(y_true_np, y_pred_np, outpath=out_dir / "mas_confusion_matrix.png")

    # Save raw outputs too
    pd.DataFrame({
        "y_true": y_true_np,
        "y_pred": y_pred_np,
        "y_score": y_score_np,
    }).to_csv(out_dir / "mas_predictions.csv", index=False)

    pd.DataFrame([metrics]).to_csv(out_dir / "mas_metrics.csv", index=False)

    print(f"\nSaved outputs to: {out_dir.resolve()}")
    print("- mas_metrics.csv")
    print("- mas_predictions.csv")
    print("- mas_metrics_bar.png")
    print("- mas_roc_curve.png (if AUC computed)")
    print("- mas_confusion_matrix.png")


if __name__ == "__main__":
    main()
