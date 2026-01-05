# evaluate_mas.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from orchestrator import build_orchestrator
from agents.data_agent import FEATURES, TARGET
# ===========================================================================

@dataclass
class EvalConfig:
  dataset_path: Path = Path("data/eval/diabetes_eval_merged.csv")
  model_path: Path = Path("artifacts/diabetes_model.joblib")
  preprocessor_path: Path = Path("artifacts/preprocessor.joblib")

  # classification threshold (for Accuracy/Precision/Recall/F1)
  threshold: float = 0.50

  # output artifacts for notebook exploration
  out_dir: Path = Path("artifacts/eval")
  run_id_prefix: str = "eval"


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (np.floating, float, int, np.integer)):
            return float(x)
        return float(str(x))
    except Exception:
        return None


def _extract_mas_prediction(final_output: Dict[str, Any]) -> Tuple[float, int]:
    """
    Uses the MAS clinical risk score as probability prediction.
    Returns: (y_proba, y_pred)
    """
    result = final_output.get("result", {})
    clinical = result.get("clinical", {})

    y_proba = _safe_float(clinical.get("risk_T2D_now"))
    if y_proba is None:
        raise ValueError("Could not extract clinical.risk_T2D_now from MAS output.")

    # y_pred derived from threshold outside (but we compute here as well if needed)
    return y_proba, 0  # placeholder; caller applies threshold


def _coerce_ground_truth(v: Any) -> int:
    """
    Ground truth in your datasets is expected to be 0/1 under TARGET = 'diabetes_present'.
    """
    if v is None:
        raise ValueError("Missing ground truth label in dataset row.")
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "diabetes", "t2d", "positive"}:
        return 1
    if s in {"0", "false", "no", "n", "negative", "none"}:
        return 0
    # last attempt
    return int(float(s))


def evaluate_mas(cfg: EvalConfig) -> Dict[str, Any]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.dataset_path)

    if TARGET not in df.columns:
        raise ValueError(f"TARGET='{TARGET}' not found in dataset columns: {list(df.columns)}")

    # Build orchestrator (expects both model + preprocessor)
    orch = build_orchestrator(
        model_path=cfg.model_path,
        preprocessor_path=cfg.preprocessor_path,
        enable_explanations=False,     # eval metrics don’t need LLM
        use_checkpointer=False,
        sqlite_path=None,
    )

    y_true: List[int] = []
    y_proba: List[float] = []
    y_pred: List[int] = []

    rows_out: List[Dict[str, Any]] = []

    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        gt = _coerce_ground_truth(row.get(TARGET))

        # Prefer evaluation mode: orchestrator extracts patient_raw + ground_truth from df row
        out = orch.invoke(
            run_id=f"{cfg.run_id_prefix}_{i:05d}",
            mode="evaluation",
            dset_df=df,
            dset_row_index=i,
            patient_raw=None,
            labs_raw={},
        )

        proba, _ = _extract_mas_prediction(out)
        pred = 1 if proba >= cfg.threshold else 0

        y_true.append(gt)
        y_proba.append(float(proba))
        y_pred.append(pred)

        rows_out.append(
            {
                "row_index": i,
                "y_true": gt,
                "y_proba": float(proba),
                "y_pred": pred,
                # lightweight trace hooks (useful in error analysis later)
                "triage_label": out.get("result", {}).get("clinical", {}).get("triage_label"),
                "diagnostic_label": out.get("result", {}).get("diagnostic", {}).get("label"),
            }
        )

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC requires both classes present
    auc = roc_auc_score(y_true, y_proba) if len(set(y_true)) == 2 else float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "n": len(y_true),
        "threshold": cfg.threshold,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm,  # [[tn, fp], [fn, tp]]
    }

    # Terminal logging
    print("\n=== MAS Evaluation (binary) ===")
    print(f"N: {metrics['n']}")
    print(f"Threshold: {metrics['threshold']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Confusion: {metrics['confusion_matrix']}")

    # Save artifacts for notebook
    pred_path = cfg.out_dir / "mas_predictions.csv"
    metrics_path = cfg.out_dir / "mas_metrics.json"

    pd.DataFrame(rows_out).to_csv(pred_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved: {pred_path}")
    print(f"Saved: {metrics_path}")

    return {"metrics": metrics, "predictions_path": str(pred_path), "metrics_path": str(metrics_path)}


if __name__ == "__main__":
    cfg = EvalConfig()
    evaluate_mas(cfg)
