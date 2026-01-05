# evaluate_pima.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
  roc_auc_score,
  confusion_matrix,
)

# from orchestrator import build_orchestrator
from agents.data_agent import load_pima, TARGET
from orchestrator import StrataOrchestrator, OrchestrationConfig, OrchestrationLogger
from agents.data_agent import DataHandlingAgent
from agents.clinical_agent import ClinicalAssessmentAgent
from agents.lab_agent import LaboratoryAgent
from agents.diagnostic_agent import DiagnosticAgent
# =======================================================================
# Label mapping (MAS -> binary)
# DiagnosticAgent labels: {"normal", "high_risk", "T2D", "uncertain"}
POSITIVE_LABELS_DEFAULT = {"high_risk", "T2D", "uncertain"}  # conservative


def mas_label_to_binary(label: str, positive_labels: set[str]) -> int:
  return 1 if label in positive_labels else 0


def safe_float(x: Any) -> float:
  try:
    return float(x)
  except Exception:
    return float("nan")
  

def build_orchestrator_eval_only(*, model_path: Path, preprocessor_path: Path) -> StrataOrchestrator:
  """
  Evaluation-only orchestrator builder.
  - Does NOT create ExplanationAgent / LLM client (avoids API key issues).
  - Does NOT modify orchestrator.py or any agents.
  """
  preprocessor = joblib.load(preprocessor_path)
  data_agent = DataHandlingAgent(preprocessor=preprocessor)

  model = joblib.load(model_path)

  # feature names for contributor mapping (optional but good)
  feature_names = list(preprocessor.get_feature_names_out()) if hasattr(preprocessor, "get_feature_names_out") else None
  clinical_agent = ClinicalAssessmentAgent(model=model, feature_names=feature_names)

  lab_agent = LaboratoryAgent()
  diagnostic_agent = DiagnosticAgent()

  cfg = OrchestrationConfig(use_checkpointer=False, sqlite_path=None)

  return StrataOrchestrator(
    data_agent=data_agent,
    clinical_agent=clinical_agent,
    lab_agent=lab_agent,
    diagnostic_agent=diagnostic_agent,
    explanation_agent=None,  # key line: no explanation node work
    logger=OrchestrationLogger(),
    config=cfg,
  )


def run_eval(*, pima_csv: Path, model_path: Path, preprocessor_path: Path, test_size: float, random_state: int, include_uncertain_as_positive: bool, out_dir: Path,) -> Dict[str, Any]:
  out_dir.mkdir(parents=True, exist_ok=True)

  # 1) Load + canonicalise dataset (FEATURES + TARGET)
  df = load_pima(pima_csv)

  # Basic sanity: drop rows with missing target
  df = df.dropna(subset=[TARGET]).copy()

  # Ensure target is int 0/1
  df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)

  # 2) Split: held-out test set (stratified)
  train_df, test_df = train_test_split(
    df,
    test_size=test_size,
    random_state=random_state,
    stratify=df[TARGET],
  )
  test_df = test_df.reset_index(drop=True)

  # 3) Build orchestrator (uses your trained model + fitted preprocessor)
  orch = build_orchestrator_eval_only(
    model_path=model_path,
    preprocessor_path=preprocessor_path,
    # enable_explanations=False,   # metrics do not require explanations
    # use_checkpointer=False,
    # sqlite_path=None,
  )

  # 4) Run MAS over test set (offline batch)
  positive_labels = set(POSITIVE_LABELS_DEFAULT)
  if not include_uncertain_as_positive and "uncertain" in positive_labels:
    positive_labels.remove("uncertain")

  y_true: List[int] = []
  y_pred: List[int] = []
  y_prob: List[float] = []

  rows_log: List[Dict[str, Any]] = []

  for i in range(len(test_df)):
    out = orch.invoke(
      run_id="eval_pima",
      mode="evaluation",
      dset_df=test_df,
      dset_row_index=i,
      labs_raw={},  # pima has no real labs payload; features already present
    )

    agg = out["result"]
    gt = agg.get("ground_truth", None)

    # Model probability (ClinicalAssessmentAgent output)
    risk = safe_float(agg["clinical"]["risk_T2D_now"])

    # Final label (DiagnosticAgent output)
    diag_label = str(agg["diagnostic"]["label"])

    y_true.append(int(gt) if gt is not None else 0)
    y_prob.append(risk)
    y_pred.append(mas_label_to_binary(diag_label, positive_labels))

    rows_log.append(
      {
        "row_index": i,
        "ground_truth": int(gt) if gt is not None else 0,
        "risk_T2D_now": risk,
        "diagnostic_label": diag_label,
        "y_pred_binary": y_pred[-1],
      }
    )

  # 5) Compute metrics (3.6.1 contract)
  y_true_arr = np.array(y_true, dtype=int)
  y_pred_arr = np.array(y_pred, dtype=int)
  y_prob_arr = np.array(y_prob, dtype=float)

  # Guard ROC-AUC if probabilities are degenerate or contain NaN
  roc_auc = None
  if np.isfinite(y_prob_arr).all() and len(np.unique(y_true_arr)) == 2:
    try:
      roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except Exception:
      roc_auc = None

  metrics = {
    "dataset": "pima",
    "n_test": int(len(test_df)),
    "positive_labels": sorted(list(positive_labels)),
    "split": {"test_size": test_size, "random_state": random_state, "stratified": True},
    "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
    "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
    "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
    "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    "roc_auc": roc_auc,
    "confusion_matrix": {
      "tn_fp_fn_tp": [int(x) for x in confusion_matrix(y_true_arr, y_pred_arr).ravel()]
      if len(np.unique(y_true_arr)) == 2
      else None
    },
  }

  # 6) Save artefacts for Results chapter
  pd.DataFrame(rows_log).to_csv(out_dir / "pima_eval_predictions.csv", index=False)
  with open(out_dir / "pima_eval_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

  return metrics


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--pima_csv", type=Path, required=True, help="Path to Pima CSV (e.g., diabetes.csv)")
  ap.add_argument("--model_path", type=Path, required=True, help="Path to trained model artifact .joblib")
  ap.add_argument("--preprocessor_path", type=Path, required=True, help="Path to fitted preprocessor .joblib")
  ap.add_argument("--test_size", type=float, default=0.2)
  ap.add_argument("--random_state", type=int, default=42)
  ap.add_argument(
    "--uncertain_positive",
    action="store_true",
    help="If set, treats 'uncertain' as positive (conservative). Default: False.",
  )
  ap.add_argument("--out_dir", type=Path, default=Path("artifacts/eval_pima"))
  args = ap.parse_args()

  metrics = run_eval(
    pima_csv=args.pima_csv,
    model_path=args.model_path,
    preprocessor_path=args.preprocessor_path,
    test_size=args.test_size,
    random_state=args.random_state,
    include_uncertain_as_positive=args.uncertain_positive,
    out_dir=args.out_dir,
  )

  print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
  main()
