# eval_offline.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from agents.data_agent import (
  FEATURES, TARGET,
  load_diabetes_prediction, load_pima, load_mohammed, load_diabetes_readmission
)
from orchestrator import build_orchestrator
# ================================================================================

def load_all(diabetes_path: Path, pima_path: Path, mohammed_path: Path, readmission_path: Path) -> pd.DataFrame:
  d1 = load_diabetes_prediction(diabetes_path)
  d2 = load_pima(pima_path)
  d3 = load_mohammed(mohammed_path)
  d4 = load_diabetes_readmission(readmission_path)
  df = pd.concat([d1, d2, d3, d4], ignore_index=True).dropna(subset=[TARGET])
  return df


def label_to_binary(label: str, *, treat_uncertain_as_positive: bool = True) -> int:
  # DiagnosticAgent labels in your code: "normal", "high_risk", "T2D", "uncertain"
  if label in ("high_risk", "T2D"):
    return 1
  if label == "uncertain":
    return 1 if treat_uncertain_as_positive else 0
  return 0


def main() -> None:
  # ---- paths (edit to match yours)
  diabetes_path = Path("data/raw/concluded/diabetes_dset1.csv")
  pima_path = Path("data/raw/concluded/pima_indians.csv")
  mohammed_path = Path("data/raw/concluded/mohammed.csv")
  readmission_path = Path("data/raw/concluded/diabetes_dset2.csv")

  model_path = Path("artifacts/diabetes_model.joblib")
  prep_path = Path("artifacts/preprocessor.joblib")

  # ------------------------------------------ 
  # load + split (must match training seed)
  dset = load_all(diabetes_path, pima_path, mohammed_path, readmission_path)

  x = dset[FEATURES].copy()
  y = dset[TARGET].astype(int).copy()

  # keep split deterministic + stratified (matches your train_model.py)
  x_train, x_test, y_train, y_test = train_test_split(
      x, y, test_size=0.2, random_state=42, stratify=y
  )

  # ---- build orchestrator (disable explanations)
  orch = build_orchestrator(
      model_path=model_path,
      preprocessor_path=prep_path,
      enable_explanations=False,
      use_checkpointer=False,
      sqlite_path=None,
  )

  y_pred = []
  y_prob = []


  # ------------------------------------- 
  # batch run MAS over test set
  # Use evaluation mode so ground_truth is available in trace (optional but clean)
  test_dset = x_test.copy()
  test_dset[TARGET] = y_test.values

  for i in range(len(test_dset)):
    out = orch.invoke(
      run_id=f"eval_{i}",
      mode="evaluation",
      dset_df=test_dset,
      dset_row_index=i,
      labs_raw={},  # no labs for now
    )

    diag_label = out["result"]["diagnostic"]["label"]
    risk_prob = out["result"]["clinical"]["risk_T2D_now"]

    y_pred.append(label_to_binary(diag_label, treat_uncertain_as_positive=True))
    y_prob.append(float(risk_prob))

  # ---- compute metrics (3.6.1)
  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred, zero_division=0)
  rec = recall_score(y_test, y_pred, zero_division=0)
  f1 = f1_score(y_test, y_pred, zero_division=0)

  # ROC-AUC needs probabilities
  auc = roc_auc_score(y_test, y_prob)

  cm = confusion_matrix(y_test, y_pred)

  print("\n=== MAS Quantitative Metrics (Test Set) ===")
  print(f"Accuracy : {acc:.4f}")
  print(f"Precision: {prec:.4f}")
  print(f"Recall   : {rec:.4f}")
  print(f"F1-score : {f1:.4f}")
  print(f"ROC-AUC  : {auc:.4f}")
  print("\nConfusion Matrix:")
  print(cm)


if __name__ == "__main__":
    main()
