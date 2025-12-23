from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal

import pandas as pd

from orchestrator import build_orchestrator
# ======================================================================================


MODE: Literal["inference", "evaluation"] = "inference"

DATASET_PATH = Path("data/raw/concluded/diabetes.csv")
MODEL_PATH = Path("artifacts/clinical_model.joblib")


def main() -> None:
  orch = build_orchestrator(
    model_path = MODEL_PATH,
    enable_explanations = True,
    use_checkpointer = False,
    sqlite_path = None,
  )

  if MODE == "inference":
    patient_raw: Dict[str, Any] = {
      "age": 45,
      "bmi": 31.2,
      "glucose": 158,
      "hypertension": 1,
      "heart_disease": 0,
      "smoking_history": "never",
    }
    out = orch.invoke(
      run_id = "run_local_001",
      mode = "inference",
      patient_raw = patient_raw,
      labs_raw = {},
    )
  else:
    if not DATASET_PATH.exists():
      raise FileNotFoundError(f"Dataset path {DATASET_PATH} does not exist for evaluation mode.")
    
    df = pd.read_csv(DATASET_PATH)
    out = orch.invoke(
      run_id = "run_eval_001",
      mode = "evaluation",
      dset_df = df,
      dset_row_index = 0,
      labs_raw = {},
    )

  print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
  main()
