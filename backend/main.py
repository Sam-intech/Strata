from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import pandas as pd

from orchestrator import build_orchestrator
from agents.data_agent import FEATURES, TARGET
# ======================================================================================
def load_dotenv(path: Path) -> None:
  if not path.exists():
    return
  for line in path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    k, v = line.split("=", 1)
    k = k.strip()
    v = v.strip().strip('"').strip("'")
    os.environ.setdefault(k, v)



MODE: Literal["inference", "evaluation"] = "inference"

# Paths are relative to the backend/ folder (i.e., run: `cd backend && python main.py`)
DATASET_PATH = Path("data/raw/eval/diabetes_eval_merged.csv")
MODEL_PATH = Path("artifacts/diabetes_model.joblib")
PREPROCESSOR_PATH = Path("artifacts/preprocessor.joblib")


def _assert_exists(path: Path, label: str) -> None:
  if not path.exists():
    raise FileNotFoundError(
      f"{label} not found at: {path}\n"
      f"- Check you are running from the backend/ directory.\n"
      f"- Or update the path in main.py.\n"
    )

# def _extract_patient_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
#   # Only keep features the pipeline expects
#   return {k: row.get(k) for k in FEATURES if k in row}


def main() -> None:
  load_dotenv(Path(".env"))

  if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
      "OPENAI_API_KEY is not set. Add it to backend/.env or export it before running.\n"
      "Example:\n"
      '  export OPENAI_API_KEY="sk-..."\n'
      "  python main.py\n"
    )

  # ------------------------------------------------------------------------------------
  # Required artifacts
  _assert_exists(MODEL_PATH, "Model artifact (MODEL_PATH)")
  _assert_exists(PREPROCESSOR_PATH, "Preprocessor artifact (PREPROCESSOR_PATH)")

  orch = build_orchestrator(
    model_path=MODEL_PATH,
    preprocessor_path=PREPROCESSOR_PATH,
    enable_explanations=True,
    use_checkpointer=False,
    sqlite_path=None,
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
      run_id="main entry",
      mode="inference",
      patient_raw=patient_raw,
      labs_raw={},
    )

  else:
    _assert_exists(DATASET_PATH, "Dataset (DATASET_PATH)")

    df = pd.read_csv(DATASET_PATH)
    out = orch.invoke(
      run_id="run_eval_001",
      mode="evaluation",
      dset_df=df,
      dset_row_index=0,
      labs_raw={},
    )

  print(json.dumps(out, indent=2, default=str))


# =====================================================================================
if __name__ == "__main__":
  main()
