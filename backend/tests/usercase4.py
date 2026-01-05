from pathlib import Path
import json
# from pprint import pprint
# import joblib
from orchestrator import build_orchestrator
# =====================================================


MODEL_PATH = Path("artifacts/diabetes_model.joblib")
PREP_PATH = Path("artifacts/preprocessor.joblib")


def main():
  orch = build_orchestrator(
    model_path = MODEL_PATH,
    preprocessor_path = PREP_PATH,
    enable_explanations = True,
    use_checkpointer = False,
    sqlite_path = None,
  )

  patient_raw = {
    "gender": "female",
    "age": 39,
    "bmi": 29.0,
    "glucose": 9.8,     # mmol/L -> should convert to ~176 mg/dL
    "hba1c": 52,        # mmol/mol -> should convert to ~4.76%
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never"
  }

  out = orch.invoke(
    run_id = "usercase4",
    mode = "inference",
    patient_raw = patient_raw,
    labs_raw = {},
  )

  # print(out)
  print(json.dumps(out, indent=2, ensure_ascii=False))
  # pprint(out, sort_dicts=False)


# ============================================
if __name__ == "__main__":
    main()
