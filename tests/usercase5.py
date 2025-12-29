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
    "age": 28,
    "bmi": 24.0,
    "glucose": 140,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "YES"  # should normalise to "current" (or your chosen bucket)
  }

  out = orch.invoke(
    run_id = "usercase5",
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
