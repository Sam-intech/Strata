from pathlib import Path
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
    "gender": "male",
    "age": 45,
    "bmi": 31.2,
    "glucose": 158,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "never",
  }

  out = orch.invoke(
    run_id = "smoke_run_001",
    mode = "inference",
    patient_raw = patient_raw,
    labs_raw = {},
  )

  print(out)


# ============================================
if __name__ == "__main__":
    main()
