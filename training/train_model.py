import argparse
from pathlib import Path
# from typing import Optional

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from agents.data_agent import (FEATURES, TARGET, load_diabetes_prediction, load_pima, load_mohammed, load_diabetes_readmission, build_preprocessor)
from agents.clinical_agent import ClinicalAssessmentAgent
# ===========================================================================================================================



def _must_exist(p: Path, name: str) -> Path:
  if not p.exists():
    raise FileNotFoundError(f"{name} not found: {p.resolve()}")
  return p


def load_datasets(diabetes_path: Path, pima_path: Path, mohammed_path: Path, readmission_path: Path) -> pd.DataFrame:
  # dsets = []
  dsets: list[pd.DataFrame] = []

  # main dataset (the kaggle 'diabetes prediction' style one)
  dset_diabetes = load_diabetes_prediction(diabetes_path)
  dsets.append(dset_diabetes)

  dset_pima = load_pima(pima_path)
  dsets.append(dset_pima)

  dset_mohammed = load_mohammed(_must_exist(mohammed_path, "mohammed_path"))
  dsets.append(dset_mohammed)

  dset_readmission = load_diabetes_readmission(_must_exist(readmission_path, "readmission_path"))
  dsets.append(dset_readmission)

  all_dset = pd.concat(dsets, ignore_index=True)

  # drop all missing rows label just incase
  all_dset = all_dset.dropna(subset=[TARGET])

  return all_dset



def train(diabetes_path: str, pima_path: str, mohammed_path: str, readmission_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42) -> None:
  output_dir_path = Path(output_dir)
  output_dir_path.mkdir(parents=True, exist_ok=True)

  # Load datasets
  print("[*] Loading datasets...")
  dsets = load_datasets(
    diabetes_path = Path(diabetes_path),
    pima_path = Path(pima_path),
    mohammed_path = Path(mohammed_path),
    readmission_path = Path(readmission_path),
  )

  print(f"[*] Combined dataset shape: {dsets.shape}")
  features = dsets[FEATURES]
  target = dsets[TARGET].astype(int)

  # Force categorical columns to strings (prevents OneHotEncoder mixed type crash)
  cat_cols = ["gender", "hypertension", "heart_disease", "smoking_history"]
  for c in cat_cols:
    if c in features.columns:
      features[c] = features[c].astype("string").fillna("unknown")



  # Split data
  print("[*] Splitting data...")
  features_train, features_test, target_train, target_test = train_test_split(
    features, 
    target, 
    test_size = test_size, 
    random_state = random_state, 
    stratify = target
  )


  # Build and fit preprocessor
  print("[*] Building preprocessor...")
  preprocessor = build_preprocessor()

  print("[*] Fitting preprocessor on training data...")
  preprocessor.fit(features_train)

  features_train_proc = preprocessor.transform(features_train)
  features_test_proc = preprocessor.transform(features_test)


  # Train baseline model -----
  print("[*] Training logistic regression model...")
  model = LogisticRegression(
    max_iter = 1000,
    class_weight = "balanced",
    # n_jobs = 1,
  )
  model.fit(features_train_proc, target_train)

  # wrap model in ClinicalAssessmentAgent for MAS useage -----
  if hasattr(preprocessor, "get_feature_names_out"):
    feature_names = list(preprocessor.get_feature_names_out())
  else:
    feature_names = FEATURES

  clinical_agent = ClinicalAssessmentAgent(
    model = model,
    feature_names = feature_names,
  )
  clinical_agent.is_fitted_ = True


  # Evaluation -----
  print("[*] Evaluating model on testing data...")  
  target_pred = model.predict(features_test_proc)
  target_proba = model.predict_proba(features_test_proc)[:, 1]

  print("\nTest classification report:")
  print(classification_report(target_test, target_pred))

  try:
    auc = roc_auc_score(target_test, target_proba)
    print(f"ROC AUC Score: {auc:.3f}")
  except ValueError:
    print("ROC AUC Could not be computed (probably only one class in target_test).")


  # Save artifacts
  preprocessor_path = output_dir_path / "preprocessor.joblib"
  model_path = output_dir_path / "diabetes_model.joblib"
  clinical_agent_path = output_dir_path / "clinical_agent.joblib"
  meta_path = output_dir_path / "metadata.txt"

  print(f"[*] Saving preprocessor to: {preprocessor_path}")
  joblib.dump(preprocessor, preprocessor_path)

  print(f"[*] Saving model to: {model_path}")
  joblib.dump(model, model_path)

  print(f"[*] Saving clinical agent wrapper to: {clinical_agent_path}")
  joblib.dump(clinical_agent, clinical_agent_path)

  with open(meta_path, "w") as f:
    f.write(f"Features: {FEATURES}\n")
    f.write(f"Target: {TARGET}\n")
    f.write(f"samples total: {len(dsets)}\n")
    f.write(f"Train size: {len(features_train)}\n")
    f.write(f"Test size: {len(features_test)}\n")
    f.write(f"diabetes_path: {Path(diabetes_path).resolve()}\n")
    f.write(f"pima_path: {Path(pima_path).resolve()}\n")
    f.write(f"mohammed_path: {Path(mohammed_path).resolve()}\n")
    f.write(f"readmission_path: {Path(readmission_path).resolve()}\n")


  print("[*] Training complete.")



def parse_args():
  parser = argparse.ArgumentParser(description="Train T2D risk/prediction model on 4 datasets.")
  parser.add_argument("--diabetes-path", required=True)
  parser.add_argument("--pima-path", required=True)
  parser.add_argument("--mohammed-path", required=True)
  parser.add_argument("--readmission-path", required=True)
  parser.add_argument("--output-dir", default="artifacts")
  parser.add_argument("--test-size", type=float, default=0.2)
  return parser.parse_args()



# ===========================================================================================================================
if __name__ == "__main__":
  args = parse_args()
  train(
    diabetes_path = args.diabetes_path,
    pima_path = args.pima_path,
    mohammed_path = args.mohammed_path,
    readmission_path = args.readmission_path,
    output_dir = args.output_dir,
    test_size = args.test_size,
  ) 