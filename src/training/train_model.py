import argparse
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from src.agents.datahandling.agent import (FEATURES, TARGET, load_diabetes_prediction, load_pima, load_mohammed, build_preprocessor)
# ===========================================================================================================================



def load_datasets(diabetes_path: Path,
                  pima_path: Path,
                  mohammed_path: Optional[Path] = None) -> pd.DataFrame:
  """
  Load and unify all datasets into a single DataFrame.
  All outputs must have the same columns as defined in FEATURES + [target]. 
  """
  dsets = []

  # main dataset (the kaggle 'diabetes prediction' style one)
  dset_diabetes = load_diabetes_prediction(diabetes_path)
  dsets.append(dset_diabetes)

  dset_pima = load_pima(pima_path)
  dsets.append(dset_pima)

  if mohammed_path is not None:
    dset_mohammed = load_mohammed(mohammed_path)
    dsets.append(dset_mohammed)

  all_dset = pd.concat(dsets, ignore_index=True)

  # drop all missing rows label just incase
  all_dset = all_dset.dropna(subset=[TARGET])

  return all_dset



def train(diabetes_path: str,
          pima_path: str,
          mohammed_path: str | None,
          output_dir: str,
          test_size: float = 0.2,
          random_state: int = 42):
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  # Load datasets
  print("[*] Loading datasets...")
  dsets = load_datasets(
      diabetes_path = Path(diabetes_path),
      pima_path = Path(pima_path),
      mohammed_path = Path(mohammed_path) if mohammed_path is not None else None,
  )

  print(f"[*] Combined dataset shape: {dsets.shape}")
  features = dsets[FEATURES]
  target = dsets[TARGET].astype(int)


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


  # Train baseline model
  print("[*] Training logistic regression model...")
  model = LogisticRegression(
    max_iter = 1000,
    class_weight = "balanced",
    # n_jobs = 1,
  )
  model.fit(features_train_proc, target_train)


  # Evaluate
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
  preprocessor_path = output_dir / "preprocessor.joblib"
  model_path = output_dir / "diabetes_model.joblib"
  meta_path = output_dir / "metadata.txt"

  print(f"[*] Saving preprocessor to: {preprocessor_path}")
  joblib.dump(preprocessor, preprocessor_path)

  print(f"[*] Saving model to: {model_path}")
  joblib.dump(model, model_path)

  with open(meta_path, "w") as f:
    f.write(f"Features: {FEATURES}\n")
    f.write(f"Target: {TARGET}\n")
    f.write(f"samples total: {len(dsets)}\n")
    f.write(f"Train size: {len(features_train)}\n")
    f.write(f"Test size: {len(features_test)}\n")


  print("[*] Training complete.")



def parse_args():
  parser = argparse.ArgumentParser(description="Train T2D risk/prediction model.")
  parser.add_argument(
    "--diabetes-path", 
    # type = str, 
    required = True, 
    # help = "Path to the diabetes prediction dataset CSV file."
  )
  parser.add_argument(
    "--pima-path", 
    # type = str, 
    required = True, 
    # help = "Path to the Pima Indians Diabetes dataset CSV file."
  )
  parser.add_argument(
    "--mohammed-path", 
    # type=str, 
    default = None,
    # help = "Optional path to the Mohammed et al. dataset CSV file."
  )
  parser.add_argument(
    "--output-dir", 
    default = "artifacts",
    # help = "Directory to save the trained model and preprocessor."
  )
  parser.add_argument(
    "--test-size", 
    type = float, 
    default = 0.2,
    # help = "Proportion of the dataset to include in the test split."
  )
  return parser.parse_args()



# ===========================================================================================================================
if __name__ == "__main__":
  args = parse_args()
  train(
    diabetes_path=args.diabetes_path,
    pima_path=args.pima_path,
    mohammed_path=args.mohammed_path,
    output_dir=args.output_dir,
    test_size=args.test_size,
  ) 