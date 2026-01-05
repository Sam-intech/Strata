from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np

from agents.data_agent import FEATURES, TARGET
# =====================================================================


# Map common source column names -> your canonical schema
COLUMN_MAP: Dict[str, str] = {
  # dataset 1
  "blood_glucose_level": "glucose",
  "hba1c_level": "hba1c",
  "diabetes": "diabetes_present",

  # dataset 2
  "outcome": "diabetes_present",
  "bloodpressure": "blood_pressure",
}

TRUE_SET = {"1", "true", "yes", "y", "positive", "pos", "diabetic", "t2d"}
FALSE_SET = {"0", "false", "no", "n", "negative", "neg", "non-diabetic", "non_diabetic", "nondiabetic", "normal", "none"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
  # lowercase + strip
  df = df.rename(columns={c: c.strip().lower() for c in df.columns})

  # apply mapping
  df = df.rename(columns={src: dst for src, dst in COLUMN_MAP.items() if src in df.columns})

  # Ensure all expected columns exist; if missing, create as NaN
  for col in FEATURES:
    if col not in df.columns:
      df[col] = np.nan

  # Ensure target exists
  if TARGET not in df.columns:
    raise ValueError(
      f"Missing target column '{TARGET}'. "
      f"Found columns: {list(df.columns)}"
    )

  # Keep only canonical columns (features + target)
  df = df[FEATURES + [TARGET]].copy()

  # Coerce target to 0/1 if it isn't already
  # Typical dataset uses 0/1 already; this just makes it robust.
  val = df[TARGET].astype(str).str.strip().str.lower()
  # df[TARGET] = df[TARGET].apply(lambda x: 1 if str(x).strip().lower() in {"1", "true", "yes"} else 0)
  df[TARGET] = val.apply(
    lambda s: 1 if s in TRUE_SET else (0 if s in FALSE_SET else int(float(s)))
  )

  return df


def build_merged_eval(dataset_paths: List[Path], out_path: Path,) -> Path:
  frames = []
  for p in dataset_paths:
    df = pd.read_csv(p)
    df = _normalize_columns(df)
    df["__source__"] = p.name
    frames.append(df)

  merged = pd.concat(frames, ignore_index=True)

  # Drop exact duplicates (common when datasets overlap)
  merged = merged.drop_duplicates(subset=FEATURES + [TARGET])

  out_path.parent.mkdir(parents=True, exist_ok=True)
  merged.to_csv(out_path, index=False)

  print(f"Merged eval saved -> {out_path}")
  print(f"Rows: {len(merged)} | Sources: {merged['__source__'].nunique()}")
  print("Missingness (top 10 cols):")
  miss = merged.isna().mean().sort_values(ascending=False).head(10)
  print(miss.to_string())

  return out_path


# =====================================================================
if __name__ == "__main__":
  # Update these two paths to your actual training datasets
  ds1 = Path("data/raw/concluded/diabetes_dset1.csv")
  ds2 = Path("data/raw/concluded/pima_indians.csv")

  out = Path("data/eval/diabetes_eval_merged.csv")
  build_merged_eval([ds1, ds2], out)
