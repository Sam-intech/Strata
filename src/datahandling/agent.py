from dataclasses import dataclass
from typing import Dict, Any, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer 
# =============================================================================


FEATURES = [
  "gender",
  "age",
  "bmi",
  "glucose",
  "hba1c",
  "blood_pressure",
  "hypertension",
  "heart_disease",
  "smoking_history",
  "insulin",
]
target = "diabetes_present"


# --- Canonical input schema for a single patient/single pateint schema (for UI/inference) --- 
class PatientInput(BaseModel):
  gender: str
  age: float = Field(ge=0, le=120)
  bmi: float = Field(gt=0)

  glucose: Optional[float] = None
  hba1c: Optional[float] = None
  blood_pressure: Optional[float] = None
  insulin: Optional[float] = None

  hypertension: Optional[bool] = None
  heart_disease: Optional[bool] = None
  smoking_history: Optional[str] = None

@dataclass
class DataHandlingOutput:
  raw: pd.DataFrame
  cleaned: pd.DataFrame
  # views: Dict[str, pd.DataFrame]
  validation_errors: Optional[Dict[str, Any]] = None



# --- Dataset-specific mappers ---
def load_diabetes_prediction(path: str) -> pd.DataFrame:
  df = pd.read_csv(path)
  df = df.rename(
    columns= {
      "HbA1c_level": "hba1c",
      "blood_glucose_level": "glucose",
      "diabetes": "diabetes_present",
    }
  )

  for col in FEATURES + [target]:
    if col not in df.columns:
      df[col] = pd.NA

  return df[
    FEATURES + [target]
  ]


def load_pima(path: str) -> pd.DataFrame:
  df = pd.read_csv(path)
  df = df.rename(
    columns={
      "Glucose": "glucose",
      "BloodPressure": "blood_pressure",
      "Insulin": "insulin",
      "BMI": "bmi",
      "Age": "age",
      "Outcome": "diabetes_present",
    }
  )
  
  for col in FEATURES + [target]:
    if col not in df.columns:
      df[col] = pd.NA

  return df[
    FEATURES + [target]
  ]


def load_mohammed(path: str) -> pd.DataFrame:
  df = pd.read_csv(path)

  mapping = {
    "HbA1c_level": "hba1c",
    "blood_glucose_level": "glucose",
    "diabetes": "diabetes_present",
  }
  df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

  for col in FEATURES + [target]:
    if col not in df.columns:
      df[col] = pd.NA

  return df[
    FEATURES + [target]
  ]



# Preprocessing Pipeline -----
def build_preprocessor() -> ColumnTransformer:
  numeric_features = [
    "age",
    "bmi",
    "glucose",
    "hba1c",
    "blood_pressure",
    "insulin",
  ]

  categorical_features = [
    "gender",
    "hypertension",
    "heart_disease",
    "smoking_history",
  ]

  numeric_pipeline = Pipeline(
    steps=[
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler()),
    ]
  )

  categorical_pipleine = Pipeline(
    steps=[
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
  )

  preprocessor = ColumnTransformer(
    transformers=[
      ("num", numeric_pipeline, numeric_features),
      ("cat", categorical_pipleine, categorical_features),
    ]
  )
  return preprocessor



# DataHandling Agent -----
class DataHandlingAgent:
  def __init__(self, preprocessor):
    self.preprocessor = preprocessor

  # Training: ingest a unified dataframe (after mapping)
  def ingest_batch(self, df_raw: pd.DataFrame) -> DataHandlingOutput:
    x = df_raw[FEATURES].copy()
    df_clean = self.preprocessor.transform(x)
    if hasattr(self.preprocessor, "get_feature_names_out"):
      cols = self.preprocessor.get_feature_names_out()
      df_clean = pd.DataFrame(df_clean, columns=cols)
    else:
      df_clean = pd.DataFrame(df_clean)

    return DataHandlingOutput(
      raw=df_raw, 
      cleaned=df_clean
    )

  # inference: single patient from UI -----
  def ingest_single(self, data: Dict[str, Any]) -> DataHandlingOutput:
    try:
      validated = PatientInput(**data)
      df_raw = pd.DataFrame([validated.model_dump()])
      errors = None
    except ValidationError as e:
      df_raw = pd.DataFrame([data])
      errors = e.errors()

    for col in FEATURES:
      if col not in df_raw.columns:
        df_raw[col] = pd.NA

    df_clean = self.preprocessor.transform(df_raw[FEATURES])
    if hasattr(self.preprocessor, "get_feature_names_out"):
      cols = self.preprocessor.get_feature_names_out()
      df_clean = pd.DataFrame(df_clean, columns=cols)
    else:
      df_clean = pd.DataFrame(df_clean)

    return DataHandlingOutput(
      raw=df_raw,
      cleaned=df_clean,
      validation_errors=errors,
    )

