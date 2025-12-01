from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

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
TARGET = "diabetes_present"


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

  # --- basic value check ---
  @field_validator("bmi")
  @classmethod
  def check_bmi_range(cls, v: float) -> float:
    if not 10 <= v <= 80:
      raise ValueError("BMI out plausible clinical range (10-80).")
    return v
  
  @field_validator("glucose")
  @classmethod
  def check_glucose_non_negative(cls, v: Optional[float]) -> Optional[float]:
    if v is not None and v <= 0:
      raise ValueError("Glucose must be > 0.")
    return v
  
  @field_validator("hba1c")
  @classmethod
  def check_hba1c_non_negative(cls, v: Optional[float]) -> Optional[float]:
    if v is not None and v <= 0:
      raise ValueError("HbA1c must be > 0")
    return v
  
  @field_validator("insulin")
  @classmethod
  def check_insulin_range(cls, v: Optional[float]) -> Optional[float]:
    if v is not None and not (0 < v <= 1000):
      raise ValueError("Insulin out of plausible range (0-1000)")
    return v
  

@dataclass
class DataHandlingOutput:
  raw: pd.DataFrame
  cleaned: pd.DataFrame
  views: Dict[str, pd.DataFrame]
  flags: pd.DataFrame
  validation_errors: Optional[Dict[str, Any]] = None



# --- Dataset-specific mappers ---
def load_diabetes_prediction(path: str) -> pd.DataFrame:
  dset = pd.read_csv(path)
  dset = dset.rename(
    columns= {
      "HbA1c_level": "hba1c",
      "blood_glucose_level": "glucose",
      "diabetes": "diabetes_present",
    }
  )

  for col in FEATURES + [TARGET]:
    if col not in dset.columns:
      dset[col] = pd.NA
  return dset[FEATURES + [TARGET]]


def load_pima(path: str) -> pd.DataFrame:
  dset = pd.read_csv(path)
  dset = dset.rename(
    columns={
      "Glucose": "glucose",
      "BloodPressure": "blood_pressure",
      "Insulin": "insulin",
      "BMI": "bmi",
      "Age": "age",
      "Outcome": "diabetes_present",
    }
  )
  
  for col in FEATURES + [TARGET]:
    if col not in dset.columns:
      dset[col] = pd.NA
  return dset[FEATURES + [TARGET]]


def load_mohammed(path: str) -> pd.DataFrame:
  dset = pd.read_csv(path)

  mapping = {
    "HbA1c_level": "hba1c",
    "blood_glucose_level": "glucose",
    "diabetes": "diabetes_present",
  }
  dset = dset.rename(columns={k: v for k, v in mapping.items() if k in dset.columns})

  for col in FEATURES + [TARGET]:
    if col not in dset.columns:
      dset[col] = pd.NA
  return dset[FEATURES + [TARGET]]



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
  def __init__(self, preprocessor, feature_groups: Dict[str, list]):
    self.preprocessor = preprocessor
    # self.feature_groups = feature_groups

  # internal helopers -----  
  def _normalise_units_and_flags(self, dset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dset = dset.copy()
    flags = pd.DataFrame(index = dset.index)

    # missing flags -----
    for col in ["age", "bmi", "glucose", "hba1c", "blood_pressure", "insulin"]:
      flags[f"{col}_missing"] = dset[col].isna() if col in dset.columns else True

    # glucose: mmol/L -> mg/dL if needed
    if "glucose" in dset.columns:
      flags["glucose_suspect_unit"] = False
      mask_mmol = dset["glucose"].notna() & (dset["glucose"] <= 40)
      dset.loc[mask_mmol, "glucose"] = dset.loc[mask_mmol, "glucose"] * 18
      # flag extreme values
      flags["glucose_out_of_range"] = dset["glucose"].notna() & ~dset["glucose"].between(40, 600)

    # --- HbA1c: mmol/mol -> % if needed ---
    if "hba1c" in dset.columns:
      flags["hba1c_suspect_unit"] = False
      mask_mmol = dset["hba1c"].notna() & (dset["hba1c"] > 20)
      dset.loc[mask_mmol, "hba1c"] = dset.loc[mask_mmol, "hba1c"] / 10.93
      flags.loc[mask_mmol, "hba1c_suspect_unit"] = True      
      flags["hba1c_out_of_range"] = dset["hba1c"].notna() & ~dset["hba1c"].between(3, 20)

    # --- BMI out-of-range soft flag (extra safety) ---
    if "bmi" in dset.columns:
      flags["bmi_out_of_range_soft"] = dset["bmi"].notna() & ~dset["bmi"].between(10, 80)


    return dset, flags


  # Training: ingest a unified dataframe (after mapping)
  def ingest_batch(self, dset_raw: pd.DataFrame) -> DataHandlingOutput:
    x = dset_raw[FEATURES].copy()

    x_norm, flags = self._normalise_units_and_flags(x)

    dset_clean = self.preprocessor.transform(x_norm)
    if hasattr(self.preprocessor, "get_feature_names_out"):
      cols = self.preprocessor.get_feature_names_out()
      dset_clean = pd.DataFrame(dset_clean, columns=cols, index=x_norm.index)
    else:
      dset_clean = pd.DataFrame(dset_clean, index=x_norm.index)

    # views = self._split_views(dset_clean)

    return DataHandlingOutput(
      raw = dset_raw, 
      cleaned = dset_clean,
      # views = views,
      flags = flags
    )

  # inference: single patient from UI -----
  def ingest_single(self, data: Dict[str, Any]) -> DataHandlingOutput:
    try:
      validated = PatientInput(**data)
      dset_raw = pd.DataFrame([validated.model_dump()])
      errors = None
    except ValidationError as e:
      dset_raw = pd.DataFrame([data])
      errors = e.errors()

    x = dset_raw[FEATURES].copy()
    x_norm, flags = self._normalise_units_and_flags(x)

    # for col in FEATURES:
    #   if col not in dset_raw.columns:
    #     dset_raw[col] = pd.NA

    dset_clean = self.preprocessor.transform(x_norm)
    if hasattr(self.preprocessor, "get_feature_names_out"):
      cols = self.preprocessor.get_feature_names_out()
      dset_clean = pd.DataFrame(dset_clean, columns=cols)
    else:
      dset_clean = pd.DataFrame(dset_clean)

    # views = self._split_views(dset_clean)

    return DataHandlingOutput(
      raw=dset_raw,
      cleaned=dset_clean,
      # views = views,
      flags = flags,
      validation_errors=errors,
    )

