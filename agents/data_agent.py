from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, Literal

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer 
from sklearn.utils.validation import check_is_fitted
# =============================================================================

# ------------------------------------------------------------------
# canonical schema(this is what downstream agents will rely on)
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

FEATURE_GROUPS = {
  "clinical": [
    "age",
    "gender",
    "bmi",
    "blood_pressure",
    "hypertension",
    "heart_disease",
    "smoking_history",
  ],
  "lab": [
    "glucose",
    "hba1c",
    "insulin",
  ],
}

DatasetKind = Literal["diabetes_prediction", "pima", "mohammed", "readmission"]
Mode = Literal["batch", "single"]

# --------------------------------------------------------------------------------------------------------
# Canonical input schema for a single patient/single patient schema (for UI/inference)
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
  


# ------------------------
# Structured output
@dataclass
class DataHandlingOutput:
  raw: pd.DataFrame

  # model ready features after preprocessing
  cleaned: pd.DataFrame
  flags: pd.DataFrame
  
  # downstream agents views (for both model & rule based agents)
  views: Optional[Dict[str, pd.DataFrame]] = None
  views_model: Optional[Dict[str, pd.DataFrame]] = None
  
  validation_errors: Optional[list] = None
  meta: Optional[Dict[str, Any]] = None



# -----------------------------------
# Dataset-specific mappers 
def ensure_columns(dset: pd.DataFrame, include_target: bool) -> pd.DataFrame:
  dset = dset.copy()
  needed = FEATURES + ([TARGET] if include_target else [])
  for col in needed:
    if col not in dset.columns:
      dset[col] = np.nan
  return dset[needed]


def load_diabetes_prediction(path: Union[str, Path]) -> pd.DataFrame:
  dset = pd.read_csv(path)
  dset = dset.rename(
    columns= {
      "HbA1c_level": "hba1c",
      "blood_glucose_level": "glucose",
      "diabetes": "diabetes_present",
    }
  )
  return ensure_columns(dset, include_target=True)

  # for col in FEATURES + [TARGET]:
  #   if col not in dset.columns:
  #     # dset[col] = pd.NA
  #     dset[col] = np.nan
  # return dset[FEATURES + [TARGET]]


def load_pima(path: Union[str, Path]) -> pd.DataFrame:
  dset = pd.read_csv(path)
  dset = dset.rename(
    columns={
      "Glucose": "glucose",
      "BloodPressure": "blood_pressure",
      "Insulin": "insulin",
      "BMI": "bmi",
      "Age": "age",
      "Outcome": TARGET,
    }
  )

  # Columns that don't exist in Pima → set to a sentinel, not NaN
  dset["gender"] = "unknown"
  dset["hypertension"] = "unknown"
  dset["heart_disease"] = "unknown"
  dset["smoking_history"] = "unknown"

  # If you’re also forcing hba1c into FEATURES:
  if "hba1c" not in dset.columns:
    dset["hba1c"] = np.nan  

  return ensure_columns(dset, include_target=True)

  # Ensure all expected columns exist
  # missing = [col for col in FEATURES + [TARGET] if col not in dset.columns]
  # if missing:
  #   raise ValueError(f"Pima loader missing columns: {missing}")
  
  # return dset[FEATURES + [TARGET]]
  
  # for col in FEATURES + [TARGET]:
  #   if col not in dset.columns:
  #     # dset[col] = pd.NA
  #     dset[col] = np.nan
  # return dset[FEATURES + [TARGET]]


def load_mohammed(path: Union[str, Path]) -> pd.DataFrame:
  dset = pd.read_csv(path)

  # Map core labs
  if "stab.glu" in dset.columns:
    dset["glucose"] = pd.to_numeric(dset["stab.glu"], errors="coerce")
  if "glyhb" in dset.columns:
    dset["hba1c"] = pd.to_numeric(dset["glyhb"], errors="coerce")

  # BMI from imperial units if available
  if "height" in dset.columns and "weight" in dset.columns:
    h = pd.to_numeric(dset["height"], errors="coerce")
    w = pd.to_numeric(dset["weight"], errors="coerce")
    dset["bmi"] = (w / (h ** 2)) * 703
  else:
    dset["bmi"] = np.nan

  # Blood pressure proxy (systolic)
  if "bp.1s" in dset.columns:
    dset["blood_pressure"] = pd.to_numeric(dset["bp.1s"], errors="coerce")
  else:
    dset["blood_pressure"] = np.nan

  # Required canonicals not present -> sentinel/NaN
  dset["gender"] = dset["gender"] if "gender" in dset.columns else "unknown"
  dset["age"] = pd.to_numeric(dset["age"], errors="coerce") if "age" in dset.columns else np.nan

  dset["hypertension"] = "unknown"
  dset["heart_disease"] = "unknown"
  dset["smoking_history"] = "unknown"
  dset["insulin"] = np.nan

  dset[TARGET] = np.nan  # no label

  return ensure_columns(dset, include_target=True)


def load_diabetes_readmission(path: Union[str, Path]) -> pd.DataFrame:
  dset = pd.read_csv(path)

  # Gender
  if "gender" not in dset.columns:
    dset["gender"] = "unknown"
  else:
    dset["gender"] = dset["gender"].fillna("unknown")

  # Age ranges like "[50-60)" -> midpoint 55
  def _age_midpoint(x: Any) -> float:
    if not isinstance(x, str):
      return np.nan
    x = x.strip()
    # common formats: "[50-60)" or "50-60"
    x = x.replace("[", "").replace(")", "")
    parts = x.split("-")
    if len(parts) != 2:
      return np.nan
    try:
      lo = float(parts[0])
      hi = float(parts[1])
      return (lo + hi) / 2.0
    except Exception:
      return np.nan

  if "age" in dset.columns:
    dset["age"] = dset["age"].apply(_age_midpoint)
  else:
    dset["age"] = np.nan

  # max_glu_serum -> glucose (approx)
  glu_map = {"None": np.nan, "Norm": 100.0, ">200": 250.0, ">300": 350.0}
  if "max_glu_serum" in dset.columns:
    dset["glucose"] = dset["max_glu_serum"].map(glu_map).astype(float)
  else:
    dset["glucose"] = np.nan

  # A1Cresult -> hba1c (approx)
  a1c_map = {"None": np.nan, "Norm": 5.5, ">7": 7.5, ">8": 8.5}
  if "A1Cresult" in dset.columns:
    dset["hba1c"] = dset["A1Cresult"].map(a1c_map).astype(float)
  else:
    dset["hba1c"] = np.nan

  # insulin categorical -> numeric code
  ins_map = {"No": 0.0, "Steady": 1.0, "Up": 2.0, "Down": -1.0}
  if "insulin" in dset.columns:
    dset["insulin"] = dset["insulin"].map(ins_map)
  else:
    dset["insulin"] = np.nan

  # Not available in this dataset -> sentinel/NaN
  dset["bmi"] = np.nan
  dset["blood_pressure"] = np.nan
  dset["hypertension"] = "unknown"
  dset["heart_disease"] = "unknown"
  dset["smoking_history"] = "unknown"

  dset[TARGET] = np.nan  # no label

  return ensure_columns(dset, include_target=True)



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
    steps = [
      ("imputer", SimpleImputer(strategy = "median")),
      ("scaler", StandardScaler()),
    ]
  )

  categorical_pipleine = Pipeline(
    steps = [
      ("imputer", SimpleImputer(strategy = "most_frequent")),
      ("encoder", OneHotEncoder(handle_unknown = "ignore")),
    ]
  )

  preprocessor = ColumnTransformer(
    transformers = [
      ("num", numeric_pipeline, numeric_features),
      ("cat", categorical_pipleine, categorical_features),
    ]
  )
  return preprocessor



# DataHandling Agent -----
class DataHandlingAgent:
  def __init__(self, preprocessor: ColumnTransformer, feature_groups: Optional[Dict[str, list]] = None):
    self.preprocessor = preprocessor
    self.feature_groups = feature_groups or FEATURE_GROUPS


  # Public API called by orchestrator
  def process(self, ref: Union[pd.DataFrame, str, Path, Dict[str, Any]], *, mode: Mode, dataset_kind: Optional[DatasetKind] = None, include_target: bool = False) -> DataHandlingOutput:
    if mode == "batch":
      dset = self.resolve_batch_ref(
        ref, 
        dataset_kind = dataset_kind, 
        include_target = include_target
      )
      return self.process_batch(dset, dataset_kind=dataset_kind, )
    elif mode == "single":
      return self.process_single_ref(ref, dataset_kind=dataset_kind,)
    else:
      raise ValueError(f"Unknown mode: {mode}")
    

  def process_dataset(self, ref: Union[pd.DataFrame, str, Path], *, dataset_kind: Optional[DatasetKind] = None, include_target: bool = False) -> DataHandlingOutput:
    return self.process(
      ref,
      mode = "batch",
      dataset_kind = dataset_kind,
      include_target = include_target,
    )
  

  def process_single(self, payload: Dict[str, Any]) -> DataHandlingOutput:
    return self.process(
      payload,
      mode = "single",
      # dataset_kind = None,
    )


  # internal: resloving refs & mapping to canonical
  def resolve_batch_ref(self, ref: Union[pd.DataFrame, str, Path], *, dataset_kind: Optional[DatasetKind], include_target: bool) -> pd.DataFrame:
    if isinstance(ref, pd.DataFrame):
      dset = ref.copy()

      needed = FEATURES + ([TARGET] if include_target and TARGET in dset.columns else [])
      missing = [c for c in needed if c not in dset.columns]
      if missing:
        raise ValueError(
          f"Batch DataFrame missing canonical columns: {missing}."
          f"Map to canonical schema before passing to dataset_kind + path."
        )
      return dset[needed]
    
    if dataset_kind is None:
      raise ValueError("dataset_kind is required when ref is a path/string.")

    path = Path(ref)
    if not path.exists():
      raise FileNotFoundError(f"Dataset path {path} does not exist.")
    
    if dataset_kind == "diabetes_prediction":
      dset = load_diabetes_prediction(path)
    elif dataset_kind == "pima":
      dset = load_pima(path)
    elif dataset_kind == "mohammed":
      dset = load_mohammed(path)
    elif dataset_kind == "readmission":
      dset = load_diabetes_readmission(path)
    else:
      raise ValueError(f"Unknown dataset_kind: {dataset_kind}")
    
    if include_target:
      return dset
    return dset[FEATURES]
  

  def process_single_ref(self, ref: Union[Dict[str, Any], pd.DataFrame, str, Path], *, dataset_kind: Optional[DatasetKind]) -> DataHandlingOutput:
    if isinstance(ref, dict):
      return self.process_single_payload(ref)
    
    if isinstance(ref, pd.DataFrame):
      if len(ref) != 1:
        raise ValueError("Single mode expects exactly one row in patient DataFrame.")
      
      missing = [c for c in FEATURES if c not in ref.columns]
      if missing:
        raise ValueError(f"Single DataFrame missing canonical columns: {missing}")
      x = ref[FEATURES].copy()
      x_norm, flags = self._normalise_units_and_flags(x)
      cleaned = self._transform_to_df(x_norm, index = x_norm.index)
      view_c, view_m = self._make_views(x_norm, cleaned)
      return DataHandlingOutput(
        raw = x_norm,
        cleaned = cleaned,
        views = view_c,
        views_model = view_m,
        flags = flags,
        validation_errors =  None,
        meta = {"mode": "single", "source": "dataframe"},
      )
    
    dset = self.resolve_batch_ref(ref, dataset_kind = dataset_kind, include_target = False)
    if len(dset) < 1:
      raise ValueError("Resolved dataset is empty.")
    first_row = dset.iloc[[0]].copy()
    return self.process_single_ref(first_row, dataset_kind = dataset_kind)
  


  # internal: batch/single processing
  def process_batch(self,dset: pd.DataFrame, *, dataset_kind: Optional[str]) -> DataHandlingOutput:
    x = dset[FEATURES].copy()

    x_norm, flags = self._normalise_units_and_flags(x)
    cleaned = self._transform_to_df(x_norm, index = x_norm.index)
    view_c, view_m = self._make_views(x_norm, cleaned)
    return DataHandlingOutput(
      raw = x_norm,
      cleaned = cleaned,
      views = view_c,
      views_model = view_m,
      flags = flags,
      validation_errors = None,
      meta = {
        "mode": "batch", 
        "dataset_kind": dataset_kind, 
        "n_rows": int(x_norm.shape[0]),
        "n_features_in": int(x_norm.shape[1]),
        "n_features_out": int(cleaned.shape[1]),
      }
    )
  

  def process_single_payload(self, payload: Dict[str, Any]) -> DataHandlingOutput:
    try:
      validated = PatientInput(**payload)
      raw = pd.DataFrame([validated.model_dump()])
      errors = None
    except ValidationError as e:
      raw = pd.DataFrame([payload])
      errors = e.errors()

    # ensure canonnical columns exist (missing NaN)
    for col in FEATURES:
      if col not in raw.columns:
        raw[col] = np.nan

    x = raw[FEATURES].copy()
    x_norm, flags = self._normalise_units_and_flags(x)

    cleaned = self._transform_to_df(x_norm, index = x_norm.index)
    view_c, view_m = self._make_views(x_norm, cleaned)

    return DataHandlingOutput(
      raw = x_norm,
      cleaned = cleaned,
      views = view_c,
      views_model = view_m,
      flags = flags,
      validation_errors = errors,
      meta = {"mode": "single", "source": "payload"},
    )



  # --------------------------
  # internal helpers 
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

    # ---- force categorical columns to strings (prevents OneHotEncoder mixed type crash) ----
    for col in ["gender", "hypertension", "heart_disease", "smoking_history"]:
      if col in dset.columns:
        dset[col] = dset[col].astype("string").fillna("unknown")


    return dset, flags


  def _transform_to_df(self, x_norm: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    check_is_fitted(self.preprocessor)

    arr = self.preprocessor.transform(x_norm)
    if hasattr(self.preprocessor, "get_feature_names_out"):
      cols = self.preprocessor.get_feature_names_out()
      return pd.DataFrame(arr, columns=cols, index=index)
    return pd.DataFrame(arr, index=index)

  def _make_views(self, x_canonical: pd.DataFrame,x_model: pd.DataFrame,) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    views_canonical: Dict[str, pd.DataFrame] = {}
    for group, cols in self.feature_groups.items():
      present = [c for c in cols if c in x_canonical.columns]
      views_canonical[group] = x_canonical[present].copy()

    # model views: match transformed column names against canonical base features
    # e.g., "num__age" contains "age"; "cat__gender_unknown" contains "gender"
    views_model: Dict[str, pd.DataFrame] = {}
    for group, base_feats in self.feature_groups.items():
      cols = [c for c in x_model.columns if any(bf in c for bf in base_feats)]
      views_model[group] = x_model[cols].copy()

    return views_canonical, views_model
  