# data_handling/agent.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator


# 1. Canonical input schema for a single patient
class PatientInput(BaseModel):
  # Demographics
  age: int = Field(ge=0, le=120)
  sex: str  # "male", "female", etc.

  # Anthropometrics
  bmi: float = Field(gt=0)
  waist_hip_ratio: Optional[float] = Field(default=None, gt=0)

  # Vitals
  systolic_bp: Optional[float] = None
  diastolic_bp: Optional[float] = None

  # Risk factors
  family_history: Optional[bool] = None
  physical_activity_level: Optional[str] = None
  smoking_status: Optional[str] = None
  alcohol_intake: Optional[str] = None

  # Symptoms
  polyuria: Optional[bool] = None
  polydipsia: Optional[bool] = None
  weight_loss: Optional[bool] = None
  blurred_vision: Optional[bool] = None
  fatigue: Optional[bool] = None

  # Laboratory
  hba1c: Optional[float] = None
  fasting_glucose: Optional[float] = None
  random_glucose: Optional[float] = None
  ogtt_2h_glucose: Optional[float] = None


  @field_validator("sex")
  @classmethod
  def normalise_sex(cls, v: str) -> str:
    v = v.strip().lower()
    mapping = {"m": "male", "f": "female"}
    return mapping.get(v, v)


# 2. Feature group configuration
FEATURE_GROUPS = {
  "clinical": [
    "age",
    "sex",
    "bmi",
    "waist_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "family_history",
    "physical_activity_level",
    "smoking_status",
    "alcohol_intake",
    "polyuria",
    "polydipsia",
    "weight_loss",
    "blurred_vision",
    "fatigue",
  ],
  "lab": [
    "hba1c",
    "fasting_glucose",
    "random_glucose",
    "ogtt_2h_glucose",
  ],
}


@dataclass
class DataHandlingOutput:
  raw: pd.DataFrame
  cleaned: pd.DataFrame
  views: Dict[str, pd.DataFrame]
  validation_errors: Optional[Dict[str, Any]] = None


class DataHandlingAgent:
  """
  Central gatekeeper for all patient data.

  It:
  - validates raw input
  - applies the pre-fitted preprocessing pipeline
  - splits features into views per agent
  """

  def __init__(self, preprocessor, feature_groups: Dict[str, list] = None):
    """
    preprocessor: fitted sklearn Pipeline / ColumnTransformer
    feature_groups: mapping from agent name -> list of canonical feature names
    """
    self.preprocessor = preprocessor
    self.feature_groups = feature_groups or FEATURE_GROUPS


  # ---------- Public API ----------
  def ingest_single(self, raw_input: Dict[str, Any]) -> DataHandlingOutput:
    """Used by the UI for a single patient."""
    try:
      validated = PatientInput(**raw_input)
      df_raw = pd.DataFrame([validated.model_dump()])
      errors = None
    except ValidationError as e:
      # Still try to build a DataFrame from partial data
      df_raw = pd.DataFrame([raw_input])
      errors = e.errors()

    df_clean = self._preprocess(df_raw)
    views = self._split_views(df_clean)

    return DataHandlingOutput(
      raw=df_raw,
      cleaned=df_clean,
      views=views,
      validation_errors=errors,
    )
  
  def ingest_batch(self, df_raw: pd.DataFrame) -> DataHandlingOutput:
    """Used during training/evaluation on secondary datasets."""
    # Assumes df_raw columns already roughly mapped to canonical names
    df_clean = self._preprocess(df_raw)
    views = self._split_views(df_clean)

    return DataHandlingOutput(
      raw=df_raw,
      cleaned=df_clean,
      views=views,
      validation_errors=None,
    )


  # ---------- Internal helpers ----------
  def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply fitted preprocessing: imputation, encoding, scaling.
    The preprocessor must be trained separately on your training data.
    """
    transformed = self.preprocessor.transform(df)
    if hasattr(self.preprocessor, "get_feature_names_out"):
      cols = self.preprocessor.get_feature_names_out()
      return pd.DataFrame(transformed, columns=cols)
    # Fallback: keep as numpy array with generic column names
    return pd.DataFrame(transformed)
  
  def _split_views(self, df_clean: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Selects columns for each agent based on canonical feature names.
    For encoded features, you may need a mapping from original -> encoded names.
    """
    views = {}
    for agent, features in self.feature_groups.items():
      cols = [c for c in df_clean.columns if any(f in c for f in features)]
      views[agent] = df_clean[cols].copy()
    return views
