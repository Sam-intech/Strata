from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
# ============================================================================================


TriageLabel = Literal["low_risk", "routine_follow_up", "high_risk"]

@dataclass
class ClinicalAssessmentOutput: 
  # what the clinical agent passess forward 
  risk_T2D_now: float
  triage_label: TriageLabel
  top_contributors: Dict[str, float]
  raw_proba_vector: Optional[np.ndarray] = None
  meta: Optional[Dict[str, Any]] = None



class ClinicalAssessmentAgent:
  # ML-based probabilistic risk estimator.
  def __init__(self, model: BaseEstimator, feature_names: Optional[List[str]] = None, triage_thresholds: Optional[Dict[str, float]] = None):
    """
      model: any sklearn-style estimator with predict_proba
      feature_names: column names for X, used for explanations
      triage_thresholds: dict with keys:
        - low: below this = low_risk
        - high: above this = high_risk
        in-between = routine_follow_up
    """
    self.model = model
    self.feature_names = feature_names

    default_thresholds = {"low": 0.2, "high": 0.5}
    self.triage_thresholds = triage_thresholds or default_thresholds

    self.is_fitted_ = False


  # Training -----
  def fit(self, x: pd.DataFrame, y: pd.Series) -> "ClinicalAssessmentAgent":
    if isinstance(x, pd.DataFrame):
      self.feature_names = list(x.columns)
      x_arr = x.values
    else: 
      x_arr = np.asarray(x)

    self.model.fit(x_arr, y)
    self.is_fitted_ = True
    return self



  # core logic -----
  def _triage_from_risk(self, p:float) -> TriageLabel:
    low = self.triage_thresholds["low"]
    high = self.triage_thresholds["high"]

    if p >= high:
      return "high_risk"
    if p >= low:
      return "routine_follow_up"
    return "low_risk"

  def _linear_contributions(self, x_row:np.ndarray) -> Dict[str, float]:
    if not hasattr(self.model, "coef_"):
      return{}
    coefs = self.model.coef_[0]
    contribs = coefs * x_row

    if self.feature_names is None:
      names = [f"f_{i}" for i in range(len(contribs))]
    else:
      names = self.feature_names

    idx = np.argsort(np.abs(contribs))[::-1][:5]
    return {names[i]: float(contribs[i]) for i in idx}


  # inference -----
  def predict_batch(self, cleaned_features: pd.DataFrame) -> pd.DataFrame:
    assert self.is_fitted_, "ClinicalAssessmentAgent is not fitted"

    if isinstance(cleaned_features, pd.DataFrame):
      x_arr = cleaned_features.values
    else:
      x_arr = np.asarray(cleaned_features)

    proba = self.model.predict_proba(x_arr)[:, 1]

    triage_labels = [self._triage_from_risk(p) for p in proba]

    return pd.DataFrame(
      {
        "risk_T2D_now": proba,
        "triage_label": triage_labels,
      },
      index = getattr(cleaned_features, "index", None),
    )


  def predict_single(self, cleaned_row: pd.Series) -> ClinicalAssessmentOutput:
    assert self.is_fitted_, "ClinicalAssessmentAgent is not fitted"

    if isinstance(cleaned_row, pd.Series):
      x = cleaned_row.values.reshape(1, -1)
    else:
      x = np.asarray(cleaned_row).reshape(1, -1)

    proba_vec = self.model.predict_proba(x)[0]
    risk_T2D_now = float(proba_vec[1])
    triage = self._triage_from_risk(risk_T2D_now)

    contribs = self._linear_contributions(x[0])

    return ClinicalAssessmentOutput(
      risk_T2D_now = risk_T2D_now,
      triage_label = triage,
      top_contributors = contribs,
      raw_proba_vector = proba_vec,
      meta = {"model_class": self.model.__class__.__name__},
    )