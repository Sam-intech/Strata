from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
# ===================================================================


TriageLabel = Literal["low", "medium", "high", "critical"]


@dataclass
class ClinicalAssessmentOutput:
  risk_T2D_now: float
  triage_label: TriageLabel
  top_contributors: Dict[str, float]          
  raw_proba_vector: Optional[Any] = None      
  meta: Optional[Dict[str, Any]] = None       


@dataclass
class ClinicalAgentConfig:
  medium: float = 0.20
  high: float = 0.50
  critical: float = 0.80
  top_k_contributors: int = 5


class ClinicalAssessmentAgent:
  def __init__(self, model: BaseEstimator, feature_names: Optional[Sequence[str]] = None, config: Optional[ClinicalAgentConfig] = None,):
    self.model = model
    self.config = config or ClinicalAgentConfig()

    self._feature_names_source: str = "unknown"

    if feature_names is not None:
      self.feature_names = list(feature_names)
      self._feature_names_source = "explicit"
    elif hasattr(model, "feature_names_in_"):
      self.feature_names = list(getattr(model, "feature_names_in_"))
      self._feature_names_source = "model_inferred"
    else:
      self.feature_names = None

    # Consider "fitted" if common sklearn fitted attributes exist
    self.is_fitted_: bool = bool(getattr(model, "classes_", None) is not None) or bool(
      getattr(model, "n_features_in_", None) is not None
    )



  # ---------------- 
  # core logic 
  def _triage_from_risk(self, p: float) -> TriageLabel:
    cfg = self.config
    if p >= cfg.critical:
      return "critical"
    if p >= cfg.high:
      return "high"
    if p >= cfg.medium:
      return "medium"
    return "low"


  def _align_row_to_feature_order(self, row: pd.Series) -> np.ndarray:
    if self.feature_names is None:
      return row.values.astype(float, copy=False)
    
    aligned = row.reindex(self.feature_names)

    missing = aligned.isna()
    if bool(missing.any()):
      missing_cols = list(aligned.index[missing])
      raise ValueError(f"Missing required features for ClinicalAssessmentAgent: {missing_cols}")
    
    return aligned.values.astype(float, copy = False)
  

  def _positive_class_index(self) -> Optional[int]:
    classes = getattr(self.model, "classes_", None)
    if classes is None:
      return None
    try:
      classes_list = list(classes)
      return classes_list.index(1)
    except ValueError:
      return None


  def _base_feature_from_transformed(self, name: str) -> str:
    # strip transformer prefix if present
    if "__" in name:
      name = name.split("__", 1)[1]

    # match the longest canonical feature prefix
    candidates = sorted(
      ["gender", "age", "bmi", "glucose", "hba1c", "blood_pressure", "hypertension", "heart_disease", "smoking_history", "insulin"],
      key=len,
      reverse=True,
    )
    for feat in candidates:
      if name == feat or name.startswith(feat + "_"):
        return feat

    # fallback
    return name
  

  def _humanize_feature_name(self, name: str) -> str:
    # strip transformer prefix
    if "__" in name:
      name = name.split("__", 1)[1]

    # categorical: feature_categoryValue
    base = self._base_feature_from_transformed(name)
    if name.startswith(base + "_"):
      category = name[len(base) + 1 :]
      return f"{base}={category}"

    # numeric: just base
    return base



  def _top_contributors(self, x_row: np.ndarray, *, missing_flags: Optional[Dict[str, bool]] = None) -> Dict[str, float]:
    names_raw = self.feature_names or [f"f_{i}" for i in range(len(x_row))]
    k = self.config.top_k_contributors
    missing_flags = missing_flags or {}

    # Linear
    if hasattr(self.model, "coef_"):
      coefs = getattr(self.model, "coef_")
      coefs_1 = coefs[0] if isinstance(coefs, np.ndarray) and coefs.ndim >= 2 else np.asarray(coefs).reshape(-1)

      contribs = coefs_1 * x_row
      idx_sorted = np.argsort(np.abs(contribs))[::-1]

      out: Dict[str, float] = {}
      for i in idx_sorted:
        base = self._base_feature_from_transformed(names_raw[i])
        if missing_flags.get(f"{base}_missing", False):
          continue
        out[self._humanize_feature_name(names_raw[i])] = float(contribs[i])
        if len(out) >= k:
          break
      return out

    # Tree-style importance
    if hasattr(self.model, "feature_importances_"):
      imps = np.asarray(getattr(self.model, "feature_importances_")).reshape(-1)
      idx_sorted = np.argsort(imps)[::-1]

      out: Dict[str, float] = {}
      for i in idx_sorted:
        base = self._base_feature_from_transformed(names_raw[i])
        if missing_flags.get(f"{base}_missing", False):
          continue
        out[self._humanize_feature_name(names_raw[i])] = float(imps[i])
        if len(out) >= k:
          break
      return out

    return {}




  # ---------------- 
  # public API 
  def predict_single(self, cleaned_row: pd.Series, *, missing_flags: Optional[Dict[str, bool]] = None) -> ClinicalAssessmentOutput:
    assert self.is_fitted_, "ClinicalAssessmentAgent is not fitted (loaded model missing fitted attributes)."

    x_row = np.asarray(cleaned_row).reshape(-1) if not isinstance(cleaned_row, pd.Series) else self._align_row_to_feature_order(cleaned_row)
    x = x_row.reshape(1, -1)

    raw_proba = None
    if hasattr(self.model, "predict_proba"):
      proba_vec = self.model.predict_proba(x)[0]
      raw_proba = proba_vec
      pos_idx = self._positive_class_index()
      if pos_idx is None:
        pos_idx = 1 if len(proba_vec) > 1 else 0
      risk = float(proba_vec[pos_idx])
    else:
      if not hasattr(self.model, "decision_function"):
        raise RuntimeError("Model lacks predict_proba and decision_function; cannot produce risk.")
      score = float(self.model.decision_function(x)[0])
      risk = float(1.0 / (1.0 + np.exp(-score)))

    triage = self._triage_from_risk(risk)
    contribs = self._top_contributors(x_row, missing_flags=missing_flags)

    return ClinicalAssessmentOutput(
      risk_T2D_now=risk,
      triage_label=triage,
      top_contributors=contribs,
      raw_proba_vector=raw_proba,
      meta={
        "model_class": self.model.__class__.__name__,
        "feature_names_source": self._feature_names_source,
      },
    )

  

  def predict_batch(self, cleaned_features: pd.DataFrame) -> pd.DataFrame:
    assert self.is_fitted_, "ClinicalAssessmentAgent is not fitted."

    if isinstance(cleaned_features, pd.DataFrame) and self.feature_names is not None:
      Xdf = cleaned_features.reindex(columns=self.feature_names)
      if bool(Xdf.isna().any().any()):
        # fail loudly; DataHandlingAgent should ensure this doesn't happen
        raise ValueError("NaNs detected after aligning feature order in predict_batch().")
      X = Xdf.values
    else:
      X = np.asarray(cleaned_features)

    if hasattr(self.model, "predict_proba"):
      pos_idx = self._positive_class_index()
      proba_all = self.model.predict_proba(X)
      if pos_idx is None:
        pos_idx = 1 if proba_all.shape[1] > 1 else 0
      proba = proba_all[:, pos_idx]
    else:
      score = np.asarray(self.model.decision_function(X)).reshape(-1)
      proba = 1.0 / (1.0 + np.exp(-score))

    triage = [self._triage_from_risk(float(p)) for p in proba]
    return pd.DataFrame(
      {"risk_T2D_now": proba, "triage_label": triage},
      index=getattr(cleaned_features, "index", None),
    )
  

