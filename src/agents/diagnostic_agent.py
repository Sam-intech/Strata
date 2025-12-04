from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
import math
# ===============================================================================================================


DiagnosisLabel = Literal["normal", "high_risk", "T2D", "uncertain"]


@dataclass
class DiagnosticConfig:
  """
  Config for DiagnosticAgent:
  - lab thresholds
  - borderline margins
  - confidence calibration
  """
  # HbA1c thresholds (mmol/mol)
  hba1c_diabetes: float = 48.0
  hba1c_high_risk_low: float = 42.0  # "pre-diabetes" / high-risk band

  # FPG thresholds (mmol/L)
  fpg_diabetes: float = 7.0
  fpg_high_risk_low: float = 6.1

  # OGTT 2h thresholds (mmol/L)
  ogtt_diabetes: float = 11.1
  ogtt_high_risk_low: float = 7.8

  # Borderline margin around thresholds (for "uncertain")
  borderline_margin: float = 0.3
  # How much to weight pre-test (clinical) risk in final confidence  pretest_weight: float = 0.3  # 0–1


@dataclass
class LabEvidence:
  """
  Flattened, validated labs from LaboratoryAgent.
  Assume units already standardised to mmol/mol + mmol/L.
  """
  hba1c_mmol_mol: Optional[float] = None
  fpg_mmol_l: Optional[float] = None
  ogtt_2h_mmol_l: Optional[float] = None
  random_glucose_mmol_l: Optional[float] = None

  # flags from LaboratoryAgent
  is_self_report_only: bool = False
  is_outdated: bool = False
  has_quality_issues: bool = False

  raw_meta: Optional[Dict[str, Any]] = None


@dataclass
class DiagnosticContext:
  """
  Context flags from DataHandlingAgent, etc.
  Used to handle edge cases where HbA1c is unreliable.
  """
  pregnancy: bool = False
  anaemia: bool = False
  haemoglobinopathy: bool = False
  ckd: bool = False

  raw_meta: Optional[Dict[str, Any]] = None


@dataclass
class ClinicalAssessmentSnapshot:
  """
  Minimal subset of ClinicalAssessmentAgent output needed here.
  You can swap this for the real ClinicalAssessmentOutput class.
  """
  risk_T2D_now: float
  triage_label: Optional[str] = None
  raw_proba_vector: Optional[Any] = None
  meta: Optional[Dict[str, Any]] = None


@dataclass
class DiagnosticResult:
  label: DiagnosisLabel
  confidence: float
  next_step: str
  basis: str
  reasoning_tokens: Dict[str, Any]


class DiagnosticAgent:
  """
  Hybrid rule-based diagnostic engine for T2D.

  Inputs:
    - LabEvidence (current + key historical)
    - ClinicalAssessmentSnapshot (pre-test probability)
    - DiagnosticContext (conditions that affect lab interpretation)

  Output:
    - DiagnosticResult with label, confidence, next_step, and reasoning tokens.
  """

  def __init__(self, config: Optional[DiagnosticConfig] = None):
    self.config = config or DiagnosticConfig()


  # Public API -----
  def diagnose(self, labs: LabEvidence, clinical: ClinicalAssessmentSnapshot, ctx: Optional[DiagnosticContext] = None,) -> DiagnosticResult:
    ctx = ctx or DiagnosticContext()

    # Decide which lab source is trusted / primary
    primary_basis = self._select_primary_basis(labs, ctx)

    if primary_basis is None:
      # No usable labs → fall back on risk only
      return self._diagnose_from_risk_only(clinical, labs, ctx)

    if primary_basis == "HbA1c":
      return self._diagnose_from_hba1c(labs, clinical, ctx)
    if primary_basis == "FPG":
      return self._diagnose_from_fpg(labs, clinical, ctx)
    if primary_basis == "OGTT":
      return self._diagnose_from_ogtt(labs, clinical, ctx)

    # Shouldn't happen, but humans also say that before production
    return self._diagnose_from_risk_only(clinical, labs, ctx)


  # Basis selection -----
  def _select_primary_basis(self, labs: LabEvidence, ctx: DiagnosticContext) -> Optional[str]:
    """
    Choose which lab to anchor the decision on, respecting context.

    Simplified logic:
      - Avoid HbA1c if haemoglobin issues / pregnancy.
      - Prefer HbA1c if available & reliable.
      - Else prefer FPG, then OGTT.
    """

    hba1c_reliable = (
      labs.hba1c_mmol_mol is not None
      and not ctx.pregnancy
      and not ctx.anaemia
      and not ctx.haemoglobinopathy
    )
    if hba1c_reliable:
      return "HbA1c"

    if labs.fpg_mmol_l is not None:
      return "FPG"

    if labs.ogtt_2h_mmol_l is not None:
      return "OGTT"

    # No structured diagnostic labs available
    return None


  # HbA1c-driven diagnosis ------
  def _diagnose_from_hba1c(self, labs: LabEvidence, clinical: ClinicalAssessmentSnapshot, ctx: DiagnosticContext,) -> DiagnosticResult:
    val = labs.hba1c_mmol_mol
    cfg = self.config

    if val is None:
      return self._diagnose_from_risk_only(clinical, labs, ctx)

    reasoning = {
      "basis": "HbA1c",
      "value": val,
      "diabetes_threshold": cfg.hba1c_diabetes,
      "high_risk_low": cfg.hba1c_high_risk_low,
    }

    # Borderline band around the diagnostic threshold
    if self._is_borderline(val, cfg.hba1c_diabetes, cfg.borderline_margin):
      label = "uncertain"
      next_step = "repeat_HbA1c_or_use_FPG_OGTT"
    elif val >= cfg.hba1c_diabetes:
      label = "T2D"
      next_step = "confirm_diagnosis_and_initiate_management"
    elif val >= cfg.hba1c_high_risk_low:
      label = "high_risk"
      next_step = "lifestyle_intervention_and_repeat_testing"
    else:
      label = "normal"
      next_step = "routine_monitoring"

    conf = self._calibrate_confidence_from_distance(
      value=val,
      threshold=cfg.hba1c_diabetes,
      pretest_risk=clinical.risk_T2D_now,
    )

    reasoning.update(
      {
        "chosen_label": label,
        "pretest_risk": clinical.risk_T2D_now,
        "borderline_margin": cfg.borderline_margin,
        "lab_flags": {
          "is_self_report_only": labs.is_self_report_only,
          "is_outdated": labs.is_outdated,
          "has_quality_issues": labs.has_quality_issues,
        },
      }
    )

    conf = self._penalise_for_lab_quality(conf, labs)

    return DiagnosticResult(
      label=label,
      confidence=conf,
      next_step=next_step,
      basis="HbA1c",
      reasoning_tokens=reasoning,
    )


  # FPG-driven diagnosis -----
  def _diagnose_from_fpg(self, labs: LabEvidence, clinical: ClinicalAssessmentSnapshot, ctx: DiagnosticContext,) -> DiagnosticResult:
    val = labs.fpg_mmol_l
    cfg = self.config

    if val is None:
      return self._diagnose_from_risk_only(clinical, labs, ctx)

    reasoning = {
      "basis": "FPG",
      "value": val,
      "diabetes_threshold": cfg.fpg_diabetes,
      "high_risk_low": cfg.fpg_high_risk_low,
    }

    if self._is_borderline(val, cfg.fpg_diabetes, cfg.borderline_margin):
      label = "uncertain"
      next_step = "repeat_FPG_or_perform_OGTT"
    elif val >= cfg.fpg_diabetes:
      label = "T2D"
      next_step = "confirm_diagnosis_and_initiate_management"
    elif val >= cfg.fpg_high_risk_low:
      label = "high_risk"
      next_step = "lifestyle_intervention_and_repeat_testing"
    else:
      label = "normal"
      next_step = "routine_monitoring"

    conf = self._calibrate_confidence_from_distance(
      value=val,
      threshold=cfg.fpg_diabetes,
      pretest_risk=clinical.risk_T2D_now,
    )
    reasoning.update(
      {
        "chosen_label": label,
        "pretest_risk": clinical.risk_T2D_now,
        "borderline_margin": cfg.borderline_margin,
        "lab_flags": {
          "is_self_report_only": labs.is_self_report_only,
          "is_outdated": labs.is_outdated,
          "has_quality_issues": labs.has_quality_issues,
        },
      }
    )

    conf = self._penalise_for_lab_quality(conf, labs)

    return DiagnosticResult(
      label=label,
      confidence=conf,
      next_step=next_step,
      basis="FPG",
      reasoning_tokens=reasoning,
    )


  # OGTT-driven diagnosis -----
  def _diagnose_from_ogtt(self, labs: LabEvidence, clinical: ClinicalAssessmentSnapshot, ctx: DiagnosticContext,) -> DiagnosticResult:
    val = labs.ogtt_2h_mmol_l
    cfg = self.config

    if val is None:
      return self._diagnose_from_risk_only(clinical, labs, ctx)

    reasoning = {
      "basis": "OGTT_2h",
      "value": val,
      "diabetes_threshold": cfg.ogtt_diabetes,
      "high_risk_low": cfg.ogtt_high_risk_low,
    }

    if self._is_borderline(val, cfg.ogtt_diabetes, cfg.borderline_margin):
      label = "uncertain"
      next_step = "repeat_OGTT_or_use_alternative_test"
    elif val >= cfg.ogtt_diabetes:
      label = "T2D"
      next_step = "confirm_diagnosis_and_initiate_management"
    elif val >= cfg.ogtt_high_risk_low:
      label = "high_risk"
      next_step = "lifestyle_intervention_and_repeat_testing"
    else:
      label = "normal"
      next_step = "routine_monitoring"

    conf = self._calibrate_confidence_from_distance(
      value=val,
      threshold=cfg.ogtt_diabetes,
      pretest_risk=clinical.risk_T2D_now,
    )
    reasoning.update(
      {
        "chosen_label": label,
        "pretest_risk": clinical.risk_T2D_now,
        "borderline_margin": cfg.borderline_margin,
        "lab_flags": {
          "is_self_report_only": labs.is_self_report_only,
          "is_outdated": labs.is_outdated,
          "has_quality_issues": labs.has_quality_issues,
        },
      }
    )

    conf = self._penalise_for_lab_quality(conf, labs)

    return DiagnosticResult(
      label=label,
      confidence=conf,
      next_step=next_step,
      basis="OGTT",
      reasoning_tokens=reasoning,
    )


  # Risk-only fallback -----
  def _diagnose_from_risk_only(self, clinical: ClinicalAssessmentSnapshot, labs: LabEvidence, ctx: DiagnosticContext,) -> DiagnosticResult:
    """
    Used when there are no usable diagnostic labs.
    This is explicitly "early risk" territory.
    """
    p = clinical.risk_T2D_now

    if p >= 0.8:
      label = "high_risk"
      next_step = "order_diagnostic_labs"
    elif p >= 0.4:
      label = "high_risk"
      next_step = "lifestyle_intervention_and_consider_labs"
    elif p >= 0.2:
      label = "normal"
      next_step = "monitor_and_reassess_risk"
    else:
      label = "normal"
      next_step = "routine_monitoring"

    conf = 0.5 + 0.4 * p  # very rough, capped <= 0.9
    conf = min(conf, 0.9)

    reasoning = {
      "basis": "risk_only",
      "pretest_risk": p,
      "lab_available": {
        "hba1c": labs.hba1c_mmol_mol is not None,
        "fpg": labs.fpg_mmol_l is not None,
        "ogtt_2h": labs.ogtt_2h_mmol_l is not None,
      },
      "context_flags": {
        "pregnancy": ctx.pregnancy,
        "anaemia": ctx.anaemia,
        "haemoglobinopathy": ctx.haemoglobinopathy,
        "ckd": ctx.ckd,
      },
    }

    return DiagnosticResult(
      label=label,
      confidence=conf,
      next_step=next_step,
      basis="risk_only",
      reasoning_tokens=reasoning,
    )


  # Helpers -----
  @staticmethod
  def _is_borderline(value: float, threshold: float, margin: float) -> bool:
    return abs(value - threshold) <= margin

  def _calibrate_confidence_from_distance(self, value: float, threshold: float, pretest_risk: float,) -> float:
    """
    Combine distance from threshold + pretest risk into a [0,1] confidence.
    Simple, explicit and easy to explain in the write-up.
    """
    dist = abs(value - threshold)
    # squash distance into [0,1] using a simple logistic-like curve
    lab_component = 1.0 / (1.0 + math.exp(-dist))

    # normalise lab_component to ~[0.5, 0.95]
    lab_conf = 0.5 + 0.45 * (lab_component - 0.5) / 0.5
    lab_conf = max(0.5, min(lab_conf, 0.95))

    w = self.config.pretest_weight
    combined = (1 - w) * lab_conf + w * pretest_risk
    return max(0.0, min(combined, 0.99))

  @staticmethod
  def _penalise_for_lab_quality(confidence: float, labs: LabEvidence) -> float:
    penalty = 0.0
    if labs.is_self_report_only:
      penalty += 0.1
    if labs.is_outdated:
      penalty += 0.1
    if labs.has_quality_issues:
      penalty += 0.15
        
    return max(0.0, confidence - penalty)
