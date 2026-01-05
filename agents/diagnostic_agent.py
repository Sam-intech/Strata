from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal, TypedDict, NotRequired 
import math
# ===============================================================================================================


DiagnosisLabel = Literal["normal", "high_risk", "T2D", "uncertain"]
DiagnosticBasis = Literal["HbA1c", "FPG", "OGTT_2h", "risk_only", "none"]
DiagnosticNextStep = Literal[
  "routine_monitoring",
  "monitor_and_reassess_risk",
  "order_diagnostic_labs",
  "lifestyle_intervention_and_consider_labs",
  "lifestyle_intervention_and_repeat_testing",
  "repeat_HbA1c_or_use_FPG_OGTT",
  "repeat_FPG_or_perform_OGTT",
  "repeat_OGTT_or_use_alternative_test",
  "confirm_diagnosis_and_initiate_management",
]


class OrchestratorState(TypedDict, total=False):
  # from ClinicalAssessmentAgent
  clinical_output: NotRequired[Dict[str, Any]]  # expects at least: {"risk_T2D_now": float, ...}

  # from LaboratoryAgent
  lab_output: NotRequired[Dict[str, Any]]  # expects labs + flags (see adapter)

  # optional context (from data handler / UI / patient profile)
  context: NotRequired[Dict[str, Any]]  # pregnancy/anaemia/haemoglobinopathy/ckd etc.

  # where this agent writes
  diagnostic_output: NotRequired[Dict[str, Any]]

  # optional: internal debug / trace
  trace: NotRequired[list]



# --------------------------------
# config + Evidence models
@dataclass
class DiagnosticConfig:
  hba1c_diabetes: float = 48.0
  hba1c_high_risk_low: float = 42.0  
  fpg_diabetes: float = 7.0
  fpg_high_risk_low: float = 6.1
  ogtt_diabetes: float = 11.1
  ogtt_high_risk_low: float = 7.8
  borderline_margin: float = 0.3
  pretest_weight: float = 0.3  


@dataclass
class LabEvidence:
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
  pregnancy: bool = False
  anaemia: bool = False
  haemoglobinopathy: bool = False
  ckd: bool = False

  raw_meta: Optional[Dict[str, Any]] = None


@dataclass
class ClinicalAssessmentSnapshot:
  risk_T2D_now: float
  triage_label: Optional[str] = None
  raw_proba_vector: Optional[Any] = None
  meta: Optional[Dict[str, Any]] = None


@dataclass
class DiagnosticResult:
  label: DiagnosisLabel
  confidence: float
  next_step: DiagnosticNextStep
  basis: DiagnosticBasis
  reasoning_tokens: Dict[str, Any]



# ---------------------------
class DiagnosticAgent:
  def __init__(self, config: Optional[DiagnosticConfig] = None, enable_trace: bool = False):
    self.config = config or DiagnosticConfig()
    self.enable_trace = enable_trace


  # Orchestrator-facing entry
  def __call__(self, state: OrchestratorState) -> OrchestratorState:
    clinical = self._clinical_from_state(state.get("clinical_output"))
    labs = self._labs_from_state(state.get("lab_output"))
    ctx = self._context_from_state(state.get("context"))
    
    result = self.diagnose(labs=labs, clinical=clinical, ctx=ctx)
    state["diagnostic_output"] = {
      "label": result.label,
      "confidence": result.confidence,
      "basis": result.basis,
      "next_step": result.next_step,
      "reasoning_tokens": result.reasoning_tokens,
    }


    if self.enable_trace:
      state.setdefault("trace", []).append(
        {
          "agent": "DiagnosticAgent", 
          "output": state["diagnostic_output"]
        }
      )

    return state


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
    if primary_basis == "OGTT_2h":
      return self._diagnose_from_ogtt(labs, clinical, ctx)

    return self._diagnose_from_risk_only(clinical, labs, ctx)


  # Basis selection -----
  def _select_primary_basis(self, labs: LabEvidence, ctx: DiagnosticContext) -> Optional[Literal["HbA1c", "FPG", "OGTT_2h"]]:
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
      return "OGTT_2h"

    # No structured diagnostic labs available
    return None


  # HbA1c-driven diagnosis ------
  def _diagnose_from_hba1c(self, labs: LabEvidence, clinical: ClinicalAssessmentSnapshot, ctx: DiagnosticContext,) -> DiagnosticResult:
    val = labs.hba1c_mmol_mol
    cfg = self.config

    if val is None:
      return self._diagnose_from_risk_only(clinical, labs, ctx)

    # reasoning = {
    #   "basis": "HbA1c",
    #   "value": val,
    #   "diabetes_threshold": cfg.hba1c_diabetes,
    #   "high_risk_low": cfg.hba1c_high_risk_low,
    # }

    # Borderline band around the diagnostic threshold
    if self._is_borderline(val, cfg.hba1c_diabetes, cfg.borderline_margin):
      label: DiagnosisLabel = "uncertain"
      next_step: DiagnosticNextStep = "repeat_HbA1c_or_use_FPG_OGTT"
    elif val >= cfg.hba1c_diabetes:
      label = "T2D"
      next_step = "confirm_diagnosis_and_initiate_management"
    elif val >= cfg.hba1c_high_risk_low:
      label = "high_risk"
      next_step = "lifestyle_intervention_and_repeat_testing"
    else:
      label = "normal"
      next_step = "routine_monitoring"

    # conf = self._calibrate_confidence_from_distance(
    #   value=val,
    #   threshold=cfg.hba1c_diabetes,
    #   pretest_risk=clinical.risk_T2D_now,
    # )
    conf = self._calibrate_confidence_from_distance(val, cfg.hba1c_diabetes, clinical.risk_T2D_now)
    conf = self._penalise_for_lab_quality(conf, labs)

    reasoning = self._reasoning_pack(
      basis = "HbA1c",
      value = val,
      thresholds = {
        "diabetes": cfg.hba1c_diabetes, 
        "high_risk_low": cfg.hba1c_high_risk_low
        },
      label = label,
      clinical = clinical,
      labs = labs,
      ctx = ctx,
    )

    # conf = self._penalise_for_lab_quality(conf, labs)

    return DiagnosticResult(
      label = label,
      confidence = conf,
      next_step = next_step,
      basis = "HbA1c",
      reasoning_tokens = reasoning,
    )


  # FPG-driven diagnosis -----
  def _diagnose_from_fpg(self, labs: LabEvidence, clinical: ClinicalAssessmentSnapshot, ctx: DiagnosticContext) -> DiagnosticResult:
    val = labs.fpg_mmol_l
    cfg = self.config
    if val is None:
      return self._diagnose_from_risk_only(clinical, labs, ctx)

    if self._is_borderline(val, cfg.fpg_diabetes, cfg.borderline_margin):
      label: DiagnosisLabel = "uncertain"
      next_step: DiagnosticNextStep = "repeat_FPG_or_perform_OGTT"
    elif val >= cfg.fpg_diabetes:
      label = "T2D"
      next_step = "confirm_diagnosis_and_initiate_management"
    elif val >= cfg.fpg_high_risk_low:
      label = "high_risk"
      next_step = "lifestyle_intervention_and_repeat_testing"
    else:
      label = "normal"
      next_step = "routine_monitoring"

    conf = self._calibrate_confidence_from_distance(val, cfg.fpg_diabetes, clinical.risk_T2D_now)
    conf = self._penalise_for_lab_quality(conf, labs)

    reasoning = self._reasoning_pack(
      basis = "FPG",
      value = val,
      thresholds = {
        "diabetes": cfg.fpg_diabetes, 
        "high_risk_low": cfg.fpg_high_risk_low
        },
      label = label,
      clinical = clinical,
      labs = labs,
      ctx = ctx,
    )

    return DiagnosticResult(
      label = label, 
      confidence = conf, 
      next_step = next_step, 
      basis = "FPG", 
      reasoning_tokens=reasoning
    )


  # OGTT-driven diagnosis -----
  def _diagnose_from_ogtt(self, labs: LabEvidence, clinical: ClinicalAssessmentSnapshot, ctx: DiagnosticContext) -> DiagnosticResult:
    val = labs.ogtt_2h_mmol_l
    cfg = self.config
    if val is None:
      return self._diagnose_from_risk_only(clinical, labs, ctx)

    if self._is_borderline(val, cfg.ogtt_diabetes, cfg.borderline_margin):
      label: DiagnosisLabel = "uncertain"
      next_step: DiagnosticNextStep = "repeat_OGTT_or_use_alternative_test"
    elif val >= cfg.ogtt_diabetes:
      label = "T2D"
      next_step = "confirm_diagnosis_and_initiate_management"
    elif val >= cfg.ogtt_high_risk_low:
      label = "high_risk"
      next_step = "lifestyle_intervention_and_repeat_testing"
    else:
      label = "normal"
      next_step = "routine_monitoring"

    conf = self._calibrate_confidence_from_distance(val, cfg.ogtt_diabetes, clinical.risk_T2D_now)
    conf = self._penalise_for_lab_quality(conf, labs)

    reasoning = self._reasoning_pack(
      basis = "OGTT_2h",
      value = val,
      thresholds = {
        "diabetes": cfg.ogtt_diabetes, 
        "high_risk_low": cfg.ogtt_high_risk_low
        },
      label = label,
      clinical = clinical,
      labs = labs,
      ctx = ctx,
    )

    return DiagnosticResult(
      label = label, 
      confidence = conf, 
      next_step = next_step, 
      basis = "OGTT_2h", 
      reasoning_tokens = reasoning
      )



  # Risk-only fallback -----
  def _diagnose_from_risk_only(self, clinical: ClinicalAssessmentSnapshot, labs: LabEvidence, ctx: DiagnosticContext) -> DiagnosticResult:
    p = clinical.risk_T2D_now

    if p >= 0.8:
      label: DiagnosisLabel = "high_risk"
      next_step: DiagnosticNextStep = "order_diagnostic_labs"
    elif p >= 0.4:
      label = "high_risk"
      next_step = "lifestyle_intervention_and_consider_labs"
    elif p >= 0.2:
      label = "normal"
      next_step = "monitor_and_reassess_risk"
    else:
      label = "normal"
      next_step = "routine_monitoring"

    conf = min(0.9, 0.5 + 0.4 * p)

    reasoning = {
      "basis": "risk_only",
      "value": None,
      "thresholds": {},
      "chosen_label": label,
      "pretest_risk": p,
      "lab_flags": {
        "is_self_report_only": labs.is_self_report_only,
        "is_outdated": labs.is_outdated,
        "has_quality_issues": labs.has_quality_issues,
      },
      "context_flags": {
        "pregnancy": ctx.pregnancy,
        "anaemia": ctx.anaemia,
        "haemoglobinopathy": ctx.haemoglobinopathy,
        "ckd": ctx.ckd,
      },
    }

    return DiagnosticResult(
      label = label, 
      confidence = conf, 
      next_step = next_step, 
      basis = "risk_only", 
      reasoning_tokens = reasoning
      )



  # -----------------------------------------------------
  # Adapters: read whatever your upstream agents wrote into state
  @staticmethod
  def _clinical_from_state(payload: Optional[Dict[str, Any]]) -> ClinicalAssessmentSnapshot:
    if not payload:
      # default safe fallback (forces "normal" risk-only path)
      return ClinicalAssessmentSnapshot(risk_T2D_now=0.0)
      
    risk = float(payload.get("risk_T2D_now", payload.get("risk", 0.0)))
    return ClinicalAssessmentSnapshot(
      risk_T2D_now = risk,
      triage_label = payload.get("triage_label"),
      raw_proba_vector = payload.get("raw_proba_vector"),
      meta = payload.get("meta"),
    )

  @staticmethod
  def _labs_from_state(payload: Optional[Dict[str, Any]]) -> LabEvidence:
    if not payload:
      return LabEvidence()

    # Common patterns you might be using in LabAgent output:
    # - direct keys: hba1c_mmol_mol / fpg_mmol_l / ogtt_2h_mmol_l
    # - nested "labs": {"hba1c_mmol_mol": ...}
    labs_dict = payload.get("labs", payload)

    return LabEvidence(
      hba1c_mmol_mol = _safe_float(labs_dict.get("hba1c_mmol_mol")),
      fpg_mmol_l = _safe_float(labs_dict.get("fpg_mmol_l")),
      ogtt_2h_mmol_l = _safe_float(labs_dict.get("ogtt_2h_mmol_l")),
      random_glucose_mmol_l = _safe_float(labs_dict.get("random_glucose_mmol_l")),
      is_self_report_only = bool(payload.get("is_self_report_only", False)),
      is_outdated = bool(payload.get("is_outdated", False)),
      has_quality_issues = bool(payload.get("has_quality_issues", False)),
      raw_meta=payload.get("meta"),
    )

  @staticmethod
  def _context_from_state(payload: Optional[Dict[str, Any]]) -> DiagnosticContext:
    if not payload:
      return DiagnosticContext()

    return DiagnosticContext(
      pregnancy=bool(payload.get("pregnancy", False)),
      anaemia=bool(payload.get("anaemia", False)),
      haemoglobinopathy=bool(payload.get("haemoglobinopathy", False)),
      ckd=bool(payload.get("ckd", False)),
      raw_meta=payload,
    )



  # -----------------
  # Helpers
  @staticmethod
  def _is_borderline(value: float, threshold: float, margin: float) -> bool:
    return abs(value - threshold) <= margin

  def _calibrate_confidence_from_distance(self, value: float, threshold: float, pretest_risk: float,) -> float:
    dist = abs(value - threshold)
    lab_component = 1.0 / (1.0 + math.exp(-dist))
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
  
  @staticmethod
  def _reasoning_pack(basis: DiagnosticBasis, value: float, thresholds: Dict[str, float], label: DiagnosisLabel, clinical: ClinicalAssessmentSnapshot, labs: LabEvidence, ctx: DiagnosticContext,) -> Dict[str, Any]:
    return {
      "basis": basis,
      "value": value,
      "thresholds": thresholds,
      "chosen_label": label,
      "pretest_risk": clinical.risk_T2D_now,
      "lab_flags": {
        "is_self_report_only": labs.is_self_report_only,
        "is_outdated": labs.is_outdated,
        "has_quality_issues": labs.has_quality_issues,
      },
      "context_flags": {
        "pregnancy": ctx.pregnancy,
        "anaemia": ctx.anaemia,
        "haemoglobinopathy": ctx.haemoglobinopathy,
        "ckd": ctx.ckd,
      },
    }


def _safe_float(x: Any) -> Optional[float]:
  try:
    if x is None:
      return None
    return float(x)
  except (TypeError, ValueError):
    return None
