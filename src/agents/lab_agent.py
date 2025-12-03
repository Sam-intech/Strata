from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, field_validator, ValidationError, validator
# ===============================================================================================================



TestType = Literal["none", "HbA1c", "FPG", "OGTT", "repeat_test"]
Urgency = Literal["none", "routine", "priority"]

TriageLabel = Literal["low_risk", "routine_follow_up", "high_risk"]


# config -----
@dataclass
class LabAgentConfig:
  """
    Configuration for LaboratoryAgent:
    - thresholds
    - recency windows
    - risk cut-offs
  """
  # Recency
  max_lab_age_days: int = 180  # 6 months

  # Risk thresholds (from Clinical Assessment Agent output)
  high_risk_threshold: float = 0.70
  moderate_risk_threshold: float = 0.30

  # HbA1c thresholds (mmol/mol)
  hba1c_diabetes_mmol_mol: float = 48.0
  hba1c_pre_diabetes_low_mmol_mol: float = 42.0

  # FPG thresholds (mmol/L)
  fpg_diabetes_mmol_l: float = 7.0
  fpg_pre_diabetes_low_mmol_l: float = 6.0

  # OGTT thresholds (2h plasma glucose, mmol/L)
  ogtt_diabetes_mmol_l: float = 11.1
  ogtt_pre_diabetes_low_mmol_l: float = 7.0

  # Safety ranges for crude sanity checks (to flag nonsense values)
  hba1c_min_mmol_mol: float = 20.0
  hba1c_max_mmol_mol: float = 130.0

  fpg_min_mmol_l: float = 2.0
  fpg_max_mmol_l: float = 30.0

  ogtt_min_mmol_l: float = 2.0
  ogtt_max_mmol_l: float = 30.0



# Input models
class LabMeasurement(BaseModel):
  # Single lab measurement with metadata.
  value: float = Field(..., description="Numeric value of the lab measurement")
  unit: str = Field(..., description="Unit string, e.g. 'mmol/mol', '%', 'mmol/L'")
  timestamp: Optional[datetime] = Field(
    default=None,
    description="When the sample was taken (optional but recommended)",
  )
  source: Literal["ehr", "self_report", "unknown"] = Field(
    default="unknown",
    description="Provenance of the lab result",
  )

  @field_validator("unit")
  @classmethod
  def normalize_unit(cls, v: str) -> str:
    return v.strip()
 


class LabBundle(BaseModel):
  """
  Bundle of lab values for a single patient.
  All fields optional: the system can work with missing labs and decide what to order.
  """
  hba1c: Optional[LabMeasurement] = None
  fpg: Optional[LabMeasurement] = None
  ogtt: Optional[LabMeasurement] = None
  random_glucose: Optional[LabMeasurement] = None



class ClinicalRiskInput(BaseModel):
  # Minimal input from Clinical Assessment Agent.
  risk_T2D_now: float = Field(ge=0.0, le=1.0)
  triage_label: Optional[TriageLabel] = None



class ClinicalContext(BaseModel):
  # Context flags that affect lab reliability / choice.
  pregnancy: bool = False
  anaemia: bool = False
  ckd: bool = Field(False, description="Chronic kidney disease")
  haemoglobinopathy: bool = False

  # Extra context if needed later
  other_context: Dict[str, Any] = Field(default_factory=dict)



# Outputs -----
class TestPlan(BaseModel):
  # Decision about whether and what to test.
  needs_test: bool
  test_type: TestType
  urgency: Urgency
  need_retest: bool
  rationale: str



@dataclass
class LabAgentOutput:
  """
  Output of the LaboratoryAgent.
  - test_plan: whether to order new labs and what kind
  - validated_labs: lab values after normalisation and basic QC
  - lab_interpretation_tokens: structured semantic tags for Diagnostic Agent
  - flags: misc quality flags
  - meta: optional debug/info metadata
  """
  test_plan: TestPlan
  validated_labs: LabBundle
  lab_interpretation_tokens: Dict[str, Any]
  flags: Dict[str, bool]
  meta: Optional[Dict[str, Any]] = None




# Laboratory Agent -----
class LaboratoryAgent:
  def __init__(self, config: Optional[LabAgentConfig] = None):
    self.config = config or LabAgentConfig()


  # Public Api
  def assess(self, labs: Dict[str, Any], clinical_risk: Dict[str, Any], context: Optional[Dict[str, Any]] = None,) -> LabAgentOutput:
    try:
      lab_bundle = LabBundle(**labs)
    except ValidationError as e:
      raise ValueError(f"Invalid lab input for LabBundle: {e}") from e
    
    try:
      risk_input = ClinicalRiskInput(**clinical_risk)
    except ValidationError as e:
      raise ValueError(f"Invalid clinical risk input: {e}") from e
    
    ctx = ClinicalContext(**(context or {}))

    # 1) Validate / normalise labs
    validated_labs, flags = self._validate_and_normalise_labs(lab_bundle, ctx)

    # 2) Interpret labs w.r.t thresholds
    interpretation = self._interpret_labs(validated_labs, ctx)

    # 3) Decide test plan (order new test / retest / nothing)
    test_plan = self._decide_test_plan(
      validated_labs=validated_labs,
      risk=risk_input.risk_T2D_now,
      context=ctx,
      interpretation=interpretation,
    )

    meta = {
      "config": self.config.__dict__,
      "triage_label": risk_input.triage_label,
    }

    return LabAgentOutput(
      test_plan=test_plan,
      validated_labs=validated_labs,
      lab_interpretation_tokens=interpretation,
      flags=flags,
      meta=meta,
    )


  # internal helpers -----
  def _validate_and_normalise_labs(self, labs: LabBundle, ctx: ClinicalContext,) -> tuple[LabBundle, Dict[str, bool]]:
    # Convert units, sanity-check ranges, and flag suspicious values.
    flags: Dict[str, bool] = {
      "has_any_lab": False,
      "hba1c_out_of_range": False,
      "fpg_out_of_range": False,
      "ogtt_out_of_range": False,
      "any_self_report_lab": False,
      "lab_potentially_unreliable_context": False,
    }

    # Context may render HbA1c unreliable
    if ctx.pregnancy or ctx.anaemia or ctx.ckd or ctx.haemoglobinopathy:
      flags["lab_potentially_unreliable_context"] = True

    # Copy to mutable dict for editing
    labs_data = labs.model_dump()

    # HbA1c
    if labs.hba1c is not None:
      flags["has_any_lab"] = True
      labs_data["hba1c"] = self._normalise_hba1c(labs.hba1c, flags)

    # FPG
    if labs.fpg is not None:
      flags["has_any_lab"] = True
      labs_data["fpg"] = self._normalise_glucose(
        labs.fpg,
        flags,
        kind="fpg",
        min_val=self.config.fpg_min_mmol_l,
        max_val=self.config.fpg_max_mmol_l,
      )

    # OGTT
    if labs.ogtt is not None:
      flags["has_any_lab"] = True
      labs_data["ogtt"] = self._normalise_glucose(
        labs.ogtt,
        flags,
        kind="ogtt",
        min_val=self.config.ogtt_min_mmol_l,
        max_val=self.config.ogtt_max_mmol_l,
      )

    # Random glucose (only basic sanity)
    if labs.random_glucose is not None:
      flags["has_any_lab"] = True
      labs_data["random_glucose"] = self._normalise_glucose(
        labs.random_glucose,
        flags,
        kind="random_glucose",
        min_val=self.config.fpg_min_mmol_l,
        max_val=self.config.fpg_max_mmol_l,
        strict_range=False,
      )

    # Self-report flags
    for key in ["hba1c", "fpg", "ogtt", "random_glucose"]:
      meas = labs_data.get(key)
      if meas and meas.get("source") == "self_report":
        flags["any_self_report_lab"] = True

    validated = LabBundle(**labs_data)
    return validated, flags


  def _normalise_hba1c(  self, meas: LabMeasurement, flags: Dict[str, bool],) -> Dict[str, Any]:
    """
    Convert HbA1c to mmol/mol and flag out-of-range values.

    Common clinical formula:
    mmol/mol ≈ (percent - 2.15) * 10.929
    """
    val = meas.value
    unit = meas.unit.lower()

    if unit in ["%", "percent", "pct"]:
      mmol_mol = (val - 2.15) * 10.929
    elif unit in ["mmol/mol", "mmol per mol", "mmol-per-mol"]:
      mmol_mol = val
    else:
      # Unknown unit, just leave it and hope for the best
      mmol_mol = val

    if (
      mmol_mol < self.config.hba1c_min_mmol_mol
      or mmol_mol > self.config.hba1c_max_mmol_mol
    ):
      flags["hba1c_out_of_range"] = True

    return {
      "value": mmol_mol,
      "unit": "mmol/mol",
      "timestamp": meas.timestamp,
      "source": meas.source,
    }


  def _normalise_glucose(self, meas: LabMeasurement, flags: Dict[str, bool], kind: str, min_val: float, max_val: float, strict_range: bool = True,) -> Dict[str, Any]:
    """
    Normalise glucose measurements to mmol/L.
    Handles typical mg/dL → mmol/L conversion if needed.
    """

    val = meas.value
    unit = meas.unit.lower()

    if unit in ["mmol/l", "mmol per l", "mmol-per-l"]:
      mmol_l = val
    elif unit in ["mg/dl", "mg per dl", "mg-per-dl"]:
      # 1 mmol/L glucose ≈ 18 mg/dL
      mmol_l = val / 18.0
    else:
      mmol_l = val  # unknown unit, unconverted

    if strict_range:
      if mmol_l < min_val or mmol_l > max_val:
        if kind == "fpg":
          flags["fpg_out_of_range"] = True
        elif kind == "ogtt":
          flags["ogtt_out_of_range"] = True

    return {
      "value": mmol_l,
      "unit": "mmol/L",
      "timestamp": meas.timestamp,
      "source": meas.source,
    }


  def _interpret_labs(  self, labs: LabBundle, ctx: ClinicalContext,) -> Dict[str, Any]:
    """
    Map lab values to semantic categories relative to clinical thresholds.
    
    Produces machine-readable tokens for the Diagnostic Agent / Explanation Agent.
    """
    tokens: Dict[str, Any] = {}

    # HbA1c
    if labs.hba1c is not None:
      v = labs.hba1c.value
      category = "normal"
      if v >= self.config.hba1c_diabetes_mmol_mol:
        category = "diabetic_range"
      elif v >= self.config.hba1c_pre_diabetes_low_mmol_mol:
        category = "pre_diabetes_range"

      tokens["hba1c"] = {
        "value": v,
        "unit": labs.hba1c.unit,
        "category": category,
        "thresholds": {
          "diabetes": self.config.hba1c_diabetes_mmol_mol,
          "pre_diabetes_low": self.config.hba1c_pre_diabetes_low_mmol_mol,
        },
        "timestamp": labs.hba1c.timestamp,
        "unreliable_context": bool(
          ctx.pregnancy or ctx.anaemia or ctx.ckd or ctx.haemoglobinopathy
        ),
      }

    # FPG
    if labs.fpg is not None:
      v = labs.fpg.value
      category = "normal"
      if v >= self.config.fpg_diabetes_mmol_l:
        category = "diabetic_range"
      elif v >= self.config.fpg_pre_diabetes_low_mmol_l:
        category = "pre_diabetes_range"

      tokens["fpg"] = {
        "value": v,
        "unit": labs.fpg.unit,
        "category": category,
        "thresholds": {
          "diabetes": self.config.fpg_diabetes_mmol_l,
          "pre_diabetes_low": self.config.fpg_pre_diabetes_low_mmol_l,
        },
        "timestamp": labs.fpg.timestamp,
      }

    # OGTT
    if labs.ogtt is not None:
      v = labs.ogtt.value
      category = "normal"
      if v >= self.config.ogtt_diabetes_mmol_l:
        category = "diabetic_range"
      elif v >= self.config.ogtt_pre_diabetes_low_mmol_l:
        category = "pre_diabetes_range"

      tokens["ogtt"] = {
        "value": v,
        "unit": labs.ogtt.unit,
        "category": category,
        "thresholds": {
          "diabetes": self.config.ogtt_diabetes_mmol_l,
          "pre_diabetes_low": self.config.ogtt_pre_diabetes_low_mmol_l,
        },
        "timestamp": labs.ogtt.timestamp,
      }

    # Random glucose (only qualitative; thresholds can be tuned)
    if labs.random_glucose is not None:
      v = labs.random_glucose.value
      # You can refine these if you want; this is intentionally soft
      category = "normal"
      if v >= 11.1:
        category = "very_high"
      elif v >= 7.8:
        category = "elevated"

      tokens["random_glucose"] = {
        "value": v,
        "unit": labs.random_glucose.unit,
        "category": category,
        "timestamp": labs.random_glucose.timestamp,
      }

    return tokens


  def _decide_test_plan(self, validated_labs: LabBundle, risk: float, context: ClinicalContext, interpretation: Dict[str, Any],) -> TestPlan:
    """
    Decide if a new test is needed, which type, and urgency.
    Logic is intentionally simple & explainable for MSc level.
    """
      
    now = datetime.utcnow()
    cfg = self.config

    def is_recent(meas: Optional[LabMeasurement]) -> bool:
      if meas is None or meas.timestamp is None:
        return False
      return now - meas.timestamp <= timedelta(days=cfg.max_lab_age_days)

    # Check recency of each lab
    hba1c_recent = is_recent(validated_labs.hba1c)
    fpg_recent = is_recent(validated_labs.fpg)
    ogtt_recent = is_recent(validated_labs.ogtt)

    has_any_recent_lab = hba1c_recent or fpg_recent or ogtt_recent

    # 1) If no labs and high risk → order HbA1c (or FPG/OGTT if context unreliable)
    if not has_any_recent_lab and validated_labs.hba1c is None \
      and validated_labs.fpg is None and validated_labs.ogtt is None:

      if risk >= cfg.high_risk_threshold:
        test_type = self._preferred_test_type(context)
        rationale = (
          f"No recent labs and high risk ({risk:.2f}) → "
          f"order {test_type}"
        )
        return TestPlan(
          needs_test=True,
          test_type=test_type,
          urgency="priority",
          need_retest=False,
          rationale=rationale,
        )

      if risk >= cfg.moderate_risk_threshold:
        test_type = self._preferred_test_type(context)
        rationale = (
          f"No recent labs and moderate risk ({risk:.2f}) ->" f"order {test_type}"
        )
        return TestPlan(
          needs_test = True,
          test_type = test_type,
          urgency = "routine",
          need_retest = False,
          rationale = rationale,
        )
          
      # Low risk, no labs → can defer
      return TestPlan(
          needs_test=False,
          test_type="none",
          urgency="none",
          need_retest=False,
          rationale="Low risk and no labs; monitoring without immediate testing.",
      )

    # 2) Labs exist but outdated → retest if risk not trivial
    if not has_any_recent_lab:
      if risk >= cfg.moderate_risk_threshold:
        test_type = self._preferred_test_type(context)
        rationale = (
          f"Existing labs are outdated and risk={risk:.2f} → retest {test_type}"
        )
        return TestPlan(
          needs_test=True,
          test_type="repeat_test",
          urgency="routine",
          need_retest=True,
          rationale=rationale,
        )
      else:
        return TestPlan(
          needs_test=False,
          test_type="none",
          urgency="none",
          need_retest=False,
          rationale="Only outdated labs available but risk is low; monitor.",
        )
      
    # 3) Recent lab(s) exist → no new test by default
    #    But handle borderline / conflicting ranges if you want to get fancy
    #    For now, we keep it simple and leave uncertainty to Diagnostic Agent.
    rationale_parts = ["Recent lab(s) available; no new test required."]

    if "hba1c" in interpretation:
      cat = interpretation["hba1c"]["category"]
      rationale_parts.append(f"HbA1c is in {cat.replace('_', ' ')}.")
    if "fpg" in interpretation:
      cat = interpretation["fpg"]["category"]
      rationale_parts.append(f"FPG is in {cat.replace('_', ' ')}.")
    if "ogtt" in interpretation:
      cat = interpretation["ogtt"]["category"]
      rationale_parts.append(f"OGTT is in {cat.replace('_', ' ')}.")

    rationale = " ".join(rationale_parts)

    return TestPlan(
      needs_test=False,
      test_type="none",
      urgency="none",
      need_retest=False,
      rationale=rationale,
    )


  def _preferred_test_type(self, context: ClinicalContext) -> TestType:
      """
      Decide preferred test based on context.
  
      HbA1c is default, but if it is unreliable, prefer FPG/OGTT.
      """
      if context.pregnancy or context.anaemia or context.ckd or context.haemoglobinopathy:
        # HbA1c unreliable → prefer OGTT or FPG
        return "OGTT"
      return "HbA1c"               