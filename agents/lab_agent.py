from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field, field_validator, ValidationError 
# ===============================================================================================================



TestType = Literal["none", "HbA1c", "FPG", "OGTT", "repeat_test"]
Urgency = Literal["none", "routine", "priority"]
TriageLabel = Literal["low_risk", "routine_follow_up", "high_risk"]

# ------------------
# config
@dataclass
class LabAgentConfig:
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
  ogtt_pre_diabetes_low_mmol_l: float = 7.8

  # Safety ranges for crude sanity checks (to flag nonsense values)
  hba1c_min_mmol_mol: float = 20.0
  hba1c_max_mmol_mol: float = 130.0

  fpg_min_mmol_l: float = 2.0
  fpg_max_mmol_l: float = 30.0

  ogtt_min_mmol_l: float = 2.0
  ogtt_max_mmol_l: float = 30.0

  # Random glucose qualitative bands (mmol/L)
  random_glucose_elevated: float = 7.8
  random_glucose_very_high: float = 11.1


# -------------------
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


# -------------------
# Output models 
class TestPlan(BaseModel):
  # Decision about whether and what to test.
  needs_test: bool
  test_type: TestType
  urgency: Urgency
  need_retest: bool
  rationale: str


# @dataclass
class LabAgentOutput(BaseModel):
  test_plan: TestPlan
  validated_labs: LabBundle
  lab_interpretation_tokens: Dict[str, Any]
  flags: Dict[str, bool]
  meta: Optional[Dict[str, Any]] = Field(default_factory = dict)




# ------------------------------
# Laboratory Agent 
class LaboratoryAgent:
  def __init__(self, config: Optional[LabAgentConfig] = None):
    self.config = config or LabAgentConfig()


  # Public Api
  def assess(self, labs: Dict[str, Any], clinical_risk: Dict[str, Any], context: Optional[Dict[str, Any]] = None, *, now: Optional[datetime] = None) -> LabAgentOutput:
    try:
      lab_bundle = LabBundle(**labs)
    except ValidationError as e:
      raise ValueError(f"Invalid lab input for LabBundle: {e}") from e
    
    try:
      risk_input = ClinicalRiskInput(**clinical_risk)
    except ValidationError as e:
      raise ValueError(f"Invalid clinical risk input: {e}") from e
    
    ctx = ClinicalContext(**(context or {}))

    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
      now_dt = now_dt.replace(tzinfo=timezone.utc)

    # 1) Validate / normalise labs
    validated_labs, flags, recency_meta = self._validate_and_normalise_labs(lab_bundle, ctx, now_dt)

    # 2) Interpret labs w.r.t thresholds
    interpretation = self._interpret_labs(validated_labs, ctx)

    # 3) Decide test plan (order new test / retest / nothing)
    test_plan = self._decide_test_plan(
      validated_labs = validated_labs,
      risk = risk_input.risk_T2D_now,
      context = ctx,
      interpretation = interpretation,
      now_dt = now_dt,
      flags = flags,
    )

    meta: Dict[str, Any] = {
      "config": self.config.__dict__,
      "triage_label": risk_input.triage_label,
      **recency_meta,
    }

    return LabAgentOutput(
      test_plan = test_plan,
      validated_labs = validated_labs,
      lab_interpretation_tokens = interpretation,
      flags = flags,
      meta = meta,
    )


  # internal helpers -----
  def _validate_and_normalise_labs(self, labs: LabBundle, ctx: ClinicalContext, now_dt: datetime,) -> tuple[LabBundle, Dict[str, bool], Dict[str, Any]]:
    cfg = self.config
    
    # Convert units, sanity-check ranges, and flag suspicious values.
    flags: Dict[str, bool] = {
      "has_any_lab": False,
      "any_outdated_lab": False,
      "any_recent_lab": False,
      "any_missing_timestamp": False,
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
    timestamps: list[datetime] = []

    def consider_timestamp(meas: Optional[Dict[str, Any]]) -> None:
      if not meas:
        return
      ts = meas.get("timestamp")
      if ts is None:
        flags["any_missing_timestamp"] = True
        return
      if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
        meas["timestamp"] = ts
      timestamps.append(ts)

      age_days = (now_dt - ts).days
      if age_days > cfg.max_lab_age_days:
        flags["any_outdated_lab"] = True
      else:
        flags["any_recent_lab"] = True

    # HbA1c
    if labs.hba1c is not None:
      flags["has_any_lab"] = True
      labs_data["hba1c"] = self._normalise_hba1c(labs.hba1c, flags)
      consider_timestamp(labs_data["hba1c"])

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
      consider_timestamp(labs_data["fpg"])

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
      consider_timestamp(labs_data["ogtt"])

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
      consider_timestamp(labs_data["random_glucose"])

    # Self-report flags
    for key in ["hba1c", "fpg", "ogtt", "random_glucose"]:
      meas = labs_data.get(key)
      if meas and meas.get("source") == "self_report":
        flags["any_self_report_lab"] = True
        
    validated = LabBundle(**labs_data)
    # return validated, flags

    # Meta for orchestration/eval to consume (no extra agent calls needed)
    most_recent_ts = max(timestamps) if timestamps else None
    meta: Dict[str, Any] = {
      "most_recent_lab_timestamp": most_recent_ts.isoformat() if most_recent_ts else None,
      "most_recent_lab_age_days": (now_dt - most_recent_ts).days if most_recent_ts else None,
      "max_lab_age_days": cfg.max_lab_age_days,
    }

    return validated, flags, meta


  def _normalise_hba1c(  self, meas: LabMeasurement, flags: Dict[str, bool],) -> Dict[str, Any]:
    cfg = self.config
    val = meas.value
    unit = meas.unit.lower()

    if unit in ["%", "percent", "pct"]:
      mmol_mol = (val - 2.15) * 10.929
    elif unit in ["mmol/mol", "mmol per mol", "mmol-per-mol"]:
      mmol_mol = val
    else:
      # Unknown unit, just leave it and hope for the best
      mmol_mol = val

    if mmol_mol < cfg.hba1c_min_mmol_mol or mmol_mol > cfg.hba1c_max_mmol_mol:
      flags["hba1c_out_of_range"] = True

    return {
      "value": float(mmol_mol),
      "unit": "mmol/mol",
      "timestamp": meas.timestamp,
      "source": meas.source,
    }


  def _normalise_glucose(self, meas: LabMeasurement, flags: Dict[str, bool], *, kind: str, min_val: float, max_val: float, strict_range: bool = True,) -> Dict[str, Any]:
    val = meas.value
    unit = meas.unit.lower()

    if unit in ["mmol/l", "mmol per l", "mmol-per-l"]:
      mmol_l = val
    elif unit in ["mg/dl", "mg per dl", "mg-per-dl"]:
      # 1 mmol/L glucose ≈ 18 mg/dL
      mmol_l = val / 18.0
    else:
      mmol_l = val  # unknown unit, unconverted

    if strict_range and (mmol_l < min_val or mmol_l > max_val):
      if kind == "fpg":
        flags["fpg_out_of_range"] = True
      elif kind == "ogtt":
        flags["ogtt_out_of_range"] = True

    return {
      "value": float(mmol_l),
      "unit": "mmol/L",
      "timestamp": meas.timestamp,
      "source": meas.source,
    }


  def _interpret_labs( self, labs: LabBundle, ctx: ClinicalContext,) -> Dict[str, Any]:
    cfg = self.config
    tokens: Dict[str, Any] = {}

    # HbA1c
    if labs.hba1c is not None:
      v = labs.hba1c.value
      category = "normal"
      if v >= cfg.hba1c_diabetes_mmol_mol:
        category = "diabetic_range"
      elif v >= cfg.hba1c_pre_diabetes_low_mmol_mol:
        category = "pre_diabetes_range"

      tokens["hba1c"] = {
        "value": v,
        "unit": labs.hba1c.unit,
        "category": category,
        "thresholds": {
          "diabetes": cfg.hba1c_diabetes_mmol_mol,
          "pre_diabetes_low": cfg.hba1c_pre_diabetes_low_mmol_mol,
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
      if v >= cfg.fpg_diabetes_mmol_l:
        category = "diabetic_range"
      elif v >= cfg.fpg_pre_diabetes_low_mmol_l:
        category = "pre_diabetes_range"

      tokens["fpg"] = {
        "value": v,
        "unit": labs.fpg.unit,
        "category": category,
        "thresholds": {
          "diabetes": cfg.fpg_diabetes_mmol_l,
          "pre_diabetes_low": cfg.fpg_pre_diabetes_low_mmol_l,
        },
        "timestamp": labs.fpg.timestamp,
      }

    # OGTT
    if labs.ogtt is not None:
      v = labs.ogtt.value
      category = "normal"
      if v >= cfg.ogtt_diabetes_mmol_l:
        category = "diabetic_range"
      elif v >= cfg.ogtt_pre_diabetes_low_mmol_l:
        category = "pre_diabetes_range"

      tokens["ogtt"] = {
        "value": v,
        "unit": labs.ogtt.unit,
        "category": category,
        "thresholds": {
          "diabetes": cfg.ogtt_diabetes_mmol_l,
          "pre_diabetes_low": cfg.ogtt_pre_diabetes_low_mmol_l,
        },
        "timestamp": labs.ogtt.timestamp,
      }

    # Random glucose (only qualitative; thresholds can be tuned)
    if labs.random_glucose is not None:
      v = labs.random_glucose.value
      # You can refine these if you want; this is intentionally soft
      category = "normal"
      if v >= cfg.random_glucose_very_high:
        category = "very_high"
      elif v >= cfg.random_glucose_elevated:
        category = "elevated"

      tokens["random_glucose"] = {
        "value": v,
        "unit": labs.random_glucose.unit,
        "category": category,
        "timestamp": labs.random_glucose.timestamp,
      }

    return tokens


  def _decide_test_plan(self, *, validated_labs: LabBundle, risk: float, context: ClinicalContext, interpretation: Dict[str, Any], now_dt: datetime, flags: Dict[str, bool]) -> TestPlan:      
    # now = datetime.utcnow()
    cfg = self.config

    def is_recent(meas: Optional[LabMeasurement]) -> bool:
      if meas is None or meas.timestamp is None:
        return False
      ts = meas.timestamp.replace(tzinfo = timezone.utc) if meas.timestamp.tzinfo is None else meas.timestamp
      return (now_dt - ts) <= timedelta(days = cfg.max_lab_age_days)

    # Check recency of each lab
    hba1c_recent = is_recent(validated_labs.hba1c)
    fpg_recent = is_recent(validated_labs.fpg)
    ogtt_recent = is_recent(validated_labs.ogtt)

    has_any_recent_lab = hba1c_recent or fpg_recent or ogtt_recent
    
    # Case A: no labs at all (or no timestamps -> treated as not recent)
    has_any_lab_values = any([
      validated_labs.hba1c is not None,
      validated_labs.fpg is not None,
      validated_labs.ogtt is not None,
    ])

    if not has_any_recent_lab and not has_any_lab_values:
      if risk >= cfg.high_risk_threshold:
        test_type = self._preferred_test_type(context)
        return TestPlan(
          needs_test=True,
          test_type=test_type,
          urgency="priority",
          need_retest=False,
          rationale=f"No recent labs and high risk ({risk:.2f}) → order {test_type}.",
        )

      if risk >= cfg.moderate_risk_threshold:
        test_type = self._preferred_test_type(context)
        return TestPlan(
          needs_test=True,
          test_type=test_type,
          urgency="routine",
          need_retest=False,
          rationale=f"No recent labs and moderate risk ({risk:.2f}) → order {test_type}.",
        )

      return TestPlan(
        needs_test=False,
        test_type="none",
        urgency="none",
        need_retest=False,
        rationale="Low risk and no labs; monitor without immediate testing.",
      )


    # Case B: labs exist but outdated
    if not has_any_recent_lab and has_any_lab_values:
      if risk >= cfg.moderate_risk_threshold:
        # retest same preferred type (repeat order is captured by repeat_test)
        pref = self._preferred_test_type(context)
        return TestPlan(
          needs_test=True,
          test_type="repeat_test",
          urgency="routine",
          need_retest=True,
          rationale=f"Existing labs outdated and risk={risk:.2f} → retest ({pref}).",
        )

      return TestPlan(
        needs_test=False,
        test_type="none",
        urgency="none",
        need_retest=False,
        rationale="Only outdated labs available but risk is low; monitor.",
      )
    

    # Case C: recent labs exist → no new test by default
    rationale_parts = ["Recent lab(s) available; no new test required."]

    if "hba1c" in interpretation:
      rationale_parts.append(f"HbA1c is {interpretation['hba1c']['category'].replace('_', ' ')}.")
      if interpretation["hba1c"].get("unreliable_context") and flags.get("lab_potentially_unreliable_context"):
        rationale_parts.append("HbA1c may be unreliable in current context; consider FPG/OGTT if needed.")

    if "fpg" in interpretation:
      rationale_parts.append(f"FPG is {interpretation['fpg']['category'].replace('_', ' ')}.")

    if "ogtt" in interpretation:
      rationale_parts.append(f"OGTT is {interpretation['ogtt']['category'].replace('_', ' ')}.")

    return TestPlan(
      needs_test=False,
      test_type="none",
      urgency="none",
      need_retest=False,
      rationale=" ".join(rationale_parts),
    )



  def _preferred_test_type(self, context: ClinicalContext) -> TestType:
    if context.pregnancy or context.anaemia or context.ckd or context.haemoglobinopathy:
      # HbA1c unreliable → prefer OGTT or FPG
      return "OGTT"
    return "HbA1c"               