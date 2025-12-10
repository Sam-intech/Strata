from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Literal, Callable
import math
# ==============================================================================


TriageLabel = Literal["low", "medium", "high", "critical"]
DiagnosticLabel = Literal["normal", "high_risk", "T2D", "uncertain"]


@dataclass
class ClinicalAssessmentOutput:
  risk_T2D_now: float
  triage_label: TriageLabel
  top_contributors: Dict[str, float]  # e.g. {"BMI": 0.31, "age": 0.22}
  raw_proba_vector: Optional[Any] = None
  meta: Optional[Dict[str, Any]] = None


@dataclass
class LabSummary:
  # Minimal view of labs that the ExplanationAgent cares about
  primary_marker: Optional[str] = None          # e.g. "HbA1c"
  primary_value: Optional[float] = None         # e.g. 61.0
  primary_unit: Optional[str] = None            # e.g. "mmol/mol"
  primary_threshold: Optional[float] = None     # e.g. 48.0
  recency_days: Optional[int] = None            # age of test
  provenance: Optional[str] = None              # "EHR", "self_report"
  reliability_flags: List[str] = field(default_factory=list)


@dataclass
class LaboratoryAgentOutput:
  test_plan: TestPlan
  lab_summary: LabSummary


@dataclass
class DiagnosticAgentOutput:
    diagnosis_label: DiagnosisLabel
    diagnosis_confidence: float
    next_step: Optional[str]
    reasoning_tokens: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)



# Output structure -----
@dataclass
class ClinicianReport:
  diagnosis_heading: str
  confidence_text: str
  summary: str
  reasoning: str
  clinical_implications: str
  recommended_actions: str
  data_quality_notes: Optional[str] = None

  def as_dict(self) -> Dict[str, str]:
    return {
      "diagnosis_heading": self.diagnosis_heading,
      "confidence_text": self.confidence_text,
      "summary": self.summary,
      "reasoning": self.reasoning,
      "clinical_implications": self.clinical_implications,
      "recommended_actions": self.recommended_actions,
      "data_quality_notes": self.data_quality_notes or "",
    }
  

@dataclass
class PatientSummary:
  title: str
  main_message: str
  what_it_means: str
  next_steps: str


@dataclass
class ExplanationBundle:
  clinician_report: ClinicianReport
  patient_summary: Optional[PatientSummary] = None
  raw_sections: Dict[str, str] = field(default_factory=dict) 



# configuration -----
@dataclass
class ExplanationConfig:
  high_conf_threshold: float = 0.8
  moderate_conf_threshold: float = 0.6

  # Section toggles (in case you want to turn bits off in eval)
  include_patient_friendly: bool = True
  include_data_quality_section: bool = True



# explanation agent -----
class ExplanationAgent:
  """
  Template-based Explanation Agent for T2D MAS.

  It does 3 things:
  1. Interprets diagnostic label + confidence.
  2. Links that to clinical risk factors & lab evidence.
  3. Generates structured clinician + optional patient explanations.
  """
  def __init__(self, config: Optional[ExplanationConfig] = None, llm_rewriter: Optional[Callable[[str, str]]] = None):
    """
    llm_rewriter: optional function(section_text, section_role) -> rewritten_text
    - section_role e.g. "clinician_summary", "patient_summary"
    - lets you plug in an LLM later without rewriting the core logic
    """
    self.config = config or ExplanationConfig()
    self.llm_rewriter = llm_rewriter  # Optional function to rewrite text via LLM

  # Public API -----
  def explain_case(self, 
                   clinical: ClinicalAssessmentOutput, 
                   laboratory: LaboratoryAgentOutput, 
                   diagnostic: DiagnosticAgentOutput
                   ) -> ExplanationBundle:
    # Main entry point: build explanations given upstream agent outputs.
    # 1. Build core sections (plain templates, no LLM)
    diagnosis_heading = self._build_diagnosis_heading(diagnostic)
    confidence_text = self._build_confidence_text(diagnostic)
    summary = self._build_summary_section(clinical, laboratory, diagnostic)
    reasoning = self._build_reasoning_section(clinical, laboratory, diagnostic)
    clinical_implications = self._build_implications_section(diagnostic)
    recommended_actions = self._build_recommendations_section(laboratory, diagnostic)
    data_quality_notes = self._build_data_quality_notes(laboratory) \
      if self.config.include_data_quality_section else None

    raw_sections = {
      "diagnosis_heading": diagnosis_heading,
      "confidence_text": confidence_text,
      "summary": summary,
      "reasoning": reasoning,
      "clinical_implications": clinical_implications,
      "recommended_actions": recommended_actions,
      "data_quality_notes": data_quality_notes or "",
    }

    # 2. Optional LLM pass (tightly constrained)
    if self.llm_rewriter is not None:
      summary = self.llm_rewriter(summary, "clinician_summary")
      reasoning = self.llm_rewriter(reasoning, "clinician_reasoning")
      clinical_implications = self.llm_rewriter(clinical_implications, "clinician_implications")
      recommended_actions = self.llm_rewriter(recommended_actions, "clinician_recommendations")
      if data_quality_notes:
        data_quality_notes = self.llm_rewriter(data_quality_notes, "clinician_data_quality")

    clinician_report = ClinicianReport(
      diagnosis_heading = diagnosis_heading,
      confidence_text = confidence_text,
      summary = summary,
      reasoning = reasoning,
      clinical_implications = clinical_implications,
      recommended_actions = recommended_actions,
      data_quality_notes = data_quality_notes,
    )

    patient_summary: Optional[PatientSummary] = None
    if self.config.include_patient_friendly:
      patient_summary = self._build_patient_summary(clinical, laboratory, diagnostic)

    return ExplanationBundle(
      clinician_report=clinician_report,
      patient_summary=patient_summary,
      raw_sections=raw_sections,
    )
  

# section builder: clinician -----
def build_diagnosis_heading(self, diagnostic: DiagnosticAgentOutput) -> str:
  label = diagnostic.diagnosis_label
  if label == "T2D":
    return "Diagnosis: Type 2 Diabetes"
  elif label == "high_risk":
    return "Diagnosis: High Risk of Type 2 Diabetes"
  elif label == "normal": 
    return "Diagnosis: No cuurent Evidence of Type 2 Diabetes"
  elif label == "uncertain":
    return "Diagnosis: Indeteremine/uncertain"
  else:
    return "Diagnosis: Unknown"


def build_confidence_text(self, diagnostic: DiagnosticAgentOutput) -> str:
  c = diagnostic.diagnosis_confidence
  if math.isnan(c):
    return "Confidence: Unable to determine confidence."
  
  if c >= self.config.high_conf_threshold:
    level = "high"
  elif c >= self.config.moderate_conf_threshold:
    level = "moderate"
  else:
    level = "low"

  return f"Confidence: {level} (estimated probability {c:.2f})."


def build_summary_section(self, clinical: ClinicalAssessmentOutput, laboratory: LaboratoryAgentOutput, diagnostic: DiagnosticAgentOutput) -> str: 
  label = diagnostic.diagnosis_label
  risk_pct = clinical.risk_T2D_now * 100
  lab = laboratory.lab_summary

  lab_str_parts = []
  if lab.primary_marker and lab.primary_value is not None:
    lab_str_parts.append(f"{lab.primary_marker} {lab.primary_value: .1f} {lab.primary_unit or ''}".strip())
  
  if lab.primary_threshold is not None:
    lab_str_parts.append(f"diagnostic threshold {lab.primary_threshold:.1f} {lab.primary_unit or ''}".strip())
  lab_str = ", ".join(lab_str_parts) if lab_str_parts else "no recent diagnostic laboratory result"

  if label == "T2D":
    return (
      "The system classify this case as Type 2 Diabetes. "
      f"The estimated pre-test risk from clinical features was {risk_pct: .0f}%. "
      f"Laboratory evidence shows {lab_str}, consistent with established diagnostic criteria."
    )
  
  if label == "high_risk":
    return (
      "The system identifies this patient as at high risk for developing Type 2 Diabetes. "
      f"Pre-test risk is {risk_pct:.0f}%, with risk driven by the combination of clinical features "
      "and available laboratory information."
    )
  
  if label == "normal":
    return (
      "The system finds no current evidence of Type 2 Diabetes. "
      f"The estimated pre-test risk was {risk_pct:.0f}% and available laboratory data "
      f"({lab_str}) do not meet diagnostic thresholds."
    )
  
  if label == "uncertain":
    return (
      "The system cannot provide a definitive classification in this case. "
      f"The estimated pre-test risk was {risk_pct:.0f}%, but the available laboratory "
      f"data ({lab_str}) are borderline, conflicting or incomplete."
    )
  
  return (
    f"The system produced a diagnostic label '{label}'. "
    f"Pre-test risk was {risk_pct:.0f}% and labs were summarised as: {lab_str}."
  )

  
def _build_reasoning_section(self, clinical: ClinicalAssessmentOutput, laboratory: LaboratoryAgentOutput, diagnostic: DiagnosticAgentOutput) -> str:
  # Top contributors (sorted)
  contrib = sorted(
    clinical.top_contributors.items(),
    key = lambda kv: abs(kv[1]),
    reverse = True
  )
  top_str_parts = []
  for feature, weight in contrib[:5]:
    direction = "increased" if weight > 0 else "reduced"
    top_str_parts.append(f"{feature} ({direction} risk, weight {weight:+.2f})")
  top_features_str = "; ".join(top_str_parts) if top_str_parts else "No dominant risk contributors identified."

  lab = laboratory.lab_summary
  lab_bits = []
  if lab.primary_marker and lab.primary_value is not None:
    comp = ""
    if lab.primary_threshold is not None:
      if lab.primary_value >= lab.primary_threshold:
        comp = " (above diagnostic threshold)"
      else:
        comp = " (below diagnostic threshold)"
    lab_bits.append(
      f"{lab.primary_marker} {lab.primary_value:.1f} {lab.primary_unit or ''}{comp}"
    )
  if lab.recency_days is not None:
    lab_bits.append(f"sample taken {lab.recency_days} days ago")
  if lab.provenance:
    lab_bits.append(f"source: {lab.provenance}")

  lab_str = "; ".join(lab_bits) if lab_bits else "No recent or reliable diagnostic laboratory marker."

  diag_basis = diagnostic.reasoning_tokens.get("basis")
  diag_details = []
  if diag_basis:
    diag_details.append(f"Primary diagnostic basis: {diag_basis}.")
  if "details" in diagnostic.reasoning_tokens:
    diag_details.append(str(diagnostic.reasoning_tokens["details"]))
  diag_details_str = " ".join(diag_details)

  return (
    "How the system reached this conclusion:\n"
    f"- Clinical risk model output: pre-test probability {clinical.risk_T2D_now:.2f} "
    f"with top contributors: {top_features_str}.\n"
    f"- Laboratory evidence: {lab_str}.\n"
    f"- Combined diagnostic reasoning: {diag_details_str or 'Standard threshold-based rules were applied.'}"
  )


def _build_implications_section(self, diagnostic: DiagnosticAgentOutput) -> str:
  label = diagnostic.diagnosis_label

  if label == "T2D":
      return (
          "This pattern is consistent with persistent hyperglycaemia and meets "
          "established criteria for Type 2 Diabetes. Early structured intervention "
          "is recommended to reduce long-term cardiovascular and microvascular risk."
      )
  if label == "high_risk":
      return (
          "Although formal diagnostic thresholds are not met, the current profile "
          "indicates substantially elevated risk. Without intervention, the probability "
          "of future progression to Type 2 Diabetes is increased."
      )
  if label == "normal":
      return (
          "There is no current evidence of Type 2 Diabetes. However, this does not exclude "
          "future risk, particularly if modifiable factors such as BMI, activity and diet are suboptimal."
      )
  if label == "uncertain":
      return (
          "The available evidence is insufficient or conflicting to provide a confident classification. "
          "This may reflect borderline laboratory values, potential measurement limitations, or incomplete data."
      )

  return "Clinical implications depend on further context; the label is non-standard for this pipeline."


def _build_recommendations_section(self, laboratory: LaboratoryAgentOutput, diagnostic: DiagnosticAgentOutput,) -> str:
  # Start from DiagnosticAgent recommendation if present
  base = diagnostic.next_step or ""

  tp = laboratory.test_plan
  plan_bits = []
  if tp.needs_test and tp.test_type != "none":
    plan_bits.append(
      f"Arrange {tp.urgency} {tp.test_type} testing "
      f"(reason: {tp.rationale})."
    )

  if diagnostic.diagnosis_label == "T2D":
    plan_bits.append(
      "Initiate or review a comprehensive Type 2 Diabetes management plan "
      "(lifestyle, pharmacotherapy and cardiovascular risk optimisation) in line with local guidance."
    )
  elif diagnostic.diagnosis_label == "high_risk":
    plan_bits.append(
      "Offer structured lifestyle intervention targeting weight, diet and physical activity. "
      "Consider repeat testing at an appropriate interval."
    )
  elif diagnostic.diagnosis_label == "normal":
    plan_bits.append(
      "Reinforce general lifestyle advice and consider periodic reassessment "
      "if risk factors remain or increase."
    )
  elif diagnostic.diagnosis_label == "uncertain":
    plan_bits.append(
      "Clarify uncertainty with repeat or alternative testing, and review any conditions "
      "that may interfere with HbA1c or glucose measurements."
    )

  if base:
      plan_bits.append(f"System-specific recommendation: {base}")

  return " ".join(plan_bits)


def _build_data_quality_notes(self, laboratory: LaboratoryAgentOutput) -> str:
  lab = laboratory.lab_summary
  notes = []

  if lab.provenance == "self_report":
      notes.append("Key laboratory values are self-reported and may be less reliable than verified EHR data.")
  if lab.recency_days is not None and lab.recency_days > 180:
      notes.append("Laboratory data are older than 6 months; interpretation should be cautious.")
  if lab.reliability_flags:
      notes.append("Additional data quality flags: " + ", ".join(lab.reliability_flags) + ".")

  return " ".join(notes) if notes else "No major data quality concerns identified."



# Patient-facing summary -----
def _build_patient_summary(self, clinical: ClinicalAssessmentOutput, laboratory: LaboratoryAgentOutput, diagnostic: DiagnosticAgentOutput,) -> PatientSummary:
  label = diagnostic.diagnosis_label
  risk_pct = clinical.risk_T2D_now * 100
  lab = laboratory.lab_summary

  if label == "T2D":
    title = "You meet the criteria for Type 2 Diabetes"
    main_message = (
      "Based on your blood test results and health information, your blood sugar levels are "
      "in the range used to diagnose Type 2 Diabetes."
    )
    what_it_means = (
      "This usually reflects ongoing high blood sugar over time. The goal now is to bring your "
      "levels under control and reduce the chance of future complications."
    )
    next_steps = (
      "Your care team may discuss medication, lifestyle changes and follow-up tests. "
      "It is important not to ignore this result, but it is also a condition that can be managed."
    )
  elif label == "high_risk":
    title = "You are at high risk of developing Type 2 Diabetes"
    main_message = (
      f"Your current risk is higher than average (around {risk_pct:.0f}%), "
      "based on your weight, age and other health factors."
    )
    what_it_means = (
      "You do not currently meet the criteria for Type 2 Diabetes, but without changes, your chances "
      "of developing it are increased."
    )
    next_steps = (
      "Changes in diet, activity and weight can significantly reduce this risk. "
      "Your clinician may suggest a plan and possibly further tests."
    )
  elif label == "normal":
    title = "No current evidence of Type 2 Diabetes"
    main_message = (
      "Based on your current information and any available blood tests, you do not "
      "meet the criteria for Type 2 Diabetes."
    )
    what_it_means = (
      "This is reassuring, but it does not mean the risk is zero, especially if some risk factors "
      "such as weight or family history are present."
    )
    next_steps = (
      "Maintaining a healthy lifestyle and attending routine checks will help keep your risk low."
    )
  else:  # uncertain
    title = "The result is not completely clear"
    main_message = (
      "The available information is not enough for a clear answer about Type 2 Diabetes right now."
    )
    what_it_means = (
      "This might be because test results are borderline or incomplete. It does not necessarily mean "
      "you have diabetes, but it also does not fully rule it out."
    )
    next_steps = (
      "Your clinician may organise repeat or different tests and will consider your overall health "
      "when deciding on the next steps."
    )

  # Light lab mention if present
  if lab.primary_marker and lab.primary_value is not None:
    main_message += f" Your latest {lab.primary_marker} result was {lab.primary_value:.1f} {lab.primary_unit or ''}."

  return PatientSummary(
    title=title,
    main_message=main_message,
    what_it_means=what_it_means,
    next_steps=next_steps,
  )


