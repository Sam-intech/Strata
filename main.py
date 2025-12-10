from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from pydantic import ValidationError

# import your actual agents & schemas
from src.agents.data_agent import DataHandlingAgent, PatientInput
from src.agents.clinical_agent import ClinicalAssessmentAgent, ClinicalAssessmentOutput
from src.agents.lab_agent import LaboratoryAgent
from src.agents.diagnostic_agent import DiagnosticAgent
from src.agents.explanation_agent import ExplanationAgent  # when you finish it
# ===========================================================================================================================




@dataclass
class MASResult:
  raw_input: Dict[str, Any]
  patient_clean: Dict[str, Any]
  labs_clean: Optional[Dict[str, Any]]

  clinical_output: ClinicalAssessmentOutput
  lab_output: Dict[str, Any]
  diagnostic_output: Dict[str, Any]

  explanation_clinician: Optional[str] = None
  explanation_patient: Optional[str] = None
  meta: Optional[Dict[str, Any]] = None


class MASOrchestrator:
  """
  Central workflow controller for the MAS.
  Handles:
    UI -> Data -> Clinical -> Lab -> Diagnostic -> Explanation.
  """

  def __init__(self, data_agent: DataHandlingAgent, clinical_agent: ClinicalAssessmentAgent, lab_agent: LaboratoryAgent, diagnostic_agent: DiagnosticAgent, explanation_agent: Optional[ExplanationAgent] = None,) -> None:
    self.data_agent = data_agent
    self.clinical_agent = clinical_agent
    self.lab_agent = lab_agent
    self.diagnostic_agent = diagnostic_agent
    self.explanation_agent = explanation_agent

  def run_pipeline(self, patient_raw: Dict[str, Any], labs_raw: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None,) -> MASResult:
    """
    End-to-end pipeline for a single patient.
    Adapt method names to your actual agent APIs.
    """
    context = context or {}

    # 1) Validate & normalise raw input via Pydantic schema
    try:
        patient_input = PatientInput(**patient_raw)
    except ValidationError as e:
        # In a real API you'd return a 400 here
        raise ValueError(f"Invalid patient input: {e}") from e

    # 2) Data Handling Agent
    data_out = self.data_agent.process(
        patient_input=patient_input,
        labs_raw=labs_raw,
        context=context,
    )
    # Expect something like:
    # data_out.patient_clean, data_out.labs_clean, data_out.quality_flags

    patient_clean = data_out.patient_clean
    labs_clean = data_out.labs_clean

    # 3) Clinical Assessment Agent
    clinical_out: ClinicalAssessmentOutput = self.clinical_agent.assess(
        patient_features=patient_clean,
        labs=labs_clean,
        context=context,
    )

    # 4) Laboratory Agent
    lab_out = self.lab_agent.decide(
        patient_features=patient_clean,
        labs=labs_clean,
        clinical_output=clinical_out,
        context=context,
    )
    # Expect: lab_out.test_plan, lab_out.labs_validated, lab_out.flags

    labs_for_diag = lab_out.labs_validated or labs_clean

    # 5) Diagnostic Agent
    diagnostic_out = self.diagnostic_agent.diagnose(
        labs=labs_for_diag,
        clinical_output=clinical_out,
        context=context,
        lab_metadata={
            "test_plan": lab_out.test_plan,
            "flags": lab_out.flags,
        },
    )

    # 6) Explanation Agent (optional at first)
    explanation_clinician = None
    explanation_patient = None

    if self.explanation_agent is not None:
      explanation = self.explanation_agent.generate(
        diagnostic_output=diagnostic_out,
        clinical_output=clinical_out,
        lab_output=lab_out,
        patient_features=patient_clean,
        context=context,
      )
      
    # design this however you like
    explanation_clinician = explanation.clinician_text
    explanation_patient = getattr(explanation, "patient_text", None)

    # 7) Bundle everything
    result = MASResult(
      raw_input=patient_raw,
      patient_clean=patient_clean,
      labs_clean=labs_for_diag,
      clinical_output=clinical_out,
      lab_output={
        "test_plan": lab_out.test_plan,
        "flags": lab_out.flags,
        "labs_validated": lab_out.labs_validated,
      },
      diagnostic_output=diagnostic_out.to_dict()
      if hasattr(diagnostic_out, "to_dict")
      else diagnostic_out,
      explanation_clinician=explanation_clinician,
      explanation_patient=explanation_patient,
      meta={"quality_flags": data_out.quality_flags},
    )

    return result
