from src.agents.data_agent import DataHandlingAgent
from src.agents.clinical_agent import ClinicalAssessmentAgent, load_clinical_model
from src.agents.lab_agent import LaboratoryAgent
from src.agents.diagnostic_agent import DiagnosticAgent
from src.agents.explanation_agent import ExplanationAgent

from main import MASOrchestrator
# ======================================================================================

def build_orchestrator() -> MASOrchestrator:
  data_agent = DataHandlingAgent()
  clinical_model = load_clinical_model("models/clinical.pkl")
  clinical_agent = ClinicalAssessmentAgent(model=clinical_model)
  lab_agent = LaboratoryAgent()
  diagnostic_agent = DiagnosticAgent()
  explanation_agent = ExplanationAgent()

  return MASOrchestrator(
    data_agent=data_agent,
    clinical_agent=clinical_agent,
    lab_agent=lab_agent,
    diagnostic_agent=diagnostic_agent,
    explanation_agent=explanation_agent,
  )


if __name__ == "__main__":
  orch = build_orchestrator()

  patient_raw = {
    "gender": "male",
    "age": 52,
    "bmi": 31.2,
    "glucose": 6.9,
    "hba1c": 45,
    "blood_pressure": 138,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "never",
    "insulin": 0,
  }

  result = orch.run_pipeline(patient_raw=patient_raw)

  print("Diagnosis:", result.diagnostic_output)
  print("Explanation (clinician):", result.explanation_clinician)
