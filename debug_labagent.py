from datetime import datetime, timezone
from agents.lab_agent import LaboratoryAgent

agent = LaboratoryAgent()

out = agent.assess(
  labs={
    "hba1c": {
      "value": 62,
      "unit": "mmol/mol",
      "timestamp": datetime.now(timezone.utc),
      "source": "ehr",
    }
  },
  clinical_risk={
    "risk_T2D_now": 0.78,
    "triage_label": "high_risk",
  },
  context={},
)

print(out)
print("\n--- as dict ---")
print(out.model_dump())
