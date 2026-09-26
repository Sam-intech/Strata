from datetime import date

from fastapi.testclient import TestClient

import api
# ======================================================================================


PAYLOAD = {
  "age": 52,
  "gender": "male",
  "bmi": 34.8,
  "bp_systolic": 142,
  "hypertension": 1,
  "heartDisease": 0,
  "smoking": "former",
  "labs": {
    "hba1c": 53,
    "hba1cUnit": "mmol/mol",
    "fpg": 7.4,
    "fpgUnit": "mmol/L",
  },
}


def test_health_and_infer(monkeypatch):
  # keep the test offline: no LLM explanation call
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)

  with TestClient(api.app) as client:
    health = client.get("/health").json()
    assert health["ok"] is True
    assert health["orchestrator_loaded"] is True

    res = client.post("/infer", json={"payload": PAYLOAD})
    assert res.status_code == 200, res.text

    out = res.json()["final_output"]
    assert 0.0 <= out["clinical"]["risk_T2D_now"] <= 1.0
    assert out["clinical"]["triage_label"] in {"low", "medium", "high", "critical"}
    assert out["diagnostic"]["label"]
    assert "evaluation" not in out


def test_payload_unit_conversion():
  patient = api.map_frontend_payload_to_patient_input(PAYLOAD)

  assert round(patient["hba1c"], 1) == 7.0      # 53 mmol/mol -> ~7.0 %
  assert round(patient["glucose"]) == 133       # 7.4 mmol/L -> ~133 mg/dL
  assert patient["blood_pressure"] == 142.0
  assert patient["hypertension"] is True
  assert patient["heart_disease"] is False


def test_recent_lab_date_is_used(monkeypatch):
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)
  today = date.today().isoformat()
  payload = {**PAYLOAD, "labs": {"hba1c": 53, "hba1cUnit": "mmol/mol", "hba1cDate": today}}

  with TestClient(api.app) as client:
    out = client.post("/infer", json={"payload": payload}).json()["final_output"]

  # a same-day HbA1c is recent, so no retest should be requested
  assert out["laboratory"]["test_plan"]["need_retest"] is False
