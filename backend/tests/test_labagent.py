from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from datetime import datetime, timedelta, timezone

import pytest

from agents.lab_agent import LaboratoryAgent  # adjust if your import path differs
# =====================================================================

def test_no_labs_high_risk_orders_hba1c():
  agent = LaboratoryAgent()
  now = datetime(2025, 1, 1, tzinfo=timezone.utc)

  out = agent.assess(
    labs={},
    clinical_risk={"risk_T2D_now": 0.85, "triage_label": "high_risk"},
    context={},
    now=now,
  )

  assert out.test_plan.needs_test is True
  assert out.test_plan.test_type == "HbA1c"
  assert out.test_plan.urgency == "priority"


def test_old_lab_moderate_risk_retest():
  agent = LaboratoryAgent()
  now = datetime(2025, 1, 1, tzinfo=timezone.utc)

  old_ts = now - timedelta(days=400)

  out = agent.assess(
    labs={
      "hba1c": {"value": 46, "unit": "mmol/mol", "timestamp": old_ts, "source": "ehr"}
    },
    clinical_risk={"risk_T2D_now": 0.45, "triage_label": "routine_follow_up"},
    context={},
    now=now,
  )

  assert out.test_plan.needs_test is True
  assert out.test_plan.test_type == "repeat_test"
  assert out.flags["any_outdated_lab"] is True


def test_recent_diabetic_hba1c_no_new_test_and_interpretation():
  agent = LaboratoryAgent()
  now = datetime(2025, 1, 1, tzinfo=timezone.utc)
  recent_ts = now - timedelta(days=5)

  out = agent.assess(
    labs={
      "hba1c": {"value": 62, "unit": "mmol/mol", "timestamp": recent_ts, "source": "ehr"}
    },
    clinical_risk={"risk_T2D_now": 0.20, "triage_label": "low_risk"},
    context={},
    now=now,
  )

  assert out.test_plan.needs_test is False
  assert "hba1c" in out.lab_interpretation_tokens
  assert out.lab_interpretation_tokens["hba1c"]["category"] == "diabetic_range"
  assert out.flags["any_recent_lab"] is True


def test_pregnancy_prefers_ogtt_when_high_risk_and_no_labs():
  agent = LaboratoryAgent()
  now = datetime(2025, 1, 1, tzinfo=timezone.utc)

  out = agent.assess(
    labs={},
    clinical_risk={"risk_T2D_now": 0.90, "triage_label": "high_risk"},
    context={"pregnancy": True},
    now=now,
  )

  assert out.test_plan.needs_test is True
  assert out.test_plan.test_type == "OGTT"


def test_self_report_flagged():
  agent = LaboratoryAgent()
  now = datetime(2025, 1, 1, tzinfo=timezone.utc)

  out = agent.assess(
    labs={
      "fpg": {"value": 8.0, "unit": "mmol/L", "timestamp": now, "source": "self_report"}
    },
    clinical_risk={"risk_T2D_now": 0.50, "triage_label": "routine_follow_up"},
    context={},
    now=now,
  )

  assert out.flags["any_self_report_lab"] is True





# =================
if __name__ == "__main__":
    test_no_labs_high_risk_orders_hba1c()
    test_old_lab_moderate_risk_retest()
    test_recent_diabetic_hba1c_no_new_test_and_interpretation()
    test_pregnancy_prefers_ogtt_when_high_risk_and_no_labs()
    test_self_report_flagged()
    print("All LabAgent tests passed.")


