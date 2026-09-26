from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _run_from_backend_root(monkeypatch):
  # Artifact and dataset paths in the code are relative to backend/
  monkeypatch.chdir(BACKEND_ROOT)
