# api.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from orchestrator import build_orchestrator
# ==========================================================
# app = FastAPI(title="Strata Clinical API", version="1.0.0")

# # Dev-safe localhost CORS (any port)
# app.add_middleware(
#   CORSMiddleware,
#   allow_origin_regex=r"^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$",
#   allow_credentials=False,
#   allow_methods=["*"],
#   allow_headers=["*"],
# )


MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/diabetes_model.joblib")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.joblib")

# Frontend origins (Vite default: http://localhost:5173)
# ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")



# -----------------------------------------------------------------------------
# Small unit conversion helpers (keeps model inputs consistent)
def mmolL_to_mgdl(x: float) -> float:
  return x * 18.0182

def mgdl_to_mmolL(x: float) -> float:
  return x / 18.0182

def hba1c_mmolmol_to_percent(x: float) -> float:
  # NGSP(% ) = (IFCC mmol/mol + 2.152) / 10.929
  return (x + 2.152) / 10.929

def parse_float(v: Any) -> Optional[float]:
  try:
    if v is None or v == "":
      return None
    return float(v)
  except Exception:
    return None


# -----------------------------------------------------------------------------
# Request/Response models (loose on purpose: your UI payload can evolve)
class InferRequest(BaseModel):
  payload: Dict[str, Any] = Field(default_factory=dict)


class InferResponse(BaseModel):
  run_id: str
  final_output: Dict[str, Any]


# -----------------------------------------------------------------------------
# App
app = FastAPI(title="Strata Clinical API", version="1.0.0")

app.add_middleware(
  CORSMiddleware,
  allow_origin_regex=r"^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$",
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

ORCH = None  # initialized on startup


@app.on_event("startup")
def _startup() -> None:
  global ORCH
  # Fail fast if paths are wrong (common cause of silent 500s later)
  if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")
  if not os.path.exists(PREPROCESSOR_PATH):
    raise RuntimeError(f"PREPROCESSOR_PATH not found: {PREPROCESSOR_PATH}")

  ORCH = build_orchestrator(
    model_path=MODEL_PATH,
    preprocessor_path=PREPROCESSOR_PATH,
  )


@app.get("/health")
def health() -> Dict[str, Any]:
  return {
    "ok": True,
    "model_path": MODEL_PATH,
    "preprocessor_path": PREPROCESSOR_PATH,
    "orchestrator_loaded": ORCH is not None,
  }


def build_labs_raw(front: Dict[str, Any]) -> Dict[str, Any]:
  labs = front.get("labs") or {}

  def meas(val_key: str, unit_key: str, date_key: str) -> Optional[Dict[str, Any]]:
    v = parse_float(labs.get(val_key))
    if v is None:
      return None
    out: Dict[str, Any] = {"value": v}
    u = labs.get(unit_key)
    if u:
      out["unit"] = str(u)
    d = labs.get(date_key)
    if d:
      out["date"] = str(d)
    return out

  built: Dict[str, Any] = {}

  h = meas("hba1c", "hba1cUnit", "hba1cDate")
  if h is not None:
    built["hba1c"] = h

  f = meas("fpg", "fpgUnit", "fpgDate")
  if f is not None:
    built["fpg"] = f

  o = meas("ogtt", "ogttUnit", "ogttDate")
  if o is not None:
    built["ogtt"] = o

  e = parse_float(labs.get("egfr"))
  if e is not None:
    built["egfr"] = {"value": e}

  return built

# -----------------------------------------------------------------------------
# Mapping: Frontend payload -> backend PatientInput (data_agent.py schema)
def map_frontend_payload_to_patient_input(front: Dict[str, Any]) -> Dict[str, Any]:
  labs = front.get("labs") or {}
  smoking = front.get("smoking")

  # Core
  age = parse_float(front.get("age"))
  bmi = parse_float(front.get("bmi"))

  # BP: backend expects one "blood_pressure" number (your schema uses float)
  # We'll store systolic if present, else try parse "bp" or fallback None.
  bp_sys = parse_float(front.get("bp_systolic") or front.get("bpSys"))
  bp_dia = parse_float(front.get("bp_diastolic") or front.get("bpDia"))
  blood_pressure = bp_sys if bp_sys is not None else None

  # Hypertension / heart disease: backend expects bool
  hypertension = front.get("hypertension")
  heart_disease = front.get("heartDisease") if "heartDisease" in front else front.get("heart_disease")


  hba1c_val = parse_float(labs.get("hba1c"))
  hba1c_unit = (labs.get("hba1cUnit") or labs.get("hba1c_unit") or "").strip()

  if hba1c_val is not None:
    if hba1c_unit.lower() in ["mmol/mol", "mmolmol", "mmol_per_mol"]:
      hba1c_val = hba1c_mmolmol_to_percent(hba1c_val)
    # "%" stays as-is

  glucose_val = parse_float(labs.get("fpg"))
  glucose_unit = (labs.get("fpgUnit") or labs.get("fpg_unit") or "").strip()

  if glucose_val is None:
    glucose_val = parse_float(labs.get("ogtt"))
    glucose_unit = (labs.get("ogttUnit") or labs.get("ogtt_unit") or "").strip()

  if glucose_val is not None:
    if glucose_unit.lower() in ["mmol/l", "mmoll"]:
      glucose_val = mmolL_to_mgdl(glucose_val)
    # "mg/dL" stays as-is

  patient_input: Dict[str, Any] = {
    "gender": front.get("gender"),
    "age": age,
    "bmi": bmi,
    "glucose": glucose_val,
    "hba1c": hba1c_val,
    "blood_pressure": blood_pressure,
    "hypertension": bool(int(hypertension)) if isinstance(hypertension, (int, str)) and str(hypertension).isdigit() else bool(hypertension) if hypertension is not None else None,
    "heart_disease": bool(int(heart_disease)) if isinstance(heart_disease, (int, str)) and str(heart_disease).isdigit() else bool(heart_disease) if heart_disease is not None else None,
    "smoking_history": smoking,
    # Optional (not currently collected in your form):
    "insulin": None,
  }

  # Remove None values so pydantic + downstream stays clean
  return {k: v for k, v in patient_input.items() if v is not None}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
  global ORCH
  if ORCH is None:
    raise HTTPException(status_code=500, detail="Orchestrator not initialized")
  
  front = req.payload or {}

  try:
    patient_raw = map_frontend_payload_to_patient_input(front)
    lab_raw = build_labs_raw(front)
  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Bad payload mapping: {e}")
  
  run_id = f"run_{time.strftime('%Y_%m_%d')}_{int(time.time())}"

  try:
    final_output = ORCH.invoke(
      run_id=run_id,
      mode="inference",
      patient_raw=patient_raw,
      labs_raw=lab_raw,
    )
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

  # run_id = f"run_{time.strftime('%Y_%m_%d')}_{int(time.time())}"
  return InferResponse(run_id=run_id, final_output=final_output)


# -----------------------------------------------------------------------------
# Run locally:
#   uvicorn api:app --reload --host 0.0.0.0 --port 8000
# -----------------------------------------------------------------------------
