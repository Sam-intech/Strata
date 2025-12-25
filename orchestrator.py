from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Literal, Optional, Callable, List
from typing_extensions import TypedDict, NotRequired

import pandas as pd
import joblib
from langgraph.graph import StateGraph, START, END
try:
  from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
  SqliteSaver = None

try:
  from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
  InMemorySaver = None

# -----import agents ----
from agents.data_agent import DataHandlingAgent, FEATURES, TARGET
from agents.clinical_agent import ClinicalAssessmentAgent, ClinicalAssessmentOutput
from agents.lab_agent import LaboratoryAgent
from agents.diagnostic_agent import DiagnosticAgent, LabEvidence, ClinicalAssessmentSnapshot, DiagnosticContext
from agents.explanation_agent import ExplanationAgent
# =============================================================================


Mode = Literal["inference", "evaluation"]

# ----------------------
# centralised logging 
class OrchestrationLogger:
  def __init__(self, logger: Optional[logging.Logger] = None): 
    self._log = logger or logging.getLogger("strata.orchestrator")
    if not self._log.handlers:
      handler = logging.StreamHandler()
      formatter = logging.Formatter("[%(levelname)s] %(message)s")
      handler.setFormatter(formatter)
      self._log.addHandler(handler)
    self._log.setLevel(logging.INFO)

  def event(self, kind: str, node: str, payload: Dict[str, Any]) -> None:
    self._log.info("%s | %s | %s", kind, node, payload)


def logged_node(node_name: str, logger: OrchestrationLogger) -> Callable[[Callable[[Dict[str, Any]], Dict[str, Any]]], Callable[[Dict[str, Any]], Dict[str, Any]]]:
  def deco(fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
      logger.event("node_start", node_name, {"mode": state.get("mode"), "run_id": state.get("run_id")})
      try:
        out = fn(state)
        logger.event("node_end", node_name, {"keys_updated": list(out.keys())})
        return out
      except Exception as e:
        logger.event("node_error", node_name, {"error": repr(e)})
        raise
    
    return wrapped
  return deco



# ---------------------
# Graph State (single source of truth for routing)
class OrchestrationState(TypedDict):
  mode: Mode
  run_id: str
  thread_id: NotRequired[str]

  # dataset passthrough (evaluation)
  dset_df: NotRequired[pd.DataFrame]
  dset_row_index: NotRequired[int]
  dset_row: NotRequired[Dict[str, Any]]
  ground_truth: NotRequired[Any]

  # raw inputs 
  patient_raw: NotRequired[Dict[str, Any]]     
  labs_raw: NotRequired[Dict[str, Any]]        

  # agent outputs 
  data_output: NotRequired[Any]                
  cleaned_row: NotRequired[pd.Series]          
  clinical_output: NotRequired[ClinicalAssessmentOutput]
  lab_output: NotRequired[Any]                 
  diagnostic_output: NotRequired[Any]          
  explanation_output: NotRequired[Any]         

  # aggregation / final output 
  aggregated: NotRequired[Dict[str, Any]]
  final_output: NotRequired[Dict[str, Any]]

  # central logging payload bucket 
  logs: NotRequired[List[Dict[str, Any]]]



# ------------------------------------------------------------------
# Orchestrator (only components that knows all agents + routes data)
@dataclass
class OrchestrationConfig:
  use_checkpointer: bool = False
  sqlite_path: Optional[str] = None

class StrataOrchestrator:
  def __init__(self, data_agent: DataHandlingAgent, clinical_agent: ClinicalAssessmentAgent, lab_agent: LaboratoryAgent, diagnostic_agent: DiagnosticAgent, explanation_agent: Optional[ExplanationAgent] = None, logger: Optional[OrchestrationLogger] = None, config: Optional[OrchestrationConfig] = None,) -> None:
    self.data_agent = data_agent
    self.clinical_agent = clinical_agent
    self.lab_agent = lab_agent
    self.diagnostic_agent = diagnostic_agent
    self.explanation_agent = explanation_agent
    self.logger = logger or OrchestrationLogger()
    self.config = config or OrchestrationConfig()
    self._compiled = self._build_graph()

  # -------------
  # Public API 
  def invoke(self, *, run_id: str, mode: Mode, patient_raw: Optional[Dict[str, Any]] = None, labs_raw: Optional[Dict[str, Any]] = None, dset_df: Optional[pd.DataFrame] = None, dset_row_index: Optional[int] = None, thread_id: Optional[str] = None,) -> Dict[str, Any]:
    state: OrchestrationState = {
      "mode": mode,
      "run_id": run_id,
    }

    if thread_id is not None:
      state["thread_id"] = thread_id

    if mode == "inference":
      if patient_raw is None:
        raise ValueError("mode = 'inference' requires patient_raw")
      state["patient_raw"] = patient_raw
      state["labs_raw"] = labs_raw or {}
    else:
      if dset_df is None or dset_row_index is None:
        raise ValueError("mode = 'evaluation' requires dset_df and dset_row_index")
      state["dset_df"] = dset_df
      state["dset_row_index"] = dset_row_index
      state["labs_raw"] = labs_raw or {}

    cfg = {}
    if thread_id is not None:
      cfg = {"configurable": {"thread_id": thread_id}}

    result_state = self._compiled.invoke(state, config = cfg)
    return result_state["final_output"]

  
  # ------------------
  # Graph construction
  def _build_graph(self):
    g = StateGraph(OrchestrationState)

    # node -----
    g.add_node("dataset", logged_node("dataset", self.logger)(self.node_dataset))
    g.add_node("data_agent", logged_node("data_agent", self.logger)(self.node_data_agent))
    g.add_node("clinical_agent", logged_node("clinical_agent", self.logger)(self.node_clinical_agent))
    g.add_node("lab_agent", logged_node("lab_agent", self.logger)(self.node_lab_agent))
    g.add_node("diagnostic_agent", logged_node("diagnostic_agent", self.logger)(self.node_diagnostic_agent))
    g.add_node("aggregation", logged_node("aggregation", self.logger)(self.node_aggregation))
    g.add_node("explanation", logged_node("explanation", self.logger)(self.node_explanation))
    g.add_node("output", logged_node("output", self.logger)(self.node_output))

    # edges ------
    g.add_edge(START, "dataset")
    g.add_edge("dataset", "data_agent")
    g.add_edge("data_agent", "clinical_agent")
    g.add_edge("clinical_agent", "lab_agent")
    g.add_edge("lab_agent", "diagnostic_agent")
    g.add_edge("diagnostic_agent", "aggregation")
    g.add_edge("aggregation", "explanation")
    g.add_edge("explanation", "output")
    g.add_edge("output", END)

    # checkpointers -----
    checkpointer = None
    if self.config.use_checkpointer:
      if self.config.sqlite_path:
        if SqliteSaver is None:
          raise RuntimeError("SqliteSaver not available. Install compatible langgraph extras.")
        import sqlite3
        conn = sqlite3.connect(self.config.sqlite_path)
        checkpointer = SqliteSaver(conn)
      else:
        if InMemorySaver is None:
          raise RuntimeError("InMemorySaver not available. Install compatible langgraph version.")
        checkpointer = InMemorySaver()
        
    return g.compile(checkpointer = checkpointer)



  # -------------------
  # Nodes
  def node_dataset(self, state: OrchestrationState) -> Dict[str, Any]:
    if state["mode"] == "inference":
      return {}

    dset = state["dset_df"]
    idx = state["dset_row_index"]

    if idx < 0 or idx >= len(dset):
      raise IndexError(f"dset_row_index {idx} out of range (0..{len(dset)-1})")

    row = dset.iloc[idx].to_dict()


    # Extract patient fields -----
    patient_raw = {k: row.get(k) for k in FEATURES if k in row}
    gt = row.get(TARGET, None)

    return {
      "dset_row": row,
      "patient_raw": patient_raw,
      "ground_truth": gt,
    }


  def node_data_agent(self, state: OrchestrationState) -> Dict[str, Any]:
    patient_raw = state["patient_raw"]

    data_out = self.data_agent.ingest_single(patient_raw)

    cleaned_dset = data_out.cleaned
    if len(cleaned_dset) != 1:
      raise ValueError(f"Expected 1-row cleaned data, got {len(cleaned_dset)} rows")

    cleaned_row = cleaned_dset.iloc[0]

    return {
      "data_output": data_out,
      "cleaned_row": cleaned_row
    }


  def node_clinical_agent(self, state: OrchestrationState) -> Dict[str, Any]:
    cleaned_row = state["cleaned_row"]

    clinical_out = self.clinical_agent.predict_single(cleaned_row)

    return {
      "clinical_output": clinical_out
    }


  def node_lab_agent(self, state: OrchestrationState) -> Dict[str, Any]:
    labs_raw = state.get("labs_raw", {}) or {}

    clinical = state["clinical_output"]
    clinical_risk_payload = {
      "risk_T2D_now": clinical.risk_T2D_now,
      "triage_label": clinical.triage_label
    }

    lab_out = self.lab_agent.assess(
      labs = labs_raw,
      clinical_risk = clinical_risk_payload,
      context = {}  
    )

    print("\n[DEBUG] LabAgentOutput:")
    print(lab_out.model_dump())

    return {
      "lab_output": lab_out
    }


  def node_diagnostic_agent(self, state: OrchestrationState) -> Dict[str, Any]:
    clinical = state["clinical_output"]
    lab_out = state["lab_output"]

    # Convert LabAgentOutput -> LabEvidence (what DiagnosticAgent expects)
    labs = lab_out.validated_labs

    # Pull values in normalised units (LabAgent already normalises to mmol/mol & mmol/L)
    evidence = LabEvidence(
      hba1c_mmol_mol=(labs.hba1c.value if labs.hba1c else None),
      fpg_mmol_l=(labs.fpg.value if labs.fpg else None),
      ogtt_2h_mmol_l=(labs.ogtt.value if labs.ogtt else None),
      random_glucose_mmol_l=(labs.random_glucose.value if labs.random_glucose else None),
      is_self_report_only=bool(lab_out.flags.get("any_self_report_lab", False)),
      is_outdated=False,
      has_quality_issues=bool(lab_out.flags.get("lab_potentially_unreliable_context", False))
        or bool(lab_out.flags.get("hba1c_out_of_range", False))
        or bool(lab_out.flags.get("fpg_out_of_range", False))
        or bool(lab_out.flags.get("ogtt_out_of_range", False)),
      raw_meta=lab_out.meta,
    )

    clinical_snap = ClinicalAssessmentSnapshot(
      risk_T2D_now=clinical.risk_T2D_now,
      triage_label=clinical.triage_label,
      raw_proba_vector=clinical.raw_proba_vector,
      meta=clinical.meta,
    )

    diag_ctx = DiagnosticContext(raw_meta={"mode": state["mode"]})

    diagnostic_out = self.diagnostic_agent.diagnose(
      labs=evidence,
      clinical=clinical_snap,
      ctx=diag_ctx,
    )

    return {"diagnostic_output": diagnostic_out}


  def node_aggregation(self, state: OrchestrationState) -> Dict[str, Any]:
    data_out = state["data_output"]
    clinical = state["clinical_output"]
    lab_out = state["lab_output"]
    diag_out = state["diagnostic_output"]

    agg: Dict[str, Any] = {
      "mode": state["mode"],
      "run_id": state["run_id"],
      "patient_raw": state.get("patient_raw", {}),
      "data_validation_errors": data_out.validation_errors,
      "data_flags": data_out.flags.to_dict(orient="records")[0] if hasattr(data_out.flags, "to_dict") else {},
      "clinical": {
        "risk_T2D_now": clinical.risk_T2D_now,
        "triage_label": clinical.triage_label,
        "top_contributors": clinical.top_contributors,
        "meta": clinical.meta,
      },
      "laboratory": {
        "test_plan": lab_out.test_plan.model_dump() if hasattr(lab_out.test_plan, "model_dump") else lab_out.test_plan,
        "flags": lab_out.flags,
        "interpretation_tokens": lab_out.lab_interpretation_tokens,
        "meta": lab_out.meta,
      },
      "diagnostic": {
        "label": diag_out.label,
        "confidence": diag_out.confidence,
        "next_step": diag_out.next_step,
        "basis": diag_out.basis,
        "reasoning_tokens": diag_out.reasoning_tokens,
      },
    }

    # Dataset passthrough for evaluation
    if state["mode"] == "evaluation":
      agg["dset_row_index"] = state.get("dset_row_index")
      agg["dset_row"] = state.get("dset_row")
      agg["ground_truth"] = state.get("ground_truth")

    return {"aggregated": agg}
  

  def node_explanation(self, state: OrchestrationState) -> Dict[str, Any]:
    if self.explanation_agent is None:
      return {"explanation_output": None}

    clinical = state["clinical_output"]
    lab_out = state["lab_output"]
    diag_out = state["diagnostic_output"]

    # Your explanation_agent.py defines its own types; we keep it pragmatic:
    # if your ExplanationAgent expects specific dataclasses, adapt here.
    explanation = self.explanation_agent.explain_case(
      clinical=clinical,          # ClinicalAssessmentOutput
      laboratory=lab_out,         # LabAgentOutput
      diagnostic=diag_out,        # DiagnosticResult
    )

    return {"explanation_output": explanation}
  

  def node_output(self, state: OrchestrationState) -> Dict[str, Any]:
    agg = state["aggregated"]
    explanation = state.get("explanation_output")

    out: Dict[str, Any] = {
      "result": agg,
      "explanation": None,
    }

    if explanation is not None:
      # try to serialise in a stable way
      if hasattr(explanation, "clinician_report") and hasattr(explanation.clinician_report, "as_dict"):
        out["explanation"] = {
          "clinician_report": explanation.clinician_report.as_dict(),
          "patient_summary": explanation.patient_summary.__dict__ if getattr(explanation, "patient_summary", None) else None,
        }
      elif hasattr(explanation, "__dict__"):
        out["explanation"] = explanation.__dict__
      else:
        out["explanation"] = explanation

    return {"final_output": out}
  

# ----------------------
def build_orchestrator(*, model_path: Path, enable_explanations: bool = True, use_checkpointer: bool = False, sqlite_path: Optional[str] = None, logger: Optional[OrchestrationLogger] = None) -> "StrataOrchestrator":
  data_agent = DataHandlingAgent()

  model = joblib.load(model_path)
  clinical_agent = ClinicalAssessmentAgent(model = model)

  lab_agent = LaboratoryAgent()
  diagnostic_agent = DiagnosticAgent()

  explanation_agent = ExplanationAgent() if enable_explanations else None

  cfg = OrchestrationConfig(
    use_checkpointer = use_checkpointer,
    sqlite_path = sqlite_path,
  )


  return StrataOrchestrator(
    data_agent = data_agent,
    clinical_agent = clinical_agent,
    lab_agent = lab_agent,
    diagnostic_agent = diagnostic_agent,
    explanation_agent = explanation_agent,
    logger = logger or OrchestrationLogger(),
    config = cfg,
  )
  





