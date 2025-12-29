from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol
import json
import time
# =============================================================================



Audience = Literal["clinician", "patient"]


class LLMClient(Protocol):
  def generate(self, *, system: str, user: str, temperature: float = 0.2) -> str: ...


@dataclass
class ExplanationConfig:
  temperature: float = 0.2
  max_trace_chars: int = 20_000
  include_patient_summary: bool = True
  force_json_output: bool = True



# -----------------------------
# Explanation Agent (renderer only)
class ExplanationAgent:
  def __init__(self, llm: LLMClient, config: Optional[ExplanationConfig] = None) -> None:
    self.llm = llm
    self.config = config or ExplanationConfig()

  def render(self, *, trace: Dict[str, Any]) -> Dict[str, Any]: 
    trace_json = self._trace_to_json(trace)
    prompt = self._build_user_prompt(trace_json = trace_json, mode = trace.get("mode", "inference"))

    clinician_raw = self.llm.generate(
      system = self._system_prompt("clinician"),
      user = prompt,
      temperature = self.config.temperature,
    )

    patient_raw = None
    if self.config.include_patient_summary:
      patient_raw = self.llm.generate(
        system = self._system_prompt("patient"),
        user = prompt,
        temperature = self.config.temperature,
      )

    clinician = self._wrap_rendered(clinician_raw, audience="clinician")
    patient = self._wrap_rendered(patient_raw, audience="patient") if patient_raw else None
    
    return {
      "clinician_report": self._wrap_text_report(clinician),
      "patient_summary": self._wrap_text_report(patient) if patient else None,
      "meta": {
        "rendered_at_unix": time.time(),
        "llm_temperature": self.config.temperature,
      },
    }
  

  def _wrap_text_report(self, payload: Any) -> Dict[str, Any]:
    if payload is None:
      return {"text": ""}

    # If _wrap_rendered returned its normal dict shape
    if isinstance(payload, dict):
      if "json" in payload and isinstance(payload["json"], dict):
        return {"json": payload["json"]}
      if "text" in payload:
        return {"text": str(payload["text"])}
      # Unknown dict shape: keep it but stringify safely
      return {"text": json.dumps(payload, ensure_ascii=False)}

    # If someone passed a raw string
    if isinstance(payload, str):
      return {"text": payload.strip()}

    # Anything else
    return {"text": str(payload)}



  # ---------------------------------------------------------------------------
  # internals (prompt hygiene only)
  def _trace_to_json(self, trace: Dict[str, Any]) -> str:
    s = json.dumps(trace, ensure_ascii=False, sort_keys=True, indent=2)
    if len(s) > self.config.max_trace_chars:
      s = s[: self.config.max_trace_chars] + "\n... [TRUNCATED_FOR_PROMPT_BUDGET]"
    return s

  def _build_user_prompt(self, *, trace_json: str, mode: str) -> str:
    base = (
      "You are given a JSON TRACE produced by an orchestrator in a multi-agent system for early T2D assessment.\n"
      "Your task is to render a post-hoc explanation STRICTLY from the TRACE.\n"
      "Rules:\n"
      "- Use ONLY facts explicitly present in the TRACE.\n"
      "- Do NOT add medical thresholds, guidelines, or interpretations that are not already in the TRACE.\n"
      "- Do NOT compute or transform values.\n"
      "- If information is missing, say it is missing.\n"
      "- If the trace contains warnings/flags/errors, include them.\n\n"
      "TRACE KEYS YOU MAY CITE (if present):\n"
      "- data_validation_errors, data_flags\n"
      "- clinical: risk_T2D_now, triage_label, top_contributors, meta\n"
      "- laboratory: test_plan, flags, interpretation_tokens, meta\n"
      "- diagnostic: label, confidence, next_step, basis, reasoning_tokens\n"
    )

    if mode == "evaluation":
      base += (
        "- evaluation-only: dset_row_index, ground_truth (and optionally dset_row)\n"
      )

    base += "\nTRACE (JSON):\n" + trace_json + "\n\n"

    if self.config.force_json_output:
      base += (
        "OUTPUT FORMAT: STRICT JSON ONLY (no markdown, no extra text).\n"
        "Clinician JSON schema:\n"
        "{\n"
        '  "summary": "1-3 sentences",\n'
        '  "evidence": ["bullet 1", "bullet 2"],\n'
        '  "uncertainty_and_data_quality": ["bullet 1", "bullet 2"],\n'
        '  "next_steps": ["bullet 1", "bullet 2"]\n'
        "}\n"
        "Evidence bullets MUST reference a trace field path in parentheses, e.g. "
        '"High risk label (diagnostic.label)" or "HbA1c requested (laboratory.test_plan)".\n'
      )
    else:
      base += (
        "Output format:\n"
        "1) Summary (1-3 sentences)\n"
        "2) Evidence cited (bullets; each bullet must point to something in TRACE)\n"
        "3) Uncertainty / data-quality notes (bullets; only if present)\n"
        "4) Next steps (only if present under diagnostic.next_step or similar)\n"
      )

    return base
 
  def _system_prompt(self, audience: Audience) -> str:
    if audience == "clinician":
      return (
        "You are a clinical report renderer. "
        "Be concise, structured, and faithful to the provided trace. "
        "No new clinical guidance."
      )
    return (
      "You are a patient-friendly summary renderer. "
      "Use plain language, avoid jargon, and be faithful to the trace. "
      "No new medical advice."
    )
  
  def _wrap_rendered(self, raw: str, *, audience: Audience) -> Dict[str, Any]:
    if raw is None:
      return {"text": ""}

    txt = raw.strip()

    if self.config.force_json_output:
      try:
        obj = json.loads(txt)
        # Minimal schema validation (keep it light)
        if not isinstance(obj, dict):
          raise ValueError("Rendered JSON is not an object.")
        return {"json": obj}
      except Exception:
        # Fall back; do NOT crash the pipeline because the LLM returned non-JSON.
        return {"text": txt, "parse_error": "LLM did not return valid JSON."}

    return {"text": txt}
