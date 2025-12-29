from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol
import json
import time
# ==================================================================


class LLMClient(Protocol):
  def generate(self, *, system: str, user: str, temperature: float = 0.2) -> str: ...


@dataclass
class ExplanationConfig:
  temperature: float = 0.2
  max_trace_chars: int = 20_000
  force_json_output: bool = True


class ExplanationAgent:
  def __init__(self, llm: LLMClient, config: Optional[ExplanationConfig] = None) -> None:
    self.llm = llm
    self.config = config or ExplanationConfig()

  def render(self, *, trace: Dict[str, Any]) -> Dict[str, Any]:
    trace_json = self._trace_to_json(trace)
    prompt = self._build_user_prompt(trace_json=trace_json, mode=str(trace.get("mode", "inference")))

    clinician_raw = self.llm.generate(
      system = self._system_prompt(),
      user = prompt,
      temperature = self.config.temperature,
    )

    clinician = self._wrap_rendered(clinician_raw)

    return {
      "clinician_report": clinician,
      "meta": {
        "rendered_at_unix": time.time(),
        "llm_temperature": self.config.temperature,
        "force_json_output": self.config.force_json_output,
      },
    }

  # -----------------------------------
  # internals (prompt hygiene only)
  def _trace_to_json(self, trace: Dict[str, Any]) -> str:
    s = json.dumps(trace, ensure_ascii=False, sort_keys=True, indent=2)
    if len(s) > self.config.max_trace_chars:
      s = s[: self.config.max_trace_chars] + "\n... [TRUNCATED_FOR_PROMPT_BUDGET]"
    return s

  def _build_user_prompt(self, *, trace_json: str, mode: str) -> str:
    base = (
      "You are given a JSON TRACE produced by an orchestrator in a multi-agent system for early T2D assessment.\n"
      "Your task is to render a clinician-facing post-hoc explanation STRICTLY from the TRACE.\n"
      "Rules:\n"
      "- Use ONLY facts explicitly present in the TRACE.\n"
      "- Do NOT add medical thresholds, guidelines, or interpretations not already present in the TRACE.\n"
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
      base += "- evaluation-only: dset_row_index, ground_truth (and optionally dset_row)\n"

    base += "\nTRACE (JSON):\n" + trace_json + "\n\n"

    if self.config.force_json_output:
      base += (
        "OUTPUT FORMAT: STRICT JSON ONLY (no markdown, no extra text).\n"
        "JSON schema:\n"
        "{\n"
        '  "summary": "1-3 sentences",\n'
        '  "evidence": ["bullet 1", "bullet 2"],\n'
        '  "uncertainty_and_data_quality": ["bullet 1", "bullet 2"],\n'
        '  "next_steps": ["bullet 1", "bullet 2"]\n'
        "}\n"
        "Every evidence bullet MUST reference a trace field path in parentheses, e.g. "
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

  def _system_prompt(self) -> str:
    return (
      "You are a clinician report renderer. "
      "Be concise, structured, and faithful to the provided trace. "
      "Do not introduce new clinical guidance."
    )

  def _wrap_rendered(self, raw: str) -> Dict[str, Any]:
    txt = (raw or "").strip()

    if self.config.force_json_output:
      try:
        obj = json.loads(txt)
        if not isinstance(obj, dict):
          raise ValueError("Rendered JSON is not an object.")
        return {"json": obj}
      except Exception:
        return {"text": txt, "parse_error": "LLM did not return valid JSON."}

    return {"text": txt}
