# llm_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from openai import OpenAI
# ==============================================================


@dataclass
class OpenAILLMConfig:
  api_key: Optional[str] = REDACTED_OPENAI_KEY          # if None, OpenAI() will use env var OPENAI_API_KEY
  model: str = "gpt-4.1-mini"
  timeout_s: float = 30.0


class OpenAILLMClient:
  def __init__(self, config: Optional[OpenAILLMConfig] = None) -> None:
    self.config = config or OpenAILLMConfig()
    self.client = OpenAI(api_key=self.config.api_key)

  def generate(self, *, system: str, user: str, temperature: float = 0.2) -> str:
    resp = self.client.chat.completions.create(
      model=self.config.model,
      temperature=temperature,
      messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
      ],
      timeout=self.config.timeout_s,
    )
    return resp.choices[0].message.content or ""
