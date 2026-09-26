from __future__ import annotations
import os

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from openai import OpenAI
# ==============================================================


@dataclass
class OpenAILLMConfig:
  api_key: Optional[str] = None
  model: str = "gpt-4.1-mini"
  timeout_s: float = 30.0
  # Any OpenAI-compatible endpoint (Cloudflare Workers AI, Groq, Gemini, ...).
  # None = OpenAI itself.
  base_url: Optional[str] = None


class OpenAILLMClient:
  def __init__(self, config: Optional[OpenAILLMConfig] = None) -> None:
    self.config = config or OpenAILLMConfig()
    # self.client = OpenAI(api_key=self.config.api_key)

    api_key = self.config.api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
      raise RuntimeError(
        "LLM_API_KEY / OPENAI_API_KEY not set. Export it or provide via OpenAILLMConfig."
      )

    self.client = OpenAI(api_key=api_key, base_url=self.config.base_url)

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
