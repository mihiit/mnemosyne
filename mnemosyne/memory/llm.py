"""
Thin wrapper around Ollama so consolidation/contradiction logic doesn't
talk to the client library directly. Swap models or providers later by
changing only this file.
"""

import json
from typing import Optional

import ollama

from mnemosyne.config import MnemosyneConfig


class LocalLLM:
    def __init__(self, config: MnemosyneConfig):
        self.config = config

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=self.config.llm_model,
            messages=messages,
            options={"temperature": self.config.llm_temperature},
        )
        return response["message"]["content"]

    def complete_json(self, prompt: str, system: Optional[str] = None) -> dict:
        """For structured outputs (consolidation summaries, contradiction
        verdicts). Asks the model to return only JSON and parses it,
        with a fallback if the model wraps it in markdown fences anyway."""
        json_instruction = (
            "\n\nRespond with ONLY a valid JSON object. No preamble, "
            "no markdown code fences, no explanation."
        )
        raw = self.complete(prompt + json_instruction, system=system)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM did not return valid JSON. Raw output:\n{raw}") from e
