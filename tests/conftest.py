"""
Fakes for embedder and LLM so the test suite runs with zero network
access (no huggingface.co, no Ollama server needed). These are simple
enough to reason about by hand, which matters for tests that assert on
exact scoring/behavior.
"""

import hashlib
import json
from typing import List, Optional

import numpy as np
import pytest


class FakeEmbedder:
    """Deterministic bag-of-words-ish embedder: same text -> same vector,
    similar text (shares words) -> higher cosine similarity. Good enough
    to test retrieval/consolidation logic without downloading a real model."""

    DIM = 64

    def embed(self, text: str) -> List[float]:
        vec = np.zeros(self.DIM)
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % self.DIM
            vec[idx] += 1.0
        norm = np.linalg.norm(vec) or 1e-8
        return (vec / norm).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)) or 1e-8
        return float(np.dot(a_arr, b_arr) / denom)


class ScriptedLLM:
    """Returns pre-programmed JSON responses in sequence, so consolidation
    and contradiction-detection logic can be tested deterministically
    without a real model. Call .queue(response_dict) to add responses."""

    def __init__(self):
        self._queue = []

    def queue(self, response: dict):
        self._queue.append(response)

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        return json.dumps(self._pop())

    def complete_json(self, prompt: str, system: Optional[str] = None) -> dict:
        return self._pop()

    def _pop(self) -> dict:
        if not self._queue:
            raise AssertionError("ScriptedLLM ran out of queued responses — add more with .queue()")
        return self._queue.pop(0)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def scripted_llm():
    return ScriptedLLM()
