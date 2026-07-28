"""
Thin wrapper around sentence-transformers so the rest of the codebase
never imports the model directly. Lets us swap embedding models later
(e.g. for a benchmark ablation) by changing one place.
"""

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    # Cached so repeated instantiation (e.g. in tests) doesn't reload the
    # model from disk every time.
    return SentenceTransformer(model_name)


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = _load_model(model_name)

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        # Vectors are already normalized on encode, but guard anyway in
        # case someone passes in raw vectors from elsewhere.
        denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)) or 1e-8
        return float(np.dot(a_arr, b_arr) / denom)
