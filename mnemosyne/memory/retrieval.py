"""
Importance-weighted retrieval and forgetting.

This is what an agent actually calls during normal operation — not raw
similarity search. Retrieval combines semantic relevance with recency
and access frequency, so a highly-relevant-but-ancient-and-unused memory
doesn't drown out something equally relevant but fresher/more used.

Forgetting is a separate maintenance pass: entries whose importance
score falls below a threshold (and are old enough to not be in a grace
period) are pruned so memory doesn't grow unbounded.
"""

import math
import time
from typing import List

from mnemosyne.config import MnemosyneConfig
from mnemosyne.memory.episodic import EpisodicMemory
from mnemosyne.memory.semantic import SemanticMemory


class RetrievalEngine:
    def __init__(self, config: MnemosyneConfig, episodic: EpisodicMemory, semantic: SemanticMemory):
        self.config = config
        self.episodic = episodic
        self.semantic = semantic

    def retrieve(self, query: str, repo: str, top_k: int = None) -> dict:
        """Retrieve the most relevant memory for a given query, combining
        semantic facts (preferred — they're distilled and durable) with
        episodic entries (for specific recent detail semantic memory
        hasn't consolidated yet)."""
        top_k = top_k or self.config.retrieval_top_k

        semantic_hits = self.semantic.query(query, repo, top_k=top_k)
        episodic_hits = self.episodic.query(query, repo, top_k=top_k)

        scored_episodic = [
            {**hit, "importance": self._importance_score(hit)}
            for hit in episodic_hits
        ]
        scored_episodic.sort(key=lambda h: h["importance"], reverse=True)

        for hit in scored_episodic[:top_k]:
            self.episodic.mark_accessed(hit["id"])

        return {
            "semantic_facts": semantic_hits,
            "episodic_entries": scored_episodic[:top_k],
        }

    def _importance_score(self, hit: dict) -> float:
        """score = w_relevance * similarity + w_recency * recency_decay + w_frequency * freq_score"""
        meta = hit["metadata"]
        similarity = hit["similarity"]

        age_days = (time.time() - meta.get("timestamp", time.time())) / 86400
        recency_score = math.exp(-math.log(2) * age_days / self.config.recency_half_life_days)

        access_count = meta.get("access_count", 0)
        # log-scaled so frequency doesn't dominate for very frequently accessed entries
        freq_score = math.log1p(access_count) / math.log1p(20)  # normalize against a cap of ~20 accesses
        freq_score = min(freq_score, 1.0)

        return (
            self.config.w_relevance * similarity
            + self.config.w_recency * recency_score
            + self.config.w_frequency * freq_score
        )

    def run_forgetting_pass(self, repo: str) -> List[str]:
        """Prune low-importance episodic entries. Returns the list of
        pruned entry ids. Semantic facts are NOT pruned here — they're
        already distilled and cheap to keep; only raw episodic log
        entries grow unbounded and need pruning."""
        all_entries = self.episodic.get_since(repo, since_timestamp=0)
        to_prune = []

        for entry in all_entries:
            meta = entry["metadata"]
            age_days = (time.time() - meta.get("timestamp", time.time())) / 86400
            if age_days < self.config.min_age_before_prune_days:
                continue  # grace period — never prune very recent entries

            score = self._retention_score(meta)
            if score < self.config.forgetting_threshold:
                to_prune.append(entry["id"])

        if to_prune:
            self.episodic.collection.delete(ids=to_prune)
        return to_prune

    def _retention_score(self, meta: dict) -> float:
        """Importance score used specifically for forgetting decisions,
        where there is no query so relevance doesn't apply. Renormalizes
        the recency/frequency weights (drops w_relevance's share) so the
        0-1 scale is actually usable against forgetting_threshold, rather
        than having a fixed relevance-driven floor an old, unused entry
        can never fall below."""
        age_days = (time.time() - meta.get("timestamp", time.time())) / 86400
        recency_score = math.exp(-math.log(2) * age_days / self.config.recency_half_life_days)

        access_count = meta.get("access_count", 0)
        freq_score = min(math.log1p(access_count) / math.log1p(20), 1.0)

        weight_sum = self.config.w_recency + self.config.w_frequency
        return (
            self.config.w_recency * recency_score
            + self.config.w_frequency * freq_score
        ) / weight_sum
