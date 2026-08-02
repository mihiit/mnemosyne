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
        top_k = top_k or self.config.retrieval_top_k

        # Pull a wider candidate pool than top_k from raw similarity search,
        # so trust-based reranking has real facts to work with.
        candidate_pool_size = top_k * 3
        semantic_candidates = self.semantic.query(query, repo, top_k=candidate_pool_size)
        episodic_hits = self.episodic.query(query, repo, top_k=top_k)

        scored_semantic = [
            {**hit, "retrieval_score": self._semantic_score(hit)}
            for hit in semantic_candidates
        ]
        scored_semantic.sort(key=lambda h: h["retrieval_score"], reverse=True)

        scored_episodic = [
            {**hit, "importance": self._importance_score(hit)}
            for hit in episodic_hits
        ]
        scored_episodic.sort(key=lambda h: h["importance"], reverse=True)

        for hit in scored_episodic[:top_k]:
            self.episodic.mark_accessed(hit["id"])

        result = {
            "semantic_facts": scored_semantic[:top_k],
            "episodic_entries": scored_episodic[:top_k],
            "cross_repo_priors": [],
        }

        if not scored_semantic and self.config.enable_cross_repo_priors:
            result["cross_repo_priors"] = self.semantic.query_cross_repo_priors(
                query, exclude_repo=repo, top_k=min(top_k, 3)
            )

        return result

    def _semantic_score(self, hit: dict) -> float:
        """score = w_semantic_relevance * similarity + w_trust * trust.
        If trust-weighted retrieval is disabled (ablation), rank by raw
        similarity only."""
        similarity = hit["similarity"]
        if not self.config.enable_trust_weighted_retrieval:
            return similarity
        trust = float(hit["metadata"].get("trust", hit["metadata"].get("confidence", 0.5)))
        return (
            self.config.w_semantic_relevance * similarity
            + self.config.w_trust * trust
        )

    def _importance_score(self, hit: dict) -> float:
        meta = hit["metadata"]
        similarity = hit["similarity"]

        age_days = (time.time() - meta.get("timestamp", time.time())) / 86400
        recency_score = math.exp(-math.log(2) * age_days / self.config.recency_half_life_days)

        access_count = meta.get("access_count", 0)
        freq_score = math.log1p(access_count) / math.log1p(20)
        freq_score = min(freq_score, 1.0)

        return (
            self.config.w_relevance * similarity
            + self.config.w_recency * recency_score
            + self.config.w_frequency * freq_score
        )

    def run_forgetting_pass(self, repo: str) -> List[str]:
        all_entries = self.episodic.get_since(repo, since_timestamp=0)
        to_prune = []

        for entry in all_entries:
            meta = entry["metadata"]
            age_days = (time.time() - meta.get("timestamp", time.time())) / 86400
            if age_days < self.config.min_age_before_prune_days:
                continue

            score = self._retention_score(meta)
            if score < self.config.forgetting_threshold:
                to_prune.append(entry["id"])

        if to_prune:
            self.episodic.collection.delete(ids=to_prune)
        return to_prune

    def _retention_score(self, meta: dict) -> float:
        age_days = (time.time() - meta.get("timestamp", time.time())) / 86400
        recency_score = math.exp(-math.log(2) * age_days / self.config.recency_half_life_days)

        access_count = meta.get("access_count", 0)
        freq_score = min(math.log1p(access_count) / math.log1p(20), 1.0)

        weight_sum = self.config.w_recency + self.config.w_frequency
        return (
            self.config.w_recency * recency_score
            + self.config.w_frequency * freq_score
        ) / weight_sum