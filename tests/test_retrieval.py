import time

import chromadb
import pytest

from mnemosyne.config import MnemosyneConfig
from mnemosyne.memory.episodic import EpisodicMemory
from mnemosyne.memory.semantic import SemanticMemory
from mnemosyne.memory.retrieval import RetrievalEngine


@pytest.fixture
def engine(tmp_path, fake_embedder, scripted_llm):
    config = MnemosyneConfig(
        chroma_persist_dir=str(tmp_path / "chroma"),
        min_age_before_prune_days=0,  # disable grace period for tests
        forgetting_threshold=0.3,
    )
    client = chromadb.PersistentClient(path=config.chroma_persist_dir)
    episodic = EpisodicMemory(config, fake_embedder, client)
    semantic = SemanticMemory(config, fake_embedder, scripted_llm, client)
    return RetrievalEngine(config, episodic, semantic), episodic, semantic


def test_retrieve_returns_both_memory_types(engine):
    retrieval, episodic, semantic = engine
    episodic.add("Fixed race condition in auth.py", repo="repo-a")

    result = retrieval.retrieve("auth race condition", repo="repo-a")

    assert "semantic_facts" in result
    assert "episodic_entries" in result
    assert len(result["episodic_entries"]) == 1


def test_retrieve_marks_entries_accessed(engine):
    retrieval, episodic, _ = engine
    entry = episodic.add("Fixed race condition in auth.py", repo="repo-a")

    retrieval.retrieve("race condition auth", repo="repo-a")

    raw = episodic.collection.get(ids=[entry.id])
    assert raw["metadatas"][0]["access_count"] == 1


def test_importance_score_favors_relevance_recency_frequency(engine):
    retrieval, episodic, _ = engine

    high_sim_old_unused = {
        "similarity": 0.9,
        "metadata": {"timestamp": time.time() - 30 * 86400, "access_count": 0},
    }
    low_sim_new_frequent = {
        "similarity": 0.3,
        "metadata": {"timestamp": time.time(), "access_count": 15},
    }

    score_a = retrieval._importance_score(high_sim_old_unused)
    score_b = retrieval._importance_score(low_sim_new_frequent)

    # Relevance is weighted highest (0.6), so a strong similarity edge
    # should still win even against recency+frequency, given the configured weights.
    assert score_a > score_b


def test_forgetting_pass_prunes_low_importance_entries(engine):
    retrieval, episodic, _ = engine

    old_entry = episodic.add("Irrelevant old note nobody looked at again", repo="repo-a")
    # Force it old by directly rewriting its metadata timestamp.
    meta = episodic.collection.get(ids=[old_entry.id])["metadatas"][0]
    meta["timestamp"] = time.time() - 60 * 86400  # 60 days old
    episodic.collection.update(ids=[old_entry.id], metadatas=[meta])

    fresh_entry = episodic.add("Recent important fix", repo="repo-a")

    pruned = retrieval.run_forgetting_pass("repo-a")

    assert old_entry.id in pruned
    assert fresh_entry.id not in pruned
    assert episodic.count("repo-a") == 1
    
def test_trust_weighted_retrieval_can_outrank_pure_similarity(engine, scripted_llm):
    """A lower-similarity but highly-trusted fact should be able to
    outrank a higher-similarity but low-trust fact, once both are within
    the widened candidate pool retrieval pulls before reranking."""
    retrieval, episodic, semantic = engine

    scripted_llm.queue({"facts": [{"text": "database client uses dependency injection", "confidence": 0.95}]})
    semantic.consolidate([{"id": "e1", "text": "DI adopted for DB client, reinforced many times"}], repo="repo-a")

    scripted_llm.queue({"facts": [{"text": "database client uses dependency injection somewhat", "confidence": 0.15}]})
    scripted_llm.queue({"verdict": "different", "reasoning": "Treat as independent for this test"})
    semantic.consolidate([{"id": "e2", "text": "someone mentioned DI once, unclear"}], repo="repo-a")

    result = retrieval.retrieve("database client dependency injection", repo="repo-a", top_k=2)
    facts = result["semantic_facts"]
    assert len(facts) == 2
    # The high-trust fact should rank first despite not being a strictly
    # closer text match (it has extra unrelated words diluting similarity
    # under the fake embedder, but far higher trust).
    assert facts[0]["metadata"]["confidence"] == 0.95


def test_cross_repo_priors_only_fire_on_cold_start(engine, scripted_llm):
    retrieval, episodic, semantic = engine

    scripted_llm.queue({"facts": [{"text": "team rejected Redis for rate limiting", "confidence": 0.8}]})
    semantic.consolidate([{"id": "e1", "text": "Redis rejected for rate limiter"}], repo="repo-a")

    # repo-b is a brand-new repo with no facts of its own yet — cold start.
    result = retrieval.retrieve("should we use Redis for rate limiting", repo="repo-b", top_k=3)
    assert result["semantic_facts"] == []
    assert len(result["cross_repo_priors"]) >= 1
    assert result["cross_repo_priors"][0]["source_repo"] == "repo-a"
    assert "unconfirmed" in result["cross_repo_priors"][0]["trust_label"]

    # repo-a itself has its own fact, so it's NOT cold-start — no cross-repo priors.
    result_a = retrieval.retrieve("should we use Redis for rate limiting", repo="repo-a", top_k=3)
    assert len(result_a["semantic_facts"]) >= 1
    assert result_a["cross_repo_priors"] == []
