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
