from mnemosyne.config import MnemosyneConfig
from mnemosyne.eval.naive_baseline_memory import NaiveBaselineMemory


def test_naive_baseline_returns_no_semantic_facts_or_priors(tmp_path, fake_embedder):
    config = MnemosyneConfig(chroma_persist_dir=str(tmp_path / "chroma"))
    memory = NaiveBaselineMemory(config, embedder=fake_embedder)

    memory.remember("Fixed a race condition in token refresh", repo="repo-a")
    result = memory.recall("token refresh bug", repo="repo-a")

    assert result["semantic_facts"] == []
    assert result["cross_repo_priors"] == []
    assert len(result["episodic_entries"]) == 1
    assert "race condition" in result["episodic_entries"][0]["text"]


def test_naive_baseline_empty_collection_returns_nothing(tmp_path, fake_embedder):
    config = MnemosyneConfig(chroma_persist_dir=str(tmp_path / "chroma"))
    memory = NaiveBaselineMemory(config, embedder=fake_embedder)
    result = memory.recall("anything", repo="repo-a")
    assert result == {"semantic_facts": [], "episodic_entries": [], "cross_repo_priors": []}


def test_naive_baseline_has_no_consolidation_or_contradiction_handling(tmp_path, fake_embedder):
    config = MnemosyneConfig(chroma_persist_dir=str(tmp_path / "chroma"))
    memory = NaiveBaselineMemory(config, embedder=fake_embedder)
    memory.remember("some fact", repo="repo-a")
    assert memory.force_consolidate("repo-a") == []
    assert memory.contradictions("repo-a") == []
    assert memory.maintain("repo-a") == []