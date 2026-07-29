import chromadb
import pytest

from mnemosyne.config import MnemosyneConfig
from mnemosyne.memory.episodic import EpisodicMemory
from mnemosyne.memory.semantic import SemanticMemory
from mnemosyne.memory.store import MemoryStore


@pytest.fixture
def episodic(tmp_path, fake_embedder):
    config = MnemosyneConfig(chroma_persist_dir=str(tmp_path / "chroma"))
    client = chromadb.PersistentClient(path=config.chroma_persist_dir)
    return EpisodicMemory(config, fake_embedder, client)


def test_get_unconsolidated_excludes_marked_entries(episodic):
    e1 = episodic.add("Event one", repo="repo-a")
    e2 = episodic.add("Event two", repo="repo-a")

    unconsolidated = episodic.get_unconsolidated("repo-a")
    assert {e["id"] for e in unconsolidated} == {e1.id, e2.id}

    episodic.mark_consolidated([e1.id])

    unconsolidated = episodic.get_unconsolidated("repo-a")
    assert {e["id"] for e in unconsolidated} == {e2.id}


def test_force_consolidate_does_not_reprocess_already_consolidated_entries(tmp_path, fake_embedder, scripted_llm):
    """Regression test for the bug found during manual testing: an
    in-memory checkpoint doesn't survive across separate MemoryStore
    instances (e.g. separate script runs), causing already-consolidated
    entries to be pulled in again. Consolidation state must live on the
    entries themselves."""
    config = MnemosyneConfig(chroma_persist_dir=str(tmp_path / "chroma"))
    client = chromadb.PersistentClient(path=config.chroma_persist_dir)
    episodic = EpisodicMemory(config, fake_embedder, client)
    semantic = SemanticMemory(config, fake_embedder, scripted_llm, client)

    episodic.add("First event", repo="repo-a")

    scripted_llm.queue({"facts": [{"text": "Some durable fact", "confidence": 0.7}]})
    first_batch = episodic.get_unconsolidated("repo-a")
    semantic.consolidate(first_batch, repo="repo-a")
    episodic.mark_consolidated([e["id"] for e in first_batch])

    # Simulate a fresh process/script: build new objects against the same
    # persisted Chroma store, add one more entry, consolidate again.
    episodic_2 = EpisodicMemory(config, fake_embedder, client)
    episodic_2.add("Second event", repo="repo-a")

    second_batch = episodic_2.get_unconsolidated("repo-a")

    # The bug would have re-included "First event" here; the fix means
    # only the truly new entry shows up.
    assert len(second_batch) == 1
    assert second_batch[0]["text"] == "Second event"