import chromadb
import pytest

from mnemosyne.config import MnemosyneConfig
from mnemosyne.memory.episodic import EpisodicMemory


@pytest.fixture
def episodic(tmp_path, fake_embedder):
    config = MnemosyneConfig(chroma_persist_dir=str(tmp_path / "chroma"))
    client = chromadb.PersistentClient(path=config.chroma_persist_dir)
    return EpisodicMemory(config, fake_embedder, client)


def test_add_and_count(episodic):
    episodic.add("Fixed race condition in auth.py", repo="repo-a")
    episodic.add("Reverted PR 42", repo="repo-a")
    assert episodic.count("repo-a") == 2


def test_repo_isolation(episodic):
    episodic.add("Event in repo A", repo="repo-a")
    episodic.add("Event in repo B", repo="repo-b")
    assert episodic.count("repo-a") == 1
    assert episodic.count("repo-b") == 1


def test_query_returns_similar_entries(episodic):
    episodic.add("Fixed race condition in auth token refresh", repo="repo-a")
    episodic.add("Updated the README with install instructions", repo="repo-a")

    results = episodic.query("race condition auth token", repo="repo-a", top_k=2)
    assert len(results) == 2
    # The auth-related entry should rank above the README entry since it
    # shares more words with the query under the fake embedder.
    assert "race condition" in results[0]["text"]


def test_mark_accessed_increments_count(episodic):
    entry = episodic.add("Some event", repo="repo-a")
    episodic.mark_accessed(entry.id)
    episodic.mark_accessed(entry.id)

    raw = episodic.collection.get(ids=[entry.id])
    assert raw["metadatas"][0]["access_count"] == 2


def test_get_since_filters_by_timestamp(episodic):
    import time

    episodic.add("Old event", repo="repo-a")
    cutoff = time.time()
    time.sleep(0.01)
    episodic.add("New event", repo="repo-a")

    recent = episodic.get_since("repo-a", since_timestamp=cutoff)
    assert len(recent) == 1
    assert recent[0]["text"] == "New event"
