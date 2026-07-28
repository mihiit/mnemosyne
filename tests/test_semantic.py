import chromadb
import pytest

from mnemosyne.config import MnemosyneConfig
from mnemosyne.memory.semantic import SemanticMemory


@pytest.fixture
def semantic(tmp_path, fake_embedder, scripted_llm):
    # The fake embedder is a crude bag-of-words vector, so paraphrases
    # score lower similarity than a real embedding model would — lower
    # the threshold here to match, rather than changing production defaults.
    config = MnemosyneConfig(
        chroma_persist_dir=str(tmp_path / "chroma"),
        semantic_similarity_threshold=0.5,
    )
    client = chromadb.PersistentClient(path=config.chroma_persist_dir)
    return SemanticMemory(config, fake_embedder, scripted_llm, client)


def test_consolidate_creates_facts(semantic, scripted_llm):
    scripted_llm.queue({
        "facts": [
            {"text": "This repo uses dependency injection for DB access", "confidence": 0.8}
        ]
    })
    episodic_entries = [
        {"id": "e1", "text": "Team decided to use DI instead of singleton for DB client"},
    ]

    facts = semantic.consolidate(episodic_entries, repo="repo-a")

    assert len(facts) == 1
    assert facts[0].confidence == 0.8
    assert facts[0].source_episodic_ids == ["e1"]


def test_consolidate_handles_no_durable_facts(semantic, scripted_llm):
    scripted_llm.queue({"facts": []})
    facts = semantic.consolidate([{"id": "e1", "text": "Fixed a typo"}], repo="repo-a")
    assert facts == []


def test_contradiction_detection_flags_conflicting_fact(semantic, scripted_llm):
    # First fact: no similar fact exists yet, so _find_similar returns None
    # and no contradiction check is made.
    scripted_llm.queue({"facts": [{"text": "This repo uses singletons for DB access", "confidence": 0.7}]})
    semantic.consolidate([{"id": "e1", "text": "DB client is a singleton"}], repo="repo-a")

    # Second fact: similar in embedding space (shares words), and we
    # script the LLM to say it's a contradiction.
    scripted_llm.queue({"facts": [{"text": "This repo uses dependency injection for DB access", "confidence": 0.8}]})
    scripted_llm.queue({"verdict": "contradiction", "reasoning": "DI and singleton are conflicting patterns"})

    semantic.consolidate([{"id": "e2", "text": "Team switched DB client to DI"}], repo="repo-a")

    contradictions = semantic.get_contradictions("repo-a")
    assert len(contradictions) == 1
    assert "dependency injection" in contradictions[0]["text"]


def test_refinement_decays_old_fact_confidence(semantic, scripted_llm):
    scripted_llm.queue({"facts": [{"text": "API uses REST endpoints", "confidence": 0.9}]})
    semantic.consolidate([{"id": "e1", "text": "API uses REST"}], repo="repo-a")

    first_fact_id = semantic.collection.get(where={"repo": "repo-a"})["ids"][0]

    scripted_llm.queue({"facts": [{"text": "API uses REST endpoints for v1, GraphQL for v2", "confidence": 0.85}]})
    scripted_llm.queue({"verdict": "refinement", "reasoning": "Adds detail rather than conflicting"})
    semantic.consolidate([{"id": "e2", "text": "API added GraphQL for v2"}], repo="repo-a")

    old_meta = semantic.collection.get(ids=[first_fact_id])["metadatas"][0]
    assert old_meta["confidence"] < 0.9  # decayed, not left at original value
