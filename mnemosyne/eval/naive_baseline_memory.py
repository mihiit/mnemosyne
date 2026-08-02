"""
A deliberately naive baseline memory: every remembered text is embedded
and stored flat; recall is pure top-k cosine similarity, with no
consolidation, no trust tracking, no contradiction handling, and no
cross-repo priors. This represents the common "just embed everything
and retrieve by similarity" approach used by many simple RAG-style
agent memories, and is what Mnemosyne's episodic+semantic+trust
pipeline should be compared against to justify the added complexity.

It is NOT a reimplementation of any specific existing system (MemGPT,
Letta, Mem0) — those have their own consolidation and scoring logic
this doesn't attempt to replicate. It's the simplest reasonable
baseline: same underlying embedding model and LLM, same agent harness,
but memory that's just a flat similarity-searchable log.
"""

import time
import uuid
from typing import List, Optional

import chromadb

from mnemosyne.config import MnemosyneConfig
from mnemosyne.memory.chroma_utils import safe_query
from mnemosyne.memory.embeddings import Embedder


class NaiveBaselineMemory:
    def __init__(self, config: Optional[MnemosyneConfig] = None, embedder=None):
        self.config = config or MnemosyneConfig()
        self.client = chromadb.PersistentClient(path=self.config.chroma_persist_dir)
        self.embedder = embedder or Embedder(self.config.embedding_model)
        self.collection = self.client.get_or_create_collection(
            name="naive_baseline_memory",
            metadata={"hnsw:space": "cosine"},
        )

    def remember(self, text: str, repo: str, task_id: Optional[str] = None, tags: Optional[List[str]] = None):
        embedding = self.embedder.embed(text)
        self.collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"repo": repo, "timestamp": time.time()}],
        )

    def recall(self, query: str, repo: str, top_k: int = None) -> dict:
        top_k = top_k or self.config.retrieval_top_k
        if self.collection.count() == 0:
            return {"semantic_facts": [], "episodic_entries": [], "cross_repo_priors": []}

        embedding = self.embedder.embed(query)
        results = safe_query(
            self.collection,
            query_embeddings=[embedding],
            n_results=top_k,
            where={"repo": repo},
        )
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        entries = [
            {"id": id_, "text": doc, "importance": 1 - dist}
            for id_, doc, dist in zip(ids, docs, distances)
        ]
        return {"semantic_facts": [], "episodic_entries": entries, "cross_repo_priors": []}

    def maintain(self, repo: str):
        return []

    def contradictions(self, repo: str):
        return []

    def force_consolidate(self, repo: str):
        return []