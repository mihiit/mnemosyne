"""
Episodic memory: the raw, timestamped log of specific things that
happened while the agent worked on a repo.

Examples of what goes in here:
    "Fixed race condition in auth.py by adding a lock around token refresh"
    "Reverted PR #42 — broke integration tests in test_billing.py"
    "User rejected the singleton pattern for the DB client, asked for DI instead"

Episodic entries are cheap to write and numerous. They are NOT meant to
be the primary thing retrieved during normal operation long-term — that's
what semantic memory (distilled, durable facts) is for. Episodic memory
is the raw material that consolidation reads from.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb

from mnemosyne.config import MnemosyneConfig
from mnemosyne.memory.embeddings import Embedder


@dataclass
class EpisodicEntry:
    id: str
    text: str
    timestamp: float
    repo: str
    task_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class EpisodicMemory:
    def __init__(self, config: MnemosyneConfig, embedder: Embedder, client: chromadb.ClientAPI):
        self.config = config
        self.embedder = embedder
        self.collection = client.get_or_create_collection(
            name=config.episodic_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        text: str,
        repo: str,
        task_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> EpisodicEntry:
        entry = EpisodicEntry(
            id=str(uuid.uuid4()),
            text=text,
            timestamp=time.time(),
            repo=repo,
            task_id=task_id,
            tags=tags or [],
        )
        embedding = self.embedder.embed(text)
        self.collection.add(
            ids=[entry.id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "timestamp": entry.timestamp,
                "repo": repo,
                "task_id": task_id or "",
                "tags": ",".join(entry.tags),
                "access_count": 0,
                "last_accessed": entry.last_accessed,
            }],
        )
        return entry

    def query(self, text: str, repo: str, top_k: int = 5) -> List[dict]:
        """Raw similarity search, no importance weighting. Used mainly by
        the consolidation pass to pull recent related entries, and by
        eval scripts. Agent-facing retrieval should go through
        retrieval.py instead, which applies recency/frequency weighting."""
        query_embedding = self.embedder.embed(text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"repo": repo},
        )
        return self._format_results(results)

    def get_since(self, repo: str, since_timestamp: float) -> List[dict]:
        """Fetch all entries newer than a given timestamp — this is what
        the consolidation trigger uses to grab the latest unconsolidated
        batch."""
        results = self.collection.get(
            where={"$and": [{"repo": repo}, {"timestamp": {"$gt": since_timestamp}}]},
        )
        return [
            {"id": id_, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])
        ]

    def mark_accessed(self, entry_id: str):
        existing = self.collection.get(ids=[entry_id])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0]
        meta["access_count"] = meta.get("access_count", 0) + 1
        meta["last_accessed"] = time.time()
        self.collection.update(ids=[entry_id], metadatas=[meta])

    def count(self, repo: str) -> int:
        return len(self.collection.get(where={"repo": repo})["ids"])

    @staticmethod
    def _format_results(results) -> List[dict]:
        formatted = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for id_, doc, meta, dist in zip(ids, docs, metas, distances):
            formatted.append({
                "id": id_,
                "text": doc,
                "metadata": meta,
                "similarity": 1 - dist,  # chroma returns cosine distance
            })
        return formatted
