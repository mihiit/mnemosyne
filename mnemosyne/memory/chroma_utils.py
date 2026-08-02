"""
ChromaDB (as of the version pinned in requirements.txt) has a known
timing bug: querying a collection immediately after an add() in the
same process can intermittently raise
    chromadb.errors.InternalError: Error creating hnsw segment reader: Nothing found on disk
This isn't specific to empty collections — it can happen right after
writing the very first item into a collection too, before the index is
fully flushed to disk. A count()==0 guard catches the empty case but
not this one, so queries go through this retry wrapper instead of
calling collection.query() directly.
"""

import time
from typing import Any, Dict

import chromadb


def safe_query(collection: "chromadb.Collection", max_retries: int = 3, retry_delay: float = 0.15, **query_kwargs) -> Dict[str, Any]:
    """Call collection.query(**query_kwargs), retrying briefly on the
    known write-then-read timing error. Returns Chroma's normal empty
    result shape if all retries are exhausted, rather than propagating
    the crash — a transient index-not-ready error should degrade to
    'no results yet', not bring down a long-running benchmark."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return collection.query(**query_kwargs)
        except chromadb.errors.InternalError as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    # All retries exhausted — degrade gracefully instead of crashing.
    n_queries = len(query_kwargs.get("query_embeddings", [[]]))
    return {
        "ids": [[] for _ in range(n_queries)],
        "documents": [[] for _ in range(n_queries)],
        "metadatas": [[] for _ in range(n_queries)],
        "distances": [[] for _ in range(n_queries)],
    }