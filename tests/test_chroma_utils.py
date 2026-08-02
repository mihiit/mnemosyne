import chromadb

from mnemosyne.memory.chroma_utils import safe_query


class _FlakyCollection:
    """Simulates the real ChromaDB timing bug: raises InternalError on
    the first N calls, then succeeds — so we can test the retry logic
    without needing to actually reproduce Chroma's race condition."""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.call_count = 0

    def query(self, **kwargs):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise chromadb.errors.InternalError("Error creating hnsw segment reader: Nothing found on disk")
        return {"ids": [["fact-1"]], "documents": [["some fact"]], "metadatas": [[{}]], "distances": [[0.1]]}


def test_safe_query_retries_and_succeeds_after_transient_errors():
    collection = _FlakyCollection(fail_times=2)
    result = safe_query(collection, max_retries=3, retry_delay=0.01, query_embeddings=[[0.1, 0.2]], n_results=1)
    assert result["ids"] == [["fact-1"]]
    assert collection.call_count == 3


def test_safe_query_degrades_gracefully_when_retries_exhausted():
    collection = _FlakyCollection(fail_times=10)  # always fails
    result = safe_query(collection, max_retries=3, retry_delay=0.01, query_embeddings=[[0.1, 0.2]], n_results=1)
    assert result["ids"] == [[]]
    assert result["documents"] == [[]]
    assert collection.call_count == 3  # stopped after max_retries, didn't hang or raise