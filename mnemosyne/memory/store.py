"""
MemoryStore: the single entry point the agent (or eval scripts) talks
to. Wires together episodic memory, semantic memory, the LLM, and the
retrieval engine, and exposes the operations an agent needs:

    remember(text)   -> log an episodic event, maybe trigger consolidation
    recall(query)     -> importance-weighted retrieval across both memory types
    maintain()        -> run the forgetting pass (call periodically, e.g. once per session)

Everything else (contradiction flags, raw queries) is available on the
underlying .episodic / .semantic objects for eval scripts that need
finer-grained access.
"""

from typing import List, Optional

import chromadb

from mnemosyne.config import MnemosyneConfig
from mnemosyne.memory.embeddings import Embedder
from mnemosyne.memory.episodic import EpisodicMemory
from mnemosyne.memory.llm import LocalLLM
from mnemosyne.memory.retrieval import RetrievalEngine
from mnemosyne.memory.semantic import SemanticMemory


class MemoryStore:
    def __init__(self, config: Optional[MnemosyneConfig] = None):
        self.config = config or MnemosyneConfig()

        self.client = chromadb.PersistentClient(path=self.config.chroma_persist_dir)
        self.embedder = Embedder(self.config.embedding_model)
        self.llm = LocalLLM(self.config)

        self.episodic = EpisodicMemory(self.config, self.embedder, self.client)
        self.semantic = SemanticMemory(self.config, self.embedder, self.llm, self.client)
        self.retrieval = RetrievalEngine(self.config, self.episodic, self.semantic)

    def remember(self, text: str, repo: str, task_id: Optional[str] = None, tags: Optional[List[str]] = None):
        self.episodic.add(text, repo=repo, task_id=task_id, tags=tags)
        self._maybe_consolidate(repo)

    def recall(self, query: str, repo: str, top_k: int = None) -> dict:
        return self.retrieval.retrieve(query, repo, top_k=top_k)

    def maintain(self, repo: str) -> List[str]:
        """Run periodically (e.g. once per agent session) rather than
        after every single write — forgetting is a maintenance operation,
        not something that needs to happen in the hot path."""
        return self.retrieval.run_forgetting_pass(repo)

    def contradictions(self, repo: str) -> List[dict]:
        return self.semantic.get_contradictions(repo)

    def _maybe_consolidate(self, repo: str):
        unconsolidated = self.episodic.get_unconsolidated(repo)
        if len(unconsolidated) >= self.config.consolidation_batch_size:
            self.semantic.consolidate(unconsolidated, repo=repo)
            self.episodic.mark_consolidated([e["id"] for e in unconsolidated])

    def force_consolidate(self, repo: str):
        """Manually trigger consolidation regardless of batch size —
        useful in eval scripts where you want deterministic control over
        when consolidation happens. Only touches entries not already
        consolidated, so calling this repeatedly (e.g. across separate
        script runs in the same repo) won't re-consolidate old entries."""
        unconsolidated = self.episodic.get_unconsolidated(repo)
        facts = self.semantic.consolidate(unconsolidated, repo=repo)
        if unconsolidated:
            self.episodic.mark_consolidated([e["id"] for e in unconsolidated])
        return facts
