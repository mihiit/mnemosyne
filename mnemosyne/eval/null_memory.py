"""
A memory store that implements the same interface as MemoryStore but
does nothing — used as the "memory OFF" condition in the benchmark, so
the agent code path is otherwise identical between conditions and the
only variable being measured is whether memory is available.
"""

from mnemosyne.config import MnemosyneConfig


class NullMemoryStore:
    def __init__(self, config: MnemosyneConfig = None):
        self.config = config or MnemosyneConfig()

    def remember(self, text: str, repo: str, task_id: str = None, tags=None):
        pass

    def recall(self, query: str, repo: str, top_k: int = None) -> dict:
        return {"semantic_facts": [], "episodic_entries": [], "cross_repo_priors": []}

    def maintain(self, repo: str):
        return []

    def contradictions(self, repo: str):
        return []

    def force_consolidate(self, repo: str):
        return []