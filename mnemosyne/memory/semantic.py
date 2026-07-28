"""
Semantic memory: durable, distilled facts about a repo/project, built by
consolidating batches of episodic entries.

Example semantic facts:
    "This repo uses dependency injection for DB access, not singletons."
    "The team prefers explicit error handling over exceptions for I/O."
    "auth.py's token refresh logic is fragile — has caused 2 prior race conditions."

This is the module that does the actual "remembering wrong things,
correctly" work: consolidation (turning many episodic entries into one
durable fact) and contradiction detection (catching when a new fact
conflicts with an old one instead of silently overwriting).
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb

from mnemosyne.config import MnemosyneConfig
from mnemosyne.memory.embeddings import Embedder
from mnemosyne.memory.llm import LocalLLM


@dataclass
class SemanticFact:
    id: str
    text: str
    repo: str
    confidence: float
    created_at: float
    updated_at: float
    source_episodic_ids: List[str] = field(default_factory=list)
    contradicted_by: Optional[str] = None  # id of a fact that contradicts this one, if any


CONSOLIDATION_SYSTEM_PROMPT = """You are a memory-consolidation module for a coding agent. \
You are given a batch of raw episodic log entries from work on a codebase. \
Your job is to extract durable, general facts worth remembering long-term — \
NOT to summarize the batch. A durable fact is something that will still be \
true and useful weeks from now (a convention, a recurring gotcha, a design \
decision and its reason). Skip one-off, non-generalizable events."""

CONSOLIDATION_PROMPT_TEMPLATE = """Episodic entries:
{entries}

Extract 0-3 durable facts from these entries. For each fact, give:
- "text": the fact, stated generally and concisely (one sentence)
- "confidence": your confidence 0.0-1.0 that this generalizes beyond this one instance

Return JSON: {{"facts": [{{"text": "...", "confidence": 0.0}}, ...]}}
If nothing durable is worth extracting, return {{"facts": []}}."""

CONTRADICTION_SYSTEM_PROMPT = """You are a contradiction-detection module for a coding \
agent's memory system. You are given an existing remembered fact and a new candidate \
fact that scored highly similar to it in embedding space. Decide whether the new fact \
actually CONTRADICTS the old one, merely REFINES/UPDATES it, or is actually about a \
DIFFERENT thing (a false positive from the similarity search)."""

CONTRADICTION_PROMPT_TEMPLATE = """Existing fact: "{old_fact}"
New candidate fact: "{new_fact}"

Return JSON: {{"verdict": "contradiction" | "refinement" | "different", "reasoning": "..."}}"""


class SemanticMemory:
    def __init__(self, config: MnemosyneConfig, embedder: Embedder, llm: LocalLLM, client: chromadb.ClientAPI):
        self.config = config
        self.embedder = embedder
        self.llm = llm
        self.collection = client.get_or_create_collection(
            name=config.semantic_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def consolidate(self, episodic_entries: List[dict], repo: str) -> List[SemanticFact]:
        """Run a consolidation pass over a batch of episodic entries.
        Returns newly created facts. Facts that turn out to contradict
        an existing fact are still created, but flagged via
        contradicted_by rather than silently overwriting the old one —
        callers (e.g. the agent) can decide how to surface that."""
        if not episodic_entries:
            return []

        entries_text = "\n".join(f"- {e['text']}" for e in episodic_entries)
        prompt = CONSOLIDATION_PROMPT_TEMPLATE.format(entries=entries_text)
        result = self.llm.complete_json(prompt, system=CONSOLIDATION_SYSTEM_PROMPT)

        new_facts = []
        source_ids = [e["id"] for e in episodic_entries]
        for candidate in result.get("facts", []):
            fact = self._add_fact(
                text=candidate["text"],
                confidence=float(candidate.get("confidence", 0.5)),
                repo=repo,
                source_episodic_ids=source_ids,
            )
            new_facts.append(fact)
        return new_facts

    def _add_fact(self, text: str, confidence: float, repo: str, source_episodic_ids: List[str]) -> SemanticFact:
        # Check for a similar existing fact first — this is where
        # contradiction detection hooks in.
        similar = self._find_similar(text, repo)
        contradicted_by = None

        if similar:
            verdict = self._check_contradiction(similar["text"], text)
            if verdict["verdict"] == "contradiction":
                contradicted_by = similar["id"]
            elif verdict["verdict"] == "refinement":
                # Treat a refinement as superseding the old fact: lower
                # the old fact's confidence rather than deleting it —
                # deleting would lose history we might want later.
                self._decay_confidence(similar["id"], factor=0.5)

        fact = SemanticFact(
            id=str(uuid.uuid4()),
            text=text,
            repo=repo,
            confidence=confidence,
            created_at=time.time(),
            updated_at=time.time(),
            source_episodic_ids=source_episodic_ids,
            contradicted_by=contradicted_by,
        )
        embedding = self.embedder.embed(text)
        self.collection.add(
            ids=[fact.id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "repo": repo,
                "confidence": confidence,
                "created_at": fact.created_at,
                "updated_at": fact.updated_at,
                "source_episodic_ids": ",".join(source_episodic_ids),
                "contradicted_by": contradicted_by or "",
            }],
        )
        return fact

    def _find_similar(self, text: str, repo: str) -> Optional[dict]:
        embedding = self.embedder.embed(text)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where={"repo": repo},
        )
        ids = results.get("ids", [[]])[0]
        if not ids:
            return None
        distance = results["distances"][0][0]
        similarity = 1 - distance
        if similarity < self.config.semantic_similarity_threshold:
            return None
        return {"id": ids[0], "text": results["documents"][0][0], "similarity": similarity}

    def _check_contradiction(self, old_fact: str, new_fact: str) -> dict:
        prompt = CONTRADICTION_PROMPT_TEMPLATE.format(old_fact=old_fact, new_fact=new_fact)
        return self.llm.complete_json(prompt, system=CONTRADICTION_SYSTEM_PROMPT)

    def _decay_confidence(self, fact_id: str, factor: float):
        existing = self.collection.get(ids=[fact_id])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0]
        meta["confidence"] = float(meta.get("confidence", 0.5)) * factor
        meta["updated_at"] = time.time()
        self.collection.update(ids=[fact_id], metadatas=[meta])

    def get_contradictions(self, repo: str) -> List[dict]:
        """Return all facts currently flagged as contradicting another
        fact — this is what an agent or a human reviewer would check
        before trusting memory blindly."""
        results = self.collection.get(where={"repo": repo})
        flagged = []
        for id_, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
            if meta.get("contradicted_by"):
                flagged.append({"id": id_, "text": doc, "contradicted_by": meta["contradicted_by"]})
        return flagged

    def query(self, text: str, repo: str, top_k: int = 5) -> List[dict]:
        embedding = self.embedder.embed(text)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where={"repo": repo},
        )
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
                "similarity": 1 - dist,
            })
        return formatted
