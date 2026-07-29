"""
Semantic memory: durable, distilled facts about a repo/project, built by
consolidating batches of episodic entries.

Example semantic facts:
    "This repo uses dependency injection for DB access, not singletons."
    "The team prefers explicit error handling over exceptions for I/O."
    "auth.py's token refresh logic is fragile — has caused 2 prior race conditions."

This module does three things: consolidation (turning many episodic
entries into one durable fact), contradiction detection (catching when a
new fact conflicts with an old one instead of silently overwriting), and
— the novel piece — trust tracking via corroboration. Most memory
architectures treat "confidence" as a static number set once at creation
and never revisited. Here, trust is a live signal: every time a new
episodic entry independently reaffirms an existing fact, that fact's
trust moves up; every time one contradicts it, trust drops; every time
one merely adds detail, the old fact is marked superseded rather than
either extreme. The agent surfaces this trust level explicitly in its
responses, so retrieved memory is treated as evidence with a weight,
not as unconditional ground truth.
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
    confidence: float  # initial LLM-assigned confidence at creation
    trust: float  # live, evidence-tracked score — starts equal to confidence, moves with corroboration/contradiction
    created_at: float
    updated_at: float
    source_episodic_ids: List[str] = field(default_factory=list)
    contradicted_by: Optional[str] = None  # id of a fact that contradicts this one, if any
    corroboration_count: int = 0
    contradiction_count: int = 0
    last_reinforced_at: Optional[float] = None


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
fact that scored highly similar to it in embedding space. Decide whether the new fact:
- CONTRADICTS the old one (they conflict, can't both be true/current)
- REFINES the old one (adds detail or narrows scope, doesn't conflict)
- CORROBORATES the old one (independently restates/reaffirms the same thing, no new detail)
- is about a DIFFERENT thing (a false positive from the similarity search)"""

CONTRADICTION_PROMPT_TEMPLATE = """Existing fact: "{old_fact}"
New candidate fact: "{new_fact}"

Return JSON: {{"verdict": "contradiction" | "refinement" | "corroboration" | "different", "reasoning": "..."}}"""


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
        Returns newly created facts (corroborations of existing facts
        reinforce trust in place and are not returned as new facts).
        Facts that turn out to contradict an existing fact are still
        created, but flagged via contradicted_by rather than silently
        overwriting the old one — callers (e.g. the agent) can decide
        how to surface that."""
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
            if fact is not None:
                new_facts.append(fact)
        return new_facts

    def _add_fact(self, text: str, confidence: float, repo: str, source_episodic_ids: List[str]) -> Optional[SemanticFact]:
        # Check for a similar existing fact first — this is where
        # contradiction/corroboration detection hooks in.
        similar = self._find_similar(text, repo)
        contradicted_by = None

        if similar:
            verdict = self._check_contradiction(similar["text"], text)
            verdict_type = verdict.get("verdict")

            if verdict_type == "corroboration":
                # This is not a new fact — it's evidence for an existing
                # one. Reinforce the old fact's trust and don't create a
                # duplicate entry.
                self._reinforce(similar["id"], source_episodic_ids)
                return None
            elif verdict_type == "contradiction":
                contradicted_by = similar["id"]
                self._apply_contradiction(similar["id"])
            elif verdict_type == "refinement":
                # Treat a refinement as superseding the old fact: lower
                # the old fact's trust rather than deleting it — deleting
                # would lose history we might want later.
                self._decay_trust(similar["id"], factor=self.config.refinement_decay)
            # "different" (or any unrecognized verdict) -> fall through
            # and create a new, independent fact as normal.

        now = time.time()
        fact = SemanticFact(
            id=str(uuid.uuid4()),
            text=text,
            repo=repo,
            confidence=confidence,
            trust=confidence,
            created_at=now,
            updated_at=now,
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
                "trust": confidence,
                "created_at": fact.created_at,
                "updated_at": fact.updated_at,
                "source_episodic_ids": ",".join(source_episodic_ids),
                "contradicted_by": contradicted_by or "",
                "corroboration_count": 0,
                "contradiction_count": 0,
                "last_reinforced_at": 0.0,
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

    def _reinforce(self, fact_id: str, new_source_ids: List[str]):
        """Corroboration: move trust toward 1.0 by corroboration_boost
        of the remaining distance, so trust asymptotically approaches
        (but never quite reaches) 1.0 — repeated corroboration keeps
        strengthening a fact, with diminishing returns."""
        existing = self.collection.get(ids=[fact_id])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0]
        old_trust = float(meta.get("trust", meta.get("confidence", 0.5)))
        new_trust = old_trust + self.config.corroboration_boost * (1.0 - old_trust)

        meta["trust"] = new_trust
        meta["corroboration_count"] = int(meta.get("corroboration_count", 0)) + 1
        meta["last_reinforced_at"] = time.time()
        meta["updated_at"] = time.time()
        existing_sources = meta.get("source_episodic_ids", "")
        combined = existing_sources.split(",") if existing_sources else []
        meta["source_episodic_ids"] = ",".join(combined + new_source_ids)
        self.collection.update(ids=[fact_id], metadatas=[meta])

    def _apply_contradiction(self, fact_id: str):
        """Contradiction: multiply trust down by contradiction_penalty —
        a harder hit than a refinement decay, since a genuine conflict
        (not just added detail) means the old fact is now questionable,
        not just outdated."""
        existing = self.collection.get(ids=[fact_id])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0]
        old_trust = float(meta.get("trust", meta.get("confidence", 0.5)))
        meta["trust"] = old_trust * self.config.contradiction_penalty
        meta["contradiction_count"] = int(meta.get("contradiction_count", 0)) + 1
        meta["updated_at"] = time.time()
        self.collection.update(ids=[fact_id], metadatas=[meta])

    def _decay_trust(self, fact_id: str, factor: float):
        existing = self.collection.get(ids=[fact_id])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0]
        old_trust = float(meta.get("trust", meta.get("confidence", 0.5)))
        meta["trust"] = old_trust * factor
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

    @staticmethod
    def trust_label(meta: dict) -> str:
        """Human/agent-readable calibration label derived from trust
        score + corroboration history — this is what gets surfaced in
        the agent's prompt so it can hedge appropriately instead of
        treating every retrieved fact as equally certain."""
        trust = float(meta.get("trust", meta.get("confidence", 0.5)))
        corroborations = int(meta.get("corroboration_count", 0))

        if trust >= 0.8 and corroborations >= 2:
            return f"well-established (reinforced {corroborations}x)"
        elif trust >= 0.65:
            return "fairly confident"
        elif trust >= 0.4:
            return "stated once, not since reinforced — treat with some caution"
        else:
            return "low trust — has been contradicted or heavily superseded"

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
                "trust_label": self.trust_label(meta),
            })
        return formatted