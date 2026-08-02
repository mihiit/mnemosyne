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
from mnemosyne.memory.chroma_utils import safe_query
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
    resolved: bool = False  # True once a contradiction involving this fact has been auto-resolved or manually settled
    resolution: Optional[str] = None  # "supersedes_old" | "superseded_by_new" | "rejected_favor_of_old" | "reaffirmed_against_new" | None


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
            if verdict_type == "corroboration" and not self.config.enable_corroboration:
                verdict_type = "different"  # ablation: treat as independent, no reinforcement

            if verdict_type == "corroboration":
                self._reinforce(similar["id"], source_episodic_ids)
                return None
            elif verdict_type == "contradiction":
                contradicted_by = similar["id"]
                self._apply_contradiction(similar["id"])
            elif verdict_type == "refinement":
                self._decay_trust(similar["id"], factor=self.config.refinement_decay)

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
                "resolved": False,
                "resolution": "",
            }],
        )
        if contradicted_by and self.config.enable_active_contradiction_resolution:
            self._attempt_auto_resolution(old_id=contradicted_by, new_id=fact.id)
        return fact

    def _find_similar(self, text: str, repo: str) -> Optional[dict]:
        if self.collection.count() == 0:
            return None

        embedding = self.embedder.embed(text)
        results = safe_query(
            self.collection,
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

    def _attempt_auto_resolution(self, old_id: str, new_id: str):
        """Active contradiction resolution: instead of leaving every
        contradiction flagged forever for a human to sort out, compare
        trust levels and auto-resolve clear-cut cases. If the trust gap
        is at least contradiction_auto_resolve_margin, the higher-trust
        fact wins and the loser's trust is further suppressed. If the gap
        is smaller than that, the contradiction is genuinely ambiguous
        and is left unresolved (still visible via get_contradictions) for
        a human or the agent to judge explicitly — auto-resolving a close
        call would be worse than flagging it."""
        old_existing = self.collection.get(ids=[old_id])
        new_existing = self.collection.get(ids=[new_id])
        if not old_existing["ids"] or not new_existing["ids"]:
            return

        old_meta = old_existing["metadatas"][0]
        new_meta = new_existing["metadatas"][0]
        old_trust = float(old_meta.get("trust", old_meta.get("confidence", 0.5)))
        new_trust = float(new_meta.get("trust", new_meta.get("confidence", 0.5)))
        gap = new_trust - old_trust

        if gap >= self.config.contradiction_auto_resolve_margin:
            old_meta["trust"] = old_trust * 0.1
            old_meta["resolved"] = True
            old_meta["resolution"] = "superseded_by_new"
            new_meta["resolved"] = True
            new_meta["resolution"] = "supersedes_old"
        elif gap <= -self.config.contradiction_auto_resolve_margin:
            new_meta["trust"] = new_trust * 0.1
            new_meta["resolved"] = True
            new_meta["resolution"] = "rejected_favor_of_old"
            old_meta["trust"] = old_trust + self.config.corroboration_boost * (1.0 - old_trust)
            old_meta["resolved"] = True
            old_meta["resolution"] = "reaffirmed_against_new"
        else:
            return

        old_meta["updated_at"] = time.time()
        new_meta["updated_at"] = time.time()
        self.collection.update(ids=[old_id, new_id], metadatas=[old_meta, new_meta])

    def resolve_contradiction_manually(self, fact_id: str, other_fact_id: str, winner_id: str):
        """Let a human (or the agent, if explicitly instructed) resolve an
        ambiguous contradiction that auto-resolution left pending."""
        loser_id = other_fact_id if winner_id == fact_id else fact_id
        winner_existing = self.collection.get(ids=[winner_id])
        loser_existing = self.collection.get(ids=[loser_id])
        if not winner_existing["ids"] or not loser_existing["ids"]:
            return

        winner_meta = winner_existing["metadatas"][0]
        loser_meta = loser_existing["metadatas"][0]
        loser_trust = float(loser_meta.get("trust", loser_meta.get("confidence", 0.5)))

        winner_meta["resolved"] = True
        winner_meta["resolution"] = "manually_confirmed"
        loser_meta["resolved"] = True
        loser_meta["resolution"] = "manually_rejected"
        loser_meta["trust"] = loser_trust * 0.1
        winner_meta["updated_at"] = time.time()
        loser_meta["updated_at"] = time.time()
        self.collection.update(ids=[winner_id, loser_id], metadatas=[winner_meta, loser_meta])

    def query_cross_repo_priors(self, text: str, exclude_repo: str, top_k: int = 3) -> List[dict]:
        """Cold-start bootstrap: when a repo has no relevant memory of its
        own yet, look at what other repos have learned as a starting
        point — NOT as established fact for this repo, just a suggestion
        worth checking. This is deliberately narrow: a same-process,
        same-embedding-space lookup across repos already in this Chroma
        store. It is NOT general meta-learning or transfer learning across
        arbitrary codebases — just reuse of patterns already collected
        locally, clearly labeled as unconfirmed."""
        if self.collection.count() == 0:
            return []

        embedding = self.embedder.embed(text)
        results = safe_query(
            self.collection,
            query_embeddings=[embedding],
            n_results=top_k + 15,
        )
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        priors = []
        for id_, doc, meta, dist in zip(ids, docs, metas, distances):
            if meta.get("repo") == exclude_repo:
                continue
            priors.append({
                "id": id_,
                "text": doc,
                "source_repo": meta.get("repo"),
                "similarity": 1 - dist,
                "trust_label": "cross-repo prior — unconfirmed in this repo, treat as a hint only",
            })
            if len(priors) >= top_k:
                break
        return priors

    def get_contradictions(self, repo: str) -> List[dict]:
        """Return contradictions that are still PENDING review."""
        results = self.collection.get(where={"repo": repo})
        flagged = []
        for id_, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
            if meta.get("contradicted_by") and not meta.get("resolved"):
                flagged.append({
                    "id": id_,
                    "text": doc,
                    "contradicted_by": meta["contradicted_by"],
                    "trust": meta.get("trust"),
                })
        return flagged

    @staticmethod
    def trust_label(meta: dict) -> str:
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
        if self.collection.count() == 0:
            return []

        embedding = self.embedder.embed(text)
        results = safe_query(
            self.collection,
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