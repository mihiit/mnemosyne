"""
Central configuration for Mnemosyne.

Kept as one file on purpose: every tunable that affects memory behavior
(decay rates, thresholds, model names) should be visible in one place,
since a lot of your eventual benchmarking will be "run with config A vs
config B" and you don't want these scattered across modules.
"""

from dataclasses import dataclass


@dataclass
class MnemosyneConfig:
    # --- Storage ---
    chroma_persist_dir: str = "./data/chroma"
    episodic_collection: str = "episodic_memory"
    semantic_collection: str = "semantic_memory"

    # --- Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- LLM (via Ollama, local) ---
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.2

    # --- Retrieval ---
    retrieval_top_k: int = 5

    w_relevance: float = 0.6
    w_recency: float = 0.25
    w_frequency: float = 0.15

    # Weight for trust in SEMANTIC FACT ranking specifically (separate from
    # the episodic w_relevance/w_recency/w_frequency above, since semantic
    # facts don't have a meaningful "recency of access" the same way raw
    # episodic entries do). semantic_score = w_semantic_relevance * similarity
    # + w_trust * trust. A well-established, repeatedly-corroborated fact
    # should outrank an equally-relevant but shaky, once-stated one.
    w_semantic_relevance: float = 0.65
    w_trust: float = 0.35

    # --- Forgetting / decay ---
    recency_half_life_days: float = 14.0
    forgetting_threshold: float = 0.15
    min_age_before_prune_days: float = 3.0

    # --- Consolidation ---
    consolidation_batch_size: int = 10
    semantic_similarity_threshold: float = 0.80

    # --- Trust / corroboration tracking ---
    corroboration_boost: float = 0.3
    contradiction_penalty: float = 0.4
    refinement_decay: float = 0.5
    # When a new fact contradicts an existing one, if the trust gap between
    # them is at least this large, auto-resolve in favor of whichever has
    # higher trust rather than leaving both flagged indefinitely. Below
    # this margin, the contradiction is genuinely ambiguous and is left
    # pending for a human or the agent to judge explicitly.
    contradiction_auto_resolve_margin: float = 0.2
    
    # --- Ablation toggles ---
    # Each defaults to True (full system). Flip individually to isolate
    # which mechanism drives a given effect in an ablation study.
    enable_trust_weighted_retrieval: bool = True  # if False, semantic facts rank by similarity only
    enable_active_contradiction_resolution: bool = True  # if False, contradictions are only ever flagged, never auto-resolved
    enable_cross_repo_priors: bool = True  # if False, cold-start repos get no cross-repo hints
    enable_corroboration: bool = True  # if False, a "corroboration" verdict is treated as "different" (creates a duplicate fact instead of reinforcing)