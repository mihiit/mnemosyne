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
    # all-MiniLM-L6-v2: fast, free, local, 384-dim. Good default for v1.
    # Swap to a bigger model (e.g. all-mpnet-base-v2) later if retrieval
    # quality becomes the bottleneck rather than a nice-to-have.
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- LLM (via Ollama, local) ---
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.2  # low temp: consolidation/contradiction checks want consistency, not creativity

    # --- Retrieval ---
    retrieval_top_k: int = 5

    # Weights for importance-weighted retrieval scoring.
    # score = w_relevance * cosine_sim + w_recency * recency_score + w_frequency * freq_score
    w_relevance: float = 0.6
    w_recency: float = 0.25
    w_frequency: float = 0.15

    # --- Forgetting / decay ---
    # Half-life in days for recency decay (exponential decay).
    recency_half_life_days: float = 14.0
    # Entries below this importance score are candidates for pruning.
    forgetting_threshold: float = 0.15
    # Never prune entries newer than this, regardless of score (grace period).
    min_age_before_prune_days: float = 3.0

    # --- Consolidation ---
    # Number of new episodic entries that triggers a consolidation pass.
    consolidation_batch_size: int = 10
    # Cosine similarity above which a new fact is considered "about the
    # same thing" as an existing semantic fact (candidate for merge or
    # contradiction check).
    semantic_similarity_threshold: float = 0.80

    # --- Trust / corroboration tracking ---
    # When a new fact corroborates an existing one, trust moves toward 1.0
    # by this fraction of the remaining distance (e.g. 0.7 -> 0.7*(1-trust)
    # added each time). Diminishing returns as trust approaches 1.0.
    corroboration_boost: float = 0.3
    # When a new fact contradicts an existing one, trust is multiplied by
    # this factor (separate from the softer refinement decay).
    contradiction_penalty: float = 0.4
    # When a new fact merely refines/adds detail to an existing one, the
    # old fact's trust decays by this factor (superseded, not wrong).
    refinement_decay: float = 0.5