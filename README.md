# Mnemosyne

Persistent memory architecture for long-horizon coding agents. Instead of
chat-history memory, Mnemosyne targets **engineering memory** — what an
agent should remember about a specific codebase across sessions: past
bugs and fixes, rejected design decisions and why, recurring gotchas,
team conventions.

## Architecture

- **Episodic memory** (`mnemosyne/memory/episodic.py`) — raw, timestamped
  log of specific events. Cheap to write, numerous, not the primary
  thing an agent queries long-term.
- **Semantic memory** (`mnemosyne/memory/semantic.py`) — durable,
  distilled facts, built by *consolidating* batches of episodic entries
  via an LLM. This is where contradiction detection lives: a new fact
  that conflicts with an existing one gets flagged (`contradicted_by`)
  rather than silently overwriting it.
- **Retrieval + forgetting** (`mnemosyne/memory/retrieval.py`) —
  importance-weighted retrieval (relevance + recency + frequency) for
  normal queries, and a separate decay-based scoring for pruning stale
  episodic entries so memory doesn't grow unbounded.
- **MemoryStore** (`mnemosyne/memory/store.py`) — the facade an agent
  actually talks to: `remember()`, `recall()`, `maintain()`.
- **Agent harness** (`mnemosyne/agent/coding_agent.py`) — a deliberately
  minimal demo agent that reasons over task descriptions using recalled
  memory. Not the novel part of this project — exists only to give
  Mnemosyne something to be benchmarked against. v1 does not execute
  real code changes (see docstring in that file for why, and what v2
  would add).

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com) installed locally.

```bash
pip install -r requirements.txt

# pull the local LLM used for consolidation/contradiction detection and the agent
ollama pull llama3.1:8b
```

The first run will download the embedding model (`all-MiniLM-L6-v2`,
~80MB) from Hugging Face — this needs internet access once, then it's
cached locally (`~/.cache/huggingface`).

## Quick usage

```python


memory = MemoryStore()

memory.remember(
    "Fixed race condition in auth.py by adding a lock around token refresh",
    repo="my-project",
)

result = memory.recall("why does auth keep breaking", repo="my-project")
print(result["episodic_entries"])
print(result["semantic_facts"])

memory.maintain(repo="my-project")  # run forgetting pass
```

With the agent harness:

```python
from mnemosyne import MemoryStore
from mnemosyne.agent.coding_agent import MemoryAugmentedAgent

memory = MemoryStore()
agent = MemoryAugmentedAgent(memory, repo="my-project")

response = agent.run_task("The token refresh in auth.py is flaky again, diagnose it.")
print(response)

agent.run_session_maintenance()
```

## Running tests

Tests use a fake deterministic embedder and a scripted fake LLM, so they
run fully offline — no Ollama, no Hugging Face download needed:

```bash
python3 -m pytest tests/ -v
```

## Status / what's next

Core memory pipeline (episodic → consolidation → semantic, contradiction
detection, importance-weighted retrieval, forgetting) is built and unit
tested (13 tests, all passing, run offline against fakes).

Not yet done:
- End-to-end smoke test against the real embedding model + Ollama (needs
  to be run in an environment with internet + Ollama; this dev sandbox's
  network doesn't reach huggingface.co)
- Benchmarks: LongBench / Needle-in-Haystack, and the custom multi-session
  repo benchmark (memory ON vs OFF task-success comparison) — see
  `mnemosyne/eval/` (currently empty, next to build)
- Write-up / technical report
