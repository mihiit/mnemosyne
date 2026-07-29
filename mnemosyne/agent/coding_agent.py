"""
A deliberately minimal coding-agent harness. This is NOT the novel part
of the project — it exists only to give Mnemosyne something real to be
tested against. Don't invest deep effort extending this into a full
agent; the memory system is the point.

Flow per task:
    1. recall() relevant memory for the task description
    2. inject that memory into the LLM's context
    3. LLM proposes an action / diagnosis / plan (no actual code execution
       in v1 — see note below)
    4. remember() the outcome as a new episodic entry

v1 intentionally does NOT execute real code changes or run test suites —
that's a large scope increase (sandboxing, repo checkout management,
etc.) that would turn this into the Helios idea. For now the agent
reasons over task descriptions and file snippets you provide, and the
benchmark measures whether memory improves its reasoning/decisions
across a sequence of related tasks, not whether it can autonomously
edit a live repo. Wiring in real execution is a natural v2 step.
"""

from typing import Optional

from mnemosyne.memory.llm import LocalLLM
from mnemosyne.memory.store import MemoryStore

AGENT_SYSTEM_PROMPT = """You are a coding agent working on a specific repository over \
many sessions. You are given relevant memory from past sessions (durable facts and \
specific past events) before each task. Each durable fact is tagged with a trust level \
in brackets (e.g. "well-established (reinforced 3x)" vs "stated once, not since \
reinforced"). Calibrate your own confidence in your response to match: state \
well-established facts plainly, but explicitly flag when you're relying on something \
that's only been stated once and never corroborated. Use this memory to avoid repeating \
past mistakes and to stay consistent with past decisions. If memory conflicts with what \
seems reasonable for the current task, say so explicitly rather than ignoring it."""

TASK_PROMPT_TEMPLATE = """# Repo: {repo}

## Relevant memory from past sessions

### Durable facts:
{semantic_facts}

### Specific past events:
{episodic_entries}

## Current task
{task}

Respond with your plan/diagnosis for this task, referencing relevant memory where it applies."""


class MemoryAugmentedAgent:
    def __init__(self, memory: MemoryStore, repo: str):
        self.memory = memory
        self.repo = repo
        self.llm = LocalLLM(memory.config)

    def run_task(self, task: str, task_id: Optional[str] = None) -> str:
        recalled = self.memory.recall(task, repo=self.repo)
        prompt = self._build_prompt(task, recalled)

        response = self.llm.complete(prompt, system=AGENT_SYSTEM_PROMPT)

        # Log this task + outcome as a new episodic entry so future tasks
        # can build on it.
        self.memory.remember(
            text=f"Task: {task}\nAgent response: {response}",
            repo=self.repo,
            task_id=task_id,
        )
        return response

    def _build_prompt(self, task: str, recalled: dict) -> str:
        semantic_text = "\n".join(
            f"- {f['text']} [{f.get('trust_label', 'confidence unknown')}]"
            for f in recalled["semantic_facts"]
        ) or "(none yet)"

        episodic_text = "\n".join(
            f"- {e['text']} (importance {e['importance']:.2f})"
            for e in recalled["episodic_entries"]
        ) or "(none yet)"

        return TASK_PROMPT_TEMPLATE.format(
            repo=self.repo,
            semantic_facts=semantic_text,
            episodic_entries=episodic_text,
            task=task,
        )

    def run_session_maintenance(self):
        """Call once at the end of a session — runs the forgetting pass
        and forces any pending consolidation."""
        self.memory.force_consolidate(self.repo)
        pruned = self.memory.maintain(self.repo)
        return {"pruned_count": len(pruned)}
