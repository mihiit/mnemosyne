"""
A deliberately minimal coding-agent harness. This is NOT the novel part
of the project — it exists only to give Mnemosyne something real to be
tested against.
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

### Durable facts (confirmed in THIS repo):
{semantic_facts}

### Specific past events:
{episodic_entries}

### Patterns from OTHER repos (unconfirmed here — treat as hints only, not established fact for this repo):
{cross_repo_priors}

## Current task
{task}

Respond with your plan/diagnosis for this task, referencing relevant memory where it applies. If you use a cross-repo pattern, say explicitly that it's unconfirmed in this specific repo."""


class MemoryAugmentedAgent:
    def __init__(self, memory: MemoryStore, repo: str):
        self.memory = memory
        self.repo = repo
        self.llm = LocalLLM(memory.config)

    def run_task(self, task: str, task_id: Optional[str] = None) -> str:
        recalled = self.memory.recall(task, repo=self.repo)
        prompt = self._build_prompt(task, recalled)

        response = self.llm.complete(prompt, system=AGENT_SYSTEM_PROMPT)

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

        cross_repo_text = "\n".join(
            f"- {p['text']} (from {p['source_repo']}) [{p['trust_label']}]"
            for p in recalled.get("cross_repo_priors", [])
        ) or "(none)"

        return TASK_PROMPT_TEMPLATE.format(
            repo=self.repo,
            semantic_facts=semantic_text,
            episodic_entries=episodic_text,
            cross_repo_priors=cross_repo_text,
            task=task,
        )

    def run_session_maintenance(self):
        self.memory.force_consolidate(self.repo)
        pruned = self.memory.maintain(self.repo)
        return {"pruned_count": len(pruned)}