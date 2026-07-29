"""
Deliberately simple, auditable scoring: keyword presence/absence in the
agent's response text. Not a sophisticated eval, but every score is
traceable to something you can point at in the raw response — which
matters more for a first benchmark than a fancier LLM-judge approach
that would need its own validation.
"""

from dataclasses import dataclass
from typing import List, Optional

from mnemosyne.eval.tasks import BenchmarkTask


@dataclass
class TaskResult:
    task_id: str
    response: str
    success: Optional[bool]  # None for pure memory-write tasks with no scoring keywords
    matched_success_keywords: List[str]
    matched_failure_keywords: List[str]


def score_response(task: BenchmarkTask, response: str) -> TaskResult:
    lower = response.lower()

    matched_success = [kw for kw in task.success_keywords if kw.lower() in lower]
    matched_failure = [kw for kw in task.failure_keywords if kw.lower() in lower]

    if not task.success_keywords:
        # Pure memory-write task (e.g. "note this decision") — nothing to score.
        success = None
    else:
        # Require at least one success keyword AND no failure keywords.
        success = len(matched_success) > 0 and len(matched_failure) == 0

    return TaskResult(
        task_id=task.task_id,
        response=response,
        success=success,
        matched_success_keywords=matched_success,
        matched_failure_keywords=matched_failure,
    )