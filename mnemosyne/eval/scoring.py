"""
Deliberately simple, auditable scoring: keyword presence/absence in the
agent's response text. Not a sophisticated eval, but every score is
traceable to something you can point at in the raw response — which
matters more for a first benchmark than a fancier LLM-judge approach
that would need its own validation.

Negation-aware: a keyword match found inside a negated phrase (e.g.
"advise against approving", "reject the proposal to use X") doesn't
count as a genuine match. Found via a real case in the large benchmark
run: the agent correctly REJECTED a proposal while using the word
"approve" in the process ("...rather than approving..."), and the old
substring-only matcher scored that as a failure-keyword hit, incorrectly
marking a correct response as a failure.

This is a heuristic, not real NLP negation detection — it checks for a
negation cue word within a fixed character window before the match. It
won't catch every case, but it fixes the specific, confirmed failure
mode without adding a dependency or another LLM call.
"""

from dataclasses import dataclass
from typing import List, Optional

from mnemosyne.eval.tasks import BenchmarkTask

NEGATION_CUES = [
    "against", "reject", "instead of", "rather than", "avoid",
    "not ", "n't ", "without", "advise against", "recommend against",
]
NEGATION_WINDOW = 45  # characters to look back from the match for a negation cue


def _keyword_has_non_negated_match(text_lower: str, keyword_lower: str) -> bool:
    """Return True if the keyword appears at least once WITHOUT a negation
    cue immediately before it. A keyword that only ever appears inside a
    negated phrase does not count as a real match."""
    start = 0
    while True:
        idx = text_lower.find(keyword_lower, start)
        if idx == -1:
            return False
        window_text = text_lower[max(0, idx - NEGATION_WINDOW):idx]
        negated = any(cue in window_text for cue in NEGATION_CUES)
        if not negated:
            return True
        start = idx + len(keyword_lower)


@dataclass
class TaskResult:
    task_id: str
    response: str
    success: Optional[bool]  # None for pure memory-write tasks with no scoring keywords
    matched_success_keywords: List[str]
    matched_failure_keywords: List[str]


def score_response(task: BenchmarkTask, response: str) -> TaskResult:
    lower = response.lower()

    matched_success = [
        kw for kw in task.success_keywords
        if _keyword_has_non_negated_match(lower, kw.lower())
    ]
    matched_failure = [
        kw for kw in task.failure_keywords
        if _keyword_has_non_negated_match(lower, kw.lower())
    ]

    if not task.success_keywords:
        # Pure memory-write task (e.g. "note this decision") — nothing to score.
        success = None
    else:
        # Require at least one non-negated success keyword AND no non-negated failure keywords.
        success = len(matched_success) > 0 and len(matched_failure) == 0

    return TaskResult(
        task_id=task.task_id,
        response=response,
        success=success,
        matched_success_keywords=matched_success,
        matched_failure_keywords=matched_failure,
    )