"""
A fixed sequence of synthetic tasks against one fictional repo, designed
so that later tasks depend on decisions/facts established in earlier
ones. This is what lets the benchmark measure something concrete: does
having memory actually change whether the agent gets later tasks right,
versus an identical agent with memory disabled.

Each task has:
    - a description (what the agent is asked to do)
    - "success_keywords": terms that should appear in a correct response
      IF the agent is using relevant memory/context correctly
    - "failure_keywords": terms that indicate the agent repeated a past
      mistake or contradicted an earlier established decision

Keyword scoring is crude but deliberately simple and auditable — you can
read exactly why a task scored the way it did, which matters more for a
first benchmark than a fancier but opaque scoring method.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkTask:
    task_id: str
    description: str
    success_keywords: List[str] = field(default_factory=list)
    failure_keywords: List[str] = field(default_factory=list)
    note: str = ""  # what this task is actually testing, for readability of results


# Narrative: a fictional repo "billing-service" evolves over a sequence
# of sessions. Early tasks establish decisions/gotchas a good
# memory-equipped agent should carry forward into later, related tasks.
BILLING_SERVICE_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        task_id="t01",
        description=(
            "The billing-service repo currently uses a singleton pattern for the "
            "database client. The team just decided to switch to dependency "
            "injection instead, because the singleton made testing with mocked "
            "DB connections very difficult. Please note this decision."
        ),
        note="Establishes a decision + its reason. No scoring — this is a pure memory-write task.",
    ),
    BenchmarkTask(
        task_id="t02",
        description=(
            "A new engineer added a PaymentProcessor class that takes the DB "
            "client as a constructor singleton reference again. Review this "
            "and advise whether it fits the repo's conventions."
        ),
        success_keywords=["dependency injection", "singleton", "test"],
        failure_keywords=[],
        note="Tests whether the agent recalls the DI decision and flags the regression.",
    ),
    BenchmarkTask(
        task_id="t03",
        description=(
            "There's a flaky test in the auth module: token refresh occasionally "
            "fails under concurrent requests. Investigate and note the likely cause."
        ),
        success_keywords=["race condition", "concurren", "lock"],
        note="Establishes a bug + root cause (race condition in token refresh).",
    ),
    BenchmarkTask(
        task_id="t04",
        description=(
            "The token refresh flakiness is back after a recent change touched "
            "the auth module. Diagnose it again."
        ),
        success_keywords=["race condition", "lock", "concurren"],
        failure_keywords=["unknown cause", "no prior", "first time seeing"],
        note="Tests whether the agent recalls the earlier root cause instead of re-diagnosing from scratch.",
    ),
    BenchmarkTask(
        task_id="t05",
        description=(
            "The team rejected using Redis for the rate-limiter because the "
            "infra team doesn't want another stateful service to operate; they "
            "asked for an in-memory token-bucket implementation instead. Note this."
        ),
        note="Establishes a rejected-option decision + reason.",
    ),
    BenchmarkTask(
        task_id="t06",
        description=(
            "A contributor proposes adding Redis-backed rate limiting to the "
            "billing-service. Should this be approved?"
        ),
        success_keywords=["redis", "reject", "in-memory", "stateful", "infra"],
        failure_keywords=["approve", "good idea", "sounds fine"],
        note="Tests whether the agent recalls the rejected-Redis decision and pushes back appropriately.",
    ),
    BenchmarkTask(
        task_id="t07",
        description=(
            "Summarize, for a new team member, the key architectural decisions "
            "made so far for billing-service and why each was made."
        ),
        success_keywords=["dependency injection", "singleton", "race condition", "redis"],
        note="Broad recall test — checks whether multiple established facts surface together.",
    ),
]