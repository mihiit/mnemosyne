"""
Procedurally generates benchmark tasks at scale by instantiating a small
set of narrative templates (a decision + a later test of whether that
decision is remembered) across many different fictional repos and
substituted entities.

Why templated, not random: 500 truly independent Q&A pairs wouldn't test
memory at all — they'd test "can the LLM answer a question." What makes
the benchmark meaningful is that each template's test task can only be
answered correctly if the agent recalls a specific fact established
earlier IN THE SAME REPO. Templating lets us scale the *number* of these
narratives without losing that structure.

Each "repo" gets a handful of template instances (tech rejections,
pattern switches, bug root-causes, a broad-recall summary). Tasks from
different repos are fully independent of each other (recall must not
leak across repos — this also stress-tests repo isolation at scale).
"""

import random
from dataclasses import dataclass
from typing import List

from mnemosyne.eval.tasks import BenchmarkTask

# --- Substitution pools ---
# Each entry is a self-contained (old, new, reason) scenario so the
# generated text stays coherent rather than randomly word-salad.

TECH_REJECTIONS = [
    ("Redis", "in-memory token-bucket", "infra team doesn't want another stateful service"),
    ("Kafka", "a simple internal queue table", "operational overhead wasn't justified at current scale"),
    ("MongoDB", "the existing Postgres instance with JSONB columns", "team didn't want to run a second database"),
    ("a custom auth service", "the existing OAuth provider", "reinventing auth was seen as unnecessary risk"),
    ("GraphQL", "the existing REST endpoints", "the team didn't have bandwidth to maintain two API styles"),
    ("a message queue", "synchronous HTTP calls", "the added complexity wasn't justified for current traffic"),
    ("Elasticsearch", "Postgres full-text search", "team wanted to avoid operating another search cluster"),
    ("a microservices split", "the existing monolith", "team size didn't justify the coordination overhead"),
]

PATTERN_SWITCHES = [
    ("singleton pattern", "dependency injection", "database client", "made testing with mocked connections difficult"),
    ("global mutable state", "a request-scoped context object", "request handler", "caused hard-to-reproduce bugs under concurrent load"),
    ("inheritance-based plugin system", "a composition-based plugin system", "plugin loader", "made adding new plugin types require touching base classes"),
    ("synchronous blocking calls", "an async event loop", "notification service", "was causing request timeouts under load"),
    ("hardcoded configuration", "environment-based configuration", "deployment scripts", "made it error-prone to deploy to different environments"),
    ("a shared mutable cache", "a per-request cache instance", "pricing calculator", "caused stale-data bugs across concurrent requests"),
]

BUG_SCENARIOS = [
    ("token refresh flakiness", "auth module", "a race condition under concurrent refresh requests"),
    ("intermittent 500 errors", "payment webhook handler", "a missing idempotency check on retried webhook deliveries"),
    ("data inconsistency", "order processing pipeline", "a missing transaction boundary across two write operations"),
    ("memory growth over time", "background worker", "an unbounded cache that was never evicted"),
    ("occasional duplicate charges", "billing job", "the job wasn't idempotent under retry after a timeout"),
    ("slow response times under load", "search endpoint", "an N+1 query pattern that wasn't caught in code review"),
]

REPO_NAMES = [
    "billing-service", "notification-service", "user-service", "search-service",
    "payments-api", "order-pipeline", "auth-gateway", "inventory-service",
    "analytics-pipeline", "recommendation-engine", "pricing-service", "webhook-dispatcher",
]


def _tech_rejection_tasks(rng: random.Random, repo: str, idx: int) -> List[BenchmarkTask]:
    old, new, reason = rng.choice(TECH_REJECTIONS)
    write = BenchmarkTask(
        task_id=f"{repo}-tr{idx}-w",
        description=(
            f"The team considered {old} but decided against it, because {reason}. "
            f"They're going with {new} instead. Please note this decision."
        ),
        note=f"Establishes rejection of {old} in favor of {new}.",
    )
    test = BenchmarkTask(
        task_id=f"{repo}-tr{idx}-t",
        description=f"A contributor proposes using {old} again for this. Should this be approved?",
        success_keywords=[old.lower(), "reject", new.lower()],
        failure_keywords=["approve", "good idea", "sounds fine", "sounds good"],
        note=f"Tests recall of the {old}-rejection decision.",
    )
    return [write, test]


def _pattern_switch_tasks(rng: random.Random, repo: str, idx: int) -> List[BenchmarkTask]:
    old, new, component, reason = rng.choice(PATTERN_SWITCHES)
    write = BenchmarkTask(
        task_id=f"{repo}-ps{idx}-w",
        description=(
            f"The {component} currently uses a {old}. The team decided to switch to "
            f"a {new} instead, because the {old} {reason}. Please note this decision."
        ),
        note=f"Establishes switch from {old} to {new} for {component}.",
    )
    test = BenchmarkTask(
        task_id=f"{repo}-ps{idx}-t",
        description=(
            f"A new contributor added code to the {component} that reintroduces a {old}. "
            f"Review this and advise whether it fits the repo's conventions."
        ),
        success_keywords=[old.lower(), new.lower()],
        failure_keywords=[],
        note=f"Tests recall of the {old}->{new} switch and flags the regression.",
    )
    return [write, test]


def _bug_recurrence_tasks(rng: random.Random, repo: str, idx: int) -> List[BenchmarkTask]:
    symptom, module, cause = rng.choice(BUG_SCENARIOS)
    first = BenchmarkTask(
        task_id=f"{repo}-bug{idx}-first",
        description=f"There's a bug in the {module}: {symptom}. Investigate and note the likely cause.",
        success_keywords=[w for w in cause.split() if len(w) > 4][:3],
        note=f"Establishes root cause ({cause}) for {symptom} in {module}.",
    )
    second = BenchmarkTask(
        task_id=f"{repo}-bug{idx}-second",
        description=f"The {symptom} in the {module} is back after a recent change. Diagnose it again.",
        success_keywords=[w for w in cause.split() if len(w) > 4][:3],
        failure_keywords=["unknown cause", "first time seeing", "no prior"],
        note=f"Tests recall of the earlier root cause for {symptom}.",
    )
    return [first, second]


TEMPLATE_FNS = [_tech_rejection_tasks, _pattern_switch_tasks, _bug_recurrence_tasks]


def generate_repo_tasks(repo: str, rng: random.Random, templates_per_repo: int = 3) -> List[BenchmarkTask]:
    """Generate one repo's worth of tasks: a few template instances,
    interleaved, plus a final broad-recall task referencing whatever was
    established. Returns tasks in narrative order (writes before their
    corresponding tests, recall task last)."""
    tasks: List[BenchmarkTask] = []
    established_keywords: List[str] = []

    for i in range(templates_per_repo):
        template_fn = rng.choice(TEMPLATE_FNS)
        pair = template_fn(rng, repo, i)
        tasks.extend(pair)
        established_keywords.extend(pair[1].success_keywords)

    if established_keywords:
        recall_task = BenchmarkTask(
            task_id=f"{repo}-recall",
            description=(
                "Summarize, for a new team member, the key decisions made so far in this "
                "repo and why each was made."
            ),
            success_keywords=list(dict.fromkeys(established_keywords))[:4],  # dedupe, cap for a fair bar
            note="Broad recall test across everything established in this repo.",
        )
        tasks.append(recall_task)

    return tasks


def generate_benchmark_suite(num_repos: int, templates_per_repo: int = 3, seed: int = 42) -> dict:
    """Returns {repo_name: [BenchmarkTask, ...]} for num_repos fictional
    repos. Repo names cycle through REPO_NAMES with a numeric suffix
    once the base pool is exhausted, so num_repos can exceed len(REPO_NAMES)."""
    rng = random.Random(seed)
    suite = {}
    for i in range(num_repos):
        base_name = REPO_NAMES[i % len(REPO_NAMES)]
        repo_name = base_name if i < len(REPO_NAMES) else f"{base_name}-{i // len(REPO_NAMES)}"
        suite[repo_name] = generate_repo_tasks(repo_name, rng, templates_per_repo=templates_per_repo)
    return suite


if __name__ == "__main__":
    # Quick sanity check: print task counts for a given target scale.
    suite = generate_benchmark_suite(num_repos=70, templates_per_repo=3)
    total = sum(len(tasks) for tasks in suite.values())
    scored = sum(1 for tasks in suite.values() for t in tasks if t.success_keywords)
    print(f"{len(suite)} repos, {total} total tasks, {scored} scored tasks")