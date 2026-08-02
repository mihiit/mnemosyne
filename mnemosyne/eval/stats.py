"""
Statistical significance testing for comparing two conditions (e.g.
memory ON vs OFF, or one ablation config vs another) on the SAME set of
tasks. This is what turns "77% vs 53%" into a claim a reviewer can
actually evaluate, rather than an eyeballed percentage difference.

McNemar's test is the standard choice here: it's designed exactly for
paired binary outcomes (same tasks, two conditions, success/fail each),
and only looks at the DISCORDANT pairs (tasks where the two conditions
disagree) — which is the right lens, since agreement doesn't tell you
anything about which condition is better.

Uses only the Python standard library (no scipy dependency) via a
Wilson-score-adjusted continuity-corrected chi-square approximation,
which is standard practice for McNemar's test and accurate for the
discordant-pair counts typical of a 100-1000 task benchmark.
"""

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class McNemarResult:
    n_a_only: int  # succeeded under condition A, failed under condition B
    n_b_only: int  # succeeded under condition B, failed under condition A
    n_agree: int  # both succeeded or both failed (uninformative for the test)
    statistic: Optional[float]  # chi-square statistic (None if too few discordant pairs for the approximation)
    p_value: Optional[float]
    significant_at_05: Optional[bool]
    note: str


def mcnemar_test(results_a: List[Optional[bool]], results_b: List[Optional[bool]]) -> McNemarResult:
    """results_a / results_b: parallel lists of per-task success (True/False),
    same task order, same length. None entries (unscored tasks) are skipped
    in both lists together — pass only the scored subset if you've already
    filtered, or pass full lists and this will filter internally."""
    if len(results_a) != len(results_b):
        raise ValueError("results_a and results_b must be the same length (same tasks, same order)")

    n_a_only = 0  # A succeeded, B failed
    n_b_only = 0  # B succeeded, A failed
    n_agree = 0

    for a, b in zip(results_a, results_b):
        if a is None or b is None:
            continue  # unscored task (e.g. a pure memory-write task) — not part of the comparison
        if a and not b:
            n_a_only += 1
        elif b and not a:
            n_b_only += 1
        else:
            n_agree += 1

    n_discordant = n_a_only + n_b_only
    if n_discordant == 0:
        return McNemarResult(
            n_a_only=0, n_b_only=0, n_agree=n_agree,
            statistic=None, p_value=None, significant_at_05=None,
            note="No discordant pairs — conditions never disagreed on any task, so no significance test applies.",
        )

    if n_discordant < 25:
        # The chi-square approximation is unreliable below ~25 discordant
        # pairs; flag this rather than reporting a misleading p-value.
        note = (
            f"Only {n_discordant} discordant pairs — below the ~25 typically "
            "needed for the chi-square approximation to be reliable. Report "
            "the raw counts, not a p-value, or use an exact binomial test instead."
        )
        return McNemarResult(
            n_a_only=n_a_only, n_b_only=n_b_only, n_agree=n_agree,
            statistic=None, p_value=None, significant_at_05=None, note=note,
        )

    # Continuity-corrected McNemar's chi-square statistic (1 degree of freedom).
    statistic = ((abs(n_a_only - n_b_only) - 1) ** 2) / (n_a_only + n_b_only)
    p_value = _chi2_sf_1df(statistic)

    return McNemarResult(
        n_a_only=n_a_only, n_b_only=n_b_only, n_agree=n_agree,
        statistic=statistic, p_value=p_value, significant_at_05=(p_value < 0.05),
        note=f"{n_discordant} discordant pairs; chi-square approximation applies.",
    )


def _chi2_sf_1df(x: float) -> float:
    """Survival function (1 - CDF) of the chi-square distribution with 1
    degree of freedom, i.e. P(X > x). For df=1, chi-square is the square
    of a standard normal, so this reduces to 2 * (1 - Phi(sqrt(x))),
    computed via the standard erf-based normal CDF — no scipy needed."""
    if x <= 0:
        return 1.0
    z = math.sqrt(x)
    # 1 - Phi(z) via the complementary error function.
    upper_tail = 0.5 * math.erfc(z / math.sqrt(2))
    return 2 * upper_tail


def wilson_confidence_interval(successes: int, n: int, confidence: float = 0.95) -> tuple:
    """Wilson score interval for a binomial proportion — a better-calibrated
    confidence interval than the naive successes/n +/- 1.96*sqrt(p(1-p)/n)
    normal approximation, especially useful when success_rate is near 0 or 1
    or n is only moderately large (as in a ~100-task benchmark)."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-9 else _z_for_confidence(confidence)
    p_hat = successes / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _z_for_confidence(confidence: float) -> float:
    # Inverse standard normal CDF via a simple bisection on erf — fine for
    # the handful of confidence levels this would realistically be called with.
    lo, hi = 0.0, 10.0
    target = confidence
    for _ in range(60):
        mid = (lo + hi) / 2
        cdf = 0.5 * (1 + math.erf(mid / math.sqrt(2)))
        prob = 2 * cdf - 1  # P(-mid < Z < mid)
        if prob < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2