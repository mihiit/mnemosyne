"""
Computes McNemar's test and Wilson confidence intervals from an existing
benchmark_large_results(_rescored).json — no new LLM calls. This turns
the raw success-rate numbers into a statistically defensible claim.

    python -m mnemosyne.eval.compute_significance benchmark_large_results_rescored.json
"""

import argparse
import json

from mnemosyne.eval.stats import mcnemar_test, wilson_confidence_interval


def extract_paired_results(data: dict) -> tuple:
    """Returns (off_results, on_results) as parallel lists of
    Optional[bool], aligned by (repo, task_id) so the pairing is exact
    even if dict ordering differs between the two conditions."""
    off_by_key = {}
    for repo, results in data["memory_off"].items():
        for r in results:
            off_by_key[(repo, r["task_id"])] = r["success"]

    on_by_key = {}
    for repo, results in data["memory_on"].items():
        for r in results:
            on_by_key[(repo, r["task_id"])] = r["success"]

    shared_keys = sorted(set(off_by_key) & set(on_by_key))
    off_results = [off_by_key[k] for k in shared_keys]
    on_results = [on_by_key[k] for k in shared_keys]
    return off_results, on_results


def main():
    parser = argparse.ArgumentParser(description="Compute statistical significance from existing benchmark results")
    parser.add_argument("input_file", help="Path to benchmark_large_results.json or its _rescored version")
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)

    off_results, on_results = extract_paired_results(data)
    scored_pairs = [(a, b) for a, b in zip(off_results, on_results) if a is not None and b is not None]
    n_scored = len(scored_pairs)
    off_successes = sum(1 for a, b in scored_pairs if a)
    on_successes = sum(1 for a, b in scored_pairs if b)

    print(f"Paired, scored tasks: {n_scored}")
    print(f"memory OFF: {off_successes}/{n_scored} = {off_successes/n_scored:.3f}" if n_scored else "no scored tasks")
    print(f"memory ON:  {on_successes}/{n_scored} = {on_successes/n_scored:.3f}" if n_scored else "")

    off_ci = wilson_confidence_interval(off_successes, n_scored)
    on_ci = wilson_confidence_interval(on_successes, n_scored)
    print(f"\n95% Wilson CI, memory OFF: [{off_ci[0]:.3f}, {off_ci[1]:.3f}]")
    print(f"95% Wilson CI, memory ON:  [{on_ci[0]:.3f}, {on_ci[1]:.3f}]")
    if off_ci[1] < on_ci[0]:
        print("-> Confidence intervals do NOT overlap: strong evidence of a real difference.")
    else:
        print("-> Confidence intervals overlap: McNemar's test (below) is the more precise check, since it uses the paired structure rather than treating the two conditions as independent samples.")

    print("\n=== McNemar's test (paired, same tasks both conditions) ===")
    result = mcnemar_test(off_results, on_results)
    print(f"Tasks where OFF succeeded but ON failed: {result.n_a_only}")
    print(f"Tasks where ON succeeded but OFF failed: {result.n_b_only}")
    print(f"Tasks where both agreed (uninformative): {result.n_agree}")
    print(f"Note: {result.note}")
    if result.p_value is not None:
        print(f"Chi-square statistic: {result.statistic:.4f}")
        print(f"p-value: {result.p_value:.6f}")
        print(f"Significant at alpha=0.05: {result.significant_at_05}")


if __name__ == "__main__":
    main()