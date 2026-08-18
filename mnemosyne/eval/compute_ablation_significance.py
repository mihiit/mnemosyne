"""
Computes pairwise McNemar's tests between the 'full' config and each
other config in ablation_results.json — no new LLM calls. Critical
before drawing ANY conclusion from the ablation summary: aggregate
percentages alone don't tell you whether a gap is real or noise, and
with 6 configs compared pairwise, a Bonferroni correction is applied
so we don't over-claim significance from multiple comparisons.

    python -m mnemosyne.eval.compute_ablation_significance ablation_results.json
"""

import argparse
import json

from mnemosyne.eval.stats import mcnemar_test


def extract_paired(data: dict, config_a: str, config_b: str) -> tuple:
    a_by_key, b_by_key = {}, {}
    for repo, results in data[config_a].items():
        for r in results:
            a_by_key[(repo, r["task_id"])] = r["success"]
    for repo, results in data[config_b].items():
        for r in results:
            b_by_key[(repo, r["task_id"])] = r["success"]
    shared = sorted(set(a_by_key) & set(b_by_key))
    return [a_by_key[k] for k in shared], [b_by_key[k] for k in shared]


def main():
    parser = argparse.ArgumentParser(description="Pairwise significance testing across ablation configs")
    parser.add_argument("input_file")
    parser.add_argument("--baseline-config", default="full", help="Config every other config is compared against")
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)

    configs = [k for k in data.keys() if not k.startswith("_")]
    other_configs = [c for c in configs if c != args.baseline_config]
    n_comparisons = len(other_configs)
    bonferroni_alpha = 0.05 / n_comparisons if n_comparisons else 0.05

    print(f"Baseline config: {args.baseline_config}")
    print(f"Comparing against {n_comparisons} other configs.")
    print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.4f} (0.05 / {n_comparisons})\n")

    for other in other_configs:
        a_results, b_results = extract_paired(data, args.baseline_config, other)
        scored = [(a, b) for a, b in zip(a_results, b_results) if a is not None and b is not None]
        n = len(scored)
        a_rate = sum(1 for a, b in scored if a) / n if n else None
        b_rate = sum(1 for a, b in scored if b) / n if n else None

        result = mcnemar_test(a_results, b_results)
        print(f"=== {args.baseline_config} vs {other} ===")
        print(f"  {args.baseline_config}: {a_rate:.3f}  |  {other}: {b_rate:.3f}  (n={n})")
        print(f"  Discordant: {args.baseline_config}-only={result.n_a_only}, {other}-only={result.n_b_only}")
        if result.p_value is not None:
            sig_uncorrected = result.significant_at_05
            sig_corrected = result.p_value < bonferroni_alpha
            print(f"  p={result.p_value:.6f} | significant at 0.05: {sig_uncorrected} | significant after Bonferroni correction: {sig_corrected}")
        else:
            print(f"  {result.note}")
        print()


if __name__ == "__main__":
    main()