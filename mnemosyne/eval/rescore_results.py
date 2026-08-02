"""
Rescores an existing benchmark_large_results.json using the current
(negation-aware) scorer, without making any new LLM calls. Use this
after fixing a scoring bug to see the corrected numbers against data
you already collected.

    python -m mnemosyne.eval.rescore_results benchmark_large_results.json

Regenerates the exact same task suite that produced the results file
(same num_repos/templates_per_repo/seed used originally — defaults
below match the 20-repo run; pass different values if you ran with
different settings) so each stored response is rescored against its
real task definition (success_keywords/failure_keywords), not just
re-summarized as-is.
"""

import argparse
import json

from mnemosyne.eval.scoring import score_response
from mnemosyne.eval.task_generator import generate_benchmark_suite


def build_task_lookup(num_repos: int, templates_per_repo: int, seed: int) -> dict:
    suite = generate_benchmark_suite(num_repos, templates_per_repo, seed)
    lookup = {}
    for tasks in suite.values():
        for task in tasks:
            lookup[task.task_id] = task
    return lookup


def rescore_condition(condition_results: dict, task_lookup: dict) -> dict:
    rescored = {}
    changed = []
    for repo, results in condition_results.items():
        new_results = []
        for r in results:
            task = task_lookup.get(r["task_id"])
            if task is None:
                new_results.append(r)
                continue
            rescored_result = score_response(task, r["response"])
            new_entry = {
                "task_id": rescored_result.task_id,
                "success": rescored_result.success,
                "matched_success_keywords": rescored_result.matched_success_keywords,
                "matched_failure_keywords": rescored_result.matched_failure_keywords,
                "response": r["response"],
            }
            if r.get("success") != new_entry["success"]:
                changed.append((r["task_id"], r.get("success"), new_entry["success"]))
            new_results.append(new_entry)
        rescored[repo] = new_results
    return rescored, changed


def summarize(all_repo_results: dict) -> dict:
    scored = []
    for repo_results in all_repo_results.values():
        scored.extend(r for r in repo_results if r["success"] is not None)
    successes = [r for r in scored if r["success"]]
    return {
        "total_scored_tasks": len(scored),
        "successes": len(successes),
        "success_rate": len(successes) / len(scored) if scored else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Rescore existing benchmark results with the current scorer")
    parser.add_argument("input_file", help="Path to existing benchmark_large_results.json")
    parser.add_argument("--num-repos", type=int, default=20)
    parser.add_argument("--templates-per-repo", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Where to write rescored results (default: overwrite input with _rescored suffix)")
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)

    task_lookup = build_task_lookup(args.num_repos, args.templates_per_repo, args.seed)

    print("Rescoring memory_off...")
    off_rescored, off_changed = rescore_condition(data["memory_off"], task_lookup)
    print("Rescoring memory_on...")
    on_rescored, on_changed = rescore_condition(data["memory_on"], task_lookup)

    off_summary = summarize(off_rescored)
    on_summary = summarize(on_rescored)

    print("\n=== Old summary (from input file) ===")
    if "_summary" in data:
        print("memory OFF:", data["_summary"]["memory_off"])
        print("memory ON: ", data["_summary"]["memory_on"])

    print("\n=== New (rescored) summary ===")
    print(f"memory OFF: {off_summary['successes']}/{off_summary['total_scored_tasks']} = {off_summary['success_rate']}")
    print(f"memory ON:  {on_summary['successes']}/{on_summary['total_scored_tasks']} = {on_summary['success_rate']}")

    print(f"\n{len(off_changed)} memory_off task(s) changed score:")
    for task_id, old, new in off_changed:
        print(f"  {task_id}: {old} -> {new}")
    print(f"\n{len(on_changed)} memory_on task(s) changed score:")
    for task_id, old, new in on_changed:
        print(f"  {task_id}: {old} -> {new}")

    output_path = args.output or args.input_file.replace(".json", "_rescored.json")
    data["memory_off"] = off_rescored
    data["memory_on"] = on_rescored
    data["_summary"] = {"memory_off": off_summary, "memory_on": on_summary}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nRescored results written to {output_path}")


if __name__ == "__main__":
    main()