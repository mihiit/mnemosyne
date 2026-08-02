"""
Runs the same benchmark suite across multiple memory configurations to
isolate which mechanism drives the memory-ON improvement:

    full                  - everything enabled (the reported system)
    no_trust_retrieval    - trust-weighted retrieval disabled (similarity-only ranking)
    no_active_resolution  - contradictions only ever flagged, never auto-resolved
    no_cross_repo         - no cold-start cross-repo priors
    no_corroboration      - corroboration verdicts treated as independent new facts
    naive_baseline        - flat similarity-only memory, no consolidation/trust/resolution at all

    python -m mnemosyne.eval.run_ablation --num-repos 15 --configs full no_trust_retrieval naive_baseline

Checkpointed the same way as run_benchmark_large.py — safe to interrupt
and resume, and each config gets its own isolated Chroma directory so
none of them share or contaminate state.
"""

import argparse
import json
import os
import time
from pathlib import Path

from mnemosyne.config import MnemosyneConfig
from mnemosyne.memory.store import MemoryStore
from mnemosyne.agent.coding_agent import MemoryAugmentedAgent
from mnemosyne.eval.naive_baseline_memory import NaiveBaselineMemory
from mnemosyne.eval.scoring import score_response
from mnemosyne.eval.task_generator import generate_benchmark_suite

ABLATION_CONFIGS = {
    "full": {},
    "no_trust_retrieval": {"enable_trust_weighted_retrieval": False},
    "no_active_resolution": {"enable_active_contradiction_resolution": False},
    "no_cross_repo": {"enable_cross_repo_priors": False},
    "no_corroboration": {"enable_corroboration": False},
}


def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(path: str, data: dict):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def build_memory(config_name: str, chroma_dir: str):
    if config_name == "naive_baseline":
        config = MnemosyneConfig(chroma_persist_dir=chroma_dir)
        return NaiveBaselineMemory(config)
    overrides = ABLATION_CONFIGS[config_name]
    config = MnemosyneConfig(chroma_persist_dir=chroma_dir, **overrides)
    return MemoryStore(config)


def run_repo(config_name: str, repo: str, tasks: list, chroma_root: str) -> list:
    chroma_dir = str(Path(chroma_root) / config_name / repo)
    memory = build_memory(config_name, chroma_dir)
    agent = MemoryAugmentedAgent(memory, repo=repo)

    results = []
    for task in tasks:
        response = agent.run_task(task.description, task_id=task.task_id)
        result = score_response(task, response)
        results.append({
            "task_id": result.task_id,
            "success": result.success,
            "matched_success_keywords": result.matched_success_keywords,
            "matched_failure_keywords": result.matched_failure_keywords,
            "response": result.response,
        })
        if hasattr(agent.memory, "force_consolidate"):
            agent.memory.force_consolidate(repo)
    return results


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
    parser = argparse.ArgumentParser(description="Ablation study across Mnemosyne mechanisms")
    parser.add_argument("--num-repos", type=int, default=15)
    parser.add_argument("--templates-per-repo", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="ablation_results.json")
    parser.add_argument("--chroma-root", default="./data/chroma_ablation")
    parser.add_argument(
        "--configs", nargs="+",
        default=["full", "no_trust_retrieval", "no_active_resolution", "no_cross_repo", "no_corroboration", "naive_baseline"],
        help="Which configs to run (space-separated)",
    )
    args = parser.parse_args()

    suite = generate_benchmark_suite(args.num_repos, args.templates_per_repo, args.seed)
    total_tasks = sum(len(t) for t in suite.values())
    print(f"Generated {len(suite)} repos, {total_tasks} total tasks per config.")
    print(f"Running configs: {args.configs}")

    checkpoint = load_checkpoint(args.output)
    start_time = time.time()

    for config_name in args.configs:
        checkpoint.setdefault(config_name, {})
        print(f"\n=== Config: {config_name} ===")
        for i, (repo, tasks) in enumerate(suite.items()):
            if repo in checkpoint[config_name]:
                print(f"  [{i+1}/{len(suite)}] {repo}: already done, skipping")
                continue
            print(f"  [{i+1}/{len(suite)}] {repo}: running {len(tasks)} tasks...")
            results = run_repo(config_name, repo, tasks, args.chroma_root)
            checkpoint[config_name][repo] = results
            save_checkpoint(args.output, checkpoint)
            elapsed_min = (time.time() - start_time) / 60
            print(f"    done. Elapsed so far: {elapsed_min:.1f} min")

    print("\n=== Final summary across configs ===")
    summary = {}
    for config_name in args.configs:
        s = summarize(checkpoint[config_name])
        summary[config_name] = s
        print(f"{config_name}: {s['successes']}/{s['total_scored_tasks']} = {s['success_rate']}")

    checkpoint["_summary"] = summary
    save_checkpoint(args.output, checkpoint)
    print(f"\nFull results in {args.output}")


if __name__ == "__main__":
    main()