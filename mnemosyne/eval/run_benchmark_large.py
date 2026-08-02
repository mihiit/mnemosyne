"""
Runs the procedurally generated multi-repo benchmark suite, memory ON
vs OFF, with per-task checkpointing so a crash or interruption partway
through a long run doesn't lose completed work.

    python -m mnemosyne.eval.run_benchmark_large --num-repos 20
    python -m mnemosyne.eval.run_benchmark_large --num-repos 125   # ~875 tasks, ~500 scored

Resumable: rerunning with the same --output picks up where it left off
(already-completed repo+condition pairs are skipped). Delete the output
file to start fresh.

Rough time budget on CPU-only Ollama (based on observed ~20-60s/call):
    num_repos=20  -> ~140 tasks  -> a few hours
    num_repos=70  -> ~490 tasks  -> most of a day
    num_repos=125 -> ~875 tasks  -> likely 20-40+ hours

Start smaller, confirm it runs cleanly, then scale up.
"""

import argparse
import json
import os
import time
from pathlib import Path

from mnemosyne.config import MnemosyneConfig
from mnemosyne.memory.store import MemoryStore
from mnemosyne.agent.coding_agent import MemoryAugmentedAgent
from mnemosyne.eval.null_memory import NullMemoryStore
from mnemosyne.eval.scoring import score_response
from mnemosyne.eval.task_generator import generate_benchmark_suite


def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"memory_off": {}, "memory_on": {}}


def save_checkpoint(path: str, data: dict):
    # Write to a temp file then replace, so a crash mid-write can't
    # corrupt the checkpoint that already has hours of progress in it.
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def run_repo_condition(repo: str, tasks: list, use_memory: bool, chroma_dir: str) -> list:
    config = MnemosyneConfig(chroma_persist_dir=chroma_dir)
    memory = MemoryStore(config) if use_memory else NullMemoryStore(config)
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
        if use_memory:
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
    parser = argparse.ArgumentParser(description="Large-scale Mnemosyne memory ON vs OFF benchmark")
    parser.add_argument("--num-repos", type=int, default=20, help="Number of fictional repos to generate")
    parser.add_argument("--templates-per-repo", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="benchmark_large_results.json")
    parser.add_argument("--chroma-root", default="./data/chroma_large_bench")
    args = parser.parse_args()

    suite = generate_benchmark_suite(args.num_repos, args.templates_per_repo, args.seed)
    total_tasks = sum(len(t) for t in suite.values())
    print(f"Generated {len(suite)} repos, {total_tasks} total tasks.")

    checkpoint = load_checkpoint(args.output)
    start_time = time.time()

    for condition_key, use_memory in [("memory_off", False), ("memory_on", True)]:
        print(f"\n=== Condition: {condition_key} ===")
        for i, (repo, tasks) in enumerate(suite.items()):
            if repo in checkpoint[condition_key]:
                print(f"  [{i+1}/{len(suite)}] {repo}: already done, skipping")
                continue

            print(f"  [{i+1}/{len(suite)}] {repo}: running {len(tasks)} tasks...")
            chroma_dir = str(Path(args.chroma_root) / condition_key / repo)
            results = run_repo_condition(repo, tasks, use_memory, chroma_dir)

            checkpoint[condition_key][repo] = results
            save_checkpoint(args.output, checkpoint)

            elapsed_min = (time.time() - start_time) / 60
            print(f"    done. Elapsed so far: {elapsed_min:.1f} min")

    print("\n=== Final summary ===")
    off_summary = summarize(checkpoint["memory_off"])
    on_summary = summarize(checkpoint["memory_on"])
    print(f"memory OFF: {off_summary['successes']}/{off_summary['total_scored_tasks']} "
          f"= {off_summary['success_rate']}")
    print(f"memory ON:  {on_summary['successes']}/{on_summary['total_scored_tasks']} "
          f"= {on_summary['success_rate']}")

    checkpoint["_summary"] = {"memory_off": off_summary, "memory_on": on_summary}
    save_checkpoint(args.output, checkpoint)
    print(f"\nFull results in {args.output}")


if __name__ == "__main__":
    main()