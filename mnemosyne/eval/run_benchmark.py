"""
Runs the BILLING_SERVICE_TASKS sequence twice against the same agent
code path — once with a real MemoryStore (memory ON), once with
NullMemoryStore (memory OFF) — and compares task success rates.

This is the benchmark that actually produces a number for the resume
claim. Run it directly:

    python -m mnemosyne.eval.run_benchmark

Requires Ollama running locally with the model set in MnemosyneConfig
pulled (default: llama3.1:8b). Uses a fresh, isolated Chroma directory
per run so repeated runs don't contaminate each other with old memory.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from mnemosyne.config import MnemosyneConfig
from mnemosyne.memory.store import MemoryStore
from mnemosyne.agent.coding_agent import MemoryAugmentedAgent
from mnemosyne.eval.null_memory import NullMemoryStore
from mnemosyne.eval.scoring import score_response
from mnemosyne.eval.tasks import BILLING_SERVICE_TASKS

REPO_NAME = "billing-service"


def run_condition(use_memory: bool, chroma_dir: str) -> dict:
    config = MnemosyneConfig(chroma_persist_dir=chroma_dir)
    memory = MemoryStore(config) if use_memory else NullMemoryStore(config)
    agent = MemoryAugmentedAgent(memory, repo=REPO_NAME)

    results = []
    for task in BILLING_SERVICE_TASKS:
        response = agent.run_task(task.description, task_id=task.task_id)
        result = score_response(task, response)
        results.append(result)
        # Force consolidation after each task so later tasks in the same
        # run can actually benefit from facts extracted from earlier ones
        # — with the default batch size of 10, a 7-task run would never
        # naturally trigger consolidation otherwise.
        if use_memory:
            agent.memory.force_consolidate(REPO_NAME)

    return results


def summarize(results: list, label: str) -> dict:
    scored = [r for r in results if r.success is not None]
    successes = [r for r in scored if r.success]
    return {
        "label": label,
        "total_scored_tasks": len(scored),
        "successes": len(successes),
        "success_rate": len(successes) / len(scored) if scored else None,
        "per_task": [
            {
                "task_id": r.task_id,
                "success": r.success,
                "matched_success_keywords": r.matched_success_keywords,
                "matched_failure_keywords": r.matched_failure_keywords,
                "response": r.response,
            }
            for r in results
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Mnemosyne memory ON vs OFF benchmark")
    parser.add_argument("--output", default="benchmark_results.json", help="Path to write results JSON")
    args = parser.parse_args()

    tmp_root = tempfile.mkdtemp(prefix="mnemosyne_bench_")
    try:
        print("Running WITHOUT memory (baseline)...")
        off_dir = str(Path(tmp_root) / "off")
        off_results = run_condition(use_memory=False, chroma_dir=off_dir)
        off_summary = summarize(off_results, "memory_off")
        print(f"  memory OFF success rate: {off_summary['success_rate']}")

        print("Running WITH memory...")
        on_dir = str(Path(tmp_root) / "on")
        on_results = run_condition(use_memory=True, chroma_dir=on_dir)
        on_summary = summarize(on_results, "memory_on")
        print(f"  memory ON success rate: {on_summary['success_rate']}")

        report = {"memory_off": off_summary, "memory_on": on_summary}
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull results written to {args.output}")

        print("\n--- Per-task comparison ---")
        for off_r, on_r in zip(off_results, on_results):
            if off_r.success is None:
                continue
            print(f"{off_r.task_id}: OFF={off_r.success}  ON={on_r.success}")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()