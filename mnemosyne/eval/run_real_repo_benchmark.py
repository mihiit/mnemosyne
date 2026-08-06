"""
Runs a manually curated set of REAL, GitHub-sourced tasks against the
agent, memory ON vs OFF — the same harness pattern as the synthetic
benchmark, but every task here traces to a real issue/PR with a real
URL, not a procedurally generated template.
"""

import argparse
from dataclasses import dataclass, field
from typing import List

from mnemosyne.config import MnemosyneConfig
from mnemosyne.memory.store import MemoryStore
from mnemosyne.agent.coding_agent import MemoryAugmentedAgent
from mnemosyne.eval.null_memory import NullMemoryStore
from mnemosyne.eval.scoring import score_response
from mnemosyne.eval.tasks import BenchmarkTask


@dataclass
class RealTask:
    task_id: str
    source_url: str
    description: str
    success_keywords: List[str] = field(default_factory=list)
    failure_keywords: List[str] = field(default_factory=list)
    establishing_source_url: str = ""
    note: str = ""

    def to_benchmark_task(self) -> BenchmarkTask:
        return BenchmarkTask(
            task_id=self.task_id,
            description=self.description,
            success_keywords=self.success_keywords,
            failure_keywords=self.failure_keywords,
            note=self.note,
        )


REAL_TASK_SUITE: List[RealTask] = []


def run_condition(use_memory: bool, chroma_dir: str, repo_name: str) -> list:
    config = MnemosyneConfig(chroma_persist_dir=chroma_dir)
    memory = MemoryStore(config) if use_memory else NullMemoryStore(config)
    agent = MemoryAugmentedAgent(memory, repo=repo_name)

    results = []
    for real_task in REAL_TASK_SUITE:
        task = real_task.to_benchmark_task()
        response = agent.run_task(task.description, task_id=task.task_id)
        result = score_response(task, response)
        results.append({
            "task_id": result.task_id,
            "source_url": real_task.source_url,
            "success": result.success,
            "response": result.response,
        })
        if use_memory:
            agent.memory.force_consolidate(repo_name)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run the curated real-repo benchmark (E6)")
    parser.add_argument("--repo-name", default="expressjs-express")
    parser.add_argument("--chroma-root", default="./data/chroma_real_repo")
    args = parser.parse_args()

    if not REAL_TASK_SUITE:
        print("REAL_TASK_SUITE is empty — curate tasks in this file before running.")
        return

    print(f"Running {len(REAL_TASK_SUITE)} curated real tasks, memory OFF vs ON...")
    off_results = run_condition(False, f"{args.chroma_root}/off", args.repo_name)
    on_results = run_condition(True, f"{args.chroma_root}/on", args.repo_name)

    off_scored = [r for r in off_results if r["success"] is not None]
    on_scored = [r for r in on_results if r["success"] is not None]
    off_rate = sum(r["success"] for r in off_scored) / len(off_scored) if off_scored else None
    on_rate = sum(r["success"] for r in on_scored) / len(on_scored) if on_scored else None

    print(f"\nmemory OFF: {sum(r['success'] for r in off_scored)}/{len(off_scored)} = {off_rate}")
    print(f"memory ON:  {sum(r['success'] for r in on_scored)}/{len(on_scored)} = {on_rate}")


if __name__ == "__main__":
    main()