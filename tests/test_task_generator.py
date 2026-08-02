from mnemosyne.eval.task_generator import generate_benchmark_suite, generate_repo_tasks
import random


def test_generate_repo_tasks_produces_write_and_test_pairs():
    rng = random.Random(1)
    tasks = generate_repo_tasks("test-repo", rng, templates_per_repo=2)

    # 2 templates -> 2 write + 2 test tasks + 1 recall task = 5
    assert len(tasks) == 5
    write_tasks = [t for t in tasks if not t.success_keywords]
    scored_tasks = [t for t in tasks if t.success_keywords]
    assert len(write_tasks) == 2
    assert len(scored_tasks) == 3  # 2 test tasks + 1 recall


def test_generate_benchmark_suite_scales_with_num_repos():
    suite_small = generate_benchmark_suite(num_repos=5, templates_per_repo=2, seed=1)
    suite_large = generate_benchmark_suite(num_repos=50, templates_per_repo=2, seed=1)

    assert len(suite_small) == 5
    assert len(suite_large) == 50

    total_small = sum(len(tasks) for tasks in suite_small.values())
    total_large = sum(len(tasks) for tasks in suite_large.values())
    assert total_large > total_small


def test_generate_benchmark_suite_is_deterministic_with_seed():
    suite_a = generate_benchmark_suite(num_repos=10, templates_per_repo=3, seed=7)
    suite_b = generate_benchmark_suite(num_repos=10, templates_per_repo=3, seed=7)

    for repo in suite_a:
        texts_a = [t.description for t in suite_a[repo]]
        texts_b = [t.description for t in suite_b[repo]]
        assert texts_a == texts_b


def test_repo_names_extend_beyond_base_pool():
    from mnemosyne.eval.task_generator import REPO_NAMES
    suite = generate_benchmark_suite(num_repos=len(REPO_NAMES) + 3, templates_per_repo=1, seed=1)
    assert len(suite) == len(REPO_NAMES) + 3
    assert len(set(suite.keys())) == len(suite)  # all repo names unique, no collisions


def test_task_ids_are_unique_within_a_repo():
    rng = random.Random(3)
    tasks = generate_repo_tasks("test-repo", rng, templates_per_repo=3)
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids))