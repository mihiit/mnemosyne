from mnemosyne.eval.scoring import score_response
from mnemosyne.eval.tasks import BenchmarkTask
from mnemosyne.eval.null_memory import NullMemoryStore


def test_pure_write_task_has_no_success_value():
    task = BenchmarkTask(task_id="t01", description="Note this decision.")
    result = score_response(task, "Noted.")
    assert result.success is None


def test_success_requires_keyword_and_no_failure_keyword():
    task = BenchmarkTask(
        task_id="t02",
        description="...",
        success_keywords=["race condition"],
        failure_keywords=["unknown cause"],
    )
    good = score_response(task, "This looks like a race condition in token refresh.")
    assert good.success is True

    bad = score_response(task, "Unknown cause, first time seeing this, but might be a race condition.")
    assert bad.success is False  # failure keyword present cancels the success


def test_missing_success_keyword_fails():
    task = BenchmarkTask(task_id="t03", description="...", success_keywords=["redis"])
    result = score_response(task, "Sounds like a fine idea to me.")
    assert result.success is False


def test_null_memory_store_interface_is_inert():
    store = NullMemoryStore()
    store.remember("something", repo="repo-a")  # should not raise
    result = store.recall("query", repo="repo-a")
    assert result == {"semantic_facts": [], "episodic_entries": []}
    assert store.maintain("repo-a") == []
    assert store.contradictions("repo-a") == []
    assert store.force_consolidate("repo-a") == []