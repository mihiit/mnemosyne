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

def test_negated_failure_keyword_does_not_cancel_success():
    """Regression test for a real scoring bug found in the large-scale
    benchmark: the agent correctly REJECTED a proposal while using the
    word 'approve' in a negated phrase ('rather than approving'), and
    the old substring-only matcher incorrectly scored this as a failure."""
    task = BenchmarkTask(
        task_id="t04",
        description="...",
        success_keywords=["elasticsearch", "reject"],
        failure_keywords=["approve"],
    )
    response = (
        "I recommend rejecting the proposal to use Elasticsearch again. "
        "I strongly advise against reintroducing Elasticsearch, rather than approving the proposal."
    )
    result = score_response(task, response)
    assert result.success is True
    assert result.matched_failure_keywords == []  # "approve" only appears negated


def test_genuine_failure_keyword_still_counts():
    """Make sure the negation fix doesn't over-correct: a real,
    non-negated failure keyword should still cancel success."""
    task = BenchmarkTask(
        task_id="t05",
        description="...",
        success_keywords=["redis"],
        failure_keywords=["approve"],
    )
    response = "Sounds like a good use of Redis. I approve this proposal."
    result = score_response(task, response)
    assert result.success is False
    assert result.matched_failure_keywords == ["approve"]


def test_null_memory_store_interface_is_inert():
    store = NullMemoryStore()
    store.remember("something", repo="repo-a")  # should not raise
    result = store.recall("query", repo="repo-a")
    assert result == {"semantic_facts": [], "episodic_entries": [], "cross_repo_priors": []}
    assert store.maintain("repo-a") == []
    assert store.contradictions("repo-a") == []
    assert store.force_consolidate("repo-a") == []