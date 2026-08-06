from mnemosyne.eval.fetch_real_repo_candidates import flag_decision_candidates


def test_flags_issues_with_decision_language():
    issues = [
        {"number": 1, "title": "Switch from callback API to promises",
         "body": "We decided to migrate instead of maintaining both.",
         "comments": 12, "html_url": "x"},
        {"number": 2, "title": "Fix typo in README", "body": "small fix",
         "comments": 1, "html_url": "y"},
    ]
    result = flag_decision_candidates(issues)
    assert len(result) == 1
    assert result[0]["number"] == 1
    assert "decided" in result[0]["matched_keywords"]


def test_flags_multiple_keyword_matches():
    issues = [
        {"number": 3, "title": "Revert PR 42", "body": "This broke tests, reverting.",
         "comments": 8, "html_url": "z"},
    ]
    result = flag_decision_candidates(issues)
    assert len(result) == 1
    assert "revert" in result[0]["matched_keywords"]


def test_handles_missing_body_gracefully():
    issues = [
        {"number": 4, "title": "We decided to deprecate this", "body": None,
         "comments": 3, "html_url": "w"},
    ]
    result = flag_decision_candidates(issues)
    assert len(result) == 1


def test_no_matches_returns_empty_list():
    issues = [
        {"number": 5, "title": "Add new endpoint", "body": "Adds a GET route.",
         "comments": 2, "html_url": "v"},
    ]
    assert flag_decision_candidates(issues) == []