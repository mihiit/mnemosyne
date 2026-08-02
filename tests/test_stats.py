from mnemosyne.eval.stats import mcnemar_test, wilson_confidence_interval


def test_mcnemar_no_discordant_pairs():
    a = [True, True, False, False]
    b = [True, True, False, False]
    result = mcnemar_test(a, b)
    assert result.n_a_only == 0
    assert result.n_b_only == 0
    assert result.p_value is None


def test_mcnemar_flags_small_discordant_count():
    a = [True, False, True] + [True] * 20
    b = [False, True, False] + [True] * 20
    result = mcnemar_test(a, b)
    assert result.p_value is None
    assert "25" in result.note


def test_mcnemar_significant_with_large_clear_gap():
    a = [True] * 5 + [False] * 40 + [True] * 30
    b = [False] * 5 + [True] * 40 + [True] * 30
    result = mcnemar_test(a, b)
    assert result.n_a_only == 5
    assert result.n_b_only == 40
    assert result.p_value is not None
    assert result.p_value < 0.05
    assert result.significant_at_05 is True


def test_mcnemar_not_significant_with_balanced_discordance():
    a = [True] * 20 + [False] * 20 + [True] * 20
    b = [False] * 20 + [True] * 20 + [True] * 20
    result = mcnemar_test(a, b)
    assert result.p_value is not None
    assert result.p_value > 0.05
    assert result.significant_at_05 is False


def test_mcnemar_skips_none_entries():
    a = [True, None, False, True]
    b = [False, True, None, True]
    result = mcnemar_test(a, b)
    assert result.n_a_only == 1
    assert result.n_b_only == 0
    assert result.n_agree == 1


def test_mcnemar_rejects_mismatched_lengths():
    import pytest
    with pytest.raises(ValueError):
        mcnemar_test([True, False], [True])


def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson_confidence_interval(77, 100)
    assert lo < 0.77 < hi
    assert 0.0 <= lo
    assert hi <= 1.0


def test_wilson_interval_narrows_with_more_data():
    lo_small, hi_small = wilson_confidence_interval(5, 10)
    lo_large, hi_large = wilson_confidence_interval(500, 1000)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_wilson_interval_zero_n():
    assert wilson_confidence_interval(0, 0) == (0.0, 0.0)