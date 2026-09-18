"""Tests for the calibration-size learning curve.

The curve decides whether an expensive re-run is justified, so the sampler it rests on is pinned:
subsets must be reproducible, the right size, drawn without replacement, taken only from the fit
split, and separated by a salt so two datasets do not share draws.
"""
import numpy as np
import jevbench_learning_curve as lc


def test_subsets_are_reproducible_disjoint_and_from_fit_only():
    fit = np.arange(100, 132)
    test = np.arange(200, 232)
    first = lc.subsets(fit, 8, 5, 7)
    again = lc.subsets(fit, 8, 5, 7)
    assert len(first) == 5
    for left, right in zip(first, again):
        np.testing.assert_array_equal(left, right)
    fit_set, test_set = set(fit.tolist()), set(test.tolist())
    for subset in first:
        assert len(subset) == 8
        assert len(set(subset.tolist())) == 8, "sampled without replacement"
        assert set(subset.tolist()) <= fit_set
        assert not set(subset.tolist()) & test_set, "calibration must never see held-out cases"


def test_a_different_salt_gives_different_draws():
    fit = np.arange(100, 132)
    assert not any(np.array_equal(a, b) for a, b in zip(lc.subsets(fit, 8, 3, 7), lc.subsets(fit, 8, 3, 8)))


def test_sizes_and_repeats_are_the_ones_the_note_reports():
    assert lc.SIZES == (4, 8, 16, 24, 32), "the note quotes these sizes"
    assert lc.REPEATS >= 5, "too few subsets would make the curve noise rather than a trend"
