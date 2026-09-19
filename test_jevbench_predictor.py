"""Tests for the calibrated correctness predictor and its decision rule.

The rule is the part worth pinning: with a wrong answer costing 1 and withholding costing 0.25,
returning the panel answer is worthwhile exactly when its calibrated probability clears 0.75. If
that threshold drifts from the declared loss, every decision number in the note becomes wrong.
"""
import numpy as np
import pandas as pd
import pytest
import jevbench_predictor as jp


def test_answer_threshold_follows_the_declared_loss():
    assert jp.DEFER == 0.25
    assert jp.ANSWER_THRESHOLD == pytest.approx(1.0 - jp.DEFER)


def test_decision_loss_answers_only_above_the_threshold():
    probability = np.array([0.9, 0.8, 0.75, 0.6])
    correct = np.array([1, 0, 1, 1])
    losses = jp.decision_loss(probability, correct)
    # strictly above 0.75 answers: the first two; 0.75 itself withholds.
    np.testing.assert_allclose(losses, [0.0, 1.0, jp.DEFER, jp.DEFER])


def test_decision_loss_is_never_below_a_single_answer_or_a_single_deferral():
    probability = np.linspace(0, 1, 11)
    correct = np.ones(11, int)
    losses = jp.decision_loss(probability, correct)
    answered = probability > jp.ANSWER_THRESHOLD
    assert losses[answered].max() <= 1.0 and losses[~answered].max() == pytest.approx(jp.DEFER)
    assert losses[~answered].max() < 1.0, "withholding is cheaper than a wrong answer"


def test_constant_rate_is_beta_smoothed_and_never_certain():
    outer = pd.DataFrame({"correct": [1, 1, 1, 1]})
    predict = jp.constant_rate(outer)
    value = predict(pd.DataFrame({"anything": range(3)}))
    assert value.shape == (3,)
    assert 0.0 < value[0] < 1.0, "a single-class slice must not become certainty"
    assert value[0] == pytest.approx((4 + 0.5) / 5)


def test_calibration_slice_keeps_the_scorer_and_calibrator_apart():
    fit = pd.DataFrame({"correct": [0, 1] * 16, "value": np.arange(32, dtype=float)})
    inner, outer = jp.calibration_slice(fit, split=0.5, seed=7)
    assert len(inner) == 16 and len(outer) == 16
    assert not set(inner.index) & set(outer.index), "the calibrator must not see the scorer cases"
    assert set(inner.index) | set(outer.index) == set(fit.index)


def test_agreement_rate_falls_back_to_the_base_rate_for_unseen_support():
    fit = pd.DataFrame({"support": [3, 3, 2], "correct": [1, 1, 0]})
    predict = jp.agreement_rate(fit)
    values = predict(pd.DataFrame({"support": [3, 2, 1]}))
    assert values[0] > values[1] > 0
    base = (fit["correct"].sum() + 1.0) / (len(fit) + 2.0)
    assert values[2] == pytest.approx(base)
