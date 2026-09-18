"""Tests for the competence-signal analysis.

The first draft of this script had two silent defects that the numbers revealed: inverted AUC ranks,
which scored a good ranker below chance, and a paired difference that was not a Brier difference.
These tests exist so that neither can come back unnoticed.
"""
import numpy as np
import pytest
import jevbench_competence as jc


def test_auc_is_one_for_a_perfect_ranker_and_zero_for_an_inverted_one():
    labels = np.array([0, 0, 1, 1])
    assert jc.binary_metrics([0.1, 0.2, 0.8, 0.9], labels)["auc"] == pytest.approx(1.0)
    assert jc.binary_metrics([0.9, 0.8, 0.2, 0.1], labels)["auc"] == pytest.approx(0.0)
    assert jc.binary_metrics([0.5] * 4, labels)["auc"] == pytest.approx(0.5)


def test_per_case_brier_matches_the_two_class_convention():
    probability = np.array([0.25, 0.75])
    truth = np.array([0.0, 1.0])
    per_case = jc.brier_per_case(probability, truth)
    np.testing.assert_allclose(per_case, [2 * 0.25 ** 2, 2 * 0.25 ** 2])
    from jevbench.analyze import metrics
    stacked = np.column_stack([1 - probability, probability])
    assert float(per_case.mean()) == pytest.approx(metrics(stacked, truth.astype(int))["brier"])


def test_case_features_report_the_plurality_the_agreement_and_the_confidence():
    probs = np.array([[0.6, 0.3, 0.1], [0.7, 0.2, 0.1], [0.5, 0.3, 0.2]])
    row = jc.case_features(probs)
    assert row["plurality"] == 0 and row["support"] == 3 and row["distinct"] == 1
    assert row["max_probability"] == pytest.approx(0.7)
    assert row["min_probability"] == pytest.approx(0.5)
    assert row["mean_probability"] == pytest.approx((0.6 + 0.7 + 0.5) / 3)
    mixed = jc.case_features(np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]))
    assert mixed["support"] == 2 and mixed["distinct"] == 2
    assert mixed["max_probability"] == pytest.approx(0.8)


def test_agreement_estimator_is_smoothed_and_falls_back_to_the_base_rate():
    import pandas as pd
    frame = pd.DataFrame({"agreement": ["a", "a", "b"], "correct": [1, 1, 0]})
    predict = jc.fit_agreement(frame, prior=0.5, weight=2.0)
    values = predict(pd.DataFrame({"agreement": ["a", "b", "unseen"]}))
    assert 0 < values[1] < values[0] < 1, "seen patterns keep their order, unseen falls back"
    base = (frame["correct"].sum() + 1.0) / (len(frame) + 2.0)
    assert values[2] == pytest.approx(base)
