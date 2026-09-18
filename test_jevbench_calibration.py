"""Tests for the calibration variant experiment.

These pin the estimator invariants the comparison depends on: that each variant still produces a
joint distribution, that the uniform-label variant really removes the task-specific label
marginal, that shrinkage really moves the accuracy estimate toward its target, and that the
bagged world stays a distribution. Synthetic probabilities only; no artifact or model is read.
"""
import numpy as np
import pytest
import jevbench_calibration as jc


def synthetic(cases=16, models=3, options=3, seed=7, accuracy=0.7):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, options, cases)
    probs = rng.random((cases, models, options))
    probs = probs / probs.sum(-1, keepdims=True)
    for model in range(models):
        for case in range(cases):
            if rng.random() < accuracy:
                probs[case, model, labels[case]] = 0.9
                probs[case, model] /= probs[case, model].sum()
    return probs, labels


def test_joint_is_a_probability_distribution():
    probs, labels = synthetic()
    joint, k, yk, accuracies = jc.build_joint(probs, labels, bins=2, strength=32.0)
    assert joint.shape == (yk, k * 2, k * 2, k * 2)
    assert joint.sum() == pytest.approx(1.0)
    assert (joint >= 0).all()
    assert len(accuracies) == 3 and all(0 < a < 1 for a in accuracies)


def test_infinite_strength_keeps_the_channels_conditionally_independent():
    """The estimator factors the three observation channels given the truth. At infinite
    strength nothing else is added, so that factorisation must hold exactly; at finite strength
    the joint case counts break it, which is the whole point of the data term."""
    probs, labels = synthetic(cases=12)
    infinite, _, _, _ = jc.build_joint(probs, labels, bins=2, strength=float("inf"))
    conditional = infinite / infinite.sum(axis=(1, 2, 3), keepdims=True)
    marginal_0 = conditional.sum(axis=(2, 3), keepdims=True)
    marginal_1 = conditional.sum(axis=(1, 3), keepdims=True)
    marginal_2 = conditional.sum(axis=(1, 2), keepdims=True)
    np.testing.assert_allclose(conditional, marginal_0 * marginal_1 * marginal_2, atol=1e-12)
    finite, _, _, _ = jc.build_joint(probs, labels, bins=2, strength=8.0)
    assert not np.allclose(finite, infinite), "the data term must change the estimate"


def test_uniform_label_marginal_is_actually_uniform_in_the_prior():
    probs, labels = synthetic()
    empirical, k, yk, _ = jc.build_joint(probs, labels, bins=2, strength=float("inf"))
    uniform, _, _, _ = jc.build_joint(probs, labels, bins=2, strength=float("inf"), label_marginal="uniform")
    empirical_marginal = empirical.sum(axis=(1, 2, 3))
    uniform_marginal = uniform.sum(axis=(1, 2, 3))
    np.testing.assert_allclose(uniform_marginal, 1.0 / yk, atol=1e-12)
    assert not np.allclose(empirical_marginal, uniform_marginal), "the fitted marginal stays task specific"

def test_shrinkage_moves_accuracy_toward_the_pooled_target():
    probs, labels = synthetic(accuracy=0.2)
    plain, _, _, raw = jc.build_joint(probs, labels, bins=2, strength=32.0)
    shrunk, _, _, pulled = jc.build_joint(probs, labels, bins=2, strength=32.0, accuracy_shrink=True,
                                          pooled_accuracy=0.9)
    assert all(pulled[m] > raw[m] for m in range(3)), "estimates must move up toward 0.9"
    assert all(pulled[m] < 0.9 for m in range(3)), "and must not overshoot the target"
    per_model = np.array([0.9, 0.9, 0.9])
    vector, _, _, pulled_vector = jc.build_joint(probs, labels, bins=2, strength=32.0, accuracy_shrink=True,
                                                 pooled_accuracy=per_model)
    np.testing.assert_allclose(pulled, pulled_vector, atol=1e-12)


def test_bagged_world_stays_a_distribution():
    probs, labels = synthetic(cases=8)
    joint, k, yk = jc.bagged_joint(probs, labels, 2, 32.0, False, 8, 160918)
    assert joint.shape == (yk, k * 2, k * 2, k * 2)
    assert joint.sum() == pytest.approx(1.0)


def test_compiler_is_borrowed_from_the_analysis_not_reimplemented():
    """The comparison must isolate estimation, so planning stays the analysis compiler."""
    source = __import__("inspect").getsource(jc)
    assert "object.__new__(JointWorld)" in source
    assert "def compile" not in source, "a second compiler would confound the comparison"
