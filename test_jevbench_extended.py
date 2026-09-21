"""Contract tests for the v1.7 candidate registry and adapters.

These tests do not download weights.  The hosted workflow performs the real
CPU forwards; local CI still checks the immutable metadata and probability
normalisation that protect those measurements.
"""
from __future__ import annotations

import numpy as np
import pytest

from jevbench.extended_adapters import _choice_request, _normalise_probabilities
from jevbench.extended_registry import EXTENDED_MODELS, EXTENDED_MODEL_ORDER, RLCD
from jevbench.rlcd_smoke import exact_match


def test_extended_registry_has_requested_native_models_and_immutable_revisions():
    assert set(EXTENDED_MODEL_ORDER) == {"decider_2b", "kev_08b", "laya_typed"}
    for name in EXTENDED_MODEL_ORDER:
        spec = EXTENDED_MODELS[name]
        assert len(spec["revision"]) == 40
        assert spec["repo"]
        assert spec["kind"] in {"decider", "kev", "laya"}
    assert EXTENDED_MODELS["kev_08b"]["base_revision"] == "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    assert EXTENDED_MODELS["laya_typed"]["source_revision"] == "42626c348753fbb17572a813127df2278a1ec527"


def test_rlcd_is_explicitly_separate_from_native_heads():
    assert len(RLCD["revision"]) == 40
    assert "no native decision head" in RLCD["source_kind"]
    assert RLCD["base_revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def test_choice_request_preserves_candidate_ids_and_text():
    request = {
        "question": "Which team?",
        "options": [{"id": "billing", "text": "Payments"}, {"id": "access", "text": "Login"}],
    }
    assert _choice_request(request) == {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "Payments", "access": "Login"},
    }


@pytest.mark.parametrize("values", [[0.2, 0.8], [2.0, 8.0], [1.0, 1.0, 2.0]])
def test_probability_normalisation_returns_a_simplex(values):
    result = _normalise_probabilities(values)
    assert np.isfinite(result).all()
    assert (np.asarray(result) >= 0).all()
    assert sum(result) == pytest.approx(1.0)


def test_probability_normalisation_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        _normalise_probabilities([0.2, float("nan")])


@pytest.mark.parametrize(
    "parsed",
    [
        {"department": "billing"},
        {"department": {"value": "billing", "prob": 0.9}},
    ],
)
def test_rlcd_exact_match_accepts_upstream_value_shapes(parsed):
    assert exact_match(parsed, {"department": "billing"})
