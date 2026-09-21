"""Contract tests for the v1.8 expanded cohort."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jevbench.expanded_adapters import _choice_request, _normalise_probabilities
from jevbench.expanded_registry import DATASETS, EXPANDED_MODELS, EXPANDED_MODEL_ORDER, LEADERBOARDS, SIZES


def test_expanded_registry_has_three_english_and_one_multilingual_checkpoint():
    assert set(EXPANDED_MODEL_ORDER) == {"decider_2b", "kev_08b", "laya_typed", "laya_multilingual"}
    assert EXPANDED_MODELS["laya_multilingual"]["repo"] == "convaiinnovations/laya-multilingual"
    assert EXPANDED_MODELS["laya_multilingual"]["revision"] == "052592a15d198d9ad47da779604259b10b47b7aa"
    assert EXPANDED_MODELS["laya_multilingual"]["source_revision"] == "42626c348753fbb17572a813127df2278a1ec527"


def test_expanded_fixture_contract_has_200_test_cases_and_declared_leaderboards():
    assert SIZES == {"fit": 128, "dev": 64, "test": 200}
    assert set(DATASETS) == {"boolq", "ocnli", "clinc", "tmmluplus"}
    assert LEADERBOARDS == {"english": ("boolq", "clinc"), "ood_chinese": ("ocnli", "tmmluplus")}
    protocol = json.loads(Path("expanded_protocol.json").read_text())
    assert protocol["sizes_per_dataset"] == {"fit": 128, "dev": 64, "test": 200}
    assert protocol["leaderboards"]["ood_chinese"] == ["ocnli", "tmmluplus"]
    assert protocol["inference"]["primary_cases_per_model"] == 1568
    assert protocol["inference"]["total_forwards_per_model"] == 1604


def test_expanded_protocol_models_match_registry():
    protocol = json.loads(Path("expanded_protocol.json").read_text())
    for name in EXPANDED_MODEL_ORDER:
        assert protocol["models"][name]["repo"] == EXPANDED_MODELS[name]["repo"]
        assert protocol["models"][name]["revision"] == EXPANDED_MODELS[name]["revision"]


def test_expanded_choice_request_preserves_ids_and_text():
    request = {"question": "Which team?", "options": [{"id": "billing", "text": "Payments"}, {"id": "access", "text": "Login"}]}
    assert _choice_request(request) == {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "Payments", "access": "Login"},
    }


@pytest.mark.parametrize("values", [[0.2, 0.8], [2.0, 8.0], [1.0, 1.0, 2.0]])
def test_expanded_probability_normalisation_returns_simplex(values):
    result = _normalise_probabilities(values)
    assert np.isfinite(result).all()
    assert (np.asarray(result) >= 0).all()
    assert sum(result) == pytest.approx(1.0)
