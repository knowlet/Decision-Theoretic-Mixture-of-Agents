"""Contract tests for the v1.8 expanded cohort."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jevbench import expanded_analyze
from jevbench.expanded_adapters import _choice_request, _normalise_probabilities
from jevbench.expanded_adapters import sha
from jevbench.expanded_analyze import validate_contract
from jevbench.expanded_registry import DATASETS, EXPANDED_MODELS, EXPANDED_MODEL_ORDER, LEADERBOARDS, SIZES, json_contract
from jevbench.fixtures import digest, write


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


def _serialized_manifest() -> dict:
    """The manifest contract exactly as the fixture writes it: through JSON."""
    return json.loads(json.dumps({
        "datasets": DATASETS,
        "sizes_per_dataset": SIZES,
        "leaderboards": LEADERBOARDS,
    }))


def test_expanded_contract_accepts_the_json_form_the_fixture_writes():
    manifest = _serialized_manifest()
    contract = validate_contract(manifest)
    assert contract["leaderboards"] == manifest["leaderboards"]
    assert contract["sizes_per_dataset"] == manifest["sizes_per_dataset"]


def test_expanded_contract_rejects_a_drifted_leaderboard():
    manifest = _serialized_manifest()
    manifest["leaderboards"]["ood_chinese"] = ["ocnli"]
    with pytest.raises(ValueError, match="leaderboard"):
        validate_contract(manifest)


def test_expanded_contract_rejects_a_drifted_cohort_size():
    manifest = _serialized_manifest()
    manifest["sizes_per_dataset"]["test"] = 100
    with pytest.raises(ValueError, match="size"):
        validate_contract(manifest)


def _synthetic_record(row: dict, request: dict, kind: str) -> dict:
    return {
        "dataset": row["dataset"],
        "role": row["role"],
        "group": row["group"],
        "kind": kind,
        "request_sha256": digest(request),
        "score": {
            "id": request["id"],
            "option_ids": [option["id"] for option in request["options"]],
            "probabilities": [0.4, 0.6],
            "raw_logits": [],
            "total_ms": 1.0,
            "preprocess_ms": 0.0,
            "input_tokens_sum": 8,
            "max_path_tokens": 8,
            "candidate_paths": len(request["options"]),
            "backbone_calls": 1,
            "decode_steps": 0,
        },
    }


def test_expanded_analyzer_accepts_a_complete_synthetic_cohort(tmp_path):
    fixture = tmp_path / "fixture"
    evaluation = tmp_path / "evaluation"
    incoming = tmp_path / "incoming"
    out = tmp_path / "out"
    for path in (fixture, evaluation, incoming):
        path.mkdir(parents=True)

    requests = []
    gold = []
    for dataset in DATASETS:
        for role, count in SIZES.items():
            for index in range(count):
                qid = f"{dataset}:{role}:{index:03d}"
                request = {
                    "id": qid,
                    "state": f"state {qid}",
                    "question": "Which option applies?",
                    "options": [{"id": "no", "text": "No"}, {"id": "yes", "text": "Yes"}],
                }
                requests.append({
                    "request": request,
                    "sha256": digest(request),
                    "dataset": dataset,
                    "role": role,
                    "group": qid,
                    "shard": len(requests) % 12,
                })
                gold.append({
                    "id": qid,
                    "dataset": dataset,
                    "role": role,
                    "group": qid,
                    "source_id": qid,
                    "source_split": role,
                    "subject": "",
                    "label": "yes",
                    "gold_index": 1,
                    "gold_in_candidates": True,
                    "option_ids": ["no", "yes"],
                    "retrieval_ms": 0.0,
                })
    (fixture / "requests.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests))
    (evaluation / "gold.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gold))
    manifest = {
        "version": "1.8.0",
        "datasets": list(DATASETS),
        "sizes_per_dataset": dict(SIZES),
        "leaderboards": json_contract()["leaderboards"],
        "cases": len(requests),
        "requests_sha256": sha(fixture / "requests.jsonl"),
        "gold_sha256": sha(evaluation / "gold.jsonl"),
        "protocol_sha256": sha(Path("expanded_protocol.json")),
    }
    write(fixture / "manifest.json", manifest)
    write(evaluation / "manifest.json", manifest)

    reverse_ids = set()
    repeat_ids = set()
    for dataset in DATASETS:
        test = [row for row in requests if row["dataset"] == dataset and row["role"] == "test"]
        reverse_ids.update(row["request"]["id"] for row in test[:4])
        repeat_ids.update(row["request"]["id"] for row in test[:2])

    for model in EXPANDED_MODEL_ORDER:
        for shard in range(12):
            assigned = [row for row in requests if row["shard"] == shard]
            records = []
            for row in assigned:
                request = row["request"]
                records.append(_synthetic_record(row, request, "primary"))
                if request["id"] in reverse_ids:
                    reversed_request = dict(request, options=list(reversed(request["options"])))
                    records.append(_synthetic_record(row, reversed_request, "reverse"))
                if request["id"] in repeat_ids:
                    records.append(_synthetic_record(row, request, "repeat"))
            root = incoming / f"jev18-{model}-{shard}-synthetic" / "results" / model
            root.mkdir(parents=True)
            (root / "predictions.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
            write(root / "provenance.json", {
                "version": "1.8.0-expanded",
                "commit": None,
                "model": model,
                "shard": shard,
                "fixture_sha256": manifest["requests_sha256"],
                "protocol_sha256": manifest["protocol_sha256"],
                "adapter": {
                    "id": EXPANDED_MODELS[model]["repo"],
                    "revision": EXPANDED_MODELS[model]["revision"],
                },
                "adapter_file_sha256": sha(Path("jevbench/expanded_adapters.py")),
                "runner_sha256": sha(Path("jevbench/expanded_run.py")),
                "forwards": len(records),
                "evidence_forwards": len(records),
                "warmups": 1,
                "total_forwards": len(records) + 1,
                "primary": len(assigned),
                "load_and_download_ms": 1.0,
                "max_rss_kb": 1,
                "cpu_model": "synthetic",
                "platform": "synthetic",
                "run_url": None,
                "new_api_calls": 0,
                "new_training_steps": 0,
            })
            write(root / "SHA256.json", {
                name: sha(root / name) for name in ("predictions.jsonl", "provenance.json")
            })

    expanded_analyze.run(incoming, fixture, evaluation, out)

    verification = json.loads((out / "verification.json").read_text())
    assert verification["all_gates_passed"] is True
    assert verification["native_predictions"] == len(requests) * len(EXPANDED_MODEL_ORDER)
    assert verification["test_predictions"] == SIZES["test"] * len(DATASETS) * len(EXPANDED_MODEL_ORDER)

    leaderboards = pd.read_csv(out / "leaderboard_summary.csv")
    assert set(leaderboards["leaderboard"]) == {"english", "ood_chinese"}
    assert set(leaderboards["n"]) == {2 * SIZES["test"]}
    native = pd.read_csv(out / "native_summary.csv")
    assert set(native["n"]) == {SIZES["test"]}
