"""Tests for the independently implemented v1.6 evidence reader.

All inputs here are synthetic: no shard artifact, checkpoint or network call is involved. The
tests pin the reader arithmetic and, just as important, that it still refuses tampered evidence.
"""
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import verify_jevbench
from verify_jevbench import check_contracts, digest, native_rows, percentile, summarise, wilson


def synthetic(expected_primary=1):
    request = {"id": "boolq:abc", "state": "passage", "question": "is it true",
               "options": [{"id": "no", "text": "No"}, {"id": "yes", "text": "Yes"}]}
    row = {"dataset": "boolq", "role": "test", "group": "g", "shard": 0, "request": request,
           "sha256": digest(request)}
    gold = [{"id": "boolq:abc", "dataset": "boolq", "role": "test", "label": "yes",
             "option_ids": ["no", "yes"], "gold_index": 1}]
    manifest = {"cases": expected_primary, "requests_sha256": "x"}
    manifest_eval = {"gold_sha256": "y"}
    score = {"id": "boolq:abc", "option_ids": ["no", "yes"], "probabilities": [0.2, 0.8],
             "total_ms": 10.0, "backbone_calls": 1, "decode_steps": 0, "input_tokens_sum": 12}
    records = {(model, "primary"): [{"dataset": "boolq", "role": "test", "kind": "primary",
                                      "request_sha256": row["sha256"], "score": dict(score)}]
               for model in verify_jevbench.MODELS}
    return row, gold, manifest, manifest_eval, records


def test_wilson_brackets_the_point_estimate_and_is_wide_at_small_n():
    low, high = wilson(1, 2)
    assert low < 0.5 < high
    assert high - low > 0.5
    tight_low, tight_high = wilson(500, 1000)
    assert tight_high - tight_low < 0.1
    assert math.isnan(wilson(0, 0)[0])


def test_percentile_uses_interpolation_like_numpy():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.5) == 2.5
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 4.0
    assert percentile([7.0], 0.95) == 7.0


def test_native_rows_uses_the_recorded_option_order():
    row, gold, _, _, records = synthetic()
    rows = native_rows([row], gold, records)
    assert len(rows) == len(verify_jevbench.MODELS)
    assert all(r["correct"] == 1 for r in rows), "argmax points at the yes option"
    flipped = {(model, "primary"): [{"dataset": "boolq", "role": "test", "kind": "primary",
                                     "request_sha256": row["sha256"],
                                     "score": {"id": "boolq:abc", "option_ids": ["yes", "no"],
                                               "probabilities": [0.8, 0.2], "total_ms": 1.0,
                                               "backbone_calls": 1, "decode_steps": 0, "input_tokens_sum": 1}}]
               for model in verify_jevbench.MODELS}
    assert all(r["correct"] == 1 for r in native_rows([row], gold, flipped)), "order must not matter"


def test_summarise_reports_accuracy_and_latency_per_dataset_and_overall():
    rows = [{"model": "nanojev", "dataset": "boolq", "role": "test", "id": f"i{i}",
             "correct": i % 2, "latency_ms": 10.0 * (i + 1), "tokens_sum": 5} for i in range(32)]
    summary = summarise(rows)
    per_dataset = [r for r in summary if r["dataset"] == "boolq"][0]
    overall = [r for r in summary if r["dataset"] == "ALL"][0]
    assert per_dataset["n"] == 32 and per_dataset["correct"] == 16
    assert per_dataset["accuracy"] == 0.5 and overall["n"] == 32
    assert per_dataset["p50_ms"] == 165.0 and per_dataset["p95_ms"] > per_dataset["p50_ms"]


def test_clean_synthetic_evidence_passes_every_contract():
    row, gold, manifest, manifest_eval, records = synthetic()
    problems = []
    check_contracts([row], gold, manifest, manifest_eval, records, problems,
                    expected={"primary": 1, "reverse": 0, "repeat": 0})
    assert problems == []


def test_reader_refuses_tampered_evidence():
    cases = {}
    cases["digest"] = lambda r, m: r[0].update(sha256="deadbeef")
    cases["duplicate"] = lambda r, m: m[(verify_jevbench.MODELS[0], "primary")].append(
        dict(m[(verify_jevbench.MODELS[0], "primary")][0]))
    cases["simplex"] = lambda r, m: m[(verify_jevbench.MODELS[0], "primary")][0]["score"].update(
        probabilities=[0.5, 0.6])
    cases["option_order"] = lambda r, m: m[(verify_jevbench.MODELS[0], "primary")][0]["score"].update(
        option_ids=["yes", "no"])
    cases["decode"] = lambda r, m: m[(verify_jevbench.MODELS[0], "primary")][0]["score"].update(
        decode_steps=7)
    cases["latency"] = lambda r, m: m[(verify_jevbench.MODELS[0], "primary")][0]["score"].update(
        total_ms=0.0)
    for name, tamper in cases.items():
        row, gold, manifest, manifest_eval, records = synthetic()
        tamper([row], records)
        problems = []
        check_contracts([row], gold, manifest, manifest_eval, records, problems,
                        expected={"primary": 1, "reverse": 0, "repeat": 0})
        assert problems, f"tampered evidence was accepted: {name}"


def test_reader_refuses_a_missing_primary_record():
    row, gold, manifest, manifest_eval, records = synthetic()
    records[(verify_jevbench.MODELS[0], "primary")] = []
    problems = []
    check_contracts([row], gold, manifest, manifest_eval, records, problems,
                    expected={"primary": 1, "reverse": 0, "repeat": 0})
    assert any("without a primary record" in p for p in problems)


@pytest.mark.parametrize("probabilities", [
    [math.nan, 0.5], [0.5, math.nan], [math.nan, math.nan],
    [math.inf, 0], [0, math.inf], [-math.inf, 1], [1, -math.inf],
    [math.inf, -math.inf], [math.inf, math.nan], [math.nan, -math.inf],
    [1e308, 1e308], [10**400, 0], [10**400, 0.5],
    [None, 1], ["0.2", 0.8], [False, True], [[0.2], 0.8],
    {"0": 0, "1": 1}, None, "01", 1, [], [1],
    [0.5, 0.6], [-1e-20, 1], [0.5, 0.500002],
])
def test_reader_refuses_invalid_probability_vectors(probabilities):
    row, gold, manifest, manifest_eval, records = synthetic()
    model = verify_jevbench.MODELS[0]
    records[(model, "primary")][0]["score"]["probabilities"] = probabilities
    problems = []
    check_contracts([row], gold, manifest, manifest_eval, records, problems,
                    expected={"primary": 1, "reverse": 0, "repeat": 0})
    assert len(problems) == 1
    assert model in problems[0] and row["request"]["id"] in problems[0]
    assert "probabilit" in problems[0]


@pytest.mark.parametrize("probabilities", [
    [0, 1], [1, 0], [-0.0, 1.0], [0.2, 0.8], (0.2, 0.8),
    [0.5, 0.5000005], [1.0000005, 0],
])
def test_reader_preserves_valid_probability_boundaries(probabilities):
    row, gold, manifest, manifest_eval, records = synthetic()
    records[(verify_jevbench.MODELS[0], "primary")][0]["score"]["probabilities"] = probabilities
    problems = []
    check_contracts([row], gold, manifest, manifest_eval, records, problems,
                    expected={"primary": 1, "reverse": 0, "repeat": 0})
    assert problems == []


@pytest.fixture
def cli_evidence(tmp_path):
    """Small local evidence matching the CLI's unchanged coverage requirements."""
    row, gold, _, _, records = synthetic()
    requests, labels, primary = [], [], []
    for index in range(verify_jevbench.EXPECTED_PRIMARY):
        request = dict(row["request"], id=f"boolq:synthetic-{index}")
        request_row = dict(row, request=request, sha256=digest(request))
        requests.append(request_row)
        labels.append(dict(gold[0], id=request["id"]))
        record = records[(verify_jevbench.MODELS[0], "primary")][0]
        primary.append(dict(record, request_sha256=request_row["sha256"],
                            score=dict(record["score"], id=request["id"])))

    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(item) + "\n" for item in rows))

    fixture, evaluation, incoming = (tmp_path / name for name in ("fixture", "evaluation", "incoming"))
    fixture.mkdir()
    evaluation.mkdir()
    write_jsonl(fixture / "requests.jsonl", requests)
    write_jsonl(evaluation / "gold.jsonl", labels)
    (fixture / "manifest.json").write_text(json.dumps({
        "cases": len(requests), "requests_sha256": verify_jevbench.sha256_file(fixture / "requests.jsonl")}))
    (evaluation / "manifest.json").write_text(json.dumps({
        "gold_sha256": verify_jevbench.sha256_file(evaluation / "gold.jsonl")}))
    probes = [dict(primary[0], kind=kind)
              for kind, count in (("reverse", verify_jevbench.EXPECTED_REVERSE),
                                  ("repeat", verify_jevbench.EXPECTED_REPEAT))
              for _ in range(count)]
    for shard in range(12):
        for model in verify_jevbench.MODELS:
            path = incoming / f"jev16-shard-{shard:02d}" / "results" / model / "predictions.jsonl"
            path.parent.mkdir(parents=True)
            write_jsonl(path, primary + probes if shard == 0 else [])
    return incoming, fixture, evaluation, tmp_path / "out"


@pytest.mark.parametrize("raw_probabilities,valid", [
    ("[0, 1]", True), ("[-0.0, 1.0]", True), ("[1.0000005, 0]", True),
    ("[NaN, 0.5]", False), ("[0.5, NaN]", False),
    ("[Infinity, 0]", False), ("[-Infinity, 1]", False),
    ("[Infinity, NaN]", False), ("[NaN, -Infinity]", False),
    ("[1e309, 0]", False), ("[0, -1e309]", False),
    ("null", False), ('["0.2", 0.8]', False), ("[false, true]", False),
    ('{"0": 0, "1": 1}', False), ("[]", False),
])
def test_cli_rejects_invalid_probabilities_without_deriving_summary(cli_evidence, raw_probabilities, valid):
    incoming, fixture, evaluation, out = cli_evidence
    path = incoming / "jev16-shard-00" / "results" / verify_jevbench.MODELS[0] / "predictions.jsonl"
    original = path.read_text()
    path.write_text(original.replace('"probabilities": [0.2, 0.8]',
                                     f'"probabilities": {raw_probabilities}', 1))
    completed = subprocess.run([
        sys.executable, str(Path(verify_jevbench.__file__).resolve()),
        "--incoming", str(incoming), "--fixture", str(fixture),
        "--evaluation", str(evaluation), "--out", str(out),
    ], capture_output=True, text=True, check=False)
    assert completed.returncode == (0 if valid else 1), completed.stdout + completed.stderr
    assert "Traceback" not in completed.stderr
    report = json.loads((out / "independent_native.json").read_text())
    assert report["shards"] == 12 and report["primary_records"] == 960
    assert report["probe_records"] == 72
    if valid:
        assert report["problems"] == []
        assert report["summary"]
        assert "independent re-derivation found no contract problem" in completed.stdout
    else:
        assert len(report["problems"]) == 1
        assert "nanojev" in report["problems"][0] and "boolq:synthetic-0" in report["problems"][0]
        assert report["summary"] == []
        assert "independent re-derivation found no contract problem" not in completed.stdout


def test_reader_does_not_import_the_analysis_package():
    source = inspect.getsource(verify_jevbench)
    assert "import jevbench" not in source and "from jevbench" not in source, (
        "the independent reader must not reuse the analysis package it checks")
