"""Tests for the independently implemented v1.6 evidence reader.

All inputs here are synthetic: no shard artifact, checkpoint or network call is involved. The
tests pin the reader arithmetic and, just as important, that it still refuses tampered evidence.
"""
import inspect
import math
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


def test_reader_does_not_import_the_analysis_package():
    source = inspect.getsource(verify_jevbench)
    assert "import jevbench" not in source and "from jevbench" not in source, (
        "the independent reader must not reuse the analysis package it checks")
