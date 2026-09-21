"""Fail-closed analysis for the v1.7 candidate native-head cohort."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import sha
from .extended_registry import EXTENDED_MODELS, EXTENDED_MODEL_ORDER
from .fixtures import digest, read_jsonl, write
from .run import validate_score


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence <= high if high == 1.0 else confidence < high)
        if mask.any():
            value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return value


def _artifact_roots(incoming: Path, model: str) -> list[Path]:
    return sorted(path.parent for path in incoming.rglob("SHA256.json") if path.parent.name == model)


def _primary_row(model: str, fixture_row: dict, gold: dict, record: dict, provenance: dict) -> dict:
    score = record["score"]
    answer = validate_score(score, fixture_row["request"])
    probabilities = dict(zip(score["option_ids"], score["probabilities"]))
    label = gold["label"]
    p_label = float(probabilities.get(label, 0.0))
    return {
        "model": model,
        "dataset": fixture_row["dataset"],
        "role": fixture_row["role"],
        "id": fixture_row["request"]["id"],
        "group": fixture_row["group"],
        "correct": int(answer == label),
        "prediction": answer,
        "label": label,
        "latency_ms": float(score["total_ms"]),
        "nll": -math.log(max(p_label, 1e-12)),
        "brier": float(sum((float(value) - (1.0 if key == label else 0.0)) ** 2 for key, value in probabilities.items())),
        "confidence": max(float(value) for value in probabilities.values()),
        "tokens_sum": int(score["input_tokens_sum"]),
        "max_path_tokens": int(score["max_path_tokens"]),
        "candidate_paths": int(score["candidate_paths"]),
        "cpu_model": provenance["cpu_model"],
    }


def run(incoming: Path, fixture: Path, evaluation: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fixture_manifest = json.loads((fixture / "manifest.json").read_text())
    requests = read_jsonl(fixture / "requests.jsonl")
    gold = read_jsonl(evaluation / "gold.jsonl")
    evaluation_manifest = json.loads((evaluation / "manifest.json").read_text())
    if sha(fixture / "requests.jsonl") != fixture_manifest["requests_sha256"]:
        raise ValueError("fixture digest mismatch")
    if sha(evaluation / "gold.jsonl") != evaluation_manifest["gold_sha256"]:
        raise ValueError("gold digest mismatch")
    qmap = {row["request"]["id"]: row for row in requests}
    gmap = {row["id"]: row for row in gold}
    if len(qmap) != 320 or set(qmap) != set(gmap):
        raise ValueError("fixture/gold coverage mismatch")

    rows: list[dict] = []
    provenance: list[dict] = []
    probe_counts = {"reverse": 0, "repeat": 0}
    for model in EXTENDED_MODEL_ORDER:
        roots = _artifact_roots(incoming, model)
        if len(roots) != 12:
            raise ValueError(f"expected 12 {model} shard artifacts, found {len(roots)}")
        primary_ids: set[str] = set()
        reverse_ids: set[str] = set()
        repeat_ids: set[str] = set()
        shards: set[int] = set()
        for root in roots:
            checks = json.loads((root / "SHA256.json").read_text())
            for name, expected in checks.items():
                path = root / name
                if path.name != name or sha(path) != expected:
                    raise ValueError(f"bad inference artifact digest: {path}")
            prov = json.loads((root / "provenance.json").read_text())
            provenance.append(prov)
            if prov["shard"] in shards:
                raise ValueError(f"duplicate {model} shard artifact: {prov['shard']}")
            shards.add(prov["shard"])
            if prov["model"] != model or prov["adapter"]["id"] != EXTENDED_MODELS[model]["repo"]:
                raise ValueError("model provenance mismatch")
            if prov["adapter"]["revision"] != EXTENDED_MODELS[model]["revision"]:
                raise ValueError("checkpoint revision mismatch")
            if prov["new_api_calls"] != 0 or prov["new_training_steps"] != 0:
                raise ValueError("unexpected external calls or training steps")
            records = read_jsonl(root / "predictions.jsonl")
            if len(records) != prov["forwards"]:
                raise ValueError("forward count does not match provenance")
            for record in records:
                score = record["score"]
                qid = score["id"]
                if qid not in qmap:
                    raise ValueError("unknown request id in inference artifact")
                original = qmap[qid]
                if original["shard"] != prov["shard"] or original["role"] != record["role"] or original["dataset"] != record["dataset"]:
                    raise ValueError("shard/request metadata mismatch")
                request = json.loads(json.dumps(original["request"]))
                if record["kind"] == "reverse":
                    if qid in reverse_ids:
                        raise ValueError("duplicate reverse probe")
                    reverse_ids.add(qid)
                    request["options"].reverse()
                    probe_counts["reverse"] += 1
                elif record["kind"] == "repeat":
                    if qid in repeat_ids:
                        raise ValueError("duplicate repeat probe")
                    repeat_ids.add(qid)
                    probe_counts["repeat"] += 1
                elif record["kind"] != "primary":
                    raise ValueError("unknown inference record kind")
                if digest(request) != record["request_sha256"]:
                    raise ValueError("record request digest mismatch")
                validate_score(score, request)
                if record["kind"] == "primary":
                    if qid in primary_ids:
                        raise ValueError("duplicate primary inference")
                    primary_ids.add(qid)
                    rows.append(_primary_row(model, original, gmap[qid], record, prov))
        if primary_ids != set(qmap):
            raise ValueError(f"primary coverage incomplete for {model}")
        expected_reverse = {
            row["request"]["id"]
            for dataset in ("boolq", "ocnli", "clinc", "tmmluplus")
            for row in [r for r in requests if r["dataset"] == dataset and r["role"] == "test"][:4]
        }
        expected_repeat = {
            row["request"]["id"]
            for dataset in ("boolq", "ocnli", "clinc", "tmmluplus")
            for row in [r for r in requests if r["dataset"] == dataset and r["role"] == "test"][:2]
        }
        if shards != set(range(12)) or reverse_ids != expected_reverse or repeat_ids != expected_repeat:
            raise ValueError(f"probe/shard coverage mismatch for {model}")
        if sum(p["primary"] for p in provenance if p["model"] == model) != 320:
            raise ValueError(f"primary provenance count mismatch for {model}")
        if sum(p["forwards"] for p in provenance if p["model"] == model) != 344:
            raise ValueError(f"evidence forward count mismatch for {model}")
        if sum(p["warmups"] for p in provenance if p["model"] == model) != 12:
            raise ValueError(f"warmup count mismatch for {model}")

    frame = pd.DataFrame(rows)
    test = frame[frame["role"] == "test"]
    summary: list[dict] = []
    for (model, dataset), group in test.groupby(["model", "dataset"], sort=True):
        summary.append({
            "model": model,
            "dataset": dataset,
            "n": len(group),
            "correct": int(group["correct"].sum()),
            "accuracy": float(group["correct"].mean()),
            "brier": float(group["brier"].mean()),
            "nll": float(group["nll"].mean()),
            "ece": ece(group["confidence"].to_numpy(), group["correct"].to_numpy()),
            "p50_ms": float(group["latency_ms"].median()),
            "p95_ms": float(group["latency_ms"].quantile(0.95)),
            "tokens_sum_mean": float(group["tokens_sum"].mean()),
            "max_path_tokens": int(group["max_path_tokens"].max()),
            "candidate_paths": int(group["candidate_paths"].iloc[0]),
            "paired_hosts": int(group["cpu_model"].nunique()),
        })
    frame.to_csv(out / "primary_predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "native_summary.csv", index=False)
    if len(frame) != 320 * len(EXTENDED_MODEL_ORDER) or len(test) != 128 * len(EXTENDED_MODEL_ORDER):
        raise ValueError("primary/test row counts are incomplete")
    if probe_counts != {"reverse": 16 * len(EXTENDED_MODEL_ORDER), "repeat": 8 * len(EXTENDED_MODEL_ORDER)}:
        raise ValueError(f"probe coverage mismatch: {probe_counts}")
    if any(len(group) != 32 for _, group in test.groupby(["model", "dataset"])):
        raise ValueError("test cohort is not 32 rows per model and dataset")
    write(out / "verification.json", {
        "version": "1.7.0-candidate",
        "all_gates_passed": True,
        "models": EXTENDED_MODELS,
        "native_predictions": len(frame),
        "test_predictions": len(test),
        "primary_forwards": 320 * len(EXTENDED_MODEL_ORDER),
        "evidence_forwards": 344 * len(EXTENDED_MODEL_ORDER),
        "probe_forwards": 24 * len(EXTENDED_MODEL_ORDER),
        "warmup_forwards": 12 * len(EXTENDED_MODEL_ORDER),
        "total_forwards": 356 * len(EXTENDED_MODEL_ORDER),
        "native_loading_provenance": provenance,
        "same_semantic_request_hashes": True,
        "gold_isolated_from_inference": True,
        "new_api_calls": 0,
        "model_training_steps": 0,
        "scope": "v1.7 candidate native typed-choice benchmark; separate from v1.6 pilot",
    })
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path, default=Path("incoming"))
    parser.add_argument("--fixture", type=Path, default=Path("fixture"))
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation"))
    parser.add_argument("--out", type=Path, default=Path("results/extended"))
    args = parser.parse_args()
    run(args.incoming, args.fixture, args.evaluation, args.out)
