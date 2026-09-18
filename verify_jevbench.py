"""Independently implemented reader for the v1.6 native-head evidence.

This module deliberately shares no code with jevbench.analyze: it re-reads the published
fixture, the separate gold artifact and the twelve shard prediction files, re-checks every
digest and coverage contract itself, and recomputes native accuracy and measured latency from
the raw records. Its purpose is to give the headline numbers a second implementation, in the
same spirit as the separately written Bellman reference solver used earlier in this project.

It computes no policies and makes no claim about them; a disagreement here means the two
readers disagree about what the checkpoints answered, which is worth knowing before any
policy result built on those answers is published.
"""
from __future__ import annotations
import argparse, hashlib, json, math, pathlib, sys

MODELS = ("nanojev", "decider", "system_one")
DATASETS = ("boolq", "ocnli", "clinc", "tmmluplus")
EXPECTED_PRIMARY = 320
EXPECTED_REVERSE = 16
EXPECTED_REPEAT = 8


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def read_jsonl(path: pathlib.Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(correct: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return float("nan"), float("nan")
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return centre - half, centre + half


def percentile(values, q: float) -> float:
    """Linear-interpolation quantile, the convention used by numpy and pandas defaults."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def collect_records(incoming: pathlib.Path):
    """Return {(model, kind): [records]} read from every shard directory, in shard order."""
    found = {}
    shards = sorted(path for path in incoming.glob("jev16-shard-*"))
    if len(shards) != 12:
        raise SystemExit(f"expected 12 shard directories under {incoming}, found {len(shards)}")
    for shard in shards:
        for model in MODELS:
            path = shard / "results" / model / "predictions.jsonl"
            if not path.exists():
                raise SystemExit(f"missing predictions for {model} in {shard.name}")
            for record in read_jsonl(path):
                found.setdefault((model, record["kind"]), []).append(record)
    return found, shards


def check_contracts(requests, gold, manifest, manifest_eval, records, problems, expected=None):
    """Fixture digests, coverage, and per-record probability contracts."""
    expected = expected or {"primary": EXPECTED_PRIMARY, "reverse": EXPECTED_REVERSE, "repeat": EXPECTED_REPEAT}
    by_id = {row["request"]["id"]: row for row in requests}
    gold_by_id = {row["id"]: row for row in gold}
    if len(by_id) != expected["primary"] or manifest["cases"] != expected["primary"]:
        problems.append(f"fixture has {len(by_id)} unique requests covering {manifest['cases']} cases, expected {expected['primary']}")
    if set(by_id) != set(gold_by_id):
        problems.append("gold and request id sets differ")
    for row in requests:
        if digest(row["request"]) != row["sha256"]:
            problems.append(f"request digest mismatch for {row['request']['id']}")
    for model in MODELS:
        primary = records.get((model, "primary"), [])
        reverse = records.get((model, "reverse"), [])
        repeat = records.get((model, "repeat"), [])
        if len(primary) != expected["primary"]:
            problems.append(f"{model}: {len(primary)} primary records, expected {expected['primary']}")
        if len(reverse) != expected["reverse"]:
            problems.append(f"{model}: {len(reverse)} reversal probes, expected {expected['reverse']}")
        if len(repeat) != expected["repeat"]:
            problems.append(f"{model}: {len(repeat)} repeat probes, expected {expected['repeat']}")
        seen = set()
        for record in primary:
            score = record["score"]
            rid = score["id"]
            if rid in seen:
                problems.append(f"{model}: duplicate primary record for {rid}")
            seen.add(rid)
            row = by_id.get(rid)
            if row is None:
                problems.append(f"{model}: record for unknown request {rid}")
                continue
            if record["request_sha256"] != digest(row["request"]):
                problems.append(f"{model}: request hash mismatch for {rid}")
            option_ids = [option["id"] for option in row["request"]["options"]]
            if score["option_ids"] != option_ids:
                problems.append(f"{model}: option order changed for {rid}")
            probabilities = score["probabilities"]
            if len(probabilities) != len(option_ids):
                problems.append(f"{model}: probability count mismatch for {rid}")
            elif abs(sum(probabilities) - 1) > 1e-6 or min(probabilities) < 0:
                problems.append(f"{model}: probabilities are not a simplex for {rid}")
            if score.get("backbone_calls") != 1 or score.get("decode_steps") != 0:
                problems.append(f"{model}: unexpected backbone calls or decode steps for {rid}")
            if not (score.get("total_ms", 0) > 0):
                problems.append(f"{model}: non-positive measured latency for {rid}")
        missing = [rid for rid in by_id if rid not in seen]
        if missing:
            problems.append(f"{model}: {len(missing)} requests without a primary record")

def native_rows(requests, gold, records):
    """Accuracy per (model, dataset, role) recomputed from argmax over the recorded order."""
    gold_by_id = {row["id"]: row for row in gold}
    rows = []
    for model in MODELS:
        for record in records.get((model, "primary"), []):
            score = record["score"]
            request_id = score["id"]
            label = gold_by_id[request_id]["label"]
            option_ids = score["option_ids"]
            best = max(range(len(option_ids)), key=lambda index: score["probabilities"][index])
            rows.append({"model": model, "dataset": record["dataset"], "role": record["role"],
                         "id": request_id, "correct": int(option_ids[best] == label),
                         "latency_ms": score["total_ms"], "tokens_sum": score["input_tokens_sum"]})
    return rows


def summarise(rows):
    summary = []
    for model in MODELS:
        for dataset in DATASETS:
            group = [r for r in rows if r["model"] == model and r["dataset"] == dataset and r["role"] == "test"]
            if not group:
                continue
            correct = sum(r["correct"] for r in group)
            low, high = wilson(correct, len(group))
            latencies = [r["latency_ms"] for r in group]
            summary.append({"model": model, "dataset": dataset, "n": len(group), "correct": correct,
                            "accuracy": correct / len(group), "wilson_low": low, "wilson_high": high,
                            "p50_ms": percentile(latencies, 0.5), "p95_ms": percentile(latencies, 0.95)})
        group = [r for r in rows if r["model"] == model and r["role"] == "test"]
        if group:
            correct = sum(r["correct"] for r in group)
            low, high = wilson(correct, len(group))
            summary.append({"model": model, "dataset": "ALL", "n": len(group), "correct": correct,
                            "accuracy": correct / len(group), "wilson_low": low, "wilson_high": high,
                            "p50_ms": percentile([r["latency_ms"] for r in group], 0.5),
                            "p95_ms": percentile([r["latency_ms"] for r in group], 0.95)})
    return summary


def run(incoming: pathlib.Path, fixture: pathlib.Path, evaluation: pathlib.Path, out: pathlib.Path) -> int:
    manifest = json.loads((fixture / "manifest.json").read_text())
    manifest_eval = json.loads((evaluation / "manifest.json").read_text())
    requests = read_jsonl(fixture / "requests.jsonl")
    gold = read_jsonl(evaluation / "gold.jsonl")
    problems = []
    if sha256_file(fixture / "requests.jsonl") != manifest["requests_sha256"]:
        problems.append("requests.jsonl does not match the manifest digest")
    if sha256_file(evaluation / "gold.jsonl") != manifest_eval["gold_sha256"]:
        problems.append("gold.jsonl does not match its manifest digest")
    records, shards = collect_records(incoming)
    check_contracts(requests, gold, manifest, manifest_eval, records, problems)
    rows = native_rows(requests, gold, records)
    summary = summarise(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "independent_native.json").write_text(json.dumps(
        {"version": "1.6.0", "reader": "verify_jevbench.py", "shards": len(shards),
         "primary_records": sum(len(v) for k, v in records.items() if k[1] == "primary"),
         "probe_records": sum(len(v) for k, v in records.items() if k[1] != "primary"),
         "summary": summary, "problems": problems}, indent=2) + chr(10))
    print(f"{'model':11s} {'dataset':10s} {'n':>3s} {'correct':>7s} {'accuracy':>9s}  95% Wilson              p50 ms      p95 ms")
    for row in summary:
        print(f"{row['model']:11s} {row['dataset']:10s} {row['n']:3d} {row['correct']:7d} "
              f"{row['accuracy']:>8.3f}  [{row['wilson_low']:.3f}, {row['wilson_high']:.3f}]  "
              f"{row['p50_ms']:9.1f} {row['p95_ms']:9.1f}")
    print()
    if problems:
        print("PROBLEMS")
        for problem in problems:
            print("  -", problem)
        return 1
    print("independent re-derivation found no contract problem")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=pathlib.Path, default=pathlib.Path("incoming"))
    parser.add_argument("--fixture", type=pathlib.Path, default=pathlib.Path("fixture"))
    parser.add_argument("--evaluation", type=pathlib.Path, default=pathlib.Path("evaluation"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/jev16-independent"))
    args = parser.parse_args()
    sys.exit(run(args.incoming, args.fixture, args.evaluation, args.out))
