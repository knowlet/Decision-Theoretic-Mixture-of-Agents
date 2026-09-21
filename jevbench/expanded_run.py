"""Gold-blind inference runner for the v1.8 expanded native-head cohort."""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

from .adapters import sha
from .expanded_adapters import ExpandedAdapter
from .expanded_registry import EXPANDED_MODEL_ORDER
from .fixtures import digest, read_jsonl, write
from .run import validate_score


DATASETS = ("boolq", "ocnli", "clinc", "tmmluplus")
PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "expanded_protocol.json"


def _probe_ids(rows: list[dict]) -> tuple[set[str], set[str]]:
    reverse: set[str] = set()
    repeat: set[str] = set()
    for dataset in DATASETS:
        test = [row for row in rows if row["dataset"] == dataset and row["role"] == "test"]
        reverse.update(row["request"]["id"] for row in test[:4])
        repeat.update(row["request"]["id"] for row in test[:2])
    return reverse, repeat


def run(model: str, shard: int, fixture: Path, out: Path, vendor: Path = Path("vendor")) -> None:
    if model not in EXPANDED_MODEL_ORDER:
        raise ValueError(f"unknown extended model: {model}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((fixture / "manifest.json").read_text())
    request_path = fixture / "requests.jsonl"
    if manifest.get("protocol_sha256") != sha(PROTOCOL_PATH):
        raise ValueError("expanded protocol changed")
    if sha(request_path) != manifest["requests_sha256"]:
        raise ValueError("input fixture changed")
    if (fixture / "gold.jsonl").exists():
        raise ValueError("inference environment must not receive gold artifact")
    rows = read_jsonl(request_path)
    expected_cases = int(manifest["cases"])
    if len(rows) != expected_cases or len({row["request"]["id"] for row in rows}) != expected_cases:
        raise ValueError("incomplete fixture")
    for row in rows:
        if digest(row["request"]) != row["sha256"]:
            raise ValueError("request digest mismatch")
    assigned = [row for row in rows if row["shard"] == shard]
    if not assigned:
        raise ValueError("empty shard")
    reverse_ids, repeat_ids = _probe_ids(rows)

    started = time.perf_counter()
    adapter = ExpandedAdapter(model, vendor=vendor)
    load_ms = 1000 * (time.perf_counter() - started)
    warmup = adapter.score(assigned[0]["request"])
    validate_score(warmup, assigned[0]["request"])  # warmup is recorded separately from evidence rows
    inventory: list[tuple[str, str]] = []
    with (out / "predictions.jsonl").open("w") as stream:
        for row in assigned:
            kinds = ["primary"]
            if row["request"]["id"] in reverse_ids:
                kinds.append("reverse")
            if row["request"]["id"] in repeat_ids:
                kinds.append("repeat")
            for kind in kinds:
                request = json.loads(json.dumps(row["request"]))
                if kind == "reverse":
                    request["options"] = list(reversed(request["options"]))
                score = adapter.score(request)
                validate_score(score, request)
                record = {
                    "dataset": row["dataset"],
                    "role": row["role"],
                    "group": row["group"],
                    "kind": kind,
                    "request_sha256": digest(request),
                    "score": score,
                }
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                inventory.append((request["id"], kind))
            print(model, shard, row["dataset"], row["role"], row["request"]["id"], round(score["total_ms"], 2), flush=True)

    write(out / "provenance.json", {
        "version": "1.8.0-expanded",
        "commit": os.environ.get("GITHUB_SHA"),
        "model": model,
        "shard": shard,
        "fixture_sha256": manifest["requests_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "adapter": adapter.meta,
        "adapter_file_sha256": sha(Path(__file__).with_name("expanded_adapters.py")),
        "runner_sha256": sha(__file__),
        "forwards": len(inventory),
        "evidence_forwards": len(inventory),
        "warmups": 1,
        "total_forwards": len(inventory) + 1,
        "primary": len(assigned),
        "load_and_download_ms": load_ms,
        "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cpu_model": next(
            (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
            "unknown",
        ),
        "platform": platform.platform(),
        "run_url": os.environ.get("RESEARCH_RUN_URL"),
        "new_api_calls": 0,
        "new_training_steps": 0,
    })
    write(out / "SHA256.json", {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "SHA256.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=EXPANDED_MODEL_ORDER, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--fixture", type=Path, default=Path("fixture"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, default=Path("vendor"))
    args = parser.parse_args()
    run(args.model, args.shard, args.fixture, args.out, args.vendor)
