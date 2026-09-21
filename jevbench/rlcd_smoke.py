"""Smoke harness for Qwen-2.5-1B-RLCD.

RLCD is a constrained-generation engine around Qwen2.5-1.5B, not a native
typed decision head.  Its metrics therefore stay in a separate report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


CASES = [
    {
        "id": "billing",
        "context": "The customer was charged twice for one invoice and asks for a refund.",
        "schema": {"department": {"type": "enum", "description": "Owning team", "choices": ["billing", "shipping", "access"]}},
        "expected": {"department": "billing"},
    },
    {
        "id": "urgent",
        "context": "The production database is unavailable and blocks every checkout.",
        "schema": {
            "priority": {"type": "enum", "description": "Urgency", "choices": ["low", "normal", "critical"]},
            "requires_review": {"type": "boolean", "description": "Needs immediate human review"},
        },
        "expected": {"priority": "critical", "requires_review": True},
    },
    {
        "id": "support",
        "context": "A user cannot log in after resetting the password.",
        "schema": {"department": {"type": "enum", "description": "Owning team", "choices": ["access", "billing", "sales"]}},
        "expected": {"department": "access"},
    },
]


def exact_match(parsed: dict, expected: dict) -> bool:
    """Compare primitive field values across the upstream engine response shapes."""
    for key, expected_value in expected.items():
        actual = parsed.get(key)
        if isinstance(actual, dict) and "value" in actual:
            actual = actual["value"]
        if actual != expected_value:
            return False
    return True


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def run(source: Path, out: Path) -> None:
    os.environ.setdefault("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    sys.path.insert(0, str(source))
    from core.engine import run_parallel_generation
    from core.schema import StructuredSchema

    results = []
    for case in CASES:
        schema = StructuredSchema(case["schema"])
        started = time.perf_counter()
        result = run_parallel_generation(case["context"], schema)
        elapsed = 1000 * (time.perf_counter() - started)
        parsed = result.get("parsed_json") or {}
        exact = exact_match(parsed, case["expected"])
        results.append({
            "id": case["id"],
            "elapsed_ms_wall": elapsed,
            "elapsed_ms_engine": result.get("elapsed_ms"),
            "sequential_forward_passes": result.get("sequential_forward_passes"),
            "is_valid_json": bool(result.get("is_valid_json")),
            "schema_match": bool(result.get("schema_match")),
            "exact_match": exact,
            "parsed_json": parsed,
        })
    if not results or not all(row["is_valid_json"] and row["schema_match"] for row in results):
        raise RuntimeError("RLCD parallel engine failed its schema contract")
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "1.7.0-candidate",
        "repo": "harshatheg/Qwen-2.5-1B-RLCD",
        "revision": "2af86848be75847ccb3553b0941cc51d6ef7e4e9",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "base_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "source": str(source),
        "mode": "parallel constrained generation; separate from native decision-head benchmark",
        "backend": os.environ.get("BACKEND", "auto"),
        "source_tree_sha256": tree_sha256(source),
        "cases": len(results),
        "schema_match_rate": sum(row["schema_match"] for row in results) / len(results),
        "exact_match_rate": sum(row["exact_match"] for row in results) / len(results),
        "mean_latency_ms": sum(row["elapsed_ms_wall"] for row in results) / len(results),
        "results": results,
    }
    (out / "rlcd_smoke.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/rlcd"))
    args = parser.parse_args()
    run(args.source, args.out)
