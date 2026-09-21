"""Pinned checkpoints for the v1.7 candidate benchmark.

The v1.6 registry is deliberately unchanged.  This file records a separate
cohort so a moving Hub reference cannot alter the published pilot evidence.
"""
from __future__ import annotations

EXTENDED_MODELS = {
    "decider_2b": {
        "kind": "decider",
        "repo": "Mapika/decider-2b",
        "revision": "b37f7e1ba3fbc9238004cf531fabbee2619973fd",
        "source_repo": "Mapika/decider",
        "base_repo": "Qwen/Qwen3.5-2B-Base",
        "language": ["en"],
        "license": "Apache-2.0",
    },
    "kev_08b": {
        "kind": "kev",
        "repo": "jaredpalmer/kev-0.8b",
        "revision": "c917edefdfd72b3e9ba71455584700acc70595f6",
        "source_repo": "jaredpalmer/kev",
        "source_revision": "e0bcf50153f1bda4ca6a8be5e12cbd5f9ebbce1c",
        "base_repo": "Qwen/Qwen3.5-0.8B-Base",
        "base_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        "language": ["en"],
        "license": "Apache-2.0",
    },
    "laya_typed": {
        "kind": "laya",
        "repo": "convaiinnovations/laya-typed-decisions",
        "revision": "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
        "source_repo": "NandhaKishorM/laya",
        "source_revision": "42626c348753fbb17572a813127df2278a1ec527",
        "language": ["en"],
        "license": "Apache-2.0",
    },
}

EXTENDED_MODEL_ORDER = tuple(EXTENDED_MODELS)

RLCD = {
    "repo": "harshatheg/Qwen-2.5-1B-RLCD",
    "revision": "2af86848be75847ccb3553b0941cc51d6ef7e4e9",
    "base_repo": "Qwen/Qwen2.5-1.5B-Instruct",
    "base_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "source_kind": "parallel constrained generation; no native decision head",
    "license": "Apache-2.0",
}
