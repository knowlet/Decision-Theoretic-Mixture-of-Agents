"""Pinned v1.8 expanded native decision cohort.

The v1.7 registry and workflow remain frozen.  This registry adds the
multilingual Laya checkpoint and is used only by the 200-test-case workflow.
"""
from __future__ import annotations

EXPANDED_MODELS = {
    "decider_2b": {
        "kind": "decider",
        "repo": "Mapika/decider-2b",
        "revision": "b37f7e1ba3fbc9238004cf531fabbee2619973fd",
        "source_repo": "Mapika/decider",
        "base_repo": "Qwen/Qwen3.5-2B-Base",
        "language": ["en"],
        "leaderboard_role": "english_and_ood_chinese",
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
        "leaderboard_role": "english_and_ood_chinese",
        "license": "Apache-2.0",
    },
    "laya_typed": {
        "kind": "laya",
        "repo": "convaiinnovations/laya-typed-decisions",
        "revision": "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
        "source_repo": "NandhaKishorM/laya",
        "source_revision": "42626c348753fbb17572a813127df2278a1ec527",
        "language": ["en"],
        "leaderboard_role": "english_and_ood_chinese",
        "license": "Apache-2.0",
    },
    "laya_multilingual": {
        "kind": "laya",
        "repo": "convaiinnovations/laya-multilingual",
        "revision": "052592a15d198d9ad47da779604259b10b47b7aa",
        "source_repo": "NandhaKishorM/laya",
        "source_revision": "42626c348753fbb17572a813127df2278a1ec527",
        "language": ["multilingual", "zh", "en", "de", "fr", "es", "ja", "ko"],
        "leaderboard_role": "english_and_ood_chinese",
        "license": "Apache-2.0",
    },
}

EXPANDED_MODEL_ORDER = tuple(EXPANDED_MODELS)
DATASETS = ("boolq", "ocnli", "clinc", "tmmluplus")
SIZES = {"fit": 128, "dev": 64, "test": 200}
LEADERBOARDS = {
    "english": ("boolq", "clinc"),
    "ood_chinese": ("ocnli", "tmmluplus"),
}


def json_contract() -> dict:
    """Registry contract in the JSON-normalized form used by manifests."""
    return {
        "datasets": list(DATASETS),
        "sizes_per_dataset": dict(SIZES),
        "leaderboards": {name: list(datasets) for name, datasets in LEADERBOARDS.items()},
    }
