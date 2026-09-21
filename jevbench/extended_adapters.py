"""Adapters for the v1.7 candidate native decision benchmark.

Each adapter receives the same semantic choice request and returns a probability
simplex over exactly the supplied options.  The v1.6 adapter module remains
unchanged so its published pilot evidence can still be reproduced byte for
byte.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from .adapters import cpu_accumulation, sha
from .extended_registry import EXTENDED_MODELS

MAX_TOKENS = 2048


def _tree_sha256(root: Path) -> str:
    """Hash a downloaded upstream source tree without committing it here."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _snapshot(repo: str, revision: str, allow_patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=allow_patterns, max_workers=2))


def _choice_request(request: dict) -> dict:
    return {
        "type": "choice",
        "instructions": request["question"],
        "criteria": {option["id"]: option["text"] for option in request["options"]},
    }


def _normalise_probabilities(values: list[float]) -> list[float]:
    """Accept rounded probabilities or logits, while rejecting invalid output."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("native adapter returned non-finite values")
    if (x >= 0).all() and float(x.sum()) > 0:
        x = x / x.sum()
    else:
        y = np.exp(x - x.max())
        x = y / y.sum()
    if (x < 0).any() or not np.isclose(float(x.sum()), 1.0, atol=1e-6):
        raise ValueError("native adapter did not return a probability simplex")
    return x.tolist()


class ExtendedAdapter:
    """Load one immutable native checkpoint and expose the study contract."""

    def __init__(self, name: str, vendor: Path = Path("vendor")):
        if name not in EXTENDED_MODELS:
            raise ValueError(f"unknown extended model: {name}")
        self.name = name
        self.spec = EXTENDED_MODELS[name]
        self.vendor = Path(vendor)
        self.torch = None
        self.model = None
        self.agent = None
        self.tok = None
        self.temperature = 1.0
        started = time.perf_counter()

        import torch
        import transformers

        torch.manual_seed(160918)
        torch.set_num_threads(4)
        self.torch = torch

        kind = self.spec["kind"]
        if kind == "decider":
            self._load_decider(transformers)
        elif kind == "kev":
            self._load_kev()
        elif kind == "laya":
            self._load_laya()
        else:
            raise ValueError(f"unsupported extended adapter kind: {kind}")

        self.meta = {
            "id": self.spec["repo"],
            "revision": self.spec["revision"],
            "source_repo": self.spec.get("source_repo"),
            "source_revision": self.spec.get("source_revision"),
            "base_repo": self.spec.get("base_repo"),
            "base_revision": self.spec.get("base_revision"),
            "dtype": self.dtype,
            "device": "cpu",
            "torch_threads": 4,
            "temperature": self.temperature,
            "layout": self.layout,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "load_seconds_excluding_download": time.perf_counter() - started,
            "code_sha256": self.code,
            "adapter_sha256": sha(Path(__file__)),
            "new_training_steps": 0,
            "benchmark_family": "native_typed_choice",
            "parameters": int(sum(p.numel() for p in self.model.parameters())),
        }

    def _load_decider(self, transformers) -> None:
        spec = self.spec
        path = _snapshot(
            spec["repo"],
            spec["revision"],
            ["*.json", "*.safetensors", "*.py", "*.txt", "*.model", "decider/*.py"],
        )
        sys.path.insert(0, str(path))
        from decider.infer import Decider

        cpu_accumulation()
        self.api = Decider(str(path), device="cpu", dtype=self.torch.bfloat16, use_graphs=False)
        self.model = self.api.m
        self.tok = self.model.tok
        self.model.eval()
        self.model.lm.config.use_cache = False
        self.temperature = self.api.T
        self.dtype = "bfloat16"
        self.layout = "upstream state-first decision slot; latest pinned checkpoint; no CUDA graphs"
        self.code = {str(p.relative_to(path)): sha(p) for p in (path / "decider").glob("*.py")}

    def _load_kev(self) -> None:
        spec = self.spec
        source = self.vendor / "kev"
        if not (source / "kev").is_dir():
            raise FileNotFoundError(f"pinned Kev source is missing: {source}")
        sys.path.insert(0, str(source))
        from kev.evaluate import load

        path = _snapshot(
            spec["repo"],
            spec["revision"],
            ["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"],
        )
        metadata = self.torch.load(path / "head.pt", map_location="cpu", weights_only=False)
        if metadata.get("base") != spec["base_repo"] or metadata.get("base_revision") != spec["base_revision"]:
            raise RuntimeError("Kev checkpoint base metadata does not match the pinned registry")
        self.tok, self.model = load(str(path), "cpu", dtype=self.torch.float32, merge=True, attn="eager")
        self.model.eval()
        self.dtype = "float32"
        self.layout = "upstream Qwen3.5 hybrid rows with pointer head; fp32 evaluation"
        self.code = {"source_tree": _tree_sha256(source), "head.pt": sha(path / "head.pt")}

    def _load_laya(self) -> None:
        spec = self.spec
        source = self.vendor / "laya"
        if not (source / "laya").is_dir():
            raise FileNotFoundError(f"pinned Laya source is missing: {source}")
        sys.path.insert(0, str(source))
        from laya.agent import Agent

        path = _snapshot(
            spec["repo"],
            spec["revision"],
            ["*.json", "*.safetensors", "*.txt", "tokenizer/*", "encoder/*"],
        )
        self.agent = Agent(str(path), device="cpu")
        self.model = self.agent.model
        self.dtype = "float32"
        self.layout = "upstream typed-decisions ModernBERT encoder with calibrated choice head"
        self.code = {"source_tree": _tree_sha256(source), "model.safetensors": sha(path / "model.safetensors")}

    def score(self, request: dict) -> dict:
        options = request["options"]
        if not 2 <= len(options) <= 255 or len({o["id"] for o in options}) != len(options):
            raise ValueError("invalid candidate options")
        started = time.perf_counter()
        kind = self.spec["kind"]

        if kind == "decider":
            from decider.infer import Example, Q, neutralize_options
            from decider.model import collate
            from decider.prompt import MAX_OPTIONS, build

            texts = [option["text"] for option in options]

            class Keep:
                def shuffle(self, values):
                    return None

                def sample(self, values, k):
                    return values[:k]

            normalized = neutralize_options(texts)[0] if self.api.neutralize_none else texts
            item = build(
                Example(request["state"], [Q(request["question"], normalized, 0)], "infer"),
                self.tok,
                Keep(),
                max_options=MAX_OPTIONS,
                max_ctx_tokens=1000000,
            )
            if len(item["ids"]) > MAX_TOKENS:
                raise ValueError("input exceeds common limit; not truncated")
            batch = collate([item], self.tok.pad_token_id)
            with self.torch.inference_mode():
                logits = self.model.slot_logits(
                    batch["input_ids"], batch["attention_mask"], batch["slot_idx"], batch["slot_batch"], batch["nopts"]
                )
            values = logits[0, : len(options)].float().cpu().tolist()
            probs = self.torch.softmax(
                self.torch.tensor(values, dtype=self.torch.float32) / self.temperature, dim=-1
            ).tolist()
            input_tokens_sum = len(item["ids"])
            max_path_tokens = len(item["ids"])
            candidate_paths = 1
        elif kind == "kev":
            from kev.api import SystemOneRequest, to_record

            req = SystemOneRequest(
                state=request["state"],
                model="kev-latest",
                questions={"choice": _choice_request(request)},
            )
            rec, _ = to_record(req)
            enc = self.model.encode(self.tok, rec, max_state=MAX_TOKENS, max_branch=MAX_TOKENS, strict=True)
            with self.torch.inference_mode():
                probs = self.model.probs(enc)[0].tolist()
            input_tokens_sum = len(enc["ids"])
            max_path_tokens = len(enc["ids"])
            candidate_paths = len(options)
        elif kind == "laya":
            with self.torch.inference_mode():
                response = self.agent.system_one(request["state"], {"choice": _choice_request(request)})
            answer = response["answers"]["choice"]
            probabilities = answer["probabilities"]
            probs = [float(probabilities[o["id"]]) for o in options]
            input_tokens_sum = int(response.get("usage", {}).get("input_tokens", 0))
            max_path_tokens = input_tokens_sum
            candidate_paths = len(options)
        else:
            raise AssertionError(kind)

        return {
            "id": request["id"],
            "option_ids": [o["id"] for o in options],
            "probabilities": _normalise_probabilities(probs),
            "raw_logits": [],
            "total_ms": 1000 * (time.perf_counter() - started),
            "preprocess_ms": 0.0,
            "input_tokens_sum": input_tokens_sum,
            "max_path_tokens": max_path_tokens,
            "candidate_paths": candidate_paths,
            "backbone_calls": 1,
            "decode_steps": 0,
        }
