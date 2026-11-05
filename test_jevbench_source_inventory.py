"""Offline v1.8 source-ledger regressions; no downloads or model inference."""
from __future__ import annotations

import hashlib
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from jevbench import expanded_fixtures as fixture


def synthetic_datasets(root: Path):
    """Tiny local source files; parquet decoding is replaced with DataFrames."""
    paths = {dataset: root / dataset for dataset in fixture.DATASETS}
    for path in paths.values():
        path.mkdir(parents=True)
    frames = {}
    for split, count in (("train", 192), ("validation", 200)):
        path = paths["boolq"] / "data" / f"{split}-00000.parquet"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"offline parquet decoding fixture")
        frames[path.name] = pd.DataFrame([
            {"passage": f"{split} passage {i}", "question": f"Question {i}?", "answer": bool(i % 2)}
            for i in range(count)
        ])
    for split, count in (("train.50k", 192), ("dev", 200)):
        records = [{"id": i, "sentence1": f"{split} premise {i}", "sentence2": f"Hypothesis {i}",
                    "label": ("entailment", "neutral", "contradiction")[i % 3]}
                   for i in range(count)]
        (paths["ocnli"] / f"{split}.json").write_text("".join(json.dumps(row) + "\n" for row in records))
    (paths["tmmluplus"] / "data").mkdir()
    for split, count in (("dev", 128), ("val", 64), ("test", 200)):
        pd.DataFrame([{ "question": f"{split} mathematics {i}", "A": "One", "B": "Two",
                        "C": "Three", "D": "Four", "answer": "ABCD"[i % 4]}
                      for i in range(count)]).to_csv(paths["tmmluplus"] / "data" / f"math_{split}.csv", index=False)
    clinc = {}
    for split, examples, oos in (("train", 10, 32), ("val", 1, 16), ("test", 1, 50)):
        clinc[split] = [[f"{split} example {index} for intent {label} shared words", f"intent_{label:03d}"]
                        for label in range(150) for index in range(examples)]
        clinc[f"oos_{split}"] = [[f"{split} out of scope {i}", "oos"] for i in range(oos)]
    (paths["clinc"] / "data_full.json").write_text(json.dumps(clinc))
    return paths, frames


class SourceInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, relative, content=b"source bytes"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_parent_cache_name_does_not_exclude_dataset_sources(self):
        for cache in (".cache/jev18", "custom-cache"):
            with self.subTest(cache=cache):
                path = self.write(f"{cache}/clinc/data_full.json", b'{"train": []}\n')
                actual = fixture.source_inventory({"clinc": path.parent})
                self.assertEqual(actual, [{
                    "dataset": "clinc", "file": "data_full.json",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }])

    def test_only_files_used_by_each_dataset_are_included(self):
        expected = {
            "boolq": ["data/train-00000.parquet", "data/validation-00000.parquet"],
            "ocnli": ["dev.json", "train.50k.json"],
            "clinc": ["data_full.json"],
            "tmmluplus": ["data/math_dev.csv", "data/math_test.csv", "data/math_val.csv"],
        }
        roots = {}
        for dataset, files in expected.items():
            roots[dataset] = self.root / ".cache" / dataset
            for name in files:
                self.write(f".cache/{dataset}/{name}", name.encode())
            for name in (".cache/huggingface/download/state.json", ".git/config.json",
                         "cache-control.json", "manifest.json", "data/unrelated.json",
                         "data/metadata.csv", "README.md"):
                self.write(f".cache/{dataset}/{name}", b"metadata")
        ledger = fixture.source_inventory(roots)
        for dataset, files in expected.items():
            self.assertEqual([r["file"] for r in ledger if r["dataset"] == dataset], files)
        self.assertEqual(len(ledger), sum(map(len, expected.values())))
        self.assertEqual(ledger, fixture.source_inventory(dict(reversed(list(roots.items())))))
        self.assertTrue(all(not Path(row["file"]).is_absolute() for row in ledger))

    def test_file_symlink_outside_dataset_is_rejected(self):
        outside = self.write("other/data.json")
        root = self.root / "clinc"
        root.mkdir()
        (root / "data_full.json").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes dataset root"):
            fixture.source_inventory({"clinc": root})

    def test_directory_symlink_outside_dataset_is_rejected(self):
        outside = self.write("other/train-00000.parquet")
        root = self.root / "boolq"
        root.mkdir()
        (root / "data").symlink_to(outside.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes dataset root"):
            fixture.source_inventory({"boolq": root})

    def test_other_dataset_files_cannot_be_claimed_by_a_symlink(self):
        other = self.write("ocnli/dev.json")
        root = self.root / "clinc"
        root.mkdir()
        (root / "data_full.json").symlink_to(other)
        with self.assertRaisesRegex(ValueError, "escapes dataset root"):
            fixture.source_inventory({"clinc": root})

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown fixture dataset"):
            fixture.source_inventory({"unknown": self.root})

    def test_changed_bytes_change_the_recorded_hash_and_size(self):
        path = self.write("clinc/data_full.json", b"first")
        before = fixture.source_inventory({"clinc": path.parent})
        path.write_bytes(b"second version")
        after = fixture.source_inventory({"clinc": path.parent})
        self.assertNotEqual(before[0]["sha256"], after[0]["sha256"])
        self.assertEqual(after[0]["bytes"], len(b"second version"))

    def test_ledger_repair_preserves_requests_and_gold_with_fixed_timing(self):
        paths, frames = synthetic_datasets(self.root / ".cache" / "jev18")
        before, after = self.root / "before", self.root / "after"
        # The old full-path filter produced an empty ledger under .cache.
        # Stub only that old result; use the real cohort/shortlist/serialization
        # code in both executions. Fix timing solely for this offline comparison.
        with patch.object(fixture.pd, "read_parquet", side_effect=lambda path: frames[path.name]), \
             patch.object(fixture.time, "perf_counter", return_value=0.0), \
             contextlib.redirect_stdout(io.StringIO()):
            with patch.object(fixture, "source_inventory", return_value=[]):
                fixture.prepare(paths, before)
            fixture.prepare(paths, after)
        for name in ("requests.jsonl", "gold.jsonl"):
            self.assertEqual((before / name).read_bytes(), (after / name).read_bytes())
        old = json.loads((before / "manifest.json").read_text())
        new = json.loads((after / "manifest.json").read_text())
        self.assertEqual(old.pop("sources"), [])
        self.assertEqual(len(new.pop("sources")), 8)
        self.assertEqual(old, new)
        self.assertEqual(new["cases"], 1568)


if __name__ == "__main__":
    unittest.main()
