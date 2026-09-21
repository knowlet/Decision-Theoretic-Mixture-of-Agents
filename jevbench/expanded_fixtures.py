"""Expanded v1.8 semantic fixture with 200 held-out cases per dataset.

The v1.7 fixture remains frozen.  This module creates a separate cohort so the
larger English/OOD-Chinese comparison cannot silently change the published pilot.
Gold is written to a separate analysis artifact and never enters native workers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .fixtures import REVISIONS, canonical, download, fetch, read_jsonl, row, write

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
DATASETS = ("boolq", "ocnli", "clinc", "tmmluplus")
SIZES = {"fit": 128, "dev": 64, "test": 200}
LEADERBOARDS = {
    "english": ("boolq", "clinc"),
    "ood_chinese": ("ocnli", "tmmluplus"),
}


def _choose(rows, n, excluded=(), roundrobin=False):
    excluded = set(excluded)
    seen = set()
    ordered = sorted(rows, key=lambda r: hashlib.sha256(
        canonical(["jev18-expanded-cohort", r["group"], r["source_id"]]).encode()
    ).hexdigest())
    unique = []
    for item in ordered:
        if item["group"] not in seen and item["group"] not in excluded:
            unique.append(item)
            seen.add(item["group"])
    if roundrobin:
        from collections import defaultdict

        queues = defaultdict(list)
        for item in unique:
            queues[item["subject"]].append(item)
        subjects = sorted(queues, key=lambda s: hashlib.sha256(
            canonical(["jev18-subject", s]).encode()
        ).hexdigest())
        unique = []
        for i in range(max(map(len, queues.values()))):
            for subject in subjects:
                if i < len(queues[subject]):
                    unique.append(queues[subject][i])
    if len(unique) < n:
        raise ValueError(f"Insufficient distinct groups: requested {n}, found {len(unique)}")
    return unique[:n]


def prepare(paths: dict[str, Path], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    selected = []
    audits = {}
    allrows = {}
    sources = []
    for dataset, path in paths.items():
        for candidate in sorted(path.rglob("*")):
            if candidate.is_file() and ".cache" not in candidate.parts and candidate.suffix in (".json", ".csv", ".parquet"):
                sources.append({
                    "dataset": dataset,
                    "file": str(candidate.relative_to(path)),
                    "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                    "bytes": candidate.stat().st_size,
                })

    yes_no = [{"id": "no", "text": "No"}, {"id": "yes", "text": "Yes"}]
    allrows["boolq"] = {}
    for source, role in (("train", "train"), ("validation", "test")):
        frame = pd.concat([
            pd.read_parquet(path)
            for path in sorted(paths["boolq"].glob(f"data/{source}*.parquet"))
        ])
        allrows["boolq"][role] = [
            row("boolq", f"{source}:{i}", item.passage, item.question, yes_no,
                "yes" if item.answer else "no", item.passage, source)
            for i, item in enumerate(frame.itertuples())
        ]

    allrows["ocnli"] = {}
    excluded_n = 0
    ocnli_options = [
        {"id": "entailment", "text": "蕴含：前提成立时，假设必然成立"},
        {"id": "neutral", "text": "中立：根据前提无法确定假设是否成立"},
        {"id": "contradiction", "text": "矛盾：前提成立时，假设必然不成立"},
    ]
    for source, role in (("train.50k", "train"), ("dev", "test")):
        original = read_jsonl(paths["ocnli"] / f"{source}.json")
        usable = [item for item in original if item["label"] in ("entailment", "neutral", "contradiction")]
        excluded_n += len(original) - len(usable)
        allrows["ocnli"][role] = [
            row(
                "ocnli",
                f"{source}:{item['id']}",
                f"前提：{item['sentence1']}\n假設：{item['sentence2']}",
                "根据前提，假设属于哪一种关系？",
                ocnli_options,
                item["label"],
                item["sentence1"],
                source,
                item.get("genre", ""),
            )
            for item in usable
        ]
    audits["ocnli"] = {
        "unlabeled_or_no_majority_excluded": excluded_n,
        "evaluation": "official labeled dev, NOT blind official test",
        "group": "premise text, one pair per selected premise",
    }

    allrows["tmmluplus"] = {}
    for suffix, role in (("dev", "train"), ("val", "dev"), ("test", "test")):
        rows = []
        for path in sorted(paths["tmmluplus"].glob(f"data/*_{suffix}.csv")):
            subject = path.name.removesuffix(f"_{suffix}.csv")
            frame = pd.read_csv(path, keep_default_na=False)
            for i, item in frame.iterrows():
                options = [{"id": key, "text": str(item[key])} for key in "ABCD"]
                if item["answer"] not in "ABCD" or len(item["answer"]) != 1:
                    raise ValueError("Unexpected TMMLU label")
                rows.append(row(
                    "tmmluplus", f"{subject}:{suffix}:{i}", str(item["question"]),
                    "請選出此題的正確答案。", options, item["answer"],
                    str(item["question"]) + "\n" + canonical(options), suffix, subject,
                ))
        allrows["tmmluplus"][role] = rows

    clinc = json.loads((paths["clinc"] / "data_full.json").read_text())
    allrows["clinc"] = {}
    for source, role in (("train", "train"), ("val", "dev"), ("test", "test")):
        data = clinc[source] + clinc[f"oos_{source}"]
        allrows["clinc"][role] = [
            row("clinc", f"{source}:{i}", text,
                "Which intent best describes this utterance?", [], label, text, source)
            for i, (text, label) in enumerate(data)
        ]

    for dataset in DATASETS:
        raw = allrows[dataset]
        test_groups = {item["group"] for item in raw["test"]}
        roundrobin = dataset == "tmmluplus"
        if dataset == "clinc":
            test = (
                _choose([r for r in raw["test"] if r["gold"] == "oos"], 50)
                + _choose([r for r in raw["test"] if r["gold"] != "oos"], 150)
            )
            reserved = test_groups
            fit = (
                _choose([r for r in raw["train"] if r["gold"] == "oos"], 32, reserved)
                + _choose([r for r in raw["train"] if r["gold"] != "oos"], 96, reserved)
            )
            reserved |= {r["group"] for r in fit}
            dev = (
                _choose([r for r in raw["dev"] if r["gold"] == "oos"], 16, reserved)
                + _choose([r for r in raw["dev"] if r["gold"] != "oos"], 48, reserved)
            )
        else:
            test = _choose(raw["test"], SIZES["test"], roundrobin=roundrobin)
            fit = _choose(raw["train"], SIZES["fit"], test_groups, roundrobin=roundrobin)
            dev = _choose(
                raw.get("dev", raw["train"]), SIZES["dev"],
                test_groups | {r["group"] for r in fit}, roundrobin=roundrobin,
            )
        for role, cohort in (("fit", fit), ("dev", dev), ("test", test)):
            for item in cohort:
                item["role"] = role
                selected.append(item)
        audits.setdefault(dataset, {}).update(
            source_rows={key: len(value) for key, value in raw.items()},
            selected=dict(SIZES),
        )

    clinc_selected = [r for r in selected if r["dataset"] == "clinc"]
    excluded = {r["group"] for r in clinc_selected}
    retrieval = [r for r in allrows["clinc"]["train"] if r["gold"] != "oos" and r["group"] not in excluded]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, norm="l2")
    matrix = vectorizer.fit_transform([r["state"] for r in retrieval])
    labels = sorted({r["gold"] for r in retrieval})
    if len(labels) != 150:
        raise ValueError("All 150 training intents required")
    by_label = {label: np.array([i for i, r in enumerate(retrieval) if r["gold"] == label]) for label in labels}
    for item in clinc_selected:
        started = time.perf_counter()
        similarity = (matrix @ vectorizer.transform([item["state"]]).T).toarray().ravel()
        ranking = sorted(labels, key=lambda label: (-float(similarity[by_label[label]].max()), label))[:8]
        item["retrieval_ms"] = 1000 * (time.perf_counter() - started)
        ids = sorted(ranking + ["oos"], key=lambda label: hashlib.sha256(
            canonical(["jev18-options", item["group"], label]).encode()
        ).hexdigest())
        item["options"] = [{"id": label, "text": "Out of scope: no supported intent" if label == "oos" else label.replace("_", " ")} for label in ids]
        item["question"] = "Select the utterance intent from the retrieved candidates, or out of scope when no supported intent applies."
    audits["clinc"].update(
        retrieval_training_rows=len(retrieval), retrieval_intents=len(labels), shortlist=8,
        gold_forced_into_shortlist=False,
        scope="Retrieval-assisted full-intent classification, NOT an oracle 9-way classification. OOS proportion is 25%.",
    )

    public = []
    gold = []
    for item in sorted(selected, key=lambda r: (r["dataset"], r["role"], r["group"])):
        qid = f"{item['dataset']}:{item['group'][:20]}"
        request = {"id": qid, "state": item["state"], "question": item["question"], "options": item["options"]}
        option_ids = [option["id"] for option in request["options"]]
        public.append({"request": request, "sha256": hashlib.sha256(canonical(request).encode()).hexdigest(), "dataset": item["dataset"], "role": item["role"], "group": item["group"], "shard": len(public) % 12})
        gold.append({"id": qid, "dataset": item["dataset"], "role": item["role"], "group": item["group"], "source_id": item["source_id"], "source_split": item["source_split"], "subject": item["subject"], "label": item["gold"], "gold_index": option_ids.index(item["gold"]) if item["gold"] in option_ids else len(option_ids), "gold_in_candidates": item["gold"] in option_ids, "option_ids": option_ids, "retrieval_ms": item.get("retrieval_ms", 0.0)})

    expected_total = len(DATASETS) * sum(SIZES.values())
    if len(public) != expected_total or len({r["group"] for r in public}) != expected_total:
        raise ValueError("Fixture group overlap or incomplete expanded cohort")
    for dataset in DATASETS:
        for role, count in SIZES.items():
            if sum(r["dataset"] == dataset and r["role"] == role for r in public) != count:
                raise ValueError(f"wrong {dataset}/{role} count")
    (out / "requests.jsonl").write_text("".join(canonical(r) + "\n" for r in public))
    (out / "gold.jsonl").write_text("".join(canonical(r) + "\n" for r in gold))
    write(out / "manifest.json", {
        "version": "1.8.0",
        "revisions": REVISIONS,
        "audit": audits,
        "sources": sources,
        "cases": len(public),
        "sizes_per_dataset": SIZES,
        "leaderboards": LEADERBOARDS,
        "requests_sha256": hashlib.sha256((out / "requests.jsonl").read_bytes()).hexdigest(),
        "gold_sha256": hashlib.sha256((out / "gold.jsonl").read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((REPOSITORY_ROOT / "expanded_protocol.json").read_bytes()).hexdigest(),
        "pretraining_contamination_excluded": False,
    })
    print(json.dumps(audits, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(".cache/jev18"))
    parser.add_argument("--out", type=Path, default=Path("fixture-expanded"))
    args = parser.parse_args()
    prepare(download(args.cache), args.out)
