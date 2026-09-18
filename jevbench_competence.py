"""Does a Jev-like model distribution add anything beyond panel agreement?

The route question left by the calibration work is structural: is there an observation of a
Jev-like model that carries decision-relevant information answer agreement does not already
contain. This script answers the cheapest form of it from records the pilot already published, so
no new inference is needed.

For every scored case the three native heads give a distribution over the same options. Three
estimators of one event, "the panel plurality option is correct", are fitted on the 32 fit cases per
task and scored on the 32 held-out cases: agreement structure only, confidence features only, and
both together. Proper scoring rules decide whether the extra signal survives out of sample.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from jevbench.analyze import DATASETS, NAMES, load, metrics

CONFIDENCE_FEATURES = ("max_probability", "mean_probability", "min_probability", "mean_entropy")


def case_features(probs):
    """Agreement structure and confidence summaries for one case."""
    picks = probs.argmax(-1)
    counts = np.bincount(picks, minlength=probs.shape[-1])
    plurality = int(counts.argmax())
    support = int(counts[plurality])
    distinct = int((counts > 0).sum())
    agreeing = [m for m in range(probs.shape[0]) if picks[m] == plurality]
    agree_probs = probs[agreeing, plurality]
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=-1)
    normaliser = np.log(probs.shape[-1])
    return dict(plurality=plurality, support=support, distinct=distinct,
                max_probability=float(agree_probs.max()),
                mean_probability=float(agree_probs.mean()),
                min_probability=float(agree_probs.min()),
                mean_entropy=float((entropy / normaliser).mean()))


def agreement_key(row):
    return "support%d_distinct%d" % (row["support"], row["distinct"])


def fit_agreement(frame, prior=0.5, weight=2.0):
    """Smoothed empirical correctness rate per agreement pattern, the equality-only baseline."""
    table = {}
    for key, group in frame.groupby("agreement"):
        table[key] = (group["correct"].sum() + prior * weight) / (len(group) + weight)
    overall = (frame["correct"].sum() + prior * weight) / (len(frame) + weight)
    return lambda rows: np.array([table.get(row["agreement"], overall) for row in rows.to_dict("records")])


def fit_confidence(frame, features=CONFIDENCE_FEATURES):
    x = frame[list(features)].to_numpy(float)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(x), frame["correct"].to_numpy(int))
    return lambda rows: model.predict_proba(scaler.transform(rows[list(features)].to_numpy(float)))[:, 1]


def fit_combined(frame, features=CONFIDENCE_FEATURES, prior=0.5, weight=2.0):
    """Agreement offsets from the smoothed baseline, plus confidence, in one logistic fit."""
    base = fit_agreement(frame, prior, weight)
    frame = frame.copy()
    frame["offset"] = np.log(np.clip(base(frame), 1e-6, 1 - 1e-6) / (1 - np.clip(base(frame), 1e-6, 1 - 1e-6)))
    x = np.column_stack([frame["offset"].to_numpy(float), frame[list(features)].to_numpy(float)])
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(x), frame["correct"].to_numpy(int))
    def predict(rows):
        rows = rows.copy()
        rows["offset"] = np.log(np.clip(base(rows), 1e-6, 1 - 1e-6) / (1 - np.clip(base(rows), 1e-6, 1 - 1e-6)))
        x = np.column_stack([rows["offset"].to_numpy(float), rows[list(features)].to_numpy(float)])
        return model.predict_proba(scaler.transform(x))[:, 1]
    return predict


def brier_per_case(probabilities, truth):
    """Two-class Brier per case, matching the convention used by the metrics helper."""
    p = np.asarray(probabilities, float)
    return 2.0 * (p - truth) ** 2

def binary_metrics(probabilities, labels):
    p = np.asarray(probabilities, float)
    y = np.asarray(labels, int)
    stacked = np.column_stack([1 - p, p])
    values = metrics(stacked, y)
    top = p >= 0.5
    values["accuracy"] = float((top == y.astype(bool)).mean())
    order = np.argsort(p, kind="mergesort")  # ascending and stable
    ranks = np.empty(len(p), float)
    sorted_p = p[order]
    position = 0
    while position < len(p):  # average ranks inside a tie group, or ties inflate the statistic
        stop = position
        while stop + 1 < len(p) and sorted_p[stop + 1] == sorted_p[position]:
            stop += 1
        ranks[order[position:stop + 1]] = (position + stop) / 2 + 1
        position = stop + 1
    positives = y.sum()
    negatives = len(y) - positives
    if positives and negatives:
        values["auc"] = float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    else:
        values["auc"] = float("nan")
    values["prevalence"] = float(y.mean())
    return values


def run(incoming: Path, fixture: Path, evaluation: Path, out: Path, draws: int = 2000) -> int:
    for label, root in (("incoming", incoming), ("fixture", fixture), ("evaluation", evaluation)):
        if not root.is_dir():
            raise SystemExit(f"missing {label} directory: {root}")
    manifest, requests, gold, scores, native, probes, provenance = load(incoming, fixture, evaluation)
    rows, predictions = [], {}
    for dataset in DATASETS:
        entries = [g for g in gold if g["dataset"] == dataset]
        for entry in entries:
            qid = entry["id"]
            probs = np.array([scores[name][qid]["probabilities"] for name in NAMES])
            row = case_features(probs)
            row.update(dataset=dataset, id=qid, role=entry["role"], group=entry["group"],
                       label=entry["label"], correct=int(row["plurality"] == entry["gold_index"]))
            row["agreement"] = agreement_key(row)
            rows.append(row)
    frame = pd.DataFrame(rows)
    table, paired = [], []
    for dataset in DATASETS:
        block = frame[frame.dataset == dataset]
        fit = block[block.role == "fit"]
        test = block[block.role == "test"]
        estimators = {"agreement_only": fit_agreement(fit), "confidence_only": fit_confidence(fit),
                      "agreement_plus_confidence": fit_combined(fit)}
        if dataset == DATASETS[0]:
            print("agreement patterns present in the fit split:", sorted(fit["agreement"].unique()))
        for name, predict in estimators.items():
            probability = predict(test)
            values = binary_metrics(probability, test["correct"].to_numpy())
            table.append(dict(dataset=dataset, estimator=name, n=len(test), **values))
            predictions[(dataset, name)] = probability
        baseline = predictions[(dataset, "agreement_only")]
        for name in ("confidence_only", "agreement_plus_confidence"):
            truth = test["correct"].to_numpy(float)
            difference = brier_per_case(predictions[(dataset, name)], truth) - brier_per_case(baseline, truth)
            rng = np.random.default_rng(np.random.SeedSequence([160920, DATASETS.index(dataset)]))
            boot = difference[rng.integers(0, len(difference), (draws, len(difference)))].mean(1)
            paired.append(dict(dataset=dataset, a=name, b="agreement_only",
                               brier_difference=float(difference.mean()),
                               lo=float(np.quantile(boot, 0.025)), hi=float(np.quantile(boot, 0.975))))
    results = pd.DataFrame(table)
    macro = results.groupby("estimator")[["brier", "nll", "ece", "accuracy", "auc"]].mean()
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "competence_metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "competence_comparisons.csv", index=False)
    print()
    print("=== predicting whether the panel plurality option is correct, macro over four tasks ===")
    print(macro.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print("=== paired Brier difference against agreement-only (negative means better) ===")
    comparisons = pd.DataFrame(paired).groupby(["a", "b"])[["brier_difference", "lo", "hi"]].mean()
    print(comparisons.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    for row in comparisons.itertuples():
        verdict = "resolved" if row.hi < 0 or row.lo > 0 else "not resolved"
        print(f"{row.Index[0]}: {verdict}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path, default=Path("incoming"))
    parser.add_argument("--fixture", type=Path, default=Path("fixture"))
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation"))
    parser.add_argument("--out", type=Path, default=Path("results/jev16-competence"))
    args = parser.parse_args()
    raise SystemExit(run(args.incoming, args.fixture, args.evaluation, args.out))
