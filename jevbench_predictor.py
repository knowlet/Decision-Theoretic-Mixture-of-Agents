"""A calibrated probability that the panel answer is correct, and what it buys at decision time.

The competence analysis left a specific obstacle: the confidence signal ranks better than agreement
alone but arrives miscalibrated, with worse log loss and ECE. This script tests the repair the
reviewed plan calls for first, turning those features into a usable probability and then judging it
where it matters, at the decision, not only as a score.

The decision rule is the one the declared loss implies. Answering wrongly costs 1 and deferring
costs 0.25, so returning the panel plurality is worthwhile exactly when its calibrated probability
of being correct exceeds 0.75. Each predictor is fitted on the 32 fit cases per task and applied to
the 32 held-out cases, and the comparison is paired at the case level.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from jevbench.analyze import DATASETS, NAMES, load
from jevbench_competence import binary_metrics, case_features

FEATURES = ("max_probability", "mean_probability", "min_probability", "mean_entropy")
DEFER = 0.25
ANSWER_THRESHOLD = 1.0 - DEFER


def case_table(gold, scores):
    rows = []
    for entry in gold:
        probs = np.array([scores[name][entry["id"]]["probabilities"] for name in NAMES])
        row = case_features(probs)
        row.update(dataset=entry["dataset"], id=entry["id"], role=entry["role"],
                   label=entry["label"], correct=int(row["plurality"] == entry["gold_index"]))
        rows.append(row)
    return pd.DataFrame(rows)


def agreement_rate(fit):
    rate = {}
    for support, group in fit.groupby("support"):
        rate[int(support)] = (group["correct"].sum() + 1.0) / (len(group) + 2.0)
    overall = (fit["correct"].sum() + 1.0) / (len(fit) + 2.0)
    return lambda frame: frame["support"].map(lambda s: rate.get(int(s), overall)).to_numpy(float)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logistic(fit):
    x = fit[list(FEATURES)].to_numpy(float)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(x), fit["correct"].to_numpy(int))
    return lambda frame: model.predict_proba(scaler.transform(frame[list(FEATURES)].to_numpy(float)))[:, 1]


def calibration_slice(fit, split=0.5, seed=160921):
    """Split the fit cases so the calibrator is fitted on cases the scorer did not see."""
    rng = np.random.default_rng(np.random.SeedSequence([seed]))
    order = rng.permutation(len(fit))
    cut = max(4, int(len(fit) * split))
    return fit.iloc[order[:cut]], fit.iloc[order[cut:]]


def constant_rate(outer, prior=0.5, weight=1.0):
    """Beta-smoothed constant for a calibration slice holding a single class, which does happen at
    32 fit cases. It cannot see the features, and is disclosed as such rather than hidden."""
    truth = outer["correct"].to_numpy(float)
    rate = float((truth.sum() + prior * weight) / (len(truth) + weight))
    return lambda frame: np.full(len(frame), rate)


def fit_platt(fit, split=0.5, seed=160921):
    """Fit the confidence model on part of the fit split, then calibrate its score on the rest."""
    inner, outer = calibration_slice(fit, split, seed)
    base = fit_logistic(inner)
    labels = outer["correct"].to_numpy(int)
    if len(set(labels.tolist())) < 2:
        return constant_rate(outer)
    calibrator = LogisticRegression(C=1.0, max_iter=5000)
    calibrator.fit(logit(base(outer)).reshape(-1, 1), labels)
    return lambda frame: calibrator.predict_proba(logit(base(frame)).reshape(-1, 1))[:, 1]


def fit_isotonic(fit, split=0.5, seed=160921):
    inner, outer = calibration_slice(fit, split, seed)
    base = fit_logistic(inner)
    labels = outer["correct"].to_numpy(int)
    if len(set(labels.tolist())) < 2:
        return constant_rate(outer)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(base(outer), labels)
    return lambda frame: np.clip(calibrator.predict(base(frame)), 1e-6, 1 - 1e-6)


def decision_loss(probability, correct, threshold=ANSWER_THRESHOLD, defer=DEFER):
    """Answer when the calibrated probability clears 1 - defer, otherwise withhold."""
    answered = np.asarray(probability, float) > threshold
    truth = np.asarray(correct, float)
    return np.where(answered, 1.0 - truth, defer)


def run(incoming: Path, fixture: Path, evaluation: Path, out: Path, draws: int = 2000) -> int:
    for label, root in (("incoming", incoming), ("fixture", fixture), ("evaluation", evaluation)):
        if not root.is_dir():
            raise SystemExit(f"missing {label} directory: {root}")
    manifest, requests, gold, scores, native, probes, provenance = load(incoming, fixture, evaluation)
    frame = case_table(gold, scores)
    metrics_rows, loss_rows, paired = [], [], []
    for dataset in DATASETS:
        block = frame[frame.dataset == dataset]
        fit, test = block[block.role == "fit"], block[block.role == "test"]
        predictors = {"agreement_smoothed": agreement_rate(fit), "confidence_logistic": fit_logistic(fit),
                      "platt_calibrated": fit_platt(fit), "isotonic_calibrated": fit_isotonic(fit)}
        per_case = {}
        for name, predict in predictors.items():
            probability = predict(test)
            truth = test["correct"].to_numpy()
            answered = probability > ANSWER_THRESHOLD
            among_answered = float(truth[answered].mean()) if answered.any() else float("nan")
            metrics_rows.append(dict(dataset=dataset, predictor=name, n=len(test),
                                     **binary_metrics(probability, truth),
                                     answered=float(answered.mean()),
                                     accuracy_at_answer_threshold=among_answered))
            per_case[name] = decision_loss(probability, test["correct"].to_numpy())
            loss_rows.append(dict(dataset=dataset, predictor=name, objective=float(per_case[name].mean())))
        baseline = per_case["agreement_smoothed"]
        for name in ("confidence_logistic", "platt_calibrated", "isotonic_calibrated"):
            difference = per_case[name] - baseline
            rng = np.random.default_rng(np.random.SeedSequence([160922, DATASETS.index(dataset)]))
            boot = difference[rng.integers(0, len(difference), (draws, len(difference)))].mean(1)
            paired.append(dict(dataset=dataset, a=name, b="agreement_smoothed",
                               objective_difference=float(difference.mean()),
                               lo=float(np.quantile(boot, 0.025)), hi=float(np.quantile(boot, 0.975))))
    metrics = pd.DataFrame(metrics_rows)
    losses = pd.DataFrame(loss_rows).groupby("predictor").objective.mean()
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "predictor_metrics.csv", index=False)
    losses.to_csv(out / "predictor_decision_loss.csv")
    pd.DataFrame(paired).to_csv(out / "predictor_comparisons.csv", index=False)
    print("=== predicting correctness and then deciding, macro over the four tasks ===")
    summary = metrics.groupby("predictor")[["brier", "nll", "ece", "accuracy", "auc", "answered"]].mean()
    summary["decision_objective"] = losses
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"=== decision loss against the smoothed agreement baseline, {int(ANSWER_THRESHOLD * 100)}% answer rule ===")
    comparisons = pd.DataFrame(paired).groupby(["a", "b"])[["objective_difference", "lo", "hi"]].mean()
    print(comparisons.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    for row in comparisons.itertuples():
        verdict = "resolved" if row.hi < 0 or row.lo > 0 else "not resolved"
        print(f"{row.Index[0]} against {row.Index[1]}: {verdict}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path, default=Path("incoming"))
    parser.add_argument("--fixture", type=Path, default=Path("fixture"))
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation"))
    parser.add_argument("--out", type=Path, default=Path("results/jev16-predictor"))
    args = parser.parse_args()
    raise SystemExit(run(args.incoming, args.fixture, args.evaluation, args.out))
