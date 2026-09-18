"""Sample-efficiency variants of the v1.6 acquisition controllers.

The v1.6 pilot fits its controllers on 32 cases per task and the fit-to-test diagnostic showed the
inversion that follows: the more parameters a controller has, the better it looks in sample and
the worse it does on held-out cases, while the 16-case development split that selects its prior
strength points at the wrong end of the grid.

This module asks a narrower question than "which model is best": with the same decision procedure
and the same data, can the calibration be made sample-efficient enough that the extra signals
(answer direction and confidence) stop costing more than they contribute? Each variant changes
only how the world is estimated; the Bellman compiler is the one already used by the analysis, so
any difference is attributable to estimation rather than to planning.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from jevbench.analyze import DATASETS, NAMES, load
from jevbench.control import AgreementWorld, JointWorld, observations

M = 3
STRENGTHS = (8.0, 32.0, 128.0)
KAPPA = 16.0
BOOTSTRAPS = 2000


def build_joint(probs, labels, bins=2, strength=32.0, missing=False, label_marginal="empirical",
                accuracy_shrink=None, pooled_accuracy=None):
    """Factorised joint over truth and the three observation channels.

    Mirrors the analysis estimator, with optional changes to how its parameters are estimated:
    a uniform label marginal drops k parameters per task, and accuracy shrinkage pulls each
    model accuracy toward a pooled value so 32 cases do not have to determine it alone.
    """
    p = np.asarray(probs, float)
    y = np.asarray(labels, int)
    obs = observations(p, bins)
    k = p.shape[2]
    a = k * bins
    yk = k + int(missing)
    n = len(p)
    if label_marginal == "empirical":
        counts = np.bincount(y, minlength=yk) + 0.5
        py = counts / counts.sum()
    elif label_marginal == "uniform":
        py = np.full(yk, 1.0 / yk)
    else:
        raise ValueError(label_marginal)
    likelihood = []
    accuracies = []
    for m in range(M):
        correct = obs[:, m] // bins == y
        accuracy = (correct.sum() + 1) / (n + 2)
        if accuracy_shrink is not None:
            pooled = accuracy_shrink if pooled_accuracy is None else pooled_accuracy
            if np.ndim(pooled):
                pooled = np.asarray(pooled, float)[m]
            accuracy = (n * accuracy + KAPPA * pooled) / (n + KAPPA)
        accuracies.append(float(accuracy))
        confidence = np.bincount(obs[:, m] % bins, minlength=bins) + 1
        confidence = confidence / confidence.sum()
        prior = np.zeros((yk, a))
        for truth in range(yk):
            for pred in range(k):
                q = 1 / k if truth == k else (accuracy if pred == truth else (1 - accuracy) / (k - 1))
                prior[truth, pred * bins:(pred + 1) * bins] = q * confidence
        counts = prior * 4
        np.add.at(counts, (y, obs[:, m]), 1.0)
        likelihood.append(counts / counts.sum(1, keepdims=True))
    joint = py[:, None, None, None] * likelihood[0][:, :, None, None] * likelihood[1][:, None, :, None] * likelihood[2][:, None, None, :]
    if np.isfinite(strength):
        joint = joint * strength
        np.add.at(joint, (y, obs[:, 0], obs[:, 1], obs[:, 2]), 1.0)
        joint = joint / joint.sum()
    return joint, k, yk, accuracies


def world_from(joint, k, bins, yk):
    """Reuse the analysis Bellman compiler so planning is held fixed across variants."""
    world = object.__new__(JointWorld)
    world.joint = joint
    world.k = k
    world.bins = bins
    world.a = k * bins
    world.yk = yk
    return world


def bagged_joint(probs, labels, bins, strength, missing, draws, seed, **kw):
    rng = np.random.default_rng(seed)
    joints = []
    for _ in range(draws):
        index = rng.integers(0, len(probs), len(probs))
        joint, k, yk, _ = build_joint(probs[index], labels[index], bins=bins, strength=strength, missing=missing, **kw)
        joints.append(joint)
    return np.mean(joints, axis=0), k, yk


def case_objectives(policy, obs, labels, index, agreement=False, defer=0.25):
    values = []
    for i in index:
        if agreement:
            pred, used, _ = policy.execute(lambda m, i=i: int(obs[i, m]), defer=defer)
        else:
            pred, used, _ = policy.execute(lambda m, i=i: int(obs[i, m]))
        values.append((defer if pred < 0 else float(pred != labels[i])) + 0.01 * len(used))
    return np.array(values)


def paired(frame, a, b, defer=0.25, draws=BOOTSTRAPS):
    samples, points = [], []
    for position, dataset in enumerate(DATASETS):
        x = frame[(frame.dataset == dataset) & (frame.algorithm == a) & (frame.defer == defer)].set_index("id")
        z = frame[(frame.dataset == dataset) & (frame.algorithm == b) & (frame.defer == defer)].set_index("id").reindex(x.index)
        difference = (x.objective - z.objective).to_numpy()
        rng = np.random.default_rng(np.random.SeedSequence([160918, position]))
        samples.append(difference[rng.integers(0, len(difference), (draws, len(difference)))].mean(1))
        points.append(difference.mean())
    boot = np.mean(samples, axis=0)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    family_lo, family_hi = np.quantile(boot, [0.00625, 0.99375])
    return dict(a=a, b=b, defer=defer, difference=float(np.mean(points)), lo=float(lo), hi=float(hi),
                family_lo=float(family_lo), family_hi=float(family_hi))


def run(incoming: Path, fixture: Path, evaluation: Path, out: Path) -> int:
    for label, root in (("incoming", incoming), ("fixture", fixture), ("evaluation", evaluation)):
        if not root.is_dir():
            raise SystemExit(f"missing {label} directory: {root}")
    manifest, requests, gold, scores, native, probes, provenance = load(incoming, fixture, evaluation)
    rows = []
    selected = []
    per_task = {}
    for dataset in DATASETS:
        entries = [g for g in gold if g["dataset"] == dataset]
        ids = [g["id"] for g in entries]
        probs = np.array([[scores[name][qid]["probabilities"] for name in NAMES] for qid in ids])
        labels = np.array([g["gold_index"] for g in entries])
        roles = np.array([g["role"] for g in entries])
        fit = np.flatnonzero(roles == "fit")
        per_task[dataset] = np.array([float((probs[fit, m].argmax(-1) == labels[fit]).mean()) for m in range(M)])
    for dataset in DATASETS:
        entries = [g for g in gold if g["dataset"] == dataset]
        ids = [g["id"] for g in entries]
        probs = np.array([[scores[name][qid]["probabilities"] for name in NAMES] for qid in ids])
        labels = np.array([g["gold_index"] for g in entries])
        roles = np.array([g["role"] for g in entries])
        fit, dev, test = np.flatnonzero(roles == "fit"), np.flatnonzero(roles == "dev"), np.flatnonzero(roles == "test")
        obs2, obs1 = observations(probs), observations(probs, bins=1)
        blind = AgreementWorld(probs[fit], labels[fit])
        blind_obs = probs.argmax(-1)
        pooled_accuracy = float(np.mean([(probs[fit, m].argmax(-1) == labels[fit]).mean() for m in range(M)]))
        variants = {}
        variants["agreement_only"] = (blind, blind_obs, True)
        for strength in STRENGTHS:
            joint, k, yk, _ = build_joint(probs[fit], labels[fit], 2, strength, dataset == "clinc")
            variants[f"joint_s{int(strength)}"] = (world_from(joint, k, 2, yk).compile(defer=0.25), obs2, False)
        for name, kw in (("uniform_label", dict(label_marginal="uniform")),
                         ("shrunk_accuracy", dict(accuracy_shrink=True, pooled_accuracy=pooled_accuracy))):
            joint, k, yk, _ = build_joint(probs[fit], labels[fit], 2, 32.0, dataset == "clinc", **kw)
            variants[name] = (world_from(joint, k, 2, yk).compile(defer=0.25), obs2, False)
        others = [per_task[d] for d in DATASETS if d != dataset]
        pooled_other = np.mean(others, axis=0)
        joint, k, yk, _ = build_joint(probs[fit], labels[fit], 2, 32.0, dataset == "clinc",
                                      accuracy_shrink=True, pooled_accuracy=pooled_other)
        variants["cross_task_accuracy"] = (world_from(joint, k, 2, yk).compile(defer=0.25), obs2, False)
        folds = np.array_split(np.arange(len(fit)), 4)
        cv_scores = []
        for strength in (8.0, 32.0, 128.0, float("inf")):
            fold_values = []
            for hold in folds:
                keep = np.setdiff1d(np.arange(len(fit)), hold)
                folded, folded_k, folded_yk, _ = build_joint(probs[fit][keep], labels[fit][keep], 2, strength, dataset == "clinc")
                policy = world_from(folded, folded_k, 2, folded_yk).compile(defer=0.25)
                fold_values.append(case_objectives(policy, obs2, labels, fit[hold]).mean())
            cv_scores.append((float(np.mean(fold_values)), strength))
        chosen = min(cv_scores)[1]
        chosen_joint, chosen_k, chosen_yk, _ = build_joint(probs[fit], labels[fit], 2, chosen, dataset == "clinc")
        variants["cv_strength"] = (world_from(chosen_joint, chosen_k, 2, chosen_yk).compile(defer=0.25), obs2, False)
        selected.append({"dataset": dataset, "chosen_strength": chosen, "cv_scores": cv_scores})
        joint, k, yk = bagged_joint(probs[fit], labels[fit], 2, 32.0, dataset == "clinc", 32, 160918 + len(rows))
        variants["bagged_32"] = (world_from(joint, k, 2, yk).compile(defer=0.25), obs2, False)
        for name, (policy, obs, agreement) in variants.items():
            values = case_objectives(policy, obs, labels, test, agreement=agreement)
            for value, j in zip(values, test):
                rows.append(dict(dataset=dataset, id=entries[j]["id"], algorithm=name, defer=0.25, objective=float(value)))
    frame = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "calibration_cases.csv", index=False)
    print("cross-validated strength per task:", [(row["dataset"], row["chosen_strength"]) for row in selected])
    macro = frame.groupby("algorithm").objective.mean().sort_values()
    print("=== macro objective across the four tasks (defer 0.25, 32 held-out cases each) ===")
    print(macro.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print("=== paired differences against agreement-only (positive means worse) ===")
    comparisons = [paired(frame, name, "agreement_only") for name in macro.index if name != "agreement_only"]
    table = pd.DataFrame(comparisons)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    table.to_csv(out / "calibration_comparisons.csv", index=False)
    print()
    crossings = [row["a"] for row in comparisons if row["lo"] <= 0 <= row["hi"]]
    print("variants whose interval includes zero:", crossings or "none")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path, default=Path("incoming"))
    parser.add_argument("--fixture", type=Path, default=Path("fixture"))
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation"))
    parser.add_argument("--out", type=Path, default=Path("results/jev16-calibration"))
    args = parser.parse_args()
    raise SystemExit(run(args.incoming, args.fixture, args.evaluation, args.out))
