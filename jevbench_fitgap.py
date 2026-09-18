"""Fit-to-test generalisation diagnostic for the fitted acquisition policies.

Reads the verified v1.6 evidence, refits each policy exactly as `jevbench.analyze` does, and
then scores the same fitted policy on the 32 fit cases it saw, the 16 development cases used
for strength selection, and the 32 held-out test cases. Nothing here is a new method or a new
inference run: it exists to show where a fitted controller gains and loses, because a policy
that wins in sample and loses out of sample is a different problem from one that never wins.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from jevbench.adapters import MODELS
from jevbench.analyze import DATASETS, NAMES, load, objective
from jevbench.control import AgreementWorld, JointWorld, observations

GRID = (1.0, 4.0, 8.0, 32.0, 128.0, 512.0, float("inf"))


def build(dataset, requests, gold, scores):
    rows = [g for g in gold if g["dataset"] == dataset]
    ids = [g["id"] for g in rows]
    probs = np.array([[scores[name][qid]["probabilities"] for name in NAMES] for qid in ids])
    labels = np.array([g["gold_index"] for g in rows])
    roles = np.array([g["role"] for g in rows])
    return rows, probs, labels, roles


def run(incoming: Path, fixture: Path, evaluation: Path, out: Path) -> int:
    for label, root in (("incoming", incoming), ("fixture", fixture), ("evaluation", evaluation)):
        if not root.is_dir():
            raise SystemExit(f"missing {label} directory: {root}")
    for name in ("requests.jsonl", "manifest.json"):
        if not (fixture / name).exists():
            raise SystemExit(f"fixture is missing {name}; run the fixture job first")
    manifest, requests, gold, scores, native, probes, provenance = load(incoming, fixture, evaluation)
    gap_rows, sweep_rows = [], []
    for dataset in DATASETS:
        rows, probs, labels, roles = build(dataset, requests, gold, scores)
        fit = np.flatnonzero(roles == "fit")
        dev = np.flatnonzero(roles == "dev")
        test = np.flatnonzero(roles == "test")
        obs2 = observations(probs)
        obs1 = observations(probs, bins=1)
        blind = AgreementWorld(probs[fit], labels[fit])
        blind_obs = probs.argmax(-1)
        policies = {}
        for defer in (0.25,):
            policies[("agreement_only", defer)] = (blind, blind_obs, True, None)
            for name, bins, mode in (("direction_only", 1, "bellman"), ("joint_bellman", 2, "bellman"),
                                     ("joint_myopic", 2, "myopic")):
                choices = []
                for strength in GRID[:3] + (32.0, 128.0):
                    world = JointWorld(probs[fit], labels[fit], bins=bins, strength=strength,
                                       missing=dataset == "clinc")
                    policy = world.compile(defer=defer, mode=mode)
                    choices.append((objective(policy, obs1 if bins == 1 else obs2, labels, dev), strength, policy))
                dev_loss, strength, policy = min(choices, key=lambda r: r[:2])
                policies[(name, defer)] = (policy, obs1 if bins == 1 else obs2, False, strength)
        for (name, defer), (policy, obs, agreement, strength) in sorted(policies.items(), key=lambda kv: kv[0][0]):
            values = {}
            for role, index in (("fit", fit), ("dev", dev), ("test", test)):
                values[role] = objective(policy, obs, labels, index, agreement=agreement, defer=defer)
            gap_rows.append(dict(dataset=dataset, algorithm=name, defer=defer, strength=strength,
                                 fit_objective=values["fit"], dev_objective=values["dev"],
                                 test_objective=values["test"], test_minus_fit=values["test"] - values["fit"]))
        for strength in GRID:
            world = JointWorld(probs[fit], labels[fit], bins=2, strength=strength, missing=dataset == "clinc")
            policy = world.compile(defer=0.25, mode="bellman")
            sweep_rows.append(dict(dataset=dataset, strength=strength,
                                   fit_objective=objective(policy, obs2, labels, fit),
                                   dev_objective=objective(policy, obs2, labels, dev),
                                   test_objective=objective(policy, obs2, labels, test)))
    gaps = pd.DataFrame(gap_rows)
    sweep = pd.DataFrame(sweep_rows)
    out.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(out / "fit_test_gap.csv", index=False)
    sweep.to_csv(out / "equality_prior_sweep.csv", index=False)
    print("=== objective on the cases a policy was fitted on, on development, and on test ===")
    print(gaps.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("=== macro averages across the four tasks ===")
    macro = gaps.groupby(["algorithm", "defer"])[["fit_objective", "dev_objective", "test_objective", "test_minus_fit"]].mean()
    print(macro.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print("=== joint Bellman prior strength: in-sample against out-of-sample ===")
    pivot = sweep.groupby("strength")[["fit_objective", "dev_objective", "test_objective"]].mean()
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    best_dev = pivot["dev_objective"].idxmin()
    best_test = pivot["test_objective"].idxmin()
    print(f"lowest development objective at strength {best_dev}; lowest test objective at {best_test}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path, default=Path("incoming"))
    parser.add_argument("--fixture", type=Path, default=Path("fixture"))
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation"))
    parser.add_argument("--out", type=Path, default=Path("results/jev16-fitgap"))
    args = parser.parse_args()
    raise SystemExit(run(args.incoming, args.fixture, args.evaluation, args.out))
