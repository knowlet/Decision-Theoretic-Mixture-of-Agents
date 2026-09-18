"""Calibration-size learning curve from the existing evidence.

The enlarged-fit plan costs about 3.4 times the inference of the pilot, so this script asks the
cheaper question first: using only the 32 fit cases the pilot already scored, refit each controller
on random subsets of size 4 to 32 and watch its held-out objective. If the richer controller is
converging toward the equality-only one as calibration grows, the larger split is worth running; if
the curve is flat, more of the same data will not buy anything and the run is not justified.

Subsets are drawn from the fit split only. The development and test groups are never used for
fitting, and the test cases are identical across all subset sizes, so the curve is paired.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from jevbench.analyze import DATASETS, NAMES, load
from jevbench.control import AgreementWorld, observations
from jevbench_calibration import build_joint, case_objectives, world_from

SIZES = (4, 8, 16, 24, 32)
REPEATS = 32
STRENGTH = 32.0


def subsets(fit, size: int, repeats: int, salt: int):
    """Random subsets of the fit split, reproducible, never touching another role."""
    return [np.random.default_rng(np.random.SeedSequence([salt, size, repeat])).choice(fit, size=size, replace=False)
            for repeat in range(repeats)]


def run(incoming: Path, fixture: Path, evaluation: Path, out: Path) -> int:
    for label, root in (("incoming", incoming), ("fixture", fixture), ("evaluation", evaluation)):
        if not root.is_dir():
            raise SystemExit(f"missing {label} directory: {root}")
    manifest, requests, gold, scores, native, probes, provenance = load(incoming, fixture, evaluation)
    rows = []
    for dataset in DATASETS:
        entries = [g for g in gold if g["dataset"] == dataset]
        ids = [g["id"] for g in entries]
        probs = np.array([[scores[name][qid]["probabilities"] for name in NAMES] for qid in ids])
        labels = np.array([g["gold_index"] for g in entries])
        roles = np.array([g["role"] for g in entries])
        fit, test = np.flatnonzero(roles == "fit"), np.flatnonzero(roles == "test")
        obs = observations(probs)
        for size in SIZES:
            test_values = {"joint": [], "agreement_only": []}
            gap_values = []
            fit_values = {"joint": [], "agreement_only": []}
            for repeat, subset in enumerate(subsets(fit, size, REPEATS, 160919 + DATASETS.index(dataset))):
                joint, k, yk, _ = build_joint(probs[subset], labels[subset], 2, STRENGTH, dataset == "clinc")
                policy = world_from(joint, k, 2, yk).compile(defer=0.25)
                blind = AgreementWorld(probs[subset], labels[subset])
                blind_obs = probs.argmax(-1)
                test_values["joint"].append(case_objectives(policy, obs, labels, test).mean())
                test_values["agreement_only"].append(case_objectives(blind, blind_obs, labels, test, agreement=True).mean())
                fit_values["joint"].append(case_objectives(policy, obs, labels, subset).mean())
                fit_values["agreement_only"].append(case_objectives(blind, blind_obs, labels, subset, agreement=True).mean())
                gap_values.append(test_values["joint"][-1] - test_values["agreement_only"][-1])
            rows.append(dict(dataset=dataset, size=size, repeats=REPEATS,
                             joint_test=float(np.mean(test_values["joint"])),
                             joint_test_sd=float(np.std(test_values["joint"])),
                             agreement_test=float(np.mean(test_values["agreement_only"])),
                             joint_fit=float(np.mean(fit_values["joint"])),
                             agreement_fit=float(np.mean(fit_values["agreement_only"])),
                             gap_mean=float(np.mean(gap_values)), gap_sd=float(np.std(gap_values))))
    frame = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "learning_curve_cases.csv", index=False)
    macro = frame.groupby("size")[["joint_fit", "agreement_fit", "joint_test", "agreement_test"]].mean()
    spread = frame.groupby("size")[["gap_mean", "gap_sd"]].mean()
    macro["joint_minus_agreement_test"] = spread["gap_mean"]
    macro["gap_sd"] = spread["gap_sd"]
    print("=== macro over the four tasks, by calibration size ===")
    print(macro.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print("=== per task, joint minus agreement-only on the 32 held-out cases ===")
    pivot = frame.pivot_table(index="size", columns="dataset", values=["joint_test", "agreement_test"])
    gap = (pivot["joint_test"] - pivot["agreement_test"]).round(4)
    print(gap.to_string())
    print()
    first, last = macro["joint_minus_agreement_test"].iloc[0], macro["joint_minus_agreement_test"].iloc[-1]
    slope = (last - first) / (SIZES[-1] - SIZES[0])
    print(f"gap at {SIZES[0]} fit cases: {first:+.4f}; at {SIZES[-1]}: {last:+.4f}; slope {slope:+.5f} per case")
    print(f"mean within-size spread across {REPEATS} subsets: {float(np.mean(macro["gap_sd"])):.4f}")
    monotone = all(b >= a - 1e-12 for a, b in zip(macro["joint_minus_agreement_test"], macro["joint_minus_agreement_test"].iloc[1:]))
    print("gap increases monotonically with calibration size:", monotone)
    print("gap exceeds one within-size spread at every size:", bool((macro["joint_minus_agreement_test"].abs() > macro["gap_sd"]).all()))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path, default=Path("incoming"))
    parser.add_argument("--fixture", type=Path, default=Path("fixture"))
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation"))
    parser.add_argument("--out", type=Path, default=Path("results/jev16-learning-curve"))
    args = parser.parse_args()
    raise SystemExit(run(args.incoming, args.fixture, args.evaluation, args.out))
