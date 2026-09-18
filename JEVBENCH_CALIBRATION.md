# Calibration sample efficiency: what was tried, and what it showed

The v1.6 fit-to-test diagnostic found an inversion: the more parameters a controller carries, the
better it fits its 32 calibration cases and the worse it does on held-out cases, with the simple
equality-only controller showing the smallest gap. This note reports the follow-up experiment that
asked whether better estimation of the same signals closes that gap.

`jevbench_calibration.py` changes only how the world is estimated. The Bellman compiler is the one
already used by the analysis, reached through the same object, so planning is held fixed and any
difference is attributable to estimation. Every variant is selected and evaluated exactly as the
analysis does: fitted on the 32 fit cases per task, scored on the 32 held-out cases, deferral loss
0.25, with paired group bootstraps over the four tasks.

## Macro objective on the 128 held-out cases

| Variant | What it changes | J |
|---|---|---:|
| agreement-only | nothing: the equality-only baseline | **0.1769** |
| joint, strength 32 | nothing: the analysis estimator | 0.2027 |
| joint, uniform label marginal | removes the task label marginal, k parameters per task | 0.2048 |
| joint, strength 128 | stronger shrinkage of the data term | 0.2065 |
| joint, accuracy shrunk within task | pulls each model accuracy toward the task mean | 0.2070 |
| joint, bagged over 32 resamples | averages the fitted world over bootstrap resamples | 0.2125 |
| joint, cross-validated strength | picks strength by 4-fold cross-validation inside the fit set | 0.2126 |
| joint, accuracy shrunk across tasks | pulls accuracy toward the other three tasks | 0.2138 |
| joint, strength 8 | weaker shrinkage of the data term | 0.2177 |

None of the estimation fixes beats the equality-only controller on the point estimate, and every
paired interval against it includes zero, so the data also do not license the claim that the joint
controller is worse. What the table does say is that no amount of care in estimating the extra
signals recovered the gap at this sample size.

## The strength selection is not the lever

Cross-validation inside the fit set chose different strengths per task (BoolQ 8, OCNLI infinite,
CLINC 32, TMMLU+ 32) and still landed at 0.2126, worse than the fixed 32 that the development split
happened to choose. Shrinking all the way to the prior, which the earlier sweep showed is the best
out-of-sample setting for the joint model, still leaves it behind agreement-only. So the difficulty
is not that a 16-case development split picks the wrong strength; the direction and confidence
observation model itself is not identifiable from 32 calibration cases.

## What this implies for the next step

The limiting factor is calibration data, not the estimator. The four tasks each hold far more
labelled rows than the 32 used for fitting: BoolQ about 9,400 in train, OCNLI tens of thousands,
CLINC 7,500 in scope plus 100 out-of-scope in train, TMMLU+ per-subject train splits. Enlarging the
fit split is therefore a calibration-only change: the development and test groups stay exactly as
they are, so it can be declared before any test outcome is examined, which is the condition for it
to be a legitimate revision rather than a post-hoc search.

The cost is inference, not analysis. Scoring 256 fit cases per task instead of 32 multiplies the
forwards by roughly 3.4, which the sharded runner can absorb by moving from twelve shards to about
twenty-four so each stays inside the current per-shard timeout. If the richer controller still fails
to beat the equality-only one after that, the honest conclusion changes from "under-calibrated" to
"the extra signals do not carry decision-relevant information at this scale", and the sensible role
for a Jev-like model becomes a competence signal feeding the decision layer rather than the layer
itself.

## Scope

These are read-only calculations over the same verified artifacts as `verify_jevbench.py`: no new
inference, no model training, no proprietary Jev call. The joint variants reuse the analysis compiler
deliberately, and a test asserts that no second compiler exists in the experiment module, so the
comparison isolates estimation.
