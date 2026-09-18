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


## What the estimation fixes first implied (superseded below)

At the time this experiment was written I read the inversion as an under-calibration problem: each
task holds thousands of further labelled training rows, the development and test groups would stay
untouched, and the cost would be about 3.4 times the inference. The calibration-size curve below
removes that reading, and the section after it states what replaces it.

## The calibration-size curve removes that recommendation

`jevbench_learning_curve.py` refits both controllers on random subsets of the 32 fit cases the
pilot already scored, thirty-two subsets per size, and scores every fit on the same 32 held-out
cases. Macro over the four tasks:

| Fit cases | Joint, in sample | Equality, in sample | Joint, held out | Equality, held out | Joint minus equality | Spread across subsets |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.0443 | 0.0748 | 0.2361 | 0.2513 | −0.0152 | 0.0659 |
| 8 | 0.0503 | 0.1063 | 0.2246 | 0.2289 | −0.0044 | 0.0536 |
| 16 | 0.0628 | 0.1312 | 0.2051 | 0.2078 | −0.0027 | 0.0453 |
| 24 | 0.0702 | 0.1441 | 0.2053 | 0.1943 | +0.0110 | 0.0422 |
| 32 | 0.0729 | 0.1434 | 0.2027 | 0.1769 | +0.0258 | 0.0000 |

Three things are visible. Both controllers get better on held-out cases as calibration grows
(0.2361 to 0.2027 and 0.2513 to 0.1769), so more calibration data helps the pipeline. The
equality-only controller improves faster, so the difference moves the same way at every step and
changes sign somewhere between 16 and 24 cases: the joint controller starts ahead and ends behind.
And that movement is directional rather than resolved, because the spread across thirty-two subsets
is larger than the difference at every size below the full fit set.

Per task the same movement shows up: the joint controller is ahead on BoolQ, OCNLI and TMMLU+ at
four fit cases and behind on all four by 32, with CLINC positive throughout.

This supersedes the recommendation in the previous section. I had proposed enlarging the fit split
on the reading that the extra signals were under-calibrated. Within the range the pilot covers, the
opposite trend appears: calibration growth erodes the joint controller's advantage rather than
recovering it. A linear projection would put it far behind at 256 cases, which is a hypothesis and
not evidence, but the measured direction is enough to say that spending 3.4 times the inference on
more of the same calibration data is not the next experiment worth running.

What remains open is structural rather than about sample size: which observation of a Jev-like model
carries decision-relevant information that agreement does not already contain. Treating the model as
a competence signal for the decision layer, instead of the layer itself, is the version of that
question this evidence points at, and it is cheap to test on the records already published.


`jevbench_learning_curve.py` refits both controllers on random subsets of the 32 fit cases the
pilot already scored, eight subsets per size, and scores every fit on the same 32 held-out cases.
Macro over the four tasks:

## Scope

These are read-only calculations over the same verified artifacts as `verify_jevbench.py`: no new
inference, no model training, no proprietary Jev call. The joint variants reuse the analysis compiler
deliberately, and a test asserts that no second compiler exists in the experiment module, so the
comparison isolates estimation.
