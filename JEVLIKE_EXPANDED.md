# v1.8 expanded Jev-like benchmark

This document describes the v1.8 expanded benchmark. It is separate from the
verified v1.7 pilot, so increasing the cohort cannot rewrite the earlier
evidence. The protocol below has since run to completion; its verified results
are recorded further down, and the v1.7 pilot stays published unchanged.

## Declared protocol

Each dataset contributes 128 fit cases, 64 development cases, and 200 held-out
test cases. The four datasets therefore contain 1,568 semantic cases in total:

| Dataset | Fit | Development | Held-out test | Reporting group |
|---|---:|---:|---:|---|
| BoolQ | 128 | 64 | 200 | English |
| CLINC | 128 | 64 | 200 | English |
| OCNLI | 128 | 64 | 200 | OOD-Chinese |
| TMMLU+ | 128 | 64 | 200 | OOD-Chinese |

The English leaderboard pools 400 held-out cases per model from BoolQ and
CLINC. The OOD-Chinese leaderboard pools 400 held-out cases per model from
OCNLI and TMMLU+. Per-dataset rows are retained alongside the pooled tables.
These are descriptive aggregates with equal case weighting inside each group;
they are not a claim that the tasks measure one latent language ability.

OCNLI uses the labeled official development split because the official test
labels are not public. Its result must therefore be described as a labeled-dev
transfer measurement. CLINC uses a training-derived character TF-IDF shortlist
of eight in-scope intents plus an out-of-scope option; the gold label is never
inserted into the shortlist. TMMLU+ uses the pinned dataset files and a
subject-round-robin selection rule.

## Pinned native cohort

| Name | Checkpoint revision | Declared language scope | Role |
|---|---|---|---|
| Decider-2B | Mapika/decider-2b b37f7e1ba3fbc9238004cf531fabbee2619973fd | English | Native typed choice |
| Kev-0.8B | jaredpalmer/kev-0.8b c917edefdfd72b3e9ba71455584700acc70595f6 | English | Native typed choice |
| Laya typed decisions | convaiinnovations/laya-typed-decisions f9ab0b228f0fc0f14d873dbc99038f135c2da1b2 | English | Native typed choice |
| Laya multilingual | convaiinnovations/laya-multilingual 052592a15d198d9ad47da779604259b10b47b7aa | Multilingual, including Chinese and English | Native typed choice |

The three English checkpoints are intentionally retained in the OOD-Chinese
table as transfer baselines. Adding the multilingual Laya checkpoint does not
create a controlled language-only comparison: its encoder, checkpoint, training
data, and language coverage all differ from the English checkpoints. The
multilingual row is evidence about this fixed cohort, not a causal architecture
comparison or a general multilingual superiority claim.

## Inference and evidence accounting

All models receive the same canonical request fixture. Inference workers receive
requests and metadata but never gold.jsonl. Each model runs:

- 1,568 primary forwards;
- 16 reversed-option probes and 8 exact-input repeat probes;
- 12 warmups, one for each shard;
- 1,592 recorded evidence forwards and 1,604 total forwards including warmups.

The 12-shard matrix has four models and 48 native jobs. The analyzer refuses to
produce a result unless every model has all 12 shards, all 1,568 primary cases,
200 test cases for each dataset, the declared probe coverage, matching request
and protocol hashes, valid probability simplexes, pinned checkpoint revisions,
and zero external API calls or training steps.

The reported metrics are accuracy, Brier score, NLL, ECE, CPU p50/p95 latency,
input-token counts, and candidate-path counts. CPU latency is a hosted Ubuntu
runner measurement. It is not a GPU, production-service, or proprietary Jev
latency measurement.

## Verified results

The declared protocol ran in [Actions run 35587955356](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/actions/runs/35587955356)
from commit 66f7135c717034c235caac66e711937cb0b90599. All 50 jobs passed: the
fixture job, 48 native shards, and the fail-closed analyzer. The analysis
artifact jev18-analysis-35587955356 has SHA-256
456d7fb64f9972d4f6799ee5fc1a5b4580f477b9f1cb83978bbd4f51f07c139a and reports
6,272 primary predictions, 6,368 evidence forwards, 96 probe forwards, 48
warmups, 6,416 total forwards, isolated gold labels, and zero external API
calls or training steps.

### Leaderboards, 400 held-out cases each

| Model | English accuracy | 95% Wilson | OOD-Chinese accuracy | 95% Wilson | English p50 | OOD-Chinese p50 |
|---|---:|---|---:|---|---:|---:|
| Decider-2B | 88.75% (355/400) | [85.28%, 91.49%] | 55.25% (221/400) | [50.35%, 60.05%] | 4,340 ms | 4,264 ms |
| Laya typed decisions | 78.50% (314/400) | [74.21%, 82.24%] | 40.50% (162/400) | [35.80%, 45.38%] | 674 ms | 852 ms |
| Laya multilingual | 73.75% (295/400) | [69.23%, 77.82%] | 47.25% (189/400) | [42.41%, 52.15%] | 220 ms | 187 ms |
| Kev-0.8B | 75.25% (301/400) | [70.79%, 79.23%] | 39.00% (156/400) | [34.35%, 43.86%] | 1,007 ms | 848 ms |

Decider-2B leads both leaderboards. The multilingual Laya checkpoint is the
fastest model in this cohort and is 6.75 accuracy points above the English Laya
checkpoint on OOD-Chinese, but it is 4.75 points lower on English and much
worse calibrated, so the two Laya checkpoints trade places instead of one
dominating the other.

### Pooled accuracy and calibration, 800 held-out cases each

| Model | Pooled accuracy | 95% Wilson | Brier | NLL | ECE | CPU p50 | CPU p95 |
|---|---:|---|---:|---:|---:|---:|---:|
| Decider-2B | 72.00% (576/800) | [68.79%, 75.00%] | 0.3571 | 0.6980 | 0.0683 | 4,313 ms | 5,996 ms |
| Laya multilingual | 60.50% (484/800) | [57.07%, 63.83%] | 0.5777 | 1.3258 | 0.2022 | 201 ms | 389 ms |
| Laya typed decisions | 59.50% (476/800) | [56.06%, 62.85%] | 0.4907 | 0.9402 | 0.0460 | 714 ms | 1,381 ms |
| Kev-0.8B | 57.13% (457/800) | [53.67%, 60.51%] | 0.6016 | 1.3130 | 0.2188 | 922 ms | 1,622 ms |

The pooled Laya contrast is 1 accuracy point with heavily overlapping Wilson
intervals, so the pooled ordering between those two checkpoints is not resolved
by this cohort. Laya typed decisions is the best-calibrated model here (ECE
0.0460) while Kev-0.8B is the worst (ECE 0.2188).

### Per-dataset accuracy

| Model | BoolQ | CLINC | OCNLI | TMMLU+ |
|---|---:|---:|---:|---:|
| Decider-2B | 85.00% | 92.50% | 65.50% | 45.00% |
| Laya typed decisions | 77.00% | 80.00% | 55.00% | 26.00% |
| Laya multilingual | 68.50% | 79.00% | 63.50% | 31.00% |
| Kev-0.8B | 81.50% | 69.00% | 50.50% | 27.50% |

### What the larger cohort changed against the v1.7 pilot

Decider-2B pooled accuracy is essentially reproduced, 72.7% on the 128 pilot
cases against 72.00% on these 800. Its per-dataset estimates move substantially:
OCNLI falls from 78.1% to 65.50% and BoolQ rises from 78.1% to 85.00%. The
32-case pilot numbers were therefore noisy per task even when the pooled
estimate was stable, which is the reason for this cohort.

## Workflow and status

The workflow is .github/workflows/jevlike-expanded.yml. It builds a public
gold-free fixture, runs the four native checkpoints across 48 shards, and then
runs the fail-closed analyzer. It does not publish a release automatically.
The run above is the verified evidence for this document; the results are an
Actions record rather than a tagged release.

The benchmark does not run proprietary TypeSafe Jev, reproduce CERA-MoA
co-training, generate new worker-agent answers, or establish universal MoA
superiority. RLCD remains a separate constrained-generation smoke test and is
not part of this native typed-choice leaderboard.
