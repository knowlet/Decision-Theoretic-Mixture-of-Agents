# v1.8 expanded Jev-like benchmark

This document describes the v1.8 candidate benchmark. It is separate from the
verified v1.7 pilot, so increasing the cohort cannot rewrite the earlier
evidence. The latest verified result remains v1.7 until the GitHub Actions run
for this workflow passes all gates.

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

## Workflow and status

The workflow is .github/workflows/jevlike-expanded.yml. It builds a public
gold-free fixture, runs the four native checkpoints across 48 shards, and then
runs the fail-closed analyzer. It does not publish a release automatically.
Until a successful Actions run produces the analysis artifact, this document
intentionally contains no accuracy or latency result.

The benchmark does not run proprietary TypeSafe Jev, reproduce CERA-MoA
co-training, generate new worker-agent answers, or establish universal MoA
superiority. RLCD remains a separate constrained-generation smoke test and is
not part of this native typed-choice leaderboard.
