# v1.7 Jev-like extension — verified Actions evidence

This candidate adds three pinned native decision checkpoints to the same
gold-blind semantic fixture used by the v1.6 pilot:

- Mapika/decider-2b at revision b37f7e1ba3fbc9238004cf531fabbee2619973fd.
- jaredpalmer/kev-0.8b at revision c917edefdfd72b3e9ba71455584700acc70595f6, with the upstream source pinned to e0bcf50153f1bda4ca6a8be5e12cbd5f9ebbce1c.
- convaiinnovations/laya-typed-decisions at revision f9ab0b228f0fc0f14d873dbc99038f135c2da1b2, with the upstream source pinned to 42626c348753fbb17572a813127df2278a1ec527.

The native cohort uses 320 semantic cases: 32 fit, 16 development, and 32
held-out test cases per BoolQ, OCNLI, CLINC, and TMMLU+. Each model receives the
same request file and never receives gold.jsonl. Each model produces 320 primary
records, 16 reverse probes, and 8 repeat probes, plus 12 warmups (356 total
forward passes per model). The workflow records native probabilities, accuracy,
Brier score, NLL, ECE, p50/p95 CPU latency, token counts, and option-order/repeat
probes. The results below come from the successful GitHub Actions run and are
kept separate from the v1.6 OpenJev pilot.

harshatheg/Qwen-2.5-1B-RLCD is included in a separate schema smoke job. Its
public implementation is a constrained-generation engine around
Qwen/Qwen2.5-1.5B-Instruct, with the PyTorch backend selected on the Ubuntu
runner. Its JSON validity, schema match, exact-match rate, forward count, and
latency are reported separately from native typed decision heads; they are not
accuracy numbers on the BoolQ/OCNLI/CLINC/TMMLU+ fixture.

The workflow is at .github/workflows/jevlike-extended.yml.
The immutable model registry is at jevbench/extended_registry.py.

## Verified run

The evidence run is [Actions run 35577340525](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/actions/runs/35577340525), executed from commit
`85f3bd11273024273732873ebf9da77187cb16b8`. Its fixture, native shards,
analysis job, and RLCD job completed successfully. The native verifier reports:

- 3 models × 12 shards, with 960 primary predictions and 384 held-out test predictions.
- 1,032 evidence forwards, 72 reverse/repeat probes, 36 warmups, and 1,068 total forwards.
- The same semantic request hashes across models, isolated gold labels, zero new API calls, and zero training steps.
- Native analysis artifact SHA-256: `3a8536f3dd227f675e379e08c64c713da7adaad73e54c723028f2bb4f6298e88`.
- RLCD artifact SHA-256: `eb4749c916cdb13768ed30a6279e9a51b1755e84f31e281923981d1f6451c041`.

The artifact contains `native_summary.csv`, `primary_predictions.csv`, and
`verification.json`; the separate RLCD artifact contains `rlcd_smoke.json`.

## Native held-out results

The headline accuracy is calculated on the 128 held-out test cases per model,
32 per dataset. Latencies are wall-clock CPU measurements from the recorded
test forwards. Brier, NLL, and ECE use the native probability output; ECE is
recomputed over the complete 128-case test cohort with ten fixed bins.

| Model | BoolQ | OCNLI | CLINC | TMMLU+ | Overall test accuracy | CPU p50 (ms) | CPU p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decider-2B | 78.1% | 78.1% | **90.6%** | 43.8% | **72.7% (93/128)** | 4,379.3 | 6,663.5 |
| Kev-0.8B | 75.0% | 62.5% | 68.8% | 34.4% | 60.2% (77/128) | 935.6 | 1,996.0 |
| Laya typed decisions | **81.3%** | 50.0% | 78.1% | 18.8% | 57.0% (73/128) | **802.6** | **1,682.3** |

The overall test metrics are:

| Model | Brier ↓ | NLL ↓ | ECE ↓ | Mean input tokens | Test forwards |
|---|---:|---:|---:|---:|---:|
| Decider-2B | **0.364638** | **0.610054** | **0.079744** | 116.0 | 128 |
| Kev-0.8B | 0.568567 | 1.175163 | 0.178401 | 113.7 | 128 |
| Laya typed decisions | 0.528354 | 0.924733 | 0.102875 | 149.0 | 128 |

The per-dataset calibration and latency records are retained in the published
`native_summary.csv`. Rounded values from that file are:

| Model | Dataset | Accuracy | Brier | NLL | ECE | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Decider-2B | BoolQ | 78.1% | 0.303278 | 0.434016 | 0.138912 | 5,035.7 | 8,303.8 |
| Decider-2B | CLINC | 90.6% | 0.102866 | 0.187490 | 0.070219 | 4,381.6 | 4,534.7 |
| Decider-2B | OCNLI | 78.1% | 0.384275 | 0.638306 | 0.130442 | 4,359.2 | 4,501.0 |
| Decider-2B | TMMLU+ | 43.8% | 0.668134 | 1.180404 | 0.239636 | 4,373.2 | 7,296.9 |
| Kev-0.8B | BoolQ | 75.0% | 0.412130 | 0.823689 | 0.188348 | 1,198.6 | 2,484.6 |
| Kev-0.8B | CLINC | 68.8% | 0.502325 | 1.271485 | 0.163436 | 1,003.2 | 1,131.9 |
| Kev-0.8B | OCNLI | 62.5% | 0.496863 | 0.873624 | 0.165702 | 859.4 | 959.2 |
| Kev-0.8B | TMMLU+ | 34.4% | 0.862949 | 1.731853 | 0.275662 | 926.1 | 2,094.8 |
| Laya typed decisions | BoolQ | 81.3% | 0.310361 | 0.476770 | 0.067525 | 794.0 | 1,565.1 |
| Laya typed decisions | CLINC | 78.1% | 0.325911 | 0.653337 | 0.178204 | 685.8 | 778.3 |
| Laya typed decisions | OCNLI | 50.0% | 0.698159 | 1.117961 | 0.215665 | 936.8 | 1,193.0 |
| Laya typed decisions | TMMLU+ | 18.8% | 0.778985 | 1.450866 | 0.145394 | 958.5 | 2,145.8 |

These are small, fixed transfer cohorts. The registry marks all three native
models as English, so the OCNLI and TMMLU+ rows should be read as transfer
measurements with language and domain mismatch, not as a balanced multilingual
leaderboard. CPU p50/p95 values are runner measurements and are not GPU,
production-service, or proprietary Jev latency.

## RLCD schema smoke

`harshatheg/Qwen-2.5-1B-RLCD` is evaluated separately because it performs
parallel constrained generation around `Qwen/Qwen2.5-1.5B-Instruct`; it does
not expose the native typed-choice decision head used by the three-model
cohort. On three hand-authored schema cases, the pinned PyTorch run produced:

| Cases | Valid JSON/schema | Exact match | Mean wall latency |
|---:|---:|---:|---:|
| 3 | 100% | 66.7% (2/3) | 1,528.8 ms |

This smoke test checks schema execution and latency only. Its exact-match rate
must not be compared with the BoolQ, OCNLI, CLINC, or TMMLU+ accuracy values.

## Interpretation and limits

Decider-2B has the highest held-out accuracy and the best aggregate calibration
metrics in this cohort, while Laya is the fastest on the recorded CPU test
cohort. That is a descriptive result under one fixture and one runtime; it is
not evidence that either source project is universally superior, that a typed
decision head improves reasoning, or that any of these checkpoints reproduces
proprietary Jev. The benchmark did not run live worker generation, CERA-MoA
co-training, a human preference study, or a production GPU service.

The GitHub sources are pinned separately from the deployable checkpoints:
`jaredpalmer/kev` is evaluated through `jaredpalmer/kev-0.8b`, with the source
tree pinned in the provenance; `NandhaKishorM/laya` is evaluated through
`convaiinnovations/laya-typed-decisions`, again with the source tree pinned.
This keeps the requested projects in the test protocol while making the exact
weights and source revisions reproducible.
