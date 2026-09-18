# Decision-Theoretic Mixture-of-Agents

## Latest: v1.4.0 — actual OpenJev inference, population transfer, and audited limits

**[Full English paper](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.4.0/paper.en.pdf)** · **[完整繁體中文論文](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.4.0/paper.zh-TW.pdf)** · [Versioned release](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/tag/v1.4.0)

[Successful validation/publication run](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/actions/runs/35311639545) · [All ten successful inference jobs and the original failed aggregate](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/actions/runs/35310006824) · [Full reproducibility bundle](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.4.0/reproducibility.zip) · [Verification](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.4.0/verification.json) · [Checksums](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.4.0/SHA256SUMS)

**The result does not establish a universally optimal MoA architecture.** On the 128-question pilot, a cheap cumulative-score control has the lowest point estimate; Bellman, static and myopic policies tie. Development-calibrated OpenJev 4B has no reliably established disadvantage versus the same-panel empirical selector after accounting for uncertainty, but its full fixed-three pipeline has higher loss than early-stopping Bellman under the declared objective. These are different comparisons, not interchangeable victory claims.

This version includes full CERA-MoA and TypeSafe Jev source analysis plus **actual local OpenJev selector forwards**, not a renamed traditional-classifier simulation. `TheoLeeCJ/openjev` is an independent open-model logits readout, **not proprietary TypeSafe Jev or its weights/training**. No proprietary Jev endpoint or CERA agent training was run.

## What actually ran

| Evidence layer | Executed scope | Important limit |
|---|---|---|
| v1.1 finite synthetic study | Explicit losses, correlated channels, matched controls, Bellman reference checks | Synthetic support and conditional optimality only |
| v1.2 RouterBench | 26,821 archived questions, four models, 5,398 held-out questions | Historical response selection; no new worker inference |
| v1.3/v1.4 ProEval | GSM8K, SVAMP, MMLU, StrategyQA: 5,156 questions, eight fixed model identities, 1,036 held-out questions; five files including a separate DICES audit | Pool replacement is not a within-agent training trajectory |
| v1.4 OpenJev pilot | 128 test + 64 development groups from those four binary datasets; 4B and 0.8B; 12 strategies; 474 actual local forward passes | One neural family, small reused benchmark subset, explicit CPU-accumulation variant |

The 474 completed-run forwards are 384 primary, 80 robustness probes, and ten per-worker warmups. Earlier owned feasibility probes and the incomplete superseded native-CPU attempt are not counted as completed benchmark evidence. Workers remain archived; newly generating their answers, multi-agent debate, and human preference elicitation were not performed.

**254 regression tests pass with zero failures/skips.** Each of ten inference shards also passes the same 11 upstream tests and 17 interface tests; repeated executions are not counted as new unique tests. The full ProEval experiment was rerun twice with all 28 non-timing outputs passing comparison. After downloading the published artifact, the 254-test suite also passed locally. These are computational checks, not independent scientific replication.

Inference commit: `ee971cc25e1475ac16375886c135fe155a0dc803`. Successful analysis/reporting commit: `3cd46cc160e847336e161ad8eafb735c493ed5fe`. They are deliberately recorded separately.

## Actual head-to-head benchmark

The four datasets contribute 32 held-out unique groups each. The fixed-three empirical selector, panel majority and OpenJev see the same acquired candidate panel. OpenJev additionally interprets task text and receives training-derived conditional candidate error estimates, never current-task gold. Bellman/static/myopic may instead acquire fewer answers. Therefore the fixed-panel comparison tests selection, while the policy-level comparison also tests acquisition and stopping.

`J = wrong-answer loss (1) or deferral loss (0.25) + 0.01 × acquired worker answers`.

The 0.01 is an assumed loss charge, **not dollars**. Selector compute is excluded from this headline objective and separately disclosed; this exclusion favors neural add-ons. Accuracy below is **among answered cases**, not all questions.

| Algorithm | J ↓ | Coverage | Accuracy among answered | Mean queries |
|---|---:|---:|---:|---:|
| Cumulative-score prompt control | 0.089219 | 95.31% | 93.44% | 1.500 |
| Bellman | 0.093125 | 93.75% | 93.33% | 1.500 |
| Myopic / static | 0.093125 | 93.75% | 93.33% | 1.500 |
| Single model | 0.095938 | 100.00% | 91.41% | 1.000 |
| Disagreement trigger | 0.098750 | 93.75% | 93.33% | 2.063 |
| Empirical fixed-three selector | 0.108125 | 93.75% | 93.33% | 3.000 |
| Same-panel majority | 0.123750 | 100.00% | 90.63% | 3.000 |
| OpenJev 4B, development-calibrated | 0.127656 | 95.31% | 90.98% | 3.000 |
| OpenJev 4B, raw | 0.135469 | 92.19% | 90.68% | 3.000 |
| OpenJev 0.8B, development-calibrated | 0.139375 | 100.00% | 89.06% | 3.000 |
| OpenJev 0.8B, raw | 0.203828 | 61.72% | 87.34% | 3.000 |

The cumulative-score control is TF-IDF/ridge plus a score-prefix rule. **It is not CERA-MoA**, and its lowest point estimate is exploratory, not a preregistered superiority claim. Complete 12-strategy records, including the distinct top-three majority and prompt-top-one controls, are in [summary.csv](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.4.0/summary.csv).

### Paired uncertainty, not just the best row

Differences are first minus second; positive is worse for the first method. Intervals condition on fitted policies and the stored inference records.

| Contrast | Difference in J | 95% paired interval | 98.75% multiple-comparison interval |
|---|---:|---|---|
| 4B raw − fixed-three | +0.027344 | [0.003906, 0.054688] | [−0.001953, 0.060547] |
| 4B calibrated − fixed-three | +0.019531 | [−0.001953, 0.044922] | [−0.007812, 0.052734] |
| 4B calibrated − Bellman | +0.034531 | [0.013047, 0.059922] | [0.007187, 0.067734] |
| 0.8B raw − 4B raw | +0.068359 | [0.019531, 0.117188] | [0.003906, 0.127942] |

The calibrated same-panel contrast crosses zero. That is not proof of equivalence. The Bellman pipeline comparison retains a positive interval, but **0.015 of its 0.034531 point difference is exactly the extra 1.5 worker queries**; the remaining 0.019531 is terminal loss. It cannot be advertised as pure semantic-selector superiority. [Exact contrasts](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.4.0/comparisons.csv).

### Calibration and order sensitivity

For 4B, treating the highest candidate option mass as a correctness proxy gives Brier 0.205278 and ECE 0.309274. A 64-case development success calibrator gives 0.084703 and 0.051613. The raw option score was not originally guaranteed to estimate correctness; this is not evidence about proprietary Jev's calibration. The development-calibrated decision also changes coverage.

The paper's action-ID reversal probe changes **31/32 4B choices and 28/32 0.8B choices**. These are not 31 or 28 factual failures: multiple candidate IDs can contain the same answer. A post-release local audit of the already-published records, reproducible with `audit_openjev_answer_order.py`, finds **7/32 4B and 23/32 0.8B answer-or-deferral changes**. Six and 22 respectively change deferral status. This diagnostic was not used to select methods or tune the protocol; it is separate from the originally reported action-ID metric.

### CPU and cross-host precision caveats

Measured selector median/p95 times are **23.307/34.533 seconds for 4B** and **2.452/3.617 seconds for 0.8B**, over different hosted CPU mixtures. They are not GPU or TypeSafe-service timings, nor a hardware-controlled capacity speed comparison. Arithmetic controller evaluation is roughly tens of microseconds per replayed case, excluding offline fitting; it is not an LLM-equivalent reasoning benchmark. The calibrated model's tiny extra classifier computation was not separately timed. A hypothetical selector-overhead loss grid is included rather than inventing a dollar conversion.

The unchanged upstream readout uses a disclosed CPU adapter: BF16 stored weights/outputs with FP32 accumulation for Linear/Conv1d. It is not bit-identical to native BF16. Details: [runtime amendment](OPENJEV_RUNTIME_CHANGELOG.md).

All ten inference jobs succeeded, but original aggregation correctly failed when cross-CPU refitting serialized last-bit candidate error estimates differently. **Every one of 464 recorded request hashes was subsequently recovered exactly** while holding all other fields fixed. 106 records needed recovery; the largest actual difference was 2 ULP / 1.11e-16. Between model sizes, 17/128 test requests had different literal decimal strings. Those strings can tokenize differently: the pilot matches tasks/panels but does not claim byte-identical numeric inputs or prove the rounding behavior irrelevant. The failed run is retained, not relabeled successful. [Recovery method](OPENJEV_REQUEST_RECOVERY.md) · [Recorded audit](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.4.0/numeric_recovery_summary.json).

## What the added ProEval experiment found

On all 1,036 held-out binary questions, refitted Bellman/static/myopic policies have identical per-case objective values; the four-dataset macro J is 0.089962. Frozen Bellman is 0.104914. Updating after pool replacement improves J by 0.014952, conditional 95% interval [0.009910, 0.020046]. Limited 50/200-question recalibration can instead worsen outcomes; its four-answer-per-question calibration charge is reported.

DICES is quarantined from headline binary comparisons: 439/1,500 rows violate the documented common label-range filter for the selected models, and the retained 1,061 rows have only 308 exact question groups. Its all-deferral result does not validate human preference decisions. Exact numerical normalization also exposed a machine-precision serialization issue in SVAMP; the secondary tolerance audit is reported without rewriting primary labels.

## Reading CERA-MoA and TypeSafe Jev correctly

[CERA-MoA](https://arxiv.org/abs/2609.18779) co-trains agents and router heads, using frozen mid-layer query features, reward-aligned familiarity and cumulative-prefix allocation. It has actual agent-training evidence that this archive replay does not reproduce. Its reported open-ended aggregator selects the most familiar agent's completion, not a generative synthesis of all candidates.

[TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) supplies a typed probabilistic decision interface. Type-valid does not mean semantically correct, and action-choice mass is not automatically action-success probability. Our actual OpenJev results do not benchmark that proprietary model. [Detailed primary-source comparison](LITERATURE_v1.3.md).

The paper separates competence estimation, external utility, legal-action authorization and acquisition strategy. Conditional Bellman and approximation/margin bounds are stated with assumptions. An additional dependency proposition shows that positively priced outputs never read by a final decision cannot help under action-invariant/no-side-effect assumptions. None proves universal MoA superiority or refutes co-evolving training.

## Reproduce and inspect

For the **published records**, use Actions → **Verify OpenJev shards and publish audited paper** → Run workflow. It verifies the ten immutable successful inference artifacts, reruns the analysis and population-transfer tests, and produces papers. It does not regenerate neural outputs or overwrite releases.

The older neural inference workflow and its original strict aggregate check are preserved with their failed aggregate record. Do not mistake it for the recovery workflow or expect a fresh cross-CPU run to avoid the documented serialization issue. A subsequent experiment should precompute canonical request fixtures once for all workers. See [inference scope](RUNNING_OPENJEV.md), [population transfer](RUNNING_TRANSFER.md), and [request recovery](OPENJEV_REQUEST_RECOVERY.md).

To inspect/recompute the release, extract `reproducibility.zip`, restore `verification/source.tar.gz`, then overlay the analysis/reporting source archive. Use Python 3.13 and `requirements-transfer.txt`; `recover_openjev_analysis.py --download` reuses the included `incoming` inference records. No neural weights are needed for analysis. For a source folder that also contains `vendor` or extracted duplicate source trees, use an explicit pytest file list to avoid duplicate test discovery.

## Earlier evidence and disclosure

[v1.3 population transfer](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/tag/v1.3.0) · [v1.2 RouterBench](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/tag/v1.2.0) · [v1.1 synthetic report](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/tag/v1.1.0)

A public GitHub technical report is not peer review, venue acceptance, arXiv deposit, DOI registration or independent replication. AI assistance was used for implementation, writing and checking. No institutional endorsement is claimed. Model weights, raw benchmark text and font files are not redistributed; third-party rights remain separate. The owner has not selected a blanket reuse license for this project.
