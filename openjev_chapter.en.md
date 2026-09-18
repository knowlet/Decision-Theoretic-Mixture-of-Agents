# Executed OpenJev head-to-head pilot

This section adds **new local neural inference** to the population-transfer evidence. It evaluates the independent `TheoLeeCJ/openjev` implementation at commit `b4782a6c953f05c6255706d7a219f4e032af5b58`, not proprietary TypeSafe Jev. The code reads native final-position logits for declared option tokens and normalizes them; it does not decode free text. The original `direct.score` is verified by Git blob `943f34728d1966bf2325d6e6d34d2a7043cd6a56`. CPU model loading replaces the upstream CUDA-only loader. BF16 weights and layer outputs are retained, with explicit FP32 accumulation in CPU Linear/Conv1d operations. This is not bit-identical to native BF16 CPU execution: an owned compatibility probe changed one option logit by 0.125 while retaining the selected action. The probe establishes feasibility, not a uniform precision guarantee. The backend is pinned and disclosed rather than treated as an upstream GPU reproduction.

## A common task, not an unrelated leaderboard

We retain the six binary ProEval datasets and replacement worker pool above. Within each original split, unique question groups are selected by a fixed salted hash: 32 test and 16 development groups per dataset, yielding 192 test and 96 development cases. Selection does not depend on difficulty, correctness, disagreement or confidence. The selection salt is unchanged from the earlier four-dataset pilot, so all 128 previously scored test cases are still selected: the earlier result is a subset of this one, not a replacement. They remain a subset of a single earlier corpus, **not 192 independently collected tasks**. The protocol preceded new selector scoring but followed v1.3 outcomes and owned runtime probes; this is not independent preregistration.

The **matched-panel comparison** gives empirical fixed-three, panel-majority and OpenJev the same three acquired worker answers, with the panel selected by training/development data. OpenJev additionally sees the task text and the same training-derived conditional candidate error estimates available to the empirical terminal rule. These estimates are not current-task gold. The empirical method does not interpret the task semantically, so representation and inductive bias still differ. This avoids comparing a calibrated incumbent against an uninformed neural comparator but does not equalize pretraining histories.

In the **policy-level comparison**, Bellman and other controllers can acquire fewer answers or stop. This tests loss and acquisition use, not a pure terminal-selector ablation. The pool and maximum budget are common, but a fixed-three selector always pays for three answers. No algorithm receives current gold, an unacquired response or a perfect verifier. Outputs are constrained to the acquired panel or deferral, never a newly invented answer.

## Algorithms and new inference

The main scorer is immutable Qwen3.5-4B, supported by the selected upstream implementation. Qwen3.5-0.8B is an explicit capacity substitution, not a reproduction of upstream's 4B leaderboard. Both use the same readout, task selection, state rendering and option-order procedure. No parameters are trained or quantized; overlong prompts fail instead of being truncated. Hosted CPU execution is not compared numerically with original GPU or proprietary-service timings.

`OpenJev raw` chooses the largest option score among three return actions and deferral. `OpenJev calibrated` proposes the highest-scored non-defer candidate. A regularized logistic predictor, fitted on 96 development cases, estimates whether that candidate is correct from candidate log odds and normalized action entropy. A single-class development set uses Beta(1,1) smoothing. At test, estimated success must reach 0.75 to return the candidate, reflecting deferral loss 0.25. Test labels never fit the calibrator. The development split also served incumbent parameter selection; it is not an independent evaluation sample.

Each model executes 288 primary forward passes, 48 reversed-order probes and twelve exact-input repeats. Twelve 4B workers and three 0.8B workers each perform one separately excluded warmup: **360 forwards for 4B and 351 for 0.8B, @@TOTAL_FORWARDS@@ in the completed benchmark**. Earlier owned probes and the superseded native-CPU attempt are not counted as completed benchmark evidence. Workers remain archived; proprietary Jev calls, CERA training, live worker generation and debate remain zero. Raw and calibrated variants share their readout; calibration does not query the model twice.

## Results with coverage and acquisition use

The objective is error loss 1 or deferral loss 0.25, plus 0.01 per acquired worker answer. It **does not invent a conversion of CPU seconds into dollars**. Accuracy marked with an asterisk includes only answered cases and must be read with coverage.

@@SCORE_TABLE@@

No gate requires the proposed method to win. A lower point estimate alone is not a superiority claim. Contrasts are paired on the same 192 unique groups, resampled within dataset and equally averaged across datasets. Intervals condition on fitted policies, calibration and recorded neural outputs, rather than measuring retraining uncertainty.

@@PAIRS@@

The following 98.75% percentile intervals provide a conservative Bonferroni-style adjustment across the four planned macro contrasts. Finite-sample bootstrap coverage remains approximate. Exploratory per-dataset and calibration comparisons are not promoted into confirmatory findings.

@@FAMILY@@

## Confidence, option order and actual computation

@@CALIBRATION@@

Raw candidate option mass is compared with correctness only as a diagnostic proxy: it was not originally guaranteed to represent that event. The calibrated score explicitly estimates correctness, but a small calibration set does not establish universal reliability. Type-valid output can be wrong; greater option mass is not a lower-risk certificate.

Primary option positions are deterministically counterbalanced. Reversal keeps the task, actions and labels unchanged, measuring decision flips and total variation between remapped distributions. These 48 probes per model do not create new independent questions. Each reversal and repeat runs on the original case's worker, avoiding a host-change confound within the probe.

@@ROBUSTNESS@@

@@TIMING@@

Neural times include tokenization and forward execution, excluding loading, warmup, calibration fitting and artifact writes. Empirical times are amortized Python replay/ledger overhead, not hardware-matched neural serving latency. The calibrated variant adds a small success-model computation that is not separately timed; its displayed time is a lower-bound proxy for the full calibrated path. Host CPU models are recorded per shard. None of these ratios is compared with TypeSafe's service claims or OpenJev's GPU-versus-JSON-generation speedup.

The score table omits selector compute from the acquisition objective, favoring a neural add-on. A separate sensitivity table adds assumed per-selection loss charges of 0, 0.001, 0.005, 0.01, 0.02, 0.05 and 0.1. They are hypothetical utility weights, not observed prices. Deployment claims require worker latency, concurrency, billing and an explicit value for controller compute. Arithmetic over estimated errors is not a faster language model, and richer neural representations are not free.

## A further conditional result: generated but unused answers

**Proposition 4 (unused-output irrelevance).** Suppose the final responding agent is selected before generation using only the initial query and routing scores. Other agents cannot affect that choice, the chosen prompt, its potential response distribution, the environment or subsequent decisions. If their outputs are never read and each extra activation costs strictly more than zero, deleting those activations preserves the final-answer distribution and strictly reduces expected cost whenever an extra agent would have run.

Proof: couple both systems on the same query, routing scores and retained-agent randomness. The final output is identical under the stated invariance assumptions. Extra outputs lie outside the terminal decision's dependency graph and cannot alter loss. Subtracting positive costs proves the result. The assumptions exclude synthesis, content-dependent selection, useful side effects and training interventions.

This separates CERA-MoA's open-ended inference convention—returning the highest-familiarity agent—from its training contribution. It does not imply that co-evolving training or all multi-agent inference is redundant. The OpenJev comparison reads multiple candidates, so it does not meet the unused-output premise. This is a dependency argument, not a new universal ensemble theorem.

## Scope and publication provenance

Experiment source: `@@EXPERIMENT_COMMIT@@`. Hosted execution: @@EXPERIMENT_RUN@@. Reporting and inference are recorded separately in the provenance bundle, and a reporting commit is never presented as the source of those inference calls.

This is an executed head-to-head pilot, not just an API contract. Its limits include 192 test cases, six familiar benchmarks, one archive, one neural family, declared task-independent costs and no new worker generation. It does not establish current SOTA, proprietary Jev quality, CERA training effects or individual human preferences. Stronger experiments should cross model families and independently collected datasets and factorially separate representation, terminal selection and acquisition strategy under common authorization and resource contracts.

Upstream attribution: TheoLeeCJ, *OpenJev*, MIT-licensed source with separate model dependencies, pinned revision above. https://github.com/TheoLeeCJ/openjev. Model checkpoint rights remain separate; no weights or font files are redistributed.
