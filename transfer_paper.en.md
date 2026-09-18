---
title: 'When to Update the Router'
subtitle: 'Decision-Theoretic Acquisition, Model-Pool Replacement, and the Limits of Typed Confidence'
author: 'Decision-Theoretic Mixture-of-Agents project'
date: '18 September 2026 · Version 1.3.0 · Public technical report'
lang: en
---

# Abstract

A multi-model system must distinguish learning better agents, estimating their competence, deciding whether another observation is worth its cost, and producing a software-valid action. CERA-MoA jointly trains agents and routing heads, while TypeSafe's Jev supplies typed probabilistic decisions. Neither capability alone specifies a user's loss function or proves an acquisition policy optimal. We extend a previously published finite-control study and RouterBench replay with @@N_SOURCES@@ pinned ProEval files. @@N_PRIMARY@@ binary benchmarks contribute @@PRIMARY_ROWS@@ questions, eight fixed model identities, and @@PRIMARY_TEST@@ held-out test questions. DICES is a separate, explicitly qualified ordinal-loss diagnostic. We compare nine controls and frozen, refitted, and limited-budget-refreshed policies under replacement of the four-model pool. At the prespecified cost of 0.01 per acquired answer, refitted Bellman, static and myopic controls obtain identical per-case objective values across the @@N_PRIMARY@@ binary test sets; their macro objective is @@REFIT_OBJECTIVE@@. Refitting versus freezing Bellman improves the macro objective by @@REFIT_GAIN@@, with a conditional paired bootstrap interval [@@REFIT_GAIN_LO@@, @@REFIT_GAIN_HI@@]. Small refresh sets can instead worsen decisions. One newly added task is deferral-dominated: every candidate’s archived error exceeds the declared deferral loss, so no control returns an answer on any held-out case there; it is kept in the average and reported rather than dropped. Source audits reveal numerical serialization aliases and DICES labels outside the documented range. We derive conditional finite-control, approximation and action-margin guarantees, explicitly separating them from semantic correctness and provider confidence claims. Two separate-process reproductions, @@TESTS@@ tests, source hashes and ledger gates accompany this report. No Jev inference, CERA-MoA agent training, live LLM generation, or human preference experiment was performed.

# Research question and evidence hierarchy

The motivating problem is not merely disagreement between models. An ensemble may disagree because evidence is incomplete, because agents fail together, or because the task lacks the user's preferences. Increasing the number of opinions does not resolve the latter. A controller must decide whether to query, verify, clarify, defer or stop, with a declared cost and a declared owner of the utility function.

Our v1.1 report studied a finite synthetic environment; v1.2 replayed RouterBench's historical responses [5,6]. Those versions are retained as separate releases. The synthetic heuristic lost to a simpler value-of-information cascade; on RouterBench, the Bellman advantage over strong static/myopic controls was not reliable. Those findings are not overwritten by this extension. The earlier locally shared classifier-proxy candidate is not the source of any ProEval result here.

This extension asks three narrower questions. First, do the earlier insights transfer to another response archive and additional task families? Second, is updating the competence model more valuable than deeper within-query planning after a model-pool change? Third, what evidence is needed to connect a familiarity score or typed confidence to a decision-quality claim?

The new tests concern **response acquisition and candidate selection**. They do not generate answers, fine-tune the candidate LLMs, execute their reasoning, synthesize prose, or elicit personal values. A retained record is not a fresh inference call. Replaying additional stored records improves empirical coverage, not the realism of unobserved interactions.

# CERA-MoA and Jev: related work with different intervention targets

## What CERA-MoA actually changes

CERA-MoA [1] couples routing with reinforcement-learning updates to agents. Its frozen router backbone supplies concatenated intermediate-layer query features. Each agent has a trained predictor head and a fixed random target head. A normalized projection distance gives familiarity $f_i(q)=\exp(-\lambda d_i(q))$. Relative reward advantages pull the predictor toward or push it away from the anchor. Training-time exploration terms affect ranking; the prefix activation test sums raw familiarity, not the exploration bonus. At inference, exploration is disabled.

The paper trains independent LoRA agents in shared-backbone experiments and also evaluates a heterogeneous pool. This is an actual population-learning intervention, whereas we retain immutable archived model outputs. Our replacement experiment cannot identify the causal benefit of co-training or emergent specialization.

For deterministic-answer tasks, CERA-MoA uses familiarity-weighted voting. For open-ended tasks, the reported aggregator returns the most familiar selected agent's completion. Appendix F explicitly identifies the absence of generative synthesis and long-horizon interaction as limitations. Thus it does not directly resolve the original private-preference disagreement problem.

In Table 3, the authors report 367.77 average generation tokens for adaptive routing versus 666.37 for fixed top-2, with ID/OOD scores 63.2/72.8 and 63.3/72.5, respectively. The approximately 44.8% reduction is tied to that comparator and that token accounting. Table 6 profiles a batch-1 router and batch-1 native generation on an H200, but the vLLM generation row uses batch 32. These are not a matched-batch end-to-end latency comparison. The reported configuration uses seed 42; uncertainty across training seeds is not supplied in those tables. These observations delimit the claims; they do not negate the authors' measured results.

## What Jev does and does not certify

TypeSafe describes Jev as a non-string-generating model with parallel structured output and RLCD training [2]. Its API exposes Choice, Score and Noul [3]. Choice returns an allowed option, option probabilities, and a confidence statistic derived from their distribution. A distribution over which action to take is not automatically the probability that each action succeeds.

The launch article's strongest workflow efficiency claims are provider measurements. It describes reference probabilities derived from two frontier models and acknowledges workflow-design and comparison-setting caveats. Agreement with that reference is not independent task truth or individual preference identification. Schema validity also does not imply factual correctness. Our included adapter contract checks the legal action set, prerequisites, simplex and chosen option, but does not call Jev. No Jev performance row, calibration score or speedup is reported.

## Comparison boundary

| Aspect | CERA-MoA | Jev | This report |
|---|---|---|---|
| Intervention | Agent RL plus router training | Provider-trained decision model | Offline policy fitting and pool replacement |
| Competence signal | Hidden-state familiarity | Typed distributions | Empirical error; transparent prompt predictor |
| Control | Familiarity-prefix selection | Caller-designed decision interface | Cost-sensitive query, selection and defer |
| Open-ended output | Highest-familiarity completion | No free string generation | Acquired response selection only |
| Evidence here | Primary-source analysis | Source analysis and contract tests | Executed archived-response experiments |

A plausible composition is a learned estimator, explicit utility, cost-aware control and an authorization boundary. It remains an integration hypothesis. The three systems cannot be ranked by placing unrelated reported scores in one leaderboard.

# Decision model and conditional guarantees

## Finite acquisition problem

Let $\theta$ identify the deployed agent population, $h$ the acquired observations, and $r$ the remaining acquisition budget. A legal action acquires an unused batch $S$ or terminates by returning an acquired candidate or deferring. The model $P_\theta$ describes the joint distribution of potential observations and terminal losses. Our primary objective is

$$J_\theta(\pi)=\mathbb E[\ell(a,Y)+c\,N_{\mathrm{acquired}}],$$

with $c=0.01$, error loss 1 and deferral loss 0.25. All are assumed loss units, not measured dollars. Let $R(h)$ be the smallest conditional terminal risk. Under known, action-invariant potential outcomes, additive acquisition cost, finite actions and bounded budget,

$$V(h,r)=\min\left\{R(h),\min_{S\in\mathcal A(h,r)}\left[c|S|+\mathbb E\{V(h\cup O_S,r-|S|)\mid h\}\right]\right\}.$$

**Proposition 1 (conditional Bayes optimality).** Selecting the minimizing legal action attains the minimum expected loss in this finite policy class. Proof: when $r=0$, compare the finite terminal actions. For $r>0$, every policy must stop or choose a legal first batch. Conditional on each possible batch outcome, the induction hypothesis lower-bounds its continuation by the value for the smaller budget. Minimizing these attainable lower bounds yields the recurrence. Randomized first actions cannot improve on the smallest component of their convex combination. This is the standard Bellman argument applied to the declared problem, not new universal MoA optimality.

The actual replay uses empirical expected error conditional on acquired answer-agreement patterns. Its state is deliberately small: 52 complete equality/INVALID patterns and 151 partial states, with a maximum of three queries. It does not identify arbitrary semantic distinctions. It selects an observed candidate rather than inventing a new answer.

## Distribution error, approximate decisions and margins

**Proposition 2 (model and solver error).** If every feasible policy's total loss is in $[0,M]$, $\operatorname{TV}(P,\widehat P)\leq\epsilon$, and $\widehat\pi$ is $\delta$-optimal under $\widehat P$, then

$$J_P(\widehat\pi)-J_P(\pi_P^*)\leq 2M\epsilon+\delta.$$

Proof: add and subtract $J_{\widehat P}(\widehat\pi)$ and $J_{\widehat P}(\pi_P^*)$. The two distribution-change terms are each at most $M\epsilon$ for bounded losses; the middle optimization term is at most $\delta$. The bound does not estimate $\epsilon$ from an uncalibrated confidence score. In this replay, $M=1+3c$ is a valid conservative bound for binary tasks.

**Proposition 3 (fast-policy approximation).** Suppose a surrogate estimates each feasible action's optimal continuation value with uniform error at most $\eta$ at every reachable decision state, under the same observation model, and at most $H$ decisions remain. Greedy surrogate decisions have regret at most $2H\eta$ in that model. At a state, comparing the surrogate-selected action with the truly smallest action introduces at most $2\eta$; induction adds the continuation bound. If the estimated gap between the best and second-best legal action exceeds $2\eta$, the first action is unchanged by any value perturbation inside that bound.

This provides a conditional way to justify a fast decision path and escalation when the margin is small. It is not a Jev guarantee: a uniform action-value error certificate is stronger than average calibration or API confidence. Replacing $\delta$ by $2H\eta$ connects the approximation result to Proposition 2 only when the assumptions actually hold.

## Familiarity sums and legal outputs are not truth certificates

Consider two perfectly correlated agents, each correct with probability 0.4. Their calibrated marginals sum to 0.8 while the probability that either is correct is still 0.4. Therefore summing familiarity, which is not itself established as a calibrated marginal, cannot be read as a joint-success probability. The cumulative rule can be a useful heuristic without being a minimum-loss certificate.

Likewise, `emit_0` can be a perfectly valid typed action while candidate 0 is false. Type safety constrains the action space; it does not fix the observations or the external objective. The test suite executes both counterexamples.

When an acquisition or training action changes future agent parameters, the fixed-potential-outcome recurrence no longer models that intervention. One must include population/training state and action-dependent transitions. This report changes population between evaluation phases only. It does not solve the co-evolutionary training control problem.

# Locked protocol and data

## Sources, provenance and splitting

We use Google DeepMind's ProEval archived outputs [4], pinned to commit `9d37b037d1d91f4b78a7ddd74a43a8aecb344c66`, with a distinct SHA-256 for each file. The protocol digest was committed before policy-performance execution after inspecting schema and three example records. It follows earlier v1.2 findings and is not an external registry preregistration. Schema and numerical-quality fixes are recorded separately from primary analysis.

@@DATASETS@@

GSM8K and SVAMP add numerical-answer tasks, StrategyQA adds yes/no reasoning, and MMLU adds another model/archive cohort rather than a previously unseen benchmark family. DICES is not pooled into the binary headline objective. @@N_SOURCES@@ files from one archive are not @@N_SOURCES@@ independent collection pipelines. GQA contributes image-grounded short answers; the archive stores the image identifier but not the pixels, so only answer selection is replayed and the text-only prompt controls are handicapped there. Jigsaw contributes toxicity verdicts whose ground truth is an annotator toxic-fraction thresholded strictly above 0.5, a rule that reproduces the archive’s own label column exactly on all comparable pairs. Deferral-dominated tasks in this archive: @@DEFER_DOMINATED@@. On such a task the best archived candidate error still exceeds the deferral loss of 0.25, so decision-theoretic controls withhold on every held-out case and the task carries no policy discrimination; it is retained in the primary average instead of being excluded after the fact. Nine GQA prediction cells contain a data-generation debug string instead of a model answer and are treated as unparseable answers and counted in the source audit. Exact normalized questions are group-hashed globally into 60/20/20 train/development/test assignments, including across both model pools. MMLU NumPy-array representations are parsed as literal syntax without executing constructors. Similar but non-identical prompts and pretraining contamination are not excluded by exact hashing.

The fixed legacy slots are GPT-4o, Claude 3.5 Haiku, Gemini 2.5 Flash and Gemma 3 12B. Replacement slots are GPT-5.2, Claude 4.5 Sonnet, Gemini 3 Flash and Gemma 3 27B. These archive names are not claims about current endpoints. Identities were chosen before examining accuracy. A slot replacement is not a temporal checkpoint of one continuously trained agent; provider, size and capabilities change together.

## Labels, normalization and fairness

The source's `label_*` columns are **errors**, not correctness. They supervise fitting and final scoring but never enter the acquisition callback. Numerical responses are normalized as exact rationals; yes/no and letter responses use strict documented mappings. Missing or unparsable answers remain observable INVALID values. Non-finite or out-of-range labels trigger the prespecified common complete-case filter across both pools; exclusions are counted.

Every principal control has the same four-model pool and three-query limit. None receives a perfect verifier, a gold answer, an unacquired response, or a personal-preference oracle. Terminal selection minimizes a common estimated error/deferral rule. `majority3` is explicitly a diagnostic exception: it votes and does not use the common defer rule except if all responses are invalid. The single and fixed-three controls deliberately make their specified number of acquisitions; static/myopic/Bellman can stop earlier.

## Controls and parameter selection

The controls are single, fixed-three, one-shot static subset, myopic value of information, finite Bellman lookahead, disagreement-triggered acquisition, majority-three, prompt-based top-1, and cumulative-score routing. Priors of strength 1, 10 and 100 are selected independently for each method using the development split. Uniform positive support smoothing over canonical patterns uses global training error for unobserved patterns; this is a modeling assumption, not a correctness prior derived from language semantics.

Prompt controls add query-text features unavailable to the agreement-only controls: 2,048 hashed character features, training-only TF-IDF and a four-output ridge error predictor. The cumulative thresholds are 0.7, 1.4 and 2.1, selected on development loss. These controls share terminal selection with the main methods, isolating their query acquisition choices. They are neither CERA-MoA nor Jev replications. In particular, cumulative-score routing lacks hidden states, random anchor heads, relative-advantage learning, agent RL and CERA voting, and is capped at the common budget rather than falling back to all four models.

## Pool replacement, refresh and statistics

L denotes legacy calibration/test. F applies that frozen policy and terminal model to replacement-pool answers. R refits and reselects using replacement train/development data, retaining the original test identities. Refresh-50/200 instead uses a deterministic prefix of non-test training records and the legacy-selected hyperparameter, without target-development selection. Each refreshed record costs four acquired archive answers. Costs for full calibration/development and small refresh sets are reported separately rather than treated as free.

For three declared contrasts, we bootstrap exact question groups within each binary dataset 2,000 times and average datasets equally. Intervals are conditional on fitted policies. No repeated phase or repeated execution is counted as an independent question. Ten-bin ECE and Brier score assess the reported terminal error forecast on answered binary cases only; they are not CERA familiarity or Jev calibration results. DICES receives neither metric. Controller timings measure Python replay/ledger CPU overhead, not LLM service latency.

# Results

## Recalibration matters more than deeper planning in the main cohort

@@MACRO@@

After refitting, Bellman, static and myopic obtain the same objective on every paired binary test case, with macro $J=@@REFIT_OBJECTIVE@@$. This is stronger than merely failing to reject a difference in this realized sample, but it does not establish population equivalence. Their batch schedules can differ even when selected candidates and acquisition counts match. Per-task acquisition counts are in the table below. Because the split salt and the archived answers of the four previously published tasks are unchanged, their per-task results reproduce the earlier version exactly. The added GQA task acquires nothing at all, since withholding is optimal there under the declared loss.

@@TASKS@@

@@PAIRS@@

The frozen-to-refitted Bellman improvement is @@REFIT_GAIN@@ loss units in the equal-dataset mixture. The interval does not include zero, conditional on the fitted policies. The gain combines refreshed competence estimates, changed selected candidates and altered acquisition/deferral; it cannot be attributed to co-training agents. Indeed, static and myopic obtain the same refitted objective. The cumulative-score and prompt controls do not establish that CERA or Jev is worse: neither actual system was run, and their hypotheses and information sources differ.

## Refreshing has a cost and can harm

@@REFRESH@@

For an assumed refresh charge $K$ and stationary future per-query loss reduction $\Delta>0$, the plug-in break-even count is $\lceil K/\Delta\rceil$. It is an estimated deployment horizon, not a guarantee or an actual dollar return. When the measured gain is non-positive, no finite break-even is claimed. Full train/development acquisition expenditure is also saved in `calibration_resources.csv`.

Small refresh sets worsen the MMLU and SVAMP Bellman results in this split. A larger refresh is not monotone in realized test performance. The data therefore support checking calibration updates rather than blindly replacing an incumbent whenever new feedback exists. These full-panel observations are not bandit-selected feedback and do not validate selective-feedback learning or continuing LLM training.

## Calibration is a distinct outcome

@@CALIBRATION@@

Lower decision loss does not require a lower realized ECE on every finite test set. These selective calibration metrics depend on which cases are answered and should not be compared without coverage. They use archived binary labels and a terminal-risk estimate. Jev's option distribution and confidence are not measured; DICES ordinal errors must not be converted into a binary success probability.

# Source-quality findings and robustness limits

An exact rational comparison finds 104 disagreements between Gemma 3 12B's SVAMP archived error labels and strict answer/gold comparison. A post-primary tolerance audit resolves 103 of those at relative tolerance $10^{-9}$ and absolute tolerance $10^{-12}$, consistent with values such as an integer serialized with machine-precision residuals. The primary observations, labels and selected policies are not changed after discovering this. `numeric_tolerance_audit.csv` reports all models and residual differences, including GSM8K. Exact numerical answer-equality states can therefore be sensitive to serialization; the tolerance audit is not a hidden rerun to optimize a headline result.

DICES is more problematic for broad claims. Of 1,500 source rows, 439 fail the common documented-range filter for the selected models, leaving 1,061 rows but only 308 distinct exact question groups. Observed label values include 1.25 and one 5.5 even though the archive documentation describes errors in $[0,1]$. These may reflect source-processing or metric-definition issues; we do not assign a cause without further evidence. The retained subset's Bayes controls defer everywhere at the assumed deferral loss. This is not a useful preference-decision success result. DICES is quarantined from the binary headline, and its conditional diagnostic cannot stand in for individual preferences or a representative safety population.

The strict audit uses the same archive's reference answers. Unlike v1.2's separate MMLU reference join, it is not an independent original-benchmark label verification. Exact-group splitting does not exclude near-duplicate tasks, model training contamination, or curation bias. Source quality and source permissions are documented; raw question/response text is not included in the permanent result package.

The additional price grid is diagnostic and was not searched to replace the primary price. Cheap queries can change the preferred number of agents, but a different cost grid cannot certify global optimality or real-world latency savings. A second process reproducing results verifies the computational path, not the external validity of the source population.

# Executed verification and artifact contract

The hosted source commit is `@@COMMIT@@`; execution record: @@RUN_URL@@. The suite passes **@@TESTS@@ tests** without failures or skips. Two complete processes repeat fitting, development selection, all phases and source audits. **@@OUTPUTS@@ non-timing output files** pass the comparison rules, **@@IDENTICAL@@** are byte-identical; two further timing-bearing files are compared after excluding timing columns only.

Gates reconstruct loss and acquisition charges from per-case ledgers, require identical sample sets across paired policies, reject unacquired candidate selection, check budget limits and split isolation, verify the locked protocol digest, and recheck source checksums. An independently written recursive support-filtering evaluator checks finite Bellman values without production transition tables. Counterexample tests reject the interpretations that familiarity sums are joint probabilities or typed outputs imply correctness. Jev-compatible contract tests validate only parsing and authorization, not a running service.

The evaluation job has read-only repository access. A separate publication job receives write access only after successful evidence validation and checksum verification. Existing versions are not overwritten. The published bundle contains source, derived traces, ledgers, tests, logs, provenance and both manuscripts. It excludes source CSVs, raw reasoning and font files. Public release is not peer review, acceptance, arXiv deposit or DOI registration.

# Discussion and conclusion

CERA-MoA supplies evidence for coupled population and router learning, a question our replay does not test. Jev proposes a software-oriented decision interface, which is orthogonal to knowing whether another acquisition is worth its cost. Both sharpen the problem formulation: competence estimation, action validity, statistical calibration and decision utility are separate requirements.

The executed extension supports a narrower and more useful result than architectural supremacy. Updating a stale competence model can matter, while full lookahead may add nothing over a strong static or myopic controller in the same updated model. Small calibration budgets and source-quality defects can erase that benefit. The right next-generation architecture is therefore a hypothesis about a calibrated, refresh-aware, externally governed control system, not a renamed claim that the original heuristic was optimal.

What remains untested is material: actual CERA training; live Jev decisions on a common task suite; action-dependent multi-turn generation; learned verifier reliability; measured inference/token costs; selective-feedback update bias; human utilities; and unseen independently collected task families. A real co-evolution study should factorially separate agent updates, router updates and sample allocation at matched training and serving budgets. A Jev experiment should compare identical authorized state/action contracts and evaluate semantic outcomes, calibration, coverage and end-to-end latency against independent labels. These are future experiments, not hidden claims of completion.

# References and attribution

[1] Jiaxuan Jiang, Liyuan He, Zhixuan Fang. *CERA-MoA: Co-Evolving Routing Mechanisms with Continually Learning LLM Agents*. arXiv:2609.18779v1, 2026. https://arxiv.org/abs/2609.18779.

[2] TypeSafe AI. *Introducing System One Models & Jev*. Official launch article, retrieved 2026-09-18. https://typesafe.ai/blog/introducing-system-one-models-and-jev.

[3] TypeSafe AI. *System One API and Confidence*. Official documentation, retrieved 2026-09-18. https://docs.typesafe.ai/ and https://docs.typesafe.ai/confidence.

[4] Google DeepMind. *ProEval data archive and data documentation*. Commit `9d37b037d1d91f4b78a7ddd74a43a8aecb344c66`. https://github.com/google-deepmind/proeval/tree/9d37b037d1d91f4b78a7ddd74a43a8aecb344c66/data. Archive documentation credits Huang, Zeng, Kumaresan and Wang and the underlying benchmark authors. Its materials state CC-BY 4.0 with source-specific exceptions. This work computes derived records and identifies every changed representation; it does not claim ownership of the benchmarks or model outputs.

[5] Decision-Theoretic Mixture-of-Agents project. *When to Ask Another Model*, v1.1.0 synthetic technical report, 2026. https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/tag/v1.1.0.

[6] Decision-Theoretic Mixture-of-Agents project. *Verified Historical LLM Replay*, v1.2.0, 2026. https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/tag/v1.2.0.

AI assistance was used for code, drafting and verification. No external reviewers, institutional affiliation, new model training or venue acceptance are implied. The project's owner has not selected a blanket reuse license; third-party materials retain their original rights and conditions.
