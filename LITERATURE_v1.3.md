# CERA-MoA, Jev, and decision-theoretic acquisition: scope-correct comparison

Primary sources were read on 2026-09-18. The CERA-MoA v1 PDF and complete HTML including appendices were obtained by the `Research source audit` Actions job. The TypeSafe launch page and full official API documentation were retrieved in the same job. This is an analysis of the sources, not an independent replication of their model results.

## CERA-MoA: a training contribution, not simply another fixed panel

[Jiang, He, and Fang, arXiv:2609.18779v1](https://arxiv.org/html/2609.18779v1) couple agent policy updates with query allocation and router-head updates. The router's backbone is frozen; it concatenates final non-padding hidden states at roughly half and three quarters of its depth. Each agent has a trainable predictor and a fixed random target head. Familiarity is `exp(-lambda * distance)` between their normalized outputs. A relative-reward push/pull objective trains the predictor. The agents themselves are optimized using DAPO/GSPO-style reinforcement learning, with independent LoRA adapters in the shared-backbone setting.

During training, exploration bonuses affect ordering, but the activation cutoff sums **raw familiarity**, not exploration-augmented scores. At inference the bonuses are disabled. The procedure chooses the smallest ranked prefix whose familiarity sum reaches a threshold; if none does, all agents are activated. For deterministic-answer tasks the paper uses familiarity-weighted voting. For open-ended tasks it returns the highest-familiarity selected agent's completion. Appendix F explicitly acknowledges that this bypasses generative synthesis across candidates and does not yet evaluate long-horizon interactions.

This is materially stronger than our work in demonstrating actual agent fine-tuning and specialization. Our archived-answer replay cannot reproduce that mechanism, its causal benefits, or its training expenditure. It can test the more limited operational question of what happens when a router is left frozen while the available model population changes.

Table 3 reports Qwen3-4B adaptive routing at ID/OOD averages 63.2/72.8 and 367.77 generated tokens, versus fixed top-2 at 63.3/72.5 and 666.37 tokens. The approximately 44.8% reduction is **generation tokens against that particular top-2 comparator**, not a universal end-to-end dollar or latency saving. Table 6 reports router latency 11.87 ms on one H200 at batch size 1, native generation 1350.32 ms at batch size 1, and vLLM generation 154.09 ms at batch size 32. The latter is not a matched-batch latency comparison. Appendix B specifies seed 42 and substantial GPU training; the presented tables do not supply multi-seed confidence intervals. None of our CPU results should be compared numerically with those GPU timings.

### What the threshold does not prove

Familiarity is a learned compatibility score, not a demonstrated probability of correctness. Even **calibrated marginal correctness probabilities** cannot simply be summed to obtain the probability that at least one agent is correct. If two agents are perfectly correlated and each is correct with probability 0.4, their scores sum to 0.8 but joint coverage remains 0.4. A smallest-prefix rule also ignores heterogeneous query prices, terminal loss, deferral cost and conditional future information unless these are separately incorporated. This counterexample does not refute CERA-MoA's empirical results; it refutes interpreting the threshold itself as a calibrated joint-confidence or minimum-risk certificate.

Our `cumulative_score` is deliberately only a mechanism-inspired control: a hashed-character TF-IDF/ridge error predictor followed by a cumulative-score prefix under the common three-query budget. It has no CERA hidden states, frozen random targets, relative-advantage training, LoRA, or co-evolving agents. Its final selection uses the same empirical terminal rule as our principal controls, not CERA's weighted vote. Calling it a CERA reproduction would be incorrect.

## Jev: an estimator/interface, not the external objective function

[TypeSafe's launch article](https://typesafe.ai/blog/introducing-system-one-models-and-jev) describes a non-string-generating model with parallel typed outputs and Reinforcement Learning for Calibrated Decisions (RLCD). [Official API documentation](https://docs.typesafe.ai/) exposes Choice, Score and Noul primitives. Choice supplies a legal option, a probability distribution over the specified options, and a derived confidence statistic. The [confidence documentation](https://docs.typesafe.ai/confidence) explicitly says confidence is computed from the distribution; it is not a separate verified outcome label.

Type safety and semantic correctness must be separated. A valid `emit_0` action can still emit an incorrect acquired answer. A probability distribution over **next actions** is not automatically a distribution over whether each action succeeds. Moreover, an API-valid action still needs application-level authorization and prerequisite checks against the current state. The included contract reconstructs the legal action set on every step, prohibits selecting unacquired candidates, validates the probability simplex, and rejects out-of-state actions. These are adapter tests, not evidence of Jev accuracy or service reliability.

TypeSafe's strongest speed/cost claims come from provider-run structured workflows. Its launch post says reference probabilities are constructed from the average of two frontier LLMs, rather than independently verified outcomes or individual human preferences, and acknowledges workflow-construction and comparison-setting caveats. Its typing claim therefore does not establish an absence of factual errors. Provider-side latency measurements and CPU replay timings are not comparable. No Jev endpoint, weights, or training system was executed in our experiment; we report no Jev leaderboard row or measured speedup.

## Where the approaches can fit together

| Layer | CERA-MoA | Jev | This study |
|---|---|---|---|
| Population learning | Coupled agent RL and router updates | Provider-trained model | None; immutable archived responses |
| Competence signal | Hidden-state familiarity | Typed probabilistic decisions | Empirical conditional error and a transparent prompt baseline |
| Query allocation | Cumulative familiarity prefix | Can select an allowed action if configured | Static, one-step VOI, finite-horizon Bellman, refresh controls |
| Objective/authority | Task rewards plus training design | Caller-defined questions/options | Explicit error, deferral, assumed acquisition cost and legal actions |
| Open-ended synthesis | Highest-familiarity response in the reported system | No free string generation | Selection only; no synthesis claim |
| Current evidence here | Primary-source comparison only | Primary-source comparison and contract tests only | Executed historical replay, calibration and replacement tests |

A plausible composition is a learned competence estimator (CERA-style or Jev-compatible), an externally specified utility, an authorization layer, and a cost-aware controller. That composition is a design hypothesis, not a validated integrated product. Unknown user preferences still require actual preference information; none of these model labels creates that information by itself.

## Reproducible primary-source provenance

The source-audit workflow pinned ProEval to `9d37b037d1d91f4b78a7ddd74a43a8aecb344c66`. Retrieved CERA v1 PDF SHA-256: `9e51bc6e2157a13d8d10c8f639c83edb1fb3450a1808e13217ee14ddebf99ab7`; CERA HTML: `b6e4a587e1b50f7d77dae6fd69cc30347d39f890279ac0817c6e59e97d337079`. Source code records dataset SHA-256 checks; it does not copy a third party's scientific findings into our own empirical result tables.
