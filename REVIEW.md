# Research-quality audit: v1.1.0

## Verdict

A reproducible synthetic mechanism/falsification study and technical report—not a demonstrated new state-of-the-art MoA method. The original request for a universal optimal architecture is not supported. A small, correctly specified finite problem has a standard Bellman optimum; that is not equivalent to new theory or practical superiority.

## Material weaknesses found and addressed

1. **Baseline description did not match code.** C1 and C2 are selected separately for each task type, not globally. Manuscripts and README now match the experiment. Numeric results were preserved.
2. **Unequal information permissions confounded comparisons.** New R1 uses all six sources but only one nonadaptive batch; R2/R3 compare greedy/full planning with workers only. P1's gain over R1 is only about 1.71%.
3. **The original numerical checker shared transition tables.** Added a tuple-history reference solver that computes posterior probabilities directly, independently of production state/transition lookup machinery. This is not formal verification.
4. **Sparse distributions and configuration changes were fragile.** Added probability/cost/budget validation, safe zero-mass histories, configurable deferral, and explicit candidate prerequisites. Default outputs remain unchanged.
5. **Claims exceeded the contribution's novelty.** Bellman recursion, TV risk comparison, and binary preference testing are presented as standard decision-theoretic applications. Related work now includes metalevel computation selection and the conditions behind greedy guarantees.
6. **Sampling size could mask low support diversity.** The report explicitly states 640 support points. Added calibration curves, routing errors, verifier common-mode stress, and planning-overhead diagnostics. These are all exploratory.
7. **Publishing and validation were conflated.** A release is a public technical report. It is not peer review, an arXiv deposit, a DOI, or evidence of commercial-product superiority.

## What remains unresolved

No real LLMs, natural-language synthesis, learned task classifier, human utility study, or end-to-end token/latency accounting. The primary generator grants a reliable independent verifier and informative preference elicitor, substantially shaping the results. Action-dependent prompting and candidate-dependent verifier quality are not represented. There is no evidence supporting a broad SOTA or top-conference methods claim.

The strongest current contribution is transparent falsification, explicit boundaries, negative results, fairer controls, and an executable artifact. A substantive next paper needs locked real model/tool versions, matched tool access, independently assessed tasks, measured controller overhead, and a development/held-out evaluation design established before observing outcomes. None of that is represented as already completed here.

## Reproduction terminology

The local original rerun matched 34 primary nonmetadata outputs. After supplementary execution, all 38 original nonmetadata result files remained identical. Hosted CI reruns the same code in a second runtime and checks results. This improves computational reproducibility but is not external scientific replication by independent researchers. Confidence intervals describe the synthetic generator, not its realism.

## Disclosure and rights

AI-assisted implementation and writing. Project attribution only; no fabricated personal authorship, institutions, or reviewers. The owner has not selected an explicit reuse license. Public availability and citation metadata do not themselves grant unrestricted redistribution rights.
