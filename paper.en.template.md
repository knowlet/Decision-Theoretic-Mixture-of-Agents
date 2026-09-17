---
title: "When to Ask Another Model"
subtitle: "A Finite Decision-Theoretic Falsification Study of Mixture-of-Agents"
author: "Decision-Theoretic Mixture-of-Agents Project"
date: "17 September 2026 · Version 1.1.0"
lang: en-US
---

# Abstract {-}

A multi-model system can collect more opinions without acquiring information that changes a decision. We examine this distinction using a reproducible, finite-support simulator of correlated worker errors, factual verification, private preferences, and costly deferral. Twenty independent calibration-and-test replications contain 500,000 calibration draws and 400,000 held-out cases. An initially proposed disagreement-triggered workflow, P0, attains composite loss 0.6148: better than worker-only single-model and fixed-panel controls, but worse than a myopic information-value cascade, C3, at 0.3824. A fitted finite-horizon Bellman controller, P1, attains 0.3752, a 1.90% improvement over C3. An added information-access-matched, nonadaptive batch control attains 0.3817; P1's remaining improvement is only 1.71%. Thus, most of the apparent architectural benefit is associated with information access and stopping rather than deep coordination.

We give the standard finite-horizon Bayes-optimality proof under explicit assumptions, a distribution-and-optimization error bound, and an exact binary preference-identification identity. These are applications of established decision theory, not claims of new universal optimality theorems. Learning curves, task-routing errors, tool dependencies, and verifier degradation reveal substantial sensitivity. An 86-test suite includes a separately implemented tuple-history reference solver. No language-model endpoint was called. This report is a synthetic mechanism study with an executable research artifact, not a commercial-product benchmark, a demonstration of natural-language quality, or a peer-reviewed publication.

**Keywords:** mixture-of-agents; metareasoning; value of information; optimal stopping; preference elicitation; selective prediction; reproducibility.

\clearpage
\tableofcontents
\clearpage

# Introduction

Open-ended disagreement exposes a gap between generating candidates and deciding what to do. Different model answers may reflect factual uncertainty, different implicit preferences, common misconceptions, or irrelevant variation. Asking another model can help only insofar as the resulting information changes the expected consequences of a decision enough to justify its cost. Conversely, agreement need not warrant stopping when several models share a failure mode.

The motivating proposal was a pipeline: risk routing, heterogeneous proposals, disagreement detection, factual verification or preference elicitation, and synthesis or abstention. Its components were plausible, but their presence did not establish that their ordering and triggers were optimal. We deliberately turn that proposal into a falsifiable policy, P0, rather than treating it as the answer to be vindicated.

Our first question is whether this concrete workflow improves on reasonable, simpler controls. The second is whether complete planning gives a material benefit beyond a strong myopic cascade with the same information sources. The third is how these comparisons change when assumptions about information quality, routing, and tool prerequisites are relaxed. The result is an artifact-oriented falsification study: the initial workflow loses to a simpler control, some proposed components fail to demonstrate a benefit, and changing a tool prerequisite reverses a ranking.

The contribution is the integrated experimental specification, executable comparisons, negative results, and audit trail. It is **not** a new Bellman principle, a new general-purpose LLM coordinator, or a theorem that a named MoA architecture is best. Section 4 separates statements about an exactly specified finite decision problem from statements about fitting its distribution or deploying it in language-model systems.

# Related work and positioning

Mixture-of-Agents uses layered model outputs to inform later generation [1]. Self-MoA challenges the assumption that mixing different models necessarily helps: diversity interacts with the quality of the constituent outputs [2]. Neither observation determines an optimal stopping or acquisition rule for arbitrary decision costs.

FrugalGPT studies cost-aware LLM cascades [3], and RouteLLM learns model routing from preference data [4]. These motivate using a capable, cost-sensitive cascade as a baseline rather than comparing only with majority voting. SelectiveNet jointly learns prediction and rejection [5]; our simpler terminal rule likewise makes coverage and deferral explicit, but is not a reproduction of SelectiveNet. LLM-as-a-judge evaluations identify position, verbosity, and self-enhancement biases while also finding useful agreement with human preferences in studied settings [6]. We avoid a language-model judge here by evaluating against the simulator's hidden state, at the cost of not studying linguistic synthesis.

Computational selection as a metalevel decision problem predates current MoA systems [7]. Bounded-optimal metareasoning examines how useful computational arrangements depend on environmental variability and computation cost [8]. Our controller is a small, exactly solvable application of that framework. Adaptive submodularity provides greedy guarantees under appropriate structural assumptions [9]. We do not establish those assumptions for our objective; the information-complementarity counterexample in Section 4 shows why arbitrary acquisition tasks cannot inherit a greedy guarantee by default.

This report does not reproduce OpenRouter Fusion, Hermes MoA, Sakana Fugu, or any other deployed system. Their engineering choices and model quality cannot be ranked from Bernoulli channels. The companion Chinese report retains a short, explicitly contextual discussion of their official documentation; neither manuscript attributes the synthetic scores to those products.

# Decision problem and executable policies

## State, observations, and admissibility

An observable task type $T=t$ selects a joint distribution over a binary hidden state $Y$ and six binary potential observations:

$$P_t(Y,O_A,O_B,O_C,O_D,O_V,O_E).$$

In factual tasks, $Y$ is the correct option. In preference tasks, $Y$ is the option preferred by one particular user's utility, not a universal normative truth. Channels A--D represent workers; V supplies grounded factual information; E elicits private preference information. The complete latent outcome is sampled before interaction. Acquiring a channel reveals its value but does not change the distribution of unobserved channels. This action-invariant assumption is important: adaptive prompting, generated candidates, and tool state changes in real agents often violate it.

A history $h$ contains only acquired observations. A policy can stop with action $a\in\{0,1,\bot\}$, or acquire a nonempty batch of previously unqueried channels. The symbol $\bot$ denotes costly deferral. At most four channels can be acquired, and no channel can be repeated. A batch can contain several channels in parallel. A candidate-before-verification variant additionally requires at least one worker observation from an **earlier** batch before V becomes legal.

All policies are nonanticipating: unqueried potential observations exist for paired evaluation but are not policy inputs. We model them jointly to represent dependence, not to allow a controller to inspect unpurchased information. Zero-probability histories receive a defined fallback and zero probability in expectations; the implementation validates distributions and avoids dividing by zero.

## Loss and resources

We minimize

$$J_P(\pi)=\mathbb E_P[\ell_T(a,Y)+\lambda C(\pi)+\mu L(\pi)],$$

where $\ell_t(a,Y)=w_t\mathbf1\{a\ne Y\}$ for an answer and $\ell_t(\bot,Y)=d$. The principal settings are $d=1.5$, $\lambda=0.1$, and $\mu=0.04$. Batch cost is $c(S)=\sum_{i\in S}c_i$, and batch latency is $\tau(S)=\max_{i\in S}\tau_i$. Sequential batch latencies are added.

Costs and latencies are **assigned normalized units**, not measured tokens, dollars, or seconds. Deferral has a nonzero expected loss; it is not assumed free or infallible. Differential planning, state-extraction, and finalizer costs are excluded from the primary objective. We separately measure CPU planning time and discuss the small overhead margin left by P1's observed gain.

Every policy uses the same terminal Bayes rule derived from its fitted distribution. If $p_h=P(Y=1\mid h,t)$, the risk of stopping is

$$R_t(h)=\min\{w_t p_h,w_t(1-p_h),d\}.$$

Thus, the fixed-panel baseline uses joint posterior aggregation, not deliberately weak majority voting. Accuracy is reported together with coverage: accuracy among answers alone can increase simply by deferring more cases.

## Principal policies

| ID | Information acquisition | Restrictions |
|---|---|---|
| C1 | Best single worker for each task type | Always one A--D query |
| C2 | Best fixed three-worker panel for each type | Always three A--D queries |
| C3 | Best positive one-step information value | One channel at a time; may stop |
| P0 | Initial risk/disagreement heuristic | Fixed fanout and verification triggers |
| P1 | Complete finite-horizon planning | Any admissible batch, continuation, or stop |

C1 and C2 choose their worker or panel on the **calibration distribution separately for each task type**. Their acquisition choices remain fixed within a type. Version 1.0 incorrectly described them as globally fixed across types; this revision corrects the manuscript to match the unchanged executable experiment. No test outcomes select the baseline models.

P0 stops immediately for low-stakes preferences, queries A for easy facts, and otherwise first queries A, C, and D in parallel. For factual tasks it queries V only when this panel disagrees. For high-stakes preferences it subsequently queries E. All final decisions use the shared terminal rule. This is one implementation of an underspecified proposal, not a claim to represent every disagreement-aware workflow.

C3 compares stopping with acquiring one channel and then stopping, executes the best improving acquisition, and recalculates after the observation. It uses the complete fitted joint distribution. It is not a fixed cheap-to-expensive chain. A batch-myopic ablation also considers every legal batch but assumes termination afterward. P1 instead plans through all remaining acquisitions. Under the same feasibility rules, its policy class contains the behaviors of the principal controls.

## Information-access-matched additions

C1 and C2 cannot access V or E; consequently their gap to P1 confounds acquisition structure with available information. Four exploratory controls address this limitation. R0 stops without querying. R1 selects the best single batch from all six sources, including the option to stop; it does not adapt further. R2 is C3 restricted to A--D. R3 is P1 restricted to A--D.

P1 versus R1 tests sequential adaptation with matched source permissions. R3 versus R2 tests lookahead with workers only. C3 versus R2 tests adding the information sources to an otherwise identical myopic controller. These comparisons do not constitute an additive causal decomposition: interactions between access, stopping, and task mix remain.

# Mathematical analysis

## What universal architectural optimality cannot mean

A mandatory three-query architecture has no universal strict advantage over one that may use one query. Take a fair binary state, one perfectly accurate worker of cost 0.1, and two workers that return redundant copies at the same cost. One query has zero decision error and total loss 0.1; three have the same decision quality and total loss 0.3. This counterexample rejects mandatory extra computation as a universal principle, not adaptive policies that can decline to use it.

For preference tasks, consensus does not identify private utility without relevant information. If $Y$ is fair and the complete noneliciting output vector $O$ is independent of $Y$, any rule required to answer using only $O$ has accuracy exactly one half. Conditioning on $O=o$ leaves a fair posterior. Randomization cannot improve a convex combination of these accuracies.

The quantitative extension is useful when preferences are only partly hidden. Let $Q_y(o)=P(O=o\mid Y=y)$. Then

$$\sup_a P(a(O)=Y)=\frac{1+\operatorname{TV}(Q_0,Q_1)}{2}.$$

**Proof.** For each observation, choose the label with larger joint probability. The resulting accuracy is $\frac12\sum_o\max\{Q_0(o),Q_1(o)\}$. Expanding the maximum and using normalization yields $\frac12+\frac14\sum_o|Q_0(o)-Q_1(o)|$. This is the stated expression. Deferral outcomes are not included as correct automatic answers. $\square$

## Conditional Bayes optimality

Let $r$ be the remaining acquisition budget and $\mathcal A(h,r)$ the finite set of legal nonempty batches. Define

$$V_t(h,0)=R_t(h),$$

$$\begin{aligned}
Q_t(S\mid h,r)={}&\lambda c(S)+\mu\tau(S)\\
&+\sum_z P_t(O_S=z\mid h)V_t(h\cup(S,z),r-|S|),\\
V_t(h,r)={}&\min\{R_t(h),\min_{S\in\mathcal A(h,r)}Q_t(S\mid h,r)\}.
\end{aligned}$$

**Theorem 1.** Given the true joint distribution, the specified bounded losses and charges, action-invariant observations, finite outcomes, legal actions, and finite budget, selecting the minimizer above is Bayes optimal among the admitted nonanticipating policies.

**Proof.** Induct on remaining budget. At zero budget the terminal rule minimizes the loss of all terminal actions. Suppose the statement holds for every smaller budget. Any feasible first action at $(h,r)$ either stops or selects a legal batch $S$. Stopping cannot improve on $R_t(h)$. Selecting $S$ pays its charge and leaves smaller-budget subproblems; by induction, no continuation improves their corresponding $V_t$ values. Averaging those bounds over the batch's possible outcomes gives $Q_t(S\mid h,r)$. Choosing the minimum first action and its optimal continuations attains the lower bound. Randomized first actions form convex combinations and cannot improve the minimum. $\square$

This is the standard finite-horizon Bellman argument, included to specify the scope of the guarantee. It does not prove that an LLM's self-reported probabilities are calibrated, that the fitted model equals the world, or that richer language-model architectures fall within this finite policy class. Under action-dependent observations a different transition model would be required.

The appropriate stopping condition is $R_t(h)\le\min_S Q_t(S\mid h,r)$, not a vote count or a disagreement threshold. Agreement can coexist with a valuable verification action; disagreement can coexist with no economically useful action. In the low-stakes preference task, unobserved terminal risk is $0.05/2=0.025$, whereas the cheapest nonempty query charge is 0.081. Immediate stopping is therefore optimal.

## Approximation and distribution error

**Theorem 2.** Suppose total loss for any feasible policy is in $[0,M]$. Let $\widetilde\pi$ have objective at most $\delta$ above the optimum under an estimated distribution $\widehat P$. Then

$$J_P(\widetilde\pi)-J_P(\pi_P^*)\le 2M\operatorname{TV}(P,\widehat P)+\delta.$$

**Proof.** Add and subtract $J_{\widehat P}(\widetilde\pi)$ and $J_{\widehat P}(\pi_P^*)$. Each cross-distribution term is bounded above by $M\operatorname{TV}(P,\widehat P)$ for a bounded loss. The remaining within-model difference is at most $\delta$, because the estimated-model optimum is no larger than the estimated loss of $\pi_P^*$. Summing gives the result. $\square$

The exact fitted optimizer has $\delta=0$. This bound separates estimation and optimization errors, but is generally loose. For the high-stakes factual task, the original study's average TV is approximately 0.04763, regret 0.01626, and bound 1.9614. True TV is observable in this simulator, not usually in deployment. Unmodeled action costs or dependencies are outside this guarantee.

## Complementarity, prerequisites, and complexity

Let independent fair observations $O_1,O_2$ determine $Y=O_1\oplus O_2$. One observation is individually uninformative; both reveal $Y$. With cost 0.12 per query and unit error loss, stopping costs 0.5, one query followed by stopping costs 0.62, and acquiring both costs 0.24. A singleton-myopic rule stops too soon. A prerequisite can produce an analogous failure when a first query is valuable mainly because it enables verification. This does not imply that complete lookahead always justifies its implementation overhead.

For $m$ binary sources and budget $B$, the number of feasible unordered observation histories is

$$H(m,B)=\sum_{k=0}^B {m\choose k}2^k.$$

For $m=6,B=4$, this is 473. The full ternary table has $3^6=729$ slots. We enumerate legal batches and their outcomes, so this is an intentionally small exact solver. The exponential dependence precludes treating it as a directly scalable controller for arbitrary text, long histories, or unlimited resampling.

# Experimental design

## Synthetic distribution

Five task types have mixture probabilities $(.30,.25,.15,.15,.15)$ and wrong-answer losses $(1,5,20,.05,8)$: easy facts, hard facts, critical facts, low-stakes preferences, and high-stakes preferences. Each type has a fair prior on $Y$. Type and stakes are given to all policies in the primary analysis.

For factual tasks, workers satisfy

$$O_i=Y\oplus G\oplus(F\mathbf1\{i\in\{A,B\}\})\oplus\epsilon_i.$$

Here $G$ is a global failure bit, $F$ a shared A/B family bit, and individual noises are mutually independent. Easy facts use $P(G=1)=.015$, $P(F=1)=.035$, and individual reliabilities $(.95,.92,.90,.86)$. Hard and critical facts use $.12$, $.08$, and $(.82,.80,.77,.74)$. Individual reliability refers to the latent flipped target, not marginal correctness against $Y$. V has 0.96 accuracy independently of these failures. E is uninformative for facts.

For preferences, workers follow a population preference $Z\sim\mathrm{Bernoulli}(.7)$ with reliabilities $(.92,.90,.85,.80)$. The individual's $Y$ is independent of $Z$. V is uninformative; E reports the individual's preference with accuracy .95. This deliberately creates a case where more population consensus cannot identify private utility. It does not model rich multiobjective elicitation or general social choice.

Costs for $(A,B,C,D,V,E)$ are $(1,.8,.8,.55,1.3,.65)$; latencies are $(1,.9,.8,.65,1.1,2)$. Primary V can answer directly. The separate prerequisite analysis prevents V from running until a previous batch has provided a candidate; it does not additionally model candidate-dependent verifier accuracy.

## Calibration, pairing, and inference

Each of 20 independent repetitions draws 5,000 calibration outcomes per task type. A Dirichlet pseudocount of .5 is added to each of the 128 joint cells. Calibration sufficient counts are sampled using a multinomial distribution. An independent random stream draws 20,000 test cases per repetition. Seed sequences derive from `(260917, replicate, stream)`, with separate calibration and test streams. Policies share fitted distributions and test cases within each repetition, permitting paired comparisons without granting access to unqueried observations.

Confidence intervals use the 20 replication means and a two-sided Student-$t_{19}$ interval. The four original primary comparisons use paired differences and Holm correction. The study had a local protocol before its initial run but no external preregistration. All additions in version 1.1 are exploratory; multiplicity correction does not change that status. The 400,000 draws are not 400,000 diverse natural-language problems: five types and 128 outcomes per type yield only 640 support points.

For a fixed policy we can integrate over the entire support. Exact expectations validate sampled estimates and quantify drift. They are expectations under one designed generator, not independent external benchmarks. The added learning curves use 20 new seeds at each of five calibration sizes, with exact expected-risk evaluation and 2,660,000 additional synthetic calibration draws.

# Results

## Original workflow and principal controls

| Policy | Composite loss $J$ [95% CI] | Terminal loss | Cost | Latency | Coverage |
|---|---|---|---|---|---|
| C1 | .9535 [.9500, .9569] | .8586 | .662 | .718 | 58.13% |
| C2 | 1.0076 [1.0039, 1.0113] | .7551 | 2.163 | .906 | 62.03% |
| C3 | .3824 [.3785, .3864] | .2195 | 1.143 | 1.215 | 97.68% |
| P0 | .6148 [.6114, .6182] | .3618 | 1.973 | 1.391 | 91.26% |
| P1 | .3752 [.3715, .3789] | .2145 | 1.159 | 1.119 | 97.58% |

P0 improves on worker-only controls but loses decisively to C3. P1 reduces P0's composite objective by 38.98%, but reduces C3's by only 1.90%. Its acquisition cost is **1.36% higher** than C3's, compensated by lower terminal loss and modeled latency. These are not changes in a language-model benchmark accuracy.

The paired P1--C3 difference is $-0.00727$, 95% CI $[-0.00941,-0.00513]$, with Holm-adjusted $p=9.19\times10^{-7}$. The P0--C3 difference is $+0.23235$, CI $[0.22892,0.23578]$. Statistical significance does not settle engineering value, especially when planner and state-extraction overhead is omitted.

P1's known-distribution optimum is .372812. Its fitted policies have exact mean true objective .375401; their held-out mean is .375175. The first is a conditional oracle optimum, the second averages estimation effects, and the third also includes test sampling. They must not be used interchangeably.

The largest lookahead benefit occurs in critical factual tasks. In the known distribution, P1 starts with the parallel batch D,V; C3 starts with V alone. On ordinary hard facts and high-stakes preferences, direct access to V or E removes much of the need for a worker panel. For the P0 A,C,D panel on hard facts, agreement occurs with probability 45.19%, and conditional error given agreement is 14.33%. This is a concrete common-failure counterexample, not an empirical failure rate of current LLMs.

## Stronger information-matched controls

@@MATCHED_ZH@@

P1 improves on R1 by .006534, CI $[-.008606,-.004462]$ for P1 minus R1, or approximately 1.71%. The added exploratory Holm-adjusted $p$ is $2.57\times10^{-6}$. A single well-selected all-source batch therefore captures most of P1's benefit in this environment. R3 improves on R2 by .046729, CI $[-.049046,-.044412]$: lookahead can matter without V/E, but worker-only planning remains far behind access to the high-quality new information.

The result weakens an interpretation that deep adaptive coordination is the primary mechanism. It also makes the comparison less dependent on worker-only controls. It does not establish that R1 is sufficient on every task distribution, particularly when prerequisite or complementary observations matter.

## Ablations and negative findings

Removing V yields mean $J=.7055$; removing E yields .5178; removing C/D yields .3814. Forcing the full acquisition budget yields .5815. The batch-myopic variant yields .3815, indicating that P1's remaining gain is not explained exclusively by the ability to query in parallel.

A conditional-independence approximation yields .3737, slightly **better** than the fitted full-joint P1. Its difference from P1 is not statistically significant in the original experiment (unadjusted $p=.201$). We therefore cannot claim that full correlation estimation is necessary or reliably beneficial in this setting. Under the known true distribution, the independent approximation and complete P1 happen to choose policies with the same overall objective. Accurate single-source information can make some worker correlations decision-irrelevant.

## Tool dependencies and distribution shifts

When V requires a worker result from an earlier batch, the known-distribution objectives become C3=.7527, P0=.6143, and P1=.4000. Without the prerequisite, the corresponding known-distribution values are .3893, .6143, and .3728. The prerequisite reverses C3 versus P0. These are exact known-model comparisons, not the fitted-policy numbers in the main table. They isolate an admissibility constraint, not the full behavior of a real verifier.

Frozen-policy stress tests show further sensitivity. Reducing V accuracy from .96 to .62 increases P1's exact objective to 1.3645, worse than the worker-only C1 at .9539. Raising V's cost eightfold, E's cost fourfold, and E's latency fivefold makes P1's frozen objective 1.0760, worse than P0 at .8924. Updating the true distribution and costs can restore a lower oracle value, but this is not a demonstrated drift detector or recalibration service.

## Calibration learning curves

@@LEARNING_ZH@@

![Calibration size versus exact expected loss. Means and 95% t intervals span 20 new calibration seeds; shaded intervals do not measure uncertainty about the generator's relevance to real tasks.](figures/calibration_curve.png){width=92%}

The full-joint estimator can overfit with limited data, while the independent approximation trades structural bias for reduced estimation variance. Neither estimator uniformly dominates across sample sizes. C3 need not improve monotonically as its estimated distribution converges: it is a myopic controller, and some estimation errors can accidentally encourage better future acquisitions. These patterns are generator-specific diagnostics, not evidence that less data is generally desirable.

## Routing errors and verifier coupling

The routing analysis replaces a task label with a uniformly selected incorrect label with probability $\eta$. Policies remain frozen; the evaluator retains the true task's loss. This perturbs both the selected observation model and the perceived stakes. It is not an isolated measurement of an implemented semantic classifier.

@@ROUTING_ZH@@

At 10% routing error, P1 rises from .3754 to .5376. This damage is much larger than its small in-distribution gain over C3. Perfect task and loss identification was therefore not an innocuous assumption.

A second analysis gradually couples V to the workers' global error bit. P1 rises from .3754 at zero coupling to .8072 at complete coupling. This intervention changes **both** V's marginal accuracy and dependence structure, so it must not be described as a pure causal effect of correlation. It diagnoses the failure of a frozen controller's reliable-independent-source assumption. All intermediate levels are saved in the artifact.

## Planning overhead

@@RUNTIME_ZH@@

These CPU measurements are medians of ten repeats, each constructing all five task-type policy tables. They are environment-specific, not LLM service latency. Policies can be compiled offline and reused, so charging full compilation time to each request would also be misleading.

Holding policy behavior fixed, a constant additional P1 overhead $h$ in composite-objective units removes its point-estimate advantage over C3 when $h\ge .00727$. The analogous threshold versus R1 is approximately .00653. There is no justified conversion of these units into milliseconds or dollars. Production decisions require end-to-end measurements of the actual state extractor, planner, model calls, tools, and finalizer.

# Artifact audit and reproducibility

The provided version-1 archive was checked against its manifest, its original 38 tests were executed, and the original main run reproduced all 34 common nonmetadata outputs byte-for-byte in the local environment. After the robustness refactor and supplementary execution, all 38 original nonmetadata result files, including the supplementary files, remained identical. This is same-environment computational reproduction, not an external scientist's independent replication.

The revised suite has 86 tests. Additions validate malformed probabilities and costs, sparse support, configuration-driven deferral, budgets, candidate prerequisites, the preference identity, and the approximation bound. Most importantly, a separate tuple-history reference solver computes probabilities directly from complete outcomes without importing the production MATCH, TRANS, ADD, or ternary-history machinery. It checks random joint distributions, random charges, budget two, and both prerequisite settings. Shared problem assumptions remain; this is cross-implementation testing, not Lean/Coq verification.

`verify.sh` runs the tests, executes the complete main experiment twice, compares numerical outputs with tolerances and raw arrays exactly, verifies saved reference tables, and reruns supplementary and revision analyses. It records versions, source checksums, test counts, repeat comparisons, and the exact tested commit when available. GitHub Actions performs the same commands before a versioned release. The release should be cited together with its verification artifact; publication itself does not supply scientific validation.

The research code does not require API credentials, access private data, use a GPU, or send model prompts. Only package installation, repository operations, and document-toolchain installation need network access in the hosted workflow. Manuscript tables are generated from CSV results where marked; automated checks also validate the principal numeric claims. PDF typography and font embedding can vary across toolchain versions, so numerical reproduction is not conflated with binary-identical PDFs.

# Limitations and research-quality assessment

**Construct validity.** Binary observations do not represent free-form answers, quality of candidate generation, persuasion in debate, tool arguments, code execution, or long-context synthesis. The observation process is action invariant, whereas a real coordinator changes prompts and therefore changes distributions. Success here cannot establish better language-model outputs.

**Designed information advantage.** V/E are deliberately high-quality sources in the primary generator. Worker population preferences are independent of private utility. These choices expose mechanisms but are not fitted empirical facts about deployed agents. Information-matched controls reduce one confound without removing this dependence on the designed environment.

**Policy and estimator selection.** P0 is one concrete heuristic, not the best implementation of every proposed component. Complete joint tables are low-dimensional enough to estimate and enumerate here, but not in arbitrary text spaces. True task types and stakes are granted in the primary analysis; the routing stress test illustrates rather than solves this problem.

**Inference and novelty.** Confidence intervals describe calibration and sampling variation under the chosen generator. More draws do not establish external validity. Exploratory additions were designed after seeing earlier results. Bellman optimality, the TV error decomposition, and binary classification identity are established mathematical tools applied to this setting, not original general theory.

**Publication status.** This is an AI-assisted project technical report and executable artifact. It has not been externally peer reviewed or accepted at a venue. The authorship label names the project rather than inventing individual coauthors or institutional affiliations. Hosting it in GitHub Releases is public dissemination, not arXiv registration, DOI assignment, or journal publication.

On these grounds, the report is stronger as a transparent synthetic falsification study or an artifact-oriented workshop submission than as a new state-of-the-art methods paper. The revision improves baseline fairness, traceability, robustness checks, and precision of claims. It does not close the principal gap: empirical evidence using real language models and decision tasks.

# A prospective external-validity protocol

A next study should lock a fixed set of real model versions and a shared finalizer across policies. Factual tasks should have executable or independently sourced evaluation, with all strategies granted equivalent retrieval and verification permissions. Candidate-dependent tools must enforce their prerequisites and record failures, not supply one policy with free ground truth.

Preference tasks should specify or elicit a participant's utility independently of the judging model. Evaluation should distinguish factual support, preference fidelity, preservation of unresolved tradeoffs, and unnecessary clarification. Open-ended outputs need blinded, randomized human assessment with documented disagreement, not only a single LLM judge sharing the strategies' biases.

The development split should select thresholds and estimate acquisition value before a locked evaluation. Comparisons should report quality--cost operating points, actual token and tool charges, wall-clock time, coverage, failure rates, and planner overhead. Repeated questions to the same models are not independent evidence merely because model calls are separate. This protocol has **not** been executed in the current release.

# Conclusion

The original multi-component workflow was not validated as the best architecture. One precise implementation loses to a strong myopic cascade, and a matched nonadaptive all-source control captures most of the fitted Bellman controller's benefit. Lookahead becomes more valuable under complementary information or prerequisites, while calibration error, task misclassification, degraded verification, and omitted overhead can erase its gains.

What can be proved optimal is the solution of a specified finite decision problem under stated assumptions. A more elaborate MoA diagram does not inherit that guarantee. The supported design principle is to compare stopping, verification, preference elicitation, and additional model calls by their expected decision value under explicit costs and legal actions—and to retain negative results when those comparisons favor a simpler policy.

\clearpage
# References {-}

[1] Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. *Mixture-of-Agents Enhances Large Language Model Capabilities*. 2024. arXiv:2406.04692. <https://arxiv.org/abs/2406.04692>.

[2] Wenzhe Li, Yong Lin, Mengzhou Xia, and Chi Jin. *Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial?* 2025. arXiv:2502.00674. <https://arxiv.org/abs/2502.00674>.

[3] Lingjiao Chen, Matei Zaharia, and James Zou. *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*. 2023. arXiv:2305.05176. <https://arxiv.org/abs/2305.05176>.

[4] Isaac Ong et al. *RouteLLM: Learning to Route LLMs with Preference Data*. 2024. arXiv:2406.18665. <https://arxiv.org/abs/2406.18665>.

[5] Yonatan Geifman and Ran El-Yaniv. *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*. ICML, 2019. <https://arxiv.org/abs/1901.09192>.

[6] Lianmin Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS Datasets and Benchmarks, 2023. <https://arxiv.org/abs/2306.05685>.

[7] Nicholas Hay, Stuart Russell, David Tolpin, and Solomon Eyal Shimony. *Selecting Computations: Theory and Applications*. UAI, 2012. <https://arxiv.org/abs/1207.5879>.

[8] Smitha Milli, Falk Lieder, and Thomas L. Griffiths. *When Does Bounded-Optimal Metareasoning Favor Few Cognitive Systems?* AAAI, 2017. DOI:10.1609/aaai.v31i1.11156. <https://ojs.aaai.org/index.php/AAAI/article/view/11156>.

[9] Daniel Golovin and Andreas Krause. *Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization*. JAIR, 2011. <https://arxiv.org/abs/1003.3967>.
