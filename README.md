# Decision-Theoretic Mixture-of-Agents

[![Synthetic study](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/actions/workflows/reproduce.yml/badge.svg)](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/actions/workflows/reproduce.yml) [![Historical LLM replay](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/actions/workflows/external.yml/badge.svg)](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/actions/workflows/external.yml)

**When to Ask Another Model: A Finite Decision-Theoretic Falsification Study of Mixture-of-Agents**

A reproducible study of information acquisition and stopping: a finite synthetic experiment plus a separately reported replay of real archived LLM answers. Neither layer establishes universally optimal MoA orchestration or superiority over Fusion, Hermes, or Fugu.

## Latest: v1.2.0 real historical LLM replay

**[Published evidence and bilingual report](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/tag/v1.2.0)** · **[繁體中文實證附錄](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.2.0/report.zh-TW.md)** · **[English addendum](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.2.0/report.en.md)**

[Successful hosted run](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/actions/runs/35240315919) · [Full evidence package](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.2.0/external-evidence.zip) · [Machine-readable verification](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.2.0/verification.json) · [Checksums](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.2.0/SHA256SUMS) · [Execution guide](RUNNING_EXTERNAL.md)

**Actually executed on GitHub-hosted Ubuntu:** 160 tests, zero failures/skips, and two full separate-process executions including refitting and development-set selection. All 21 required outputs passed the comparison rules; 20 were byte-identical, with only runtime metadata differing in provenance. Metric ledgers, paired case sets, exact prompt-group split isolation, and independent MMLU-reference matching are checked before publication. The tested source is [`ef7018d`](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/commit/ef7018d89cd3ad9f272e212e8833660f5d4a4214).

This layer uses RouterBench's frozen historical outputs from Mixtral-8x7B, GPT-3.5 Turbo 1106, Claude-v2, and GPT-4 1106 preview. It retains **26,821 questions / 107,284 model-response records** from MMLU, ARC-Challenge, HellaSwag, and WinoGrande, with 16,089 training, 5,334 development, and **5,398 held-out test questions**. Counts are archived records, not newly generated inference calls. All policies have the same four-model pool and a maximum of three acquisitions. No perfect verifier or human-preference oracle is provided.

| Policy | Primary composite loss ↓ | Coverage | Accuracy among answered cases | Mean acquisitions |
|---|---:|---:|---:|---:|
| Selected single model | 0.191691 | 99.78% | 82.47% | 1.000 |
| Nonadaptive all-source batch | 0.175928 | 76.51% | 87.14% | 3.000 |
| Myopic information-value cascade | 0.175218 | 79.12% | 86.84% | 2.285 |
| Bellman controller | 0.174017 | 76.07% | 87.58% | 2.770 |

The objective is wrong-answer loss (1), deferral loss (0.25), plus 10 times **historical estimated USD**. These are declared study preferences, not measured human utility; costs are not current API prices or latency. Lower coverage cannot be hidden behind selective accuracy.

**No reliable advantage over the strong simpler controls was established.** Bellman minus myopic is -0.001200, with paired prompt-group 95% bootstrap interval [-0.004219, 0.001759]; Bellman minus static is -0.001911 [-0.004791, 0.000970]. Both cross zero. This is not proof of equivalence or noninferiority. Bellman minus single is -0.017674 [-0.024048, -0.011085], but quality, deferral and acquisition costs all contribute to that difference. [Exact paired results](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.2.0/paired-comparisons.csv).

The audit independently matches **14,028/14,042 MMLU questions**, checking 56,112 response/label pairs with zero strict-score disagreements. For 13,480 questions, the correct option is remapped by text because choice order differs. This does not certify benchmark gold correctness or exclude pretraining contamination.

A frozen zero-to-five-shot stress test matches 5,394 held-out questions without target refitting. Formatting, behavior, and prompt-length costs change together; it is not isolated distribution shift. Claude-v2 has 10,330 strict-parser INVALID responses across the 26,807 eligible five-shot rows. INVALID cases remain included. Few-shot demonstration overlap is not audited. See the bilingual addendum for these limitations and the complete results.

**New LLM API calls: zero.** This is real historical-answer replay, not traditional-classifier proxy data, live generation, inter-agent debate, free-text synthesis, human preference evaluation, or a present-day model ranking. The earlier locally shared classifier-proxy candidate is not the source of the numbers above. The v1.1 PDFs below remain the original synthetic report; the v1.2 Markdown addendum and artifacts extend rather than silently rewrite its evidence.

To rerun on GitHub, open **Actions → External validity replay → Run workflow**. Manual runs upload verified artifacts; only successful source-triggered runs can publish a previously unused version. The evaluation job has read-only repository permissions; a separate gated publication job has write permission. Existing released tags/assets are not overwritten. Local commands are in [RUNNING_EXTERNAL.md](RUNNING_EXTERNAL.md).

## Original v1.1 paper and artifacts

**[English paper (PDF)](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.1.0/paper.en.pdf)** · **[完整繁體中文論文 (PDF)](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.1.0/paper.zh-TW.pdf)**

[Versioned release](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/tag/v1.1.0) · [Full reproducibility package](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.1.0/reproducibility-package.zip) · [Verification record](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.1.0/verification.json) · [Checksums](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.1.0/SHA256SUMS)

A public GitHub technical report is **not peer review, venue acceptance, an arXiv deposit, or a DOI registration**.

## Original synthetic experiment

Six binary observation channels; five task types; correlated failures; a maximum of four acquisitions; positive query costs, latency, and deferral loss. The primary experiment uses 20 independent calibration/test replications: 500,000 synthetic calibration draws and 400,000 held-out cases. There are only 640 complete support points—not 400,000 diverse natural-language questions.

**Synthetic-layer LLM endpoint calls: zero. GPU use: none.** Its scores are normalized decision losses, not API dollars, token usage, measured service latency, or real-model benchmark accuracy. The synthetic and historical-replay objectives differ and cannot be directly compared as an improvement across versions.

| Policy | Synthetic mean composite loss ↓ | Meaning |
|---|---:|---|
| C1 | 0.9535 | Per-task-type selected single worker |
| C2 | 1.0076 | Per-task-type fixed three-worker panel |
| C3 | 0.3824 | Myopic value-of-information cascade |
| P0 | 0.6148 | Original risk/disagreement heuristic |
| P1 | **0.3752** | Fitted finite-horizon Bellman controller |
| R1 | 0.3817 | Added all-source, nonadaptive batch control |

Within the synthetic experiment, P1 improves on C3 by 1.90%, and on the information-access-matched R1 by 1.71%. Removing correlation modeling does not significantly hurt in that distribution. Degraded verifiers, wrong task labels, and omitted planning overhead can erase the advantage. Negative findings are part of the result, not exceptions hidden from the report.

## Reproduce the original synthetic layer

Use Python 3.13 and the pinned dependencies. No credentials are needed.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
bash verify.sh
```

This runs the original 86-test suite, the main experiment twice, compares 34 main outputs, checks version-1 reference tables, and executes the supplementary and revision studies. The external workflow runs the expanded 160-test suite. Numeric tables use explicit tolerances across environments; within-runtime raw arrays must match exactly.

To build both original papers on Debian/Ubuntu, install Pandoc, XeLaTeX, `lmodern`, `texlive-lang-chinese`, `texlive-latex-extra`, `texlive-fonts-recommended`, Noto CJK, Liberation, and DejaVu fonts, then:

```sh
bash build_pdf.sh
python package_release.py
```

Installed fonts are used by the renderer; font files are not distributed. PDF byte identity is not a scientific-reproducibility criterion.

## Source map

| File | Responsibility |
|---|---|
| `external_replay.py` | Hash-verified historical source, parsing, split, fitted policies and held-out replay |
| `external_audit.py` | Independent MMLU-reference grading and frozen five-shot stress test |
| `external_protocol.json`, `external_audit_plan.json` | Primary replay and explicitly post-primary audit specifications |
| `ci_external_evidence.py` | Full double-run comparison, independent ledger checks, bilingual reports and commit-bound verification |
| `test_external.py`, `test_external_ci.py` | Replay correctness and fail-closed evidence-gate tests |
| `RUNNING_EXTERNAL.md` | Hosted and local execution instructions |
| `study.py` | Synthetic generator, terminal risk, baseline policies, Bellman solver and main experiments |
| `test_study.py`, `test_revision.py` | Original 38 tests and 48 revision tests, including a separate tuple-history solver |
| `supplement.py` | Candidate-before-verification and updated-oracle sensitivity |
| `revision.py` | Matched controls, learning curves, routing error, verifier coupling and CPU diagnostics |
| `protocol.json`, `revision_protocol.json` | Original local protocol and explicitly exploratory post-v1.0 additions |
| `verify.sh`, `verify_results.py` | Synthetic checks and provenance |
| `paper.en.template.md` | Full original English manuscript source |
| `paper_template.md`, `revision_section.zh.md` | Full original Traditional Chinese manuscript source |
| `REVIEW.md` | Research-quality audit of the v1.1 synthetic report |

## Scope of the proof

Bellman induction establishes Bayes optimality **only** for the declared finite action space, known observation law, fixed costs/losses, and action-invariant potential observations. A fitted or approximate policy additionally incurs distribution and optimization error. The original report derives the standard bound `regret <= 2 M TV(P, P_hat) + delta` and a binary preference-information identity. None proves universal architectural superiority or a real-world advantage of the historical-replay controller.

## Revision and disclosure

v1.1 corrected a documentation mismatch: C1/C2 were selected separately by task type, not globally. Its unchanged primary experiment reproduced the original numeric results and added matched controls, sparse-support fixes and independent solver checks. Those additions were exploratory, not retroactively preregistered.

v1.2 adds the historical-response evidence layer. This hosted execution revision preserves the existing remote primary analysis and adds two full executions, fail-closed evidence gates, public artifacts and data-derived reporting. The source and protocols are archived with the tested commit; observing the existing results before infrastructure hardening is not disguised as prospective preregistration.

AI assistance was used for drafting, implementation, and checking. There are no invented institutional affiliations, external reviewers, or acceptance claims. Citation metadata for the original report is in `CITATION.cff`; cite the v1.2 release and its tested commit for the external appendix. An explicit project reuse license has not yet been selected by the repository owner; upstream data retains its own terms. Raw upstream prompt/response text is not redistributed in the external release.
