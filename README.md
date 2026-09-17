# Decision-Theoretic Mixture-of-Agents

[![Reproduce](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/actions/workflows/reproduce.yml/badge.svg)](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/actions/workflows/reproduce.yml)

**When to Ask Another Model: A Finite Decision-Theoretic Falsification Study of Mixture-of-Agents**

A reproducible synthetic study of information acquisition, verification, private preferences, and stopping. The original multi-component heuristic is **not** optimal in the studied environment. Full planning gives only a small improvement over strong simpler controls.

## Public paper and artifacts

**[English paper (PDF)](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.1.0/paper.en.pdf)** · **[完整繁體中文論文 (PDF)](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.1.0/paper.zh-TW.pdf)**

[Versioned release](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/tag/v1.1.0) · [Full reproducibility package](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.1.0/reproducibility-package.zip) · [Verification record](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.1.0/verification.json) · [Checksums](https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/releases/download/v1.1.0/SHA256SUMS)

Release assets are produced only after the hosted verification workflow succeeds. A public GitHub technical report is **not peer review, venue acceptance, an arXiv deposit, or a DOI registration**.

## What was actually tested

Six binary observation channels; five task types; correlated failures; a maximum of four acquisitions; positive query costs, latency, and deferral loss. The primary experiment uses 20 independent calibration/test replications: 500,000 synthetic calibration draws and 400,000 held-out cases. There are only 640 complete support points—not 400,000 diverse natural-language questions.

**LLM endpoint calls: zero. GPU use: none.** Scores are normalized decision losses, not API dollars, token usage, measured service latency, or real-model benchmark accuracy. No result ranks Fusion, Hermes, Fugu, or any commercial product.

| Policy | Primary mean composite loss ↓ | Meaning |
|---|---:|---|
| C1 | 0.9535 | Per-task-type selected single worker |
| C2 | 1.0076 | Per-task-type fixed three-worker panel |
| C3 | 0.3824 | Myopic value-of-information cascade |
| P0 | 0.6148 | Original risk/disagreement heuristic |
| P1 | **0.3752** | Fitted finite-horizon Bellman controller |
| R1 | 0.3817 | Added all-source, nonadaptive batch control |

P1 improves on C3 by **1.90%**, and on the information-access-matched R1 by **1.71%**. Removing correlation modeling does not significantly hurt in the original distribution. Degraded verifiers, wrong task labels, and omitted planning overhead can erase the advantage. These negative findings are part of the result, not exceptions hidden from the report.

## Reproduce

Use Python 3.13 and the pinned dependencies. No credentials are needed.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
bash verify.sh
```

This executes **86 tests**, runs the main experiment twice, compares 34 main outputs, checks version-1 reference tables, and executes the supplementary and revision studies. It writes `verification/attestation.json`, raw paired cases, exact expectations, confidence intervals, and logs. Numeric tables use explicit tolerances across environments; within-runtime raw arrays must match exactly.

To build both papers on Debian/Ubuntu, install Pandoc, XeLaTeX, `lmodern`, `texlive-lang-chinese`, `texlive-latex-extra`, `texlive-fonts-recommended`, Noto CJK, Liberation, and DejaVu fonts, then:

```sh
bash build_pdf.sh
python package_release.py
```

Installed fonts are used by the renderer; font files are not distributed. PDF byte identity is not a scientific-reproducibility criterion.

## Source map

| File | Responsibility |
|---|---|
| `study.py` | Generator, terminal risk, baseline policies, Bellman solver, main experiments |
| `test_study.py` | Original 38 tests |
| `test_revision.py` | 48 added tests, including a separately implemented tuple-history solver |
| `supplement.py` | Candidate-before-verification and updated-oracle sensitivity |
| `revision.py` | Information-matched controls, learning curves, routing error, verifier coupling, CPU diagnostics |
| `protocol.json` | Original local protocol; not externally preregistered |
| `revision_protocol.json` | Explicitly exploratory, post-v1.0 additions |
| `verify.sh`, `verify_results.py` | Executed checks and machine-readable provenance |
| `paper.en.template.md` | Full English manuscript source |
| `paper_template.md`, `revision_section.zh.md` | Full Traditional Chinese manuscript source |
| `REVIEW.md` | Candid research-quality assessment and remaining gaps |

## Scope of the proof

Bellman induction establishes Bayes optimality **only** for the declared finite action space, known observation law, fixed costs/losses, and action-invariant potential observations. A fitted or approximate policy additionally incurs distribution and optimization error. The report derives the standard bound `regret <= 2 M TV(P, P_hat) + delta` and a binary preference-information identity. None proves universal architectural superiority.

## Revision and disclosure

v1.1 corrects an important documentation mismatch: C1/C2 were selected **separately by task type**, not globally. The unchanged primary experiment reproduced the original numeric results. The revision adds stronger matched controls, sparse-support and configuration fixes, independent solver checks, and a complete English paper. New analyses are exploratory, not retroactively preregistered.

AI assistance was used for drafting, implementation, and checking. There are no invented institutional affiliations, external reviewers, or acceptance claims. Citation metadata is provided in `CITATION.cff`. An explicit reuse license has not yet been selected by the repository owner; public visibility is not a blanket license grant.
