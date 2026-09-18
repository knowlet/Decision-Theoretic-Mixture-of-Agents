# Six-task replay and widened OpenJev pilot — v1.5.0

Public, non-peer-reviewed technical report with complete English and Traditional Chinese manuscripts, executable source, paired per-case records and fail-closed verification.

- Adds `gqa` and `jigsaw` from the same pinned ProEval revision: 8,656 questions, 1,732 held-out binary questions, six primary tasks.
- Widens the executed OpenJev selector pilot from 128 to 192 held-out groups with 96 development cases; 711 local forward passes across fifteen shards.
- The four earlier datasets keep their split salt and reproduce the v1.3 per-task table, apart from machine-precision summation in one Brier score.
- Selector request text is now canonical: estimates are rounded to eight decimal places, removing the cross-host request-hash failure that v1.4 patched with a ULP recovery search.
- Reports the deferral-dominated task instead of dropping it, and separates action-ID flips from answer or deferral changes in the order probe (42/48 action flips is only 9/48 semantic changes for 4B).
- 296 regression tests, fifteen shards with 11 upstream and 21 interface tests each, two separate-process replays and every ledger gate.

No proprietary Jev endpoint, CERA-MoA training, live worker generation or human preference experiment was run. The measured disadvantage of the calibrated selector is specific to this archive, this declared loss and this panel, and is not a claim about reasoning quality.

---

# Audited synthetic decision study — v1.1.0

Public, non-peer-reviewed technical report with complete English and Traditional Chinese manuscripts, executable source, raw synthetic cases, results, and verification provenance.

- Original default results reproduced; C1/C2 task-conditional baseline descriptions corrected.
- 86 tests, including a separately implemented tuple-history reference solver.
- Stronger all-source nonadaptive control: mean J=0.3817 versus P1=0.3752 (approximately 1.71% improvement).
- Added calibration learning curves, task-routing errors, verifier coupling, planning diagnostics, and approximation-error analysis.
- Explicitly retained negative results and limitations. New analyses are exploratory.

No LLM endpoint was called. No commercial MoA product was benchmarked. Mathematical optimality is conditional on the specified finite model. This release is not peer review, venue acceptance, an arXiv registration, or a DOI assignment.

Download `verification.json` for the tested commit, runtime, and repeat checks. `reproducibility-package.zip` contains raw arrays, source, manuscripts, tests, and computed tables. Verify assets against `SHA256SUMS`.
