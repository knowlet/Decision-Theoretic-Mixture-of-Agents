# Audited synthetic decision study — v1.1.0

Public, non-peer-reviewed technical report with complete English and Traditional Chinese manuscripts, executable source, raw synthetic cases, results, and verification provenance.

- Original default results reproduced; C1/C2 task-conditional baseline descriptions corrected.
- 86 tests, including a separately implemented tuple-history reference solver.
- Stronger all-source nonadaptive control: mean J=0.3817 versus P1=0.3752 (approximately 1.71% improvement).
- Added calibration learning curves, task-routing errors, verifier coupling, planning diagnostics, and approximation-error analysis.
- Explicitly retained negative results and limitations. New analyses are exploratory.

No LLM endpoint was called. No commercial MoA product was benchmarked. Mathematical optimality is conditional on the specified finite model. This release is not peer review, venue acceptance, an arXiv registration, or a DOI assignment.

Download `verification.json` for the tested commit, runtime, and repeat checks. `reproducibility-package.zip` contains raw arrays, source, manuscripts, tests, and computed tables. Verify assets against `SHA256SUMS`.
