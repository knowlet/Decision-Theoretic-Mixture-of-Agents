# Execute the real historical LLM replay

This path supersedes the traditional-classifier proxy as the project's main external-evidence layer. It reuses real archived RouterBench responses, not generated correctness bits. It does not call a live LLM endpoint, reproduce Fusion/Hermes/Fugu, or test open-ended human preferences.

## GitHub-hosted execution

The `External validity replay` workflow (`.github/workflows/external.yml`) runs on source changes to `develop`, and supports the Actions **Run workflow** button. A run checks out the triggering commit with credentials disabled, installs Python 3.13 and pinned scientific dependencies, and executes all original, replay, and publication-gate tests.

It then performs two complete, separate-process executions. Each execution refits probabilities, selects hyperparameters on the development split, evaluates the locked held-out split, runs ten cost/deferral settings, holds out each benchmark family, independently regrades matched MMLU questions by option text, and tests the frozen policy against five-shot archives. This is not just reevaluating a previously cached policy.

The final gate independently recomputes the primary metric ledger, checks exact prompt-group split isolation and paired case sets, requires all 21 outputs, and compares both runs. CSV and scientific JSON outputs must be byte-identical. NPZ arrays must be exactly equal with pickle loading disabled. Only runtime-specific metadata may differ in `provenance.json`. Empty, failed, skipped or unmatched tests/results prevent publication. Shell pipelines use `bash` with `pipefail` so `tee` cannot hide failures.

A separate publication job has write permission; the replay job is read-only. A release is created only after the evidence gates succeed. Existing v1.1.0/v1.2.0 assets are never overwritten by reruns. Manual runs produce artifacts without publishing. The published evidence contains original source, both result sets, JUnit, environment, source revisions/hashes, bilingual reports and per-file verification hashes. Raw third-party prompt/response text and source pickle files are not redistributed in the release.

## Local reproduction of the same steps

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-external.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0
mkdir -p verification
python -m pytest -q test_study.py test_revision.py test_external.py test_external_ci.py --junitxml=verification/all-tests.xml
python external_replay.py --download --output results/external
python external_audit.py --results results/external
python external_replay.py --output results/external-repeat
python external_audit.py --results results/external-repeat
python ci_external_evidence.py
```

Read `dist-external/report.en.md`, `report.zh-TW.md`, and `verification.json`. A git checkout is required to bind the evidence to its source commit. No API keys are needed. Data downloads require access to the fixed Hugging Face URLs declared in the source. SHA mismatches fail closed, including cached files.

## Interpretation

The primary objective uses wrong-answer loss 1, deferral loss 0.25, and ten times historical estimated USD. It is not accuracy alone. Selective accuracy applies only to answered cases and must be read alongside coverage. Confidence intervals are conditional on fitted policies and cluster exact normalized prompt groups; they do not establish representativeness, rule out pretraining contamination, or include retraining uncertainty.

The five-shot analysis simultaneously changes answer format, model behavior and prompt-length costs; it is a compound stress test. The strict parser's INVALID category remains included. The synthetic Bellman theorem is not a claim of universally optimal real LLM orchestration. A successful CI badge establishes executable computational reproducibility, not peer review or a method beating all baselines.
