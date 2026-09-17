#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0 MPLBACKEND=Agg
mkdir -p verification
python -m pytest -q test_study.py test_revision.py --junitxml=verification/junit.xml | tee verification/tests.log
python study.py --output results | tee verification/main.log
python study.py --output .repeat-results | tee verification/repeat.log
python verify_results.py --repeat .repeat-results
python supplement.py | tee verification/supplement.log
python revision.py --results results | tee verification/revision.log
python verify_results.py --repeat .repeat-results --final
