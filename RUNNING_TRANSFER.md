# Run v1.3 population-transfer experiments

In GitHub open **Actions → Population transfer and typed decisions → Run workflow**. No model API key or GPU is needed. The workflow downloads the seven immutable ProEval CSVs with SHA-256 checks (six binary tasks plus the DICES ordinal diagnostic), executes all tests, refits and evaluates twice in separate Python processes, verifies the result ledgers and builds the full English and Traditional Chinese papers. Manual runs produce artifacts. Successful source-triggered runs may publish a previously unused version; existing release assets are not overwritten.

Local reproduction (Python 3.13):

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-transfer.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0
mkdir -p verification
python -m pytest -q --junitxml=verification/transfer-tests.xml
python transfer_study.py --download --out results/transfer
python audit_transfer.py --out results/transfer
python transfer_study.py --out results/transfer-repeat
python audit_transfer.py --out results/transfer-repeat
python ci_transfer.py
```

Install the document packages listed in the workflow to run `bash build_transfer_pdf.sh`. After verified execution, `python package_transfer.py` packages exact source and derived results without redistributing the raw input CSVs or font files. Timing columns are explicitly excluded from byte-identity checks; all scientific outputs are compared.

This experiment evaluates archived answers and assumed acquisition costs. The model-pool substitution is not a within-agent continual-training trajectory. The prompt/cumulative-score controls are not CERA-MoA reproductions. `decision_contract.py` supplies Jev-compatible contract tests, not live Jev inference. No empirical claim of universal MoA optimality is made.

The original v1.1 and v1.2 studies remain independently reproducible and versioned. See `LITERATURE_v1.3.md`, `V1.3_CHANGELOG.md` and `protocols/v1.3-before-results.md` for source comparisons, changes and prospective scope.
