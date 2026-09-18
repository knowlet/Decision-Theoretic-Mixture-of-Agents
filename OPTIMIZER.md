# Answer-aware optimizer (v1.6 candidate, not a release)

This extension preserves v1.5 and offers three separable changes: richer observations, prompt-conditioned calibration, and exact-behavior compilation. It does not run or claim an improvement to proprietary Jev/CERA. Existing held-out benchmarks have been examined previously, so new performance findings are exploratory.

## Interfaces

`adaptive_router.Pool` binds ordered worker IDs to an explicit revision. `CompiledPolicy` returns an acquired worker index or -1 (defer), plus risk, query order, batches and final state. A caller supplies a callback that returns normalized strings or None for invalid answers. `execute` checks the pool before any acquisition. Check artifact hashes against a trusted manifest; JSON schema validation does not authenticate policy files.

The compiler evaluates a joint empirical error distribution, preserving correlations. Equality mode identifies answer agreement; polarity mode additionally distinguishes normalized `yes` and `no`. Polarity is only appropriate when those labels have stable task semantics. It is not a generic semantic model and cannot infer private values.

`ContextRouter` uses a training-only character-feature ridge head to split prompts into two estimated-difficulty bins. Conditional risk tables fit separate calibration records and shrink toward the global joint distribution; leaves below 32 cases use the global table. Context-head coefficients and policy tables serialize as bounded numeric JSON, never pickle.

## Reproduce

```sh
python -m pip install -r requirements-transfer.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0
mkdir -p verification
python -m pytest -q test_*.py --junitxml=verification/optimizer-tests.xml
python optimizer_study.py --download --out results/optimizer
python optimizer_diagnostics.py --out results/optimizer
python optimizer_study.py --out results/optimizer-repeat
python optimizer_diagnostics.py --out results/optimizer-repeat
python optimizer_ci.py
python optimizer_report.py
```

The Actions workflow **Answer-aware optimizer validation** executes the same path and uploads exact source, both result sets, compiled policy JSON files, tests, machine environment, hashes and bilingual reports. It has read-only repository permission and does not overwrite a release. Old workflows and old result protocols are unchanged.

## Deploy a verified table

From the repository root, after generating/extracting the evidence:

```python
from pathlib import Path
from adaptive_router import CompiledPolicy, Pool
import transfer_study as source

pool = Pool(tuple(source.POOL_NAMES['replacement']),
            '9d37b037d1d91f4b78a7ddd74a43a8aecb344c66:replacement')
policy = CompiledPolicy.from_json(
    Path('results/optimizer/policies/jigsaw-1601-compiled_legacy.json').read_text(),
    expected_pool=pool,
)
# Illustrative acquired-response provider. A real provider must normalize its
# actual response and cache the returned candidate under the same worker ID.
responses = {0: 'no', 1: 'yes', 2: 'no', 3: 'no'}
chosen, risk, queried, batches, state = policy.execute(
    lambda worker: responses[worker], pool=pool)
print('defer' if chosen == -1 else responses[chosen], queried)
```

`compiled_legacy` preserves incumbent behavior and is the lowest-risk implementation upgrade. `guarded_selected` only changes behavior after independent gate approval. `polarity_bellman`, `context_bellman` and `context_polarity` are experimental candidates; do not promote them simply because a test average looks better. Context exports require `context_from_json` and an explicit prompt.

## Loss, uncertainty and minority-class failure

The primary objective prices wrong answers equally, deferral at 0.25 and each archived answer at 0.01. These are assumed utility units, not billing. Context CPU costs are separately outside that objective. A lower mean loss can coexist with worse minority-class detection. The post-primary per-class audit is mandatory in reports: in the initial local result the apparent Jigsaw gain removed many non-toxic false positives but no toxic test case was correctly returned by the new candidates. Thus no unqualified detection-quality upgrade or automatic default replacement is claimed.

The promotion gate is independent of tuning. For the one preselected challenger, D in [-M,M], M=1.03, the empirical-Bernstein upper bound is

`mean(D) + sqrt(2 sample_var(D) log(2/alpha)/n) + 7 (2M) log(2/alpha)/(3(n-1))`.

Only a negative upper bound promotes. This is an application of Maurer & Pontil (2009), arXiv:0907.3740, under independent groups, fixed selected policies and stationary deployment. It is per-dataset, conservative, not a simultaneous all-task guarantee and not valid under arbitrary distribution shift. A known/reused benchmark does not become independent deployment evidence just because it is split again. Non-promotion and per-task regressions are valid outputs, not gate failures.

The original train/dev/test assignment is retained. Representation/risk fitting and tuning/gate partitions are group-separated using the fixed protocol. Seeds 1602/1603 repeat selection sensitivity on the same test set, not independent scientific replications. No test label selects a parameter, and the new candidates cannot see unacquired worker answers.

## Proof and verification scope

The larger state space supports the same conditional finite Bellman proof, but a richer fitted model is not guaranteed to generalize better. A separate recursive enumerator checks the polarity policy's expected risk without production transition tables. Compiled incumbents match all 52 equality/INVALID patterns, five supported legacy modes and multiple generated distributions; fitted experiment policies are checked again over all patterns and test cases. Serialization must preserve actual decisions. Model-pool revision, unknown actions, malformed states, invalid risks and unacquired outputs fail closed.

Timing compares legacy versus compiled callbacks in the same process with interleaved execution. It excludes fitting, compilation and LLM serving; it is not a model-intelligence or token-throughput claim. Reports and evidence are generated from successful hosted runs rather than hard-coded score tables.
