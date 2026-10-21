# Independent reader probability validation

Scope: offline CPU verifier correctness on PR #4, starting at
`2281caf496fb26c9c32d7c195044cc31e1659260` (2026-10-21).

## Evidence and decision

| Reference | Match and license | Decision |
| --- | --- | --- |
| [CPython 3.12.14 JSON decoder](https://github.com/python/cpython/blob/v3.12.14/Lib/json/decoder.py), `_CONSTANTS` and `JSONDecoder.__init__` | The current runtime's parser accepts NaN and both infinities by default; [PSF license](https://github.com/python/cpython/blob/v3.12.14/LICENSE). | Use the existing standard library; no copied source or new dependency. |
| Repository `jevbench/run.py::validate_score` and `jevbench/adapters.py::Adapter.score` at the starting head | The primary runner checks finiteness and adapters serialize numeric lists. | Reference the contract only; keep the independent reader independent of analysis code. |

The existing `abs(sum(p) - 1) > 1e-6 or min(p) < 0` rejection can
accept NaN because its comparisons are false. The file reader uses ordinary
`json.loads`; rejecting nonstandard constants alone would also miss overflow
from a standard JSON number such as `1e309`.

Validate finite numeric probability vectors before arithmetic, preserve integer
endpoints, direct-call tuples, signed zero, and the existing sum tolerance.
Reject booleans and malformed JSON shapes without an uncaught exception.
When contracts fail, emit diagnostics and no derived summary.

Verification uses tiny direct-contract cases and a synthetic CLI fixture with
the reader's unchanged default coverage counts and all twelve shard paths.
It uses no dataset, model, API, or GPU benchmark and proves no JevBench result.

## Red/green verification

Local runtime: Python 3.12.14 with the unchanged pinned
`requirements-transfer.txt` environment, including pytest 9.0.3.

- Original eight reader tests: exit 0, 8 passed.
- New reader suite against the starting implementation: exit 1,
  27 failed / 30 passed. Serialized `[NaN, 0.5]`, `[0.5, NaN]`,
  `[Infinity, NaN]`, and `[NaN, -Infinity]` returned CLI exit 0 and
  reported no contract problems. Malformed types could raise exceptions;
  rejected infinities still produced derived summaries.
- Same suite after repair: exit 0, 57 passed (26 invalid-vector cases,
  7 valid boundaries, 16 CLI cases, and the original 8 tests).
- Existing source-inventory suite: exit 0, 8 passed.
- `python -m compileall -q jevbench/expanded_fixtures.py
  test_jevbench_source_inventory.py verify_jevbench.py test_verify_jevbench.py`:
  exit 0. `git diff --check` and `python -m pip check`: exit 0.
- The repository has no configured standalone lint/build command; compilation
  and whitespace checks cover this Python-only patch. The existing read-only
  offline PR workflow now also runs the reader suite and compiles its files.

Coverage includes NaN in either position, signed infinities, mixed NaN/infinity,
standard JSON exponent overflow, overflowing sums and large integers, null,
booleans, strings, nested arrays, mappings, length mismatch, negative values,
integer endpoints, signed zero, tuples, and the unchanged tolerance boundary.
Other record fields and probe validation are outside this probability fix.
