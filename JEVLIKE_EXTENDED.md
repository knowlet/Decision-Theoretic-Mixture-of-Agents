# v1.7 candidate Jev-like extension

This candidate adds three pinned native decision checkpoints to the same
gold-blind semantic fixture used by the v1.6 pilot:

- Mapika/decider-2b at revision b37f7e1ba3fbc9238004cf531fabbee2619973fd.
- jaredpalmer/kev-0.8b at revision c917edefdfd72b3e9ba71455584700acc70595f6, with the upstream source pinned to e0bcf50153f1bda4ca6a8be5e12cbd5f9ebbce1c.
- convaiinnovations/laya-typed-decisions at revision f9ab0b228f0fc0f14d873dbc99038f135c2da1b2, with the upstream source pinned to 42626c348753fbb17572a813127df2278a1ec527.

The native cohort uses 320 semantic cases: 32 fit, 16 development, and 32
held-out test cases per BoolQ, OCNLI, CLINC, and TMMLU+. Each model receives the
same request file and never receives gold.jsonl. Each model produces 320 primary
records, 16 reverse probes, and 8 repeat probes, plus 12 unreported warmups
(356 total forward passes per model). The workflow records native probabilities,
accuracy, Brier score, NLL, ECE, p50/p95 CPU latency, token counts, and
option-order/repeat probes. The result is a v1.7 candidate and is not merged
into the v1.6 pilot numbers.

harshatheg/Qwen-2.5-1B-RLCD is included in a separate schema smoke job. Its
public implementation is a constrained-generation engine around
Qwen/Qwen2.5-1.5B-Instruct, with the PyTorch backend selected on the Ubuntu
runner. Its JSON validity, schema match, exact-match rate, forward count, and
latency are reported separately from native typed decision heads; they are not
accuracy numbers on the BoolQ/OCNLI/CLINC/TMMLU+ fixture.

The workflow is at .github/workflows/jevlike-extended.yml.
The immutable model registry is at jevbench/extended_registry.py.
