# Independent re-derivation of the v1.6 native-head evidence

Run `35364030597` executed the native Jev-like checkpoints and uploaded twelve paired shard
artifacts plus a separate gold artifact. `verify_jevbench.py` re-reads that evidence with an
implementation that shares no code with `jevbench.analyze`, re-checks every digest and coverage
contract itself, and recomputes native accuracy and measured latency from the raw records.

It covers the native heads only. The fitted-policy layer is not re-derived here and is not
endorsed by this note.

## Contract checks (all passed)

- `requests.jsonl` and the separate `gold.jsonl` match the digests in their manifests.
- All 320 per-request digests recompute, and request ids align exactly with the gold ids.
- Each model produced 320 primary records, 16 reversal probes and 8 repeat probes.
- Every primary record preserves the fixture option order, sums to one, makes exactly one
  backbone call with zero decode steps, and reports a positive measured latency.

## Native accuracy on 32 held-out test groups per task

| Model | BoolQ | OCNLI | CLINC | TMMLU+ | All (128) | 95% Wilson |
|---|---:|---:|---:|---:|---:|---|
| NanoJev (0.6B) | 0.656 | 0.344 | 0.344 | 0.250 | 0.398 | [0.318, 0.485] |
| decider-2b | 0.781 | 0.750 | 0.938 | 0.469 | 0.734 | [0.652, 0.803] |
| System One scorer (4B) | 0.875 | 0.719 | 0.844 | 0.312 | 0.688 | [0.603, 0.761] |

## Measured same-host latency, milliseconds per request

| Model | p50 | p95 |
|---|---:|---:|
| NanoJev (0.6B) | 1578.7 | 3970.2 |
| decider-2b | 4324.1 | 5756.3 |
| System One scorer (4B) | 13195.3 | 29613.6 |

Latency is the model call recorded by the adapter on hosted CPU with four Torch threads. It is
not GPU or service latency, and it is not integrated serving latency for any multi-model policy.

## Agreement with the analysis reader

Both readers agree exactly on every accuracy, every Wilson bound and every latency percentile
once the quantile convention is aligned: `verify_jevbench.py` uses linear interpolation, the
same convention as the numpy and pandas defaults used by `jevbench.analyze`. Before that
alignment the two differed only by the percentile definition, never by a decision.

## What the numbers do and do not say

The ordering is what the capacity difference would suggest: the 0.6B NanoJev is fastest and
least accurate overall, the 4B System One scorer is slowest on CPU, and decider-2b leads on
this mixture. Its CLINC score is the least surprising of the four tasks, because CLINC families
are inside its training mixture, and its protocol entry already records that. TMMLU+ is weak
for all three models, which is consistent with the recorded limits: knowledge-heavy multiple
choice is not what a single-pass option scorer is built for, and Chinese sits outside the
decider model declared language.

Two things are deliberately absent. There is no proprietary TypeSafe Jev call anywhere in the
evidence, and no training or fine-tuning step was performed to produce these numbers.
