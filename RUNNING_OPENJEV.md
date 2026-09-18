# Actual OpenJev versus empirical decision controls

In GitHub Actions, **OpenJev versus decision-theoretic controls** executes actual local selector inference on two immutable Qwen3.5 checkpoints. Fifteen CPU workers execute the locked 192 test / 96 development cases across six primary datasets, with disjoint shards and same-worker reversal/repeat probes. They do not call proprietary TypeSafe Jev or generate new worker answers. Read `openjev_protocol.json` and `OPENJEV_RUNTIME_CHANGELOG.md` together: the latter discloses FP32 Linear/Conv accumulation under BF16 weight/output storage and per-worker warmups.

The comparison includes nine empirical policies, an exactly matched-panel majority rule, OpenJev raw and development-calibrated variants. Inference uses the unchanged upstream `direct.score` from `TheoLeeCJ/openjev` at `b4782a6c953f05c6255706d7a219f4e032af5b58`. Every downloaded dataset and checkpoint is version-pinned, and the direct scorer Git blob is verified. No output is silently truncated, no unknown option is accepted, and a scoring failure fails the experiment rather than being replaced with mock output.

An analysis job merges every shard, fits a single 96-case development calibrator per model, checks the exact source and sample manifests, runs all regression tests, repeats the full v1.5 seven-file study twice, and emits paired losses, coverage, calibration, order robustness and CPU timings. The model-pool and selector experiments are separate evidence layers; repeated shards do not create independent new questions. The primary worker-acquisition loss omits selector compute, so a separate hypothetical selector-overhead grid is published without inventing dollar prices.

Local numerical analysis can be rerun from the published `evidence.zip` without downloading model weights:

```sh
python -m pip install -r requirements-transfer.txt
python analyze_openjev.py --inputs inputs --out results/openjev-rechecked
```

To repeat inference, install `torch==2.10.0` from the CPU wheel index, then `requirements-openjev.txt`, and download the pinned OpenJev source to `vendor/openjev`. For example:

```sh
python openjev_sharded.py shard --model qwen35-4b --index 0 --download --out results/shard
```

All twelve 4B and three 0.8B shards are required before `finalize`; do not interpret a partial shard as the full benchmark. GPU use requires a separately disclosed runtime variant, not reuse of CPU timing claims. Raw model weights, source question/answer text and fonts are excluded from release artifacts. Earlier versions and the native-CPU feasibility attempt remain in Git history and Actions logs.
