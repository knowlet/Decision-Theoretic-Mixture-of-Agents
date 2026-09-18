# Runtime-only amendment to the locked OpenJev pilot

The task, model identities, data splits, 128 test / 64 development groups, candidate panels, costs, calibration rule and contrasts in `openjev_protocol.json` are unchanged. This amendment is motivated by owned compatibility probes and wall-clock feasibility, not benchmark quality outcomes.

On hosted AMD EPYC 7763 CPU, native BF16 execution took 33.37 seconds for one 100-token owned routing prompt (separate smoke example: 37.52 seconds / 113 tokens). Explicit FP32 accumulation inside Linear and Conv1d, with unchanged BF16 stored weights and BF16 layer outputs, took 9.45 seconds for the 100-token prompt and 29.82 seconds for a 521-token prompt. On the short comparison, the selected option was unchanged, the largest option probability difference was 2.86e-6, but an option logit differed by 0.125. This is NOT a bit-identical numerical reproduction. The compatibility probe does NOT establish a uniform precision bound on benchmark cases.

The scored implementation is therefore explicitly a **CPU-accumulation variant of the original OpenJev readout**. Its `direct.score`, tokenization, prompts and source hash are unchanged; model parameters are not trained or quantized. Runtime metadata pins the adapter source. The native serial attempt is superseded to avoid timeout, not counted as a completed head-to-head.

Execution is split into eight independent 4B workers and two 0.8B workers. Each primary question is scored once per model; its order and repeat probes run on the same worker as its original call. The global 64-case calibrator is fit only after merging all workers. Warmup changes from one per model to one per worker: 8 + 2 = 10 warmups, in addition to 384 primary and 80 robustness forward passes, for 474 calls in the final benchmark execution. Warmup is excluded from latency summaries. These counts exclude earlier owned runtime probes and the incomplete superseded attempt.

The original protocol file is preserved rather than rewritten; this runtime amendment is part of the published source. No claim compares these CPU timings with TypeSafe Jev service or the upstream author's GPU timings.
