# Paired QSA research on M5 Max

The new kernel stages each selected K/V block once for two queries. A GPU planner partitions their selections into private-left, private-right and shared buckets. Inactive query SIMD groups skip QK and PV operations, retaining the original selected keys and causal tails. This avoids the redundant dot products of a plain union kernel. Softmax key order changes; this is not bit identity.

Measured against **fa76a4b's `msv_qsa_nax_precise`**, on M5 Max 128 GB, the server MLX runtime (source `1f8e74e3`): **29.5954 / 29.5001 ms stock vs 14.1949 ms paired, including the GPU plan**. S=4096, KV=36864, Q24/KV2/D256, block budget512, ratio4, BF16. Same captured selection map; seeded synthetic BF16 Q/K/V. Three timed kernel repetitions per arm, one warmup, stock–paired–stock. This is **2.08× for this attention component**, not full-model tokens/s. The original uncompressed fixture SHA is in `provenance.json`.

No new full-model throughput has been measured. The user-provided full-model baseline is 1868 tok/s on fa76a4b; >1.5× means >2802 tok/s. Given previous synchronized profiles, paired QSA alone plausibly saves around 8–10% of full forward time, not the 33.3% required. A combined candidate must clear that budget before expensive model benchmarking.

## Reproduce on Apple Silicon with NAX

Use the repository's built `lib/mlx` and source headers in `lib/mlx-src`. Run from the repository root:

```sh
python3 -c 'import lzma,pathlib; p=pathlib.Path("research/qsa_pair/blocks-4096.bin.xz"); p.with_suffix("").write_bytes(lzma.decompress(p.read_bytes()))'
clang++ -O3 -std=c++20 -mmacosx-version-min=26.3 research/qsa_pair/probe.cpp -I lib/mlx-src -L lib/mlx/lib -lmlx -Wl,-rpath,"$PWD/lib/mlx/lib" -o /tmp/qsa-pair-probe
clang++ -O3 -std=c++20 -mmacosx-version-min=26.3 research/qsa_pair/validate.cpp -I lib/mlx-src -L lib/mlx/lib -lmlx -Wl,-rpath,"$PWD/lib/mlx/lib" -o /tmp/qsa-pair-validate
/tmp/qsa-pair-validate
/tmp/qsa-pair-probe
```

Coordinate exclusive GPU use with any other local benchmark tasks. The fixture contains only block indices from an earlier public-source-code prompt: no prompt text, activations, weights or credentials. The helper C++ source is standalone, but uses the server's actual Metal kernel files.

## Numerical checks already executed

`qsa-pair-validation.jsonl`: every selected-key multiset matches exactly, on every row, for B2/S17/KV17, B2/S65/KV65599, B2/S130/KV159, B1/S8192/KV65536. This covers partial pairs, causal tails, differing selections, changing geometry, batch indexing and strided Q/K/V.

The float64 oracle checks 43,008 output elements over those shapes, including full D256 for selected rows/heads. Per-element bars match the existing upstream oracle: f32 error <= max(1.5×stock error, 2.5e-6); BF16-store error <= max(1.5×stock error, 2e-3). All pass. At S8192 the BF16 maximum error was 0.000121626 for both arms. These are kernel parity checks, not broad model-quality evaluation.

## Server integration

`MLX_SERVE_QSA_PAIR=1` opts in after the existing NAX eligibility checks. Default is off. Geometry is bounded to ratio4, KB<=512, S<=8192, B<=2, KV<=131072. Configs are built per call, so output shapes do not reuse the warmup's cached shape. `src/qsa_pair.zig` and two Metal files carry the implementation; `gatherQsa256` dispatches it. The existing NAX path remains the default.

- ReleaseFast build and focused Zig test status: see `STATUS.md`.
- Focused test: `zig build test -Dtest-filter="qsa pair"`.
- HTTP script: `tests/test_qsa_pair_prefill.py`, two different-length requests and engagement assertion. Prepared; not run yet.
- Full Zig suite, HTTP model execution and llmprobe have **not** been run for this branch. No upstream PR yet. There is no separate paired-kernel startup JIT fallback probe yet; keep the opt-in experimental until that is added.

## Rejected shortcuts and next substantial lever

`projection-probe.jsonl`: pre-dequantized dense non-expert projections retained exact outputs on the fixture but mostly improved only around 10–15%. HC-up's initial 4.675 ms stock sample fell to 1.117 ms in the control; the apparent 4× result was warmup/drift and is rejected.

`qmm-retile-probe.jsonl`: larger generic NAX tiles do not materially beat the warmed stock. `moe-aligned-probe.jsonl`: expert-aligned BM64 is exact and ~15% faster on gate/down matrix components; larger BM128/192/256 lose. Previous fused gate/up and reduction improvements remain component candidates, not proven multiplicative model gains.

A representative overlap probe was also executed before rewriting the scheduler: paired QSA (including planner) 15.56 ms, GDN-sized QMM 13.76 ms; sequential pair 29.81/30.14 ms; joint eval on one stream 28.99 ms; separate GPU streams 30.09 ms. This does **not** justify a wavefront scheduler's hoped-for 1.5×. It is one representative pair, not a proof that no scheduling change can help. Avoid claiming these two stages overlap cheaply.

Further work must remove arithmetic or memory traffic in the expert chain/HC, or change their data layout and reuse. Do not spend full-model runs trying to combine this QSA win with speculative overlap or cold-sample QMM gains: the measured budget is still below the target.

Source attribution: paired attention adapts this repository's QSA NAX implementation and its Apple MLX fragment helpers, already covered by `NOTICE` and `MLX-LICENSE`. Planner, experiment drivers and integration glue are new research code.
