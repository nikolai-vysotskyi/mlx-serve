# HC write/read fusion research

M5 Max 128 GB, fa76a4b, the server MLX runtime (source 1f8e74e3). Full HC write/read component, M8192/HC4/H2560/R320: **stock 6.14867 / 6.12183 ms → 3.91179 ms (~1.57×)**. Three timed repetitions per arm after warmup; seeded synthetic BF16 activations, affine4/gs64 projections. Both arms include write, group norm, inject, down/up GEMMs, SiLU, sigmoid, and four-stream mixing. `results.jsonl` contains the final run. No model throughput claim or server integration yet.

The new first kernel combines the pending write, group RMS normalization, the per-stream normalization weight, and the four inject dot products. The last kernel directly mixes four streams and avoids a full-width sigmoid/product intermediate. Both projection GEMMs stay on MLX's existing NAX path. The exact BF16 rounding sites of the write, norm and mean are retained. A first version using FP32 mean accumulation was rejected: MLX's Metal mean sums this axis in BF16.

On the fixture, written stream, normalized stream and mixed output are bit-identical to the stock chain. The changed inject reduction has differences up to 0.0078125 in raw/output gates. A float64 dot-product oracle checks 144 gates, including the row with largest raw difference: new gates have zero mismatches, stock one. Maximum raw errors are 0.00390571 new and 0.00390679 stock. This is limited numerical evidence, not broad quality proof. Injection reduction order remains the item to cover with varied shapes/inputs before server integration.

Run from repository root after building `lib/mlx`:

```sh
clang++ -O3 -std=c++20 -mmacosx-version-min=26.3 research/hc_prefill/probe.cpp -I lib/mlx-src -L lib/mlx/lib -lmlx -Wl,-rpath,"$PWD/lib/mlx/lib" -o /tmp/hc-prefill-probe
/tmp/hc-prefill-probe
```

The kernel currently requires H2560, HC4, BF16, contiguous data and the fixed benchmark geometry. Integration must retain the dense inject GEMM fallback for unsupported weights/shapes. Deferral must flush around PLE/capture/eval boundaries, as the existing decode fusion does. Synchronized block profiling flushes the pending write and would hide this particular fusion: use a normal forward for throughput.

Related failed experiment: fusing the up GEMM itself with mixing gave 3.2125 ms versus 2.61571 ms for the stock up+mix subchain. Keep the native projection and fuse the surrounding operations instead.
