# HC write/read fusion research

M5 Max 128 GB, fa76a4b, the server MLX runtime (source 1f8e74e3). Full HC write/read component, M8192/HC4/H2560/R320: **stock 6.14867 / 6.12183 ms → 3.91179 ms (~1.57×)**. Three timed repetitions per arm after warmup; seeded synthetic BF16 activations, affine4/gs64 projections. Both arms include write, group norm, inject, down/up GEMMs, SiLU, sigmoid, and four-stream mixing. `results.jsonl` contains the final run. The original component measurement preceded the opt-in integration described below.

The new first kernel combines the pending write, group RMS normalization, the per-stream normalization weight, and the four inject dot products. The last kernel directly mixes four streams and avoids a full-width sigmoid/product intermediate. Both projection GEMMs stay on MLX's existing NAX path. The exact BF16 rounding sites of the write, norm and mean are retained. A first version using FP32 mean accumulation was rejected: MLX's Metal mean sums this axis in BF16.

On the fixture, written stream, normalized stream and mixed output are bit-identical to the stock chain. The changed inject reduction has differences up to 0.0078125 in raw/output gates. A float64 dot-product oracle checks 144 gates, including the row with largest raw difference: new gates have zero mismatches, stock one. Maximum raw errors are 0.00390571 new and 0.00390679 stock. This is limited numerical evidence, not broad quality proof. The integration remains experimental; broader inject-reduction accuracy and model-quality coverage are pending.

Run from repository root after building `lib/mlx`:

```sh
clang++ -O3 -std=c++20 -mmacosx-version-min=26.3 research/hc_prefill/probe.cpp -I lib/mlx-src -L lib/mlx/lib -lmlx -Wl,-rpath,"$PWD/lib/mlx/lib" -o /tmp/hc-prefill-probe
/tmp/hc-prefill-probe
```

The server integration in `src/hc_prefill.zig` is opt-in with `MLX_SERVE_HC_PREFILL=1`, defaults off, and requires H2560, HC4, BF16, B1..2 and S17..8192. Other shapes retain the original chain. Pending writes flush around PLE/capture/eval boundaries. Synchronized block profiling flushes the pending write and would hide this particular fusion: use a normal forward for throughput.

ReleaseFast build passed (7/7 steps). Focused Zig test passed (9/9 build steps, 3/3 executed tests): B2/S17 and B1/S65, with and without a pending write; exact write, normalization, one-hot inject mapping and BF16 mix. The full suite has not been run; this branch remains research, with no PR yet.

Both flags together were exercised through HTTP with two different prompt lengths and two requests per server. Both enabled/disabled arms returned the correct passphrase, with no cached prompt tokens. `http-smoke.json` records the settings and engagement lines. The second, warmed request (15,715 tokens) reported 1953.6 -> 2166.7 tok/s (+10.9%). The first (13,515 tokens, includes cold work) was 1751.4 -> 1770.2. These are server-reported smoke measurements, not llmprobe or a 65K target validation. No >1.5× whole-model claim follows. The user baseline 1868 tok/s is not the denominator for these shorter requests.

Related failed experiment: fusing the up GEMM itself with mixing gave 3.2125 ms versus 2.61571 ms for the stock up+mix subchain. Keep the native projection and fuse the surrounding operations instead.

The final dispatcher also removes the old 8192-key crossover for supported paired-prefill shapes, unless a threshold override is supplied. That makes the first 8192-token chunk use paired QSA. Final same-binary warmed comparison: **1989.0 -> 2397.0 tok/s (+20.5%)**, 15,715 tokens. First requests: 1835.2 -> 2251.1; do not use these cold cells as the primary ratio. The separate threshold-override experiment gave 2383.4 warmed tok/s. All are short HTTP cells, not the required long-prompt llmprobe result.
