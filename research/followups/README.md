# Follow-up hypotheses on M5 Max, 2026-09-12

These are component investigations on the same MLX runtime as `../qsa_pair/provenance.json`. They do not establish the requested >1.5x full-model improvement. The production branch retains paired QSA and opt-in HC, not the failed kernels below. Different files came from different microbenchmark runs; compare each run's own bracketed stock arms.

| Approach | Result | Decision |
|---|---|---|
| Reload Q fragments per K tile to reduce register lifetime | 16.9513 ms, vs stock ~29.53; earlier paired kernel 14.19 | Worse than existing paired implementation |
| Pack four-token KV blocks for direct NAX fragment loads | 21.0074 ms, stock 29.8455 / 29.5859; planning and packing included | Worse than existing paired implementation |
| Exact dense expert weights, 160 tokens/expert | Inclusive 13.6615 ms, cached dense 9.1835, stock 6.04496 / 6.04138 | Reject |
| Eightfold expert batch / potential layer-major execution, 1280 tokens/expert | Inclusive 42.4547 ms, cached dense 38.1508, stock 41.2389 / 41.0247 | Insufficient gain even before restructuring the model |
| INT4 16-entry BF16 codebook per group, SIMDGROUP shuffle reuse | 8.50967 ms vs stock 7.37629 / 7.29038; bit-exact | Reject; shuffles outweigh saved dequantization arithmetic |
| Expert-aligned double-buffered weight staging | Gate 6.2935 vs 7.29279 / 7.29675; down 6.11838 vs 6.70996 / 6.71742, bit-exact | Only 10–16% component gain; does not isolate pipelining from expert alignment, which already helped similarly |
| Direct native MPP BF16 GEMM, K128 slices | Matches stock for GDN input / attention Q, slower for GDN output; includes dequantization | No speedup |
| QSA: separate QK/softmax and PV SIMD groups | Addressable union: 72.85 ms; explicit packed register bank: 19.04 ms; QK/PV pipeline: 19.69 ms | All slower than the retained 14.19 ms paired kernel; reduced spilling is an inference, not a counter measurement |
| Virtual gate/up interleaving + fused SwiGLU, BM64 | 13.9794 ms vs 15.1407 / 15.1029, bit-exact | Small component gain; not integrated |
| Interleaved gate/up, BM192 / 6×2 SIMD groups | 13.1783 ms vs 15.1456 / 15.1795, bit-exact, GPU schedule included | ~1.15× component, insufficient for a new whole-model run |
| Physically transpose dense projection weights | Including transpose/dequantization is slower on all three shapes; cached layout gains ~2% | Reject as a major lever |

Each component uses three timed repetitions after warmup. No long model runs were added for these ideas. QSA fixtures contain only block indices; activations and MoE data are synthetic. The paired probes still have the same maximum output difference 0.000244141 and RMSE 1.16102e-6 as the original paired math on that fixture. No independent broad quality claim is made for the rejected experiments.

Metal sources here are historical research variants, not runtime-imported kernels. Exact codebook dequantization and pipelining depend on the specialized E512/affine4/gs64 MoE probe geometry. The codebook kernel assumes 128 threads and BN64/BK64. Do not copy it into a general dispatcher without geometry guards.

The interleaved kernels virtually alternate gate/up weight rows so adjacent accumulator columns are gate/up pairs; SwiGLU is evaluated directly in the store epilogue. They preserve the two BF16 activation rounding sites. The wide variant keeps 32 rows per SIMD group while supporting a non-power-of-two group size and exactly one 192-row tile for most experts in the fixture. The remaining gain does not justify integrating or benchmarking this variant across the full model.
