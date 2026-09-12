# Follow-up hypotheses on M5 Max, 2026-09-12

Latest investigation: [resident gate/up/down fusion and the long-context QSA score/select budget](20260912-large-levers.md). Both candidates preserve the tested outputs but are slower; no full-model run or production change was made for them. Raw results and reproduction sources are retained here.

These are component investigations on the same MLX runtime as `../qsa_pair/provenance.json`. They do not establish the requested >1.5x full-model improvement. The production branch retains paired QSA, opt-in HC and wide GDN fusion, not the failed kernels below. Different files came from different microbenchmark runs; compare each run's own bracketed stock arms.

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

Each component uses three timed repetitions after warmup. No long model runs were added for these ideas. QSA fixtures contain only block indices; activations and MoE data are synthetic. The original paired layout probes in the table above have the same maximum output difference 0.000244141 and RMSE 1.16102e-6 as the original paired math on that fixture. No independent broad quality claim is made for the rejected experiments.

Metal sources here are historical research variants, not runtime-imported kernels. Exact codebook dequantization and pipelining depend on the specialized E512/affine4/gs64 MoE probe geometry. The codebook kernel assumes 128 threads and BN64/BK64. Do not copy it into a general dispatcher without geometry guards.

The interleaved kernels virtually alternate gate/up weight rows so adjacent accumulator columns are gate/up pairs; SwiGLU is evaluated directly in the store epilogue. They preserve the two BF16 activation rounding sites. The wide variant keeps 32 rows per SIMD group while supporting a non-power-of-two group size and exactly one 192-row tile for most experts in the fixture. The remaining gain does not justify integrating or benchmarking this variant across the full model.

Further QSA followups: full D256 in one SIMD group, with Q staged in threadgroup memory, took **62.0599 ms**, against retained paired **13.7532 ms** and stock 29.553 / 29.456 ms in the same run. It was rejected. Register pressure is a hypothesis, not a hardware-counter finding. The MPP descriptor M8 still exposed the padded M16-sized cooperative fragments in a tiny layout probe, so it did not provide the smaller register footprint needed for that design.

A CPU-only pass over the captured block selections evaluated four-query packing: 122,057 K/V tiles versus 174,432 for pairs (-30%), but 273,875 active 16-head tiles versus the pair kernel's 266,113 (+3%). Eight-query membership bucketing fragmented into too many partly filled buckets. `qsa-group-costs.json` records these counts; no GPU speedup is claimed, and no full-model run was spent on these candidates.


## Followups after the 2509.4 tok/s short HTTP checkpoint

| Candidate | Same-run component result | Decision |
|---|---|---|
| Single native FP32 × BF16 PV call, inherited relaxed precision | Paired 13.7005 / 13.8480 ms, candidate 12.4128; output RMSE vs stock rose from 1.16102e-6 to 1.92547e-5, max difference doubled | Do not integrate this accuracy tradeoff |
| Same call with relaxed precision disabled for FP32 inputs | Incorrect output (max difference 0.204834); 16.6484 ms from the invalid implementation is not a valid performance result | Cooperative-fragment layout must be re-derived for this type/mode before claiming correctness; no integration |
| Quantize online log2 maximum upward in steps of 4, skip rescaling when factor is 1 | Paired 13.8110 / 13.7857 ms, candidate 16.1306; max difference 0.000244141, RMSE 1.5375e-6 | Slower; no broader validation or model run |
| Cache exactly dequantized BF16 HC down/up weights | Fused quantized HC 4.0310 / 3.94983 ms, cached-dense HC 4.15925; mixed output bit-identical on this fixture | Slower even excluding one-time expansion; no model run |

`qsa-mixed-pv-probe.cpp`, `qsa-mixed-pv-strict-probe.cpp`, `qsa-coarse-max-probe.cpp` and `hc-cached-dense-probe.cpp` reproduce these cells from repository root, with the same compiler/runtime options as the original probes. The QSA files require the decompressed public block fixture. Each candidate is bracketed by the existing paired/fused arm; no failed implementation is adopted. Strict-mode fragment-layout mismatch is an inference from the MPP type/mode contract and failed output comparison, not a completed diagnosis.

The split QK → softmax → PV idea was also costed before implementation. On the S4096/Q24/KB512 fixture, a compact 66×32 score row already needs about 0.83 GB of FP32 scratch; materializing and reading scores/probabilities adds several GB of traffic per attention layer. It removes online-output rescaling and shortens register lifetimes, but there is no credible >1.5× prediction yet once scratch and staging are counted. No expensive model run was spent on it. To pursue it, first reduce the score scratch/traffic or demonstrate a large register/occupancy benefit on a component, then add planner admission accounting.
