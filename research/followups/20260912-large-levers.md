# Large-lever investigation after the long-context result

No new whole-model speedup is established. The unchanged PR #408 binary still has the previously measured mixed-checkpoint results: 1879 tok/s on the long llmprobe rung and 2141.6 tok/s in the 15,715-token HTTP smoke. These experiments load synthetic component inputs, not the model. No new full-model runs were performed.

## One-pass resident expert

`moe-resident.metal` computes gate, up, BF16 SwiGLU, and down within one threadgroup. It preserves both projection output casts and the two BF16 activation products; sigmoid values come from the native MLX lookup table. The 16 × 640 intermediate occupies 20 KiB of threadgroup memory. A GPU-generated expert-aligned schedule supports variable expert counts and incomplete row tiles; no routing-count readback is used.

| Geometry | Stock before | Resident | Stock after | Comparison |
|---|---:|---:|---:|---|
| E3, 53 routed rows, H2560, I640 | 0.982 ms | 1.657 ms | 0.967 ms | All output elements bit-identical |
| E512, 81,920 routed rows, H2560, I640 | 25.080 ms | 63.053 ms | 24.545 ms | All output elements bit-identical |

Reject this implementation. The large fixture represents the routed-row count for 8192 tokens / top-10, but uses random synthetic sorted expert assignments and activations. It excludes upstream routing, input duplication, final inverse permutation, expert weighting/reduction and the shared expert. It is not a full MoE layer benchmark. The candidate's GPU schedule is included in its timings.

The fusion removes gate/up/activation global materialization: about 629 MB of writes plus reads at the large fixture shape, counting each intermediate once on each side. However, it reduces the row tile from the stock 64 to 16, increasing repeated weight decoding, and uses 1024 threads with some SIMD groups inactive during the gate/up stage. These are concrete design costs; their individual contribution to the slowdown was not measured with GPU counters. Simply increasing BM is not a solution: BM32 already needs 40 KiB for the complete intermediate, while slicing I would keep larger down accumulators alive across the gate/up phases. A successor needs a different storage/reuse design, not another full-model benchmark of this kernel.

## QSA score/select budget

The existing wide score kernel holds all four indexer heads in per-thread arrays. A candidate instead reloads the query head fragments while preserving the ascending K/head accumulation order. The existing small-row `base` layout was also evaluated at prefill width as a lower-register reference. Stock kernel bodies are extracted verbatim from production `transformer.zig` at 6a0ee06 (production sources match PR head 092ce2e).

| S8192, NB16384, K512 | Score only | Score + exact select |
|---|---:|---:|
| Existing h4, before | 5.463 ms | 9.005 ms |
| Reload head fragments | 6.375 ms | 10.385 ms |
| Existing base layout at wide S | 9.153 ms | 13.155 ms |
| Existing h4, after | 5.607 ms | 9.465 ms |

Every FP32 score and selected index matched the existing h4 result exactly on both S128/NB700 and S8192/NB16384 synthetic fixtures. This is not a proof for all inputs or a model-quality evaluation. Tiny-fixture timings are dominated by submission overhead and are retained only for transparency. Each timed cell is the median of three runs after evaluating that arm once. Score-only and composed timings are independent samples; do not subtract them to report a measured select-kernel duration.

Both alternatives are slower. The existing score/select operation is about 9 ms per attention layer in this isolated 64k geometry; 12 such layers total roughly 108–114 ms of component work per chunk. This arithmetic is for prioritization, not a measured contribution in the live model graph. It gives no reason to spend a long model run on either candidate.

## Current direction

1. Treat the full expert pipeline (routing, expert GEMMs, permutation/reduction and shared expert) as the MoE optimization unit. Keep substantial row reuse while removing materialization. Earlier gate/up interleaving and down/reduce results are modest components, not an established combined >1.5x result.
2. Treat GDN's input/output projections, prework and recurrence as one budget. A large recurrence-only ratio would not describe the full GDN block. Generic dense expansion, larger QMM tiles and the old WY implementation already have negative or small results in this research tree; require a concrete change in reuse/dataflow before repeating them.
3. If source analysis cannot distinguish the next designs, collect one current mixed-model kernel profile with the accepted fusions engaged. The old synchronous block profile predates those fusions and must not be presented as current attribution. A profile that changes synchronization is diagnostic only, not a throughput A/B.

The objective remains >2802 tok/s against the user's 1868 baseline. At 1879 tok/s, an 8192-token-equivalent interval is 4.360 s; at 2802 it is 2.924 s. The next combined design must credibly remove about 1.436 s per equivalent interval to justify another target run. These are rate conversions, not measured per-chunk timings or a speed limit.

## Reproduction

From repository root, with its pinned MLX library and headers available:

```sh
clang++ -O3 -std=c++20 -mmacosx-version-min=26.3 \
  research/followups/moe-resident-probe.cpp -I lib/mlx-src -L lib/mlx/lib \
  -lmlx -Wl,-rpath,"$PWD/lib/mlx/lib" -o /tmp/moe-resident-probe
clang++ -O3 -std=c++20 -mmacosx-version-min=26.3 \
  research/followups/qsa-score-budget-probe.cpp -I lib/mlx-src -L lib/mlx/lib \
  -lmlx -Wl,-rpath,"$PWD/lib/mlx/lib" -o /tmp/qsa-score-budget-probe
```

Acquire the shared GPU lock before executing either probe and release it in an EXIT trap. No argument selects the small correctness fixture; `large` selects the large component. Raw results are `moe-resident-{small,large}.jsonl` and `qsa-score-budget-{small,large}.jsonl`. Runtime/source/file hashes are in `20260912-large-levers-provenance.json`. These kernels are standalone research and are not dispatched by mlx-serve.

## Reviewer context

beamivalice clarified that the 2060.2 / 2083.1 screenshot used a 16k prompt, greedy decoding and n=512, with byte-identical inputs between arms: https://github.com/ddalcu/mlx-serve/pull/408#issuecomment-5645323179. This identifies it as a short-context comparison; it is not a competing 64k measurement. Exact runtime/launch details and input bytes are still needed for matched cross-machine attribution.
