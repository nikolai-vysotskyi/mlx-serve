# Investigating the PLE model regression

The earlier whole-table GPU PLE rejection was a rejection of that integration, not evidence that GPU PLE cannot help. This follow-up separates kernel time, table preparation, host copying and effects on the model. **There is still no new accepted whole-model speedup or >1.5x result.** Production PR #408 remains at the HC-corrected `1b1a4df`.

## What was found

1. The old integration returned early from `NgramTable.startWarm` after creating GPU PLE, disabling the native file-cache warmer even though decode and small widths still used CPU gather. The diagnostic patch restores warming for both paths. This is a concrete integration defect, but its share of the previous regression has not been isolated in a matched A/B.
2. The no-copy MLX table is contiguous. Allocator accounting rises by **32,000,163,840 bytes** at wrapping, with no immediate increase in task footprint. Thus neither an explicit whole-table memcpy nor the custom kernel's non-contiguous-input copy explains the observed wrap.
3. Merely evaluating **three rows** then takes 10.092 s in one model process. Task page-ins increase by 1,953,134 pages, approximately **32.000 GB** at 16 KiB/page: almost the entire mapped table, rather than the three requested rows. The separate Metal implementation also reports 10.351 s for three rows while GPU execution is 0.005 ms. The cost is outside the arithmetic kernel.
4. In the managed integration, ready MLX active memory is **105.224 GB**, versus **73.224 GB** after releasing the registration. In the profiled session, machine-wide wired memory changes from about 103 GiB to 73 GiB and free pages from 2.85 GiB to 32.55 GiB after release. Startup also coincides with increased swap. These are process/VM observations, not a claim that every swapped byte belongs to this model. `task phys_footprint`, RSS, MLX active bytes and wired memory are different counters.
5. The actual managed prefill gather remains fast: 1.067 ms evaluation + 5.044 ms output copy under block profiling, or 3.086 + 4.234 ms in normal execution for 131,072 rows. This does **not** explain a multi-second model regression by itself. A CPU -> GPU -> CPU -> MLX copy roundtrip is unnecessary, but its measured cost is milliseconds here.

The MLX allocator collects cached buffers above `min(block_limit, 0.95 * recommendedMaxWorkingSetSize)`; registering the whole table reduces its headroom substantially. **A zero cache counter between prefill chunks is not proof of GC thrashing:** `generate.zig` explicitly calls `mlx_clear_cache()` there. Automatic GC, per-allocation residency commits and the precise steady-state contribution to the old 1001/1075 tok/s regression remain unisolated. The earlier interpretation of an empty cache as automatic GC was too strong.

## Causal diagnostic, not a throughput A/B

The managed diagnostic restores native warming and uses one process with G = GPU gather, C = CPU gather while retaining the GPU table, D = release the GPU table then use CPU gather. The same 12,454-token prompt is repeated with zero prefix-cache hits. This differs from the original two-shape 13,515/15,715-token HTTP run.

| Block-profiled diagnostic | G | C | D |
|---|---:|---:|---:|
| Request wall time | 7.339 s | 17.560 s | 7.914 s |
| First 8192-row-token CPU gather | GPU path | 6453.12 ms | 53.86 ms |
| Second 4261-row-token CPU gather | GPU path | 1999.88 ms | 22.22 ms |

Switching from GPU to CPU changes which mmap/PTE path has been exercised. These sequential cold/warm states are not interchangeable controls. In the non-profiled G/D diagnostic, G reports 2104.8 tok/s and 45.3 decode tok/s; the first D request reports 917.2 and 47.6, with cold CPU table access. **Do not claim a 2.3x model speedup from this pair.** The later controls were not rerun to manufacture a performance claim. Profiling also changes synchronization; rates from profiled and normal sessions must not be compared as an optimization ratio.

Raw diagnostic records are `ple-residency-{diagnostic,normal}.json`. The archived patch supports both managed and direct variants; set `MLX_SERVE_PLE_GPU=1`, and optionally point `MLX_SERVE_PLE_GPU_DIAGNOSTIC` at a file containing G, C or D. D permanently releases that process's GPU PLE state. These controls are research-only.

## Separate-queue attempt

A second implementation borrowed the original CPU mmap and issued the same gather on an independent Metal queue. This avoids registering the table in MLX's pool, but **the external 32 GB resource still exists and must be included in any memory budget**. MLX-only `/props` counters undercount it; this is not production-ready accounting.

For the first 15,715-token model request:

| Gather | Command execution/wait | GPU timestamps | Host output copy |
|---|---:|---:|---:|
| 131,072 rows | **3264.668 ms** | **0.266 ms** | 4.659 ms |
| 120,352 rows | **1912.078 ms** | **0.710 ms** | 4.074 ms |

The enormous gap localizes time outside the GPU arithmetic. It is consistent with the cost of making a whole-table resource accessible alongside the model; driver subphases were not individually instrumented. Simply changing queues is not a fix.

This model request returned an empty answer and failed the recall assertion; later requested arms were **not executed**. It is invalid as a performance result. The cause of this model-level failure is still open. A subsequent independent test checks all **20,971,520 BF16 outputs** against a C++ scalar reference on actual table rows and passes exactly, so the failure cannot be declared an arithmetic-kernel bug from that model response alone. That standalone test does not validate integration, lifetimes or every model input. Its first GPU call takes 11.291 s, warm CPU 54.88–55.41 ms, and warm GPU plus output copy 3.061 ms. See `ple-direct-{failed-model,parity-timings}.txt` and `ple-direct-parity.jsonl`.

The direct diagnostic binary is retained by fingerprint in the provenance JSON. It is not the executable left in the accepted checkout. Both managed and direct variants are archived in `ple-residency-diagnostic.patch`, applying to `1b1a4df`. ReleaseFast built; the full Zig suite was not spent on these failed experiments.

## Working successor: bounded selected-row staging

`ple-packed-staging-probe.mm` leaves the original table on the CPU. It deduplicates selected row IDs, copies each required 100-byte affine4 row (80-byte weights + 10-byte BF16 scales + 10-byte biases) into a small shared Metal buffer, and dequantizes/gathers there. It preserves the original row IDs, order and BF16 packing. There is no lower precision, skipped expert, reduced context or KV/prefix-cache shortcut.

Actual table, 131,072 seeded uniform IDs, 131,046 unique rows, three warm timings per cell:

| Complete component path | Median |
|---|---:|
| C++ scalar CPU reference before | 54.7411 ms |
| Clear row cache, lookup + gather packed rows + GPU | **16.5392 ms** |
| All rows already in this row cache: lookup + GPU | **1.73475 ms** |
| C++ scalar CPU reference after | 54.2461 ms |

All 20,971,520 BF16 outputs are bit-identical. The miss-path component ratio is **3.28–3.31x**, without relying on useful duplicate IDs or cache hits. The all-hit fixture is a distinct state, not a promised application hit rate. The original CPU mapping was exercised by the reference before these warm timings; SSD-cold staging is **not measured**. The 26.902 ms first staging call also follows the CPU reference, and must not be compared with the cold full-table registration as an equivalent startup A/B.

The table buffer is **13,107,200 bytes**, rather than 32,000,163,840 bytes (about 2441x smaller). Output is 41,943,040 bytes, plus a 524,288-byte slot-index buffer. Timings end with a CPU-readable shared output buffer; they do not include an additional integration copy into an MLX leaf. The reference is C++, not compiled production Zig. No full-model gain is claimed.

The current cache is a fixed single-batch fixture, bounded by the input row count; a persistent runtime cache needs eviction/generation ownership and immutable GPU input lifetimes. In the real first model chunk, the ID instrumentation found **67,441 unique IDs out of 131,072**, suggesting deduplication is useful on this public-source prompt. It does not establish a general-text hit rate.

Build from the research repository root and run under the shared GPU lock:

```sh
clang++ -O3 -std=c++20 -fobjc-arc -ffp-contract=off -mmacosx-version-min=26.3 \
  research/followups/ple-packed-staging-probe.mm -framework Foundation -framework Metal \
  -o /tmp/ple-packed-staging-probe
/tmp/ple-packed-staging-probe /path/to/ngram_table.bin
```

## Next substantial levers

1. **PLE: implement bounded packed-row staging in the ordinary MLX graph**, retain native warming, keep prepared row data immutable until its GPU consumer finishes, and return the MLX output directly. Then prepare future prompt chunks concurrently with the current GPU forward. This targets first-use faults and avoids making 32 GB permanently GPU-resident. First validate a real chunk's rows/outputs and memory budget; do not run another long benchmark for the standalone ratio.
2. **Full routed MoE: preserve substantial weight reuse while removing grouping and intermediate copies.** The accepted experimental counting-group + inverse reduction already gives 26.92–27.01 -> 23.53 ms on the routed-chain fixture. Integrate it only together with a stronger projection improvement; 1.14x of that component is insufficient. The current mixed profile attributes about 1.59 s/chunk to MLP, so this is the largest steady-state target. Existing large-tile and resident-expert failures are documented; repeating them without a changed storage/reuse design is not progress.
3. **Mixed-checkpoint 8-bit dense projections.** The GDN category is about 0.98 s/chunk, attention about 0.62 s, and both include projections. Earlier generic dense-projection probes used 4-bit weights; those results do not settle 8-bit dispatch or unnecessary dequantization/layout work. Isolate actual mixed-model projection shapes and dispatch, then optimize weight reuse and graph materialization. GDN recurrence alone accounts for much less than the entire GDN category.

At 1879 versus target 2802 tok/s, the rate-equivalent saving sought is about 1.436 s per 8192 tokens. This guides prioritization, not a measured per-chunk speed limit. A plausible combined candidate must attack more than the current warm PLE component; neither the new staging result nor the existing MoE grouping independently establishes >1.5x for the model.
