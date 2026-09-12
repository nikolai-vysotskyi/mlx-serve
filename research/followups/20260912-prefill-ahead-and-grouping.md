# Bounded PLE preparation ahead of prefill and MoE grouping

The target of >1.5× whole-model prefill is still unmet. A matched long HTTP
screening measured **2071.3 → 2154.0 tok/s (+3.99%) at 62,176 tokens**.
The short request did not improve. This is one sequential pair, not an
alternating confirmation or llmprobe acceptance result. Do not combine the
historical 1868 tok/s number with this different prompt to claim a speedup.

## Whole-model observations

M5 Max 128 GB, AC power, mixed-4-8bit checkpoint, context 131072, chunk 8192,
prefix cache entries 0, KV quantization off, MTP/PLD off, temperature 0,
seed 1234. Each arm starts a fresh server from the same frozen executable;
both first answer the same Paris readiness request. All four measured answers
were `MAGNOLIA-7731`, with 10 output tokens and zero cached prompt tokens.

| Prompt tokens | QSA + corrected HC + GDN | Plus bounded PLE / preparation ahead |
|---:|---:|---:|
| 13,515 | 2341.1 tok/s | 2335.0 tok/s |
| 62,176 | 2071.3 tok/s | 2154.0 tok/s |

The long peak MLX allocation was 78,143,458,918 versus 78,143,475,302 bytes
(16 KiB difference). This counter excludes CPU-only preparation allocations;
available-memory observations are also preserved in the result file. The new
path does not register the 32 GB n-gram table as a GPU buffer.

`ple-ahead-long-ab.json` contains request hashes, actual token counts, rates,
memory observations, engagement lines and CPU timing. The measured executable
SHA256 is `8e3eceadab95651e86640bd3613920d476464fe5b1ceb49f01efb24069849d18`.
`ple-ahead-measured.patch` against research parent `912ebc8` preserves the
measured runtime source before the later MoE addition. Provenance is in
`prefill-ahead-provenance.json`. The test-root registration described below
was added afterward; it does not change the server executable.

The earlier stateless packed-row screening used a separate executable:

| Prompt tokens | All three fusions disabled | Fusions enabled | Fusions + stateless PLE |
|---:|---:|---:|---:|
| 13,515 | 1864.0 | 2026.3 | 1922.1 |
| 15,715 | 1923.1 | 2201.0 | 2238.6 |

Raw observations and that binary's fingerprint are in `ple-packed-ab.json`.
The second-request incremental difference is only +1.7%, and the first is
negative. Do not cherry-pick controls across these two experiments.

## What changed, and what the CPU timing actually establishes

`MLX_SERVE_PLE_PACKED=1` selects only required affine4/gs32/160-wide table rows,
copies their 100 compressed bytes into a bounded buffer, and dequantizes them
to BF16 in the normal MLX graph. MLX owns immutable copies of the selected
bytes and slot indices. There is no readback or second Metal queue.

`MLX_SERVE_PLE_AHEAD=1`, together with PACKED, starts one CPU producer per
eligible fixed-boundary text prefill. It prepares up to two chunks ahead and
uses a request-local cache of at most 262,144 compressed rows (26.2 MB of row
payload, plus index storage). The producer has no MLX calls and does not
advance model state. The consumer recomputes the original row IDs, advances
history once, and compares every prepared row ID before consuming a buffer.
An unexpected history or forward declines safely. Cancellation joins the
producer and releases queued data. The normal table warmer remains enabled.

| Request | Producer packing | Consumer waiting | Cached row hits / misses |
|---|---:|---:|---:|
| 13,515 tokens | 206.83 ms | 160.22 ms | 16,190 / 85,755 |
| 62,176 tokens | 18.54 ms | 6.44 ms | 377,074 / 85,747 |

The cache is recreated for each request. The difference between those packing
times therefore also reflects the state of the CPU mapping/page cache; it is
not evidence of a persistent cross-request embedding cache.

Crucially, the current-path long CPU gather measured **35.94–49.57 ms per
8192-token chunk**, with 24.35 ms for the tail. Its first short request was
176.65 / 57.60 ms. The earlier synchronous block profile's 488–1788 ms cannot
be treated as the steady critical-path PLE budget. This experiment sets
`QWEN4_PROFILE_FWD=1`, which enables the host gather clock but **does not add
block synchronization for these prefill widths** (the block profiler only
synchronizes S=2..16 in this mode; `=all` would change that).

The long CPU gather adds to 306.88 ms. Removing that wait alone cannot explain
the entire approximately 1.15-second model difference in this single pair.
Other data movement and run variability are not isolated. Keep the +4% as an
observation, not a proved causal attribution. PLE is no longer the primary
candidate for the missing large gain on a warm mapping.

## Actual mixed projection weights

`mixed-dense-probe.cpp` reads the installed checkpoint's GDN input/output and
attention-Q weights. Their packed geometry proves these tensors are 8-bit,
group size 64, despite the config's global 4-bit default. Activations in this
probe are deterministic synthetic BF16 values, not a captured model state.

| Projection | Native QMM, trailing control | Dequant + matmul | Cached dequant + matmul |
|---|---:|---:|---:|
| GDN input, combined 10240+6144 outputs | 13.1024 ms | 11.6807 ms | 11.4705 ms |
| GDN output | 5.09242 ms | 4.84350 ms | 4.35417 ms |
| Attention Q | 9.95200 ms | 8.60417 ms | 8.47338 ms |

All compared BF16 outputs matched exactly on these fixtures. The first GDN
native control was 16.8975 ms; the trailing control moved, so do not use that
first number to inflate the comparison. Production keeps GDN QKV/Z separate;
the combined probe is not itself an implemented projection fusion.

The dequant+matmul route is **already enabled by upstream** for these prefill
widths (`prefillDqGemm`, default M>=2048). Persistent dense caching therefore
adds only the smaller difference between the last two columns and additional
memory. No model run or new improvement claim was made for this candidate.

## Compatible MoE work now implemented, not yet model-benchmarked

`MLX_SERVE_MOE_PREFILL_GROUP=1` opts into a guarded Qwen4/B1/S2048..8192,
E512/K10/H2560 BF16 path. Three GPU kernels perform counting grouping, prefix
offsets and scatter, producing the permutation and its inverse together.
Native gate/up/down projections and activation remain. The final kernel
gathers from sorted output, multiplies router scores and sums in the native
eight-group BF16 order; it avoids materializing the full restored/weighted
expert output. Decode and unsupported models keep the existing path.

The predecessor C++ pipeline measured about 1.14× for the routed expert chain
([earlier evidence](20260912-moe-direct-pipeline.md)). That is not a model
speedup and has **not** been added arithmetically to the PLE result. This
integration is a compatible piece for a larger design, not a >1.5× claim.

Both new module tests are explicitly imported by `src/tests.zig`. Earlier
filter-only runs returned two harness tests without discovering these new
modules and are excluded from validation. After registration each focused
command reported 3/3 executed tests, including the named candidate test:
PLE exact native BF16 values, duplicate rows, queued graph ownership,
three-chunk cache reuse, mismatch fallback and cancellation; MoE a bijection,
sorted expert IDs, a heavily skewed expert distribution, tails, and all
657,920 BF16 values of native weighted reduction. Full-suite/build results
are recorded alongside the final source fingerprints in
`prefill-ahead-validation.json`: ReleaseFast succeeded; the full suite passed
**2331 tests, 154 skipped**. One combined real-model smoke also passed, with
both PLE chunks consumed and the MoE grouping kernel engaged. Its 13,515-token
server observation was 2444.8 tok/s, **without a matched control**; this is
integration evidence, not a new speedup measurement.

## Reproduce the long PLE comparison

From the repository root, after the normal dependency build and model install:

```sh
.zig-toolchain/zig build -Doptimize=ReleaseFast
PLE_BENCH_TAG=ple-ahead-long-ab PLE_BENCH_ARMS=current,ahead \
PLE_BENCH_LONG=1 PLE_BENCH_CPU_TIMING=1 \
python3 research/followups/bench-prefill-ahead.py \
  --binary zig-out/bin/mlx-serve --out /tmp/ple-ahead-results \
  --lock /path/to/your/shared-gpu-lock.sh
```

The lock script must accept `acquire OWNER` / `release OWNER`; coordinate with
other users of this GPU. The driver refuses an occupied port and terminates
only its own server. It records results before assertions and forces the new
MoE flag off. Prompt source slice and request hashes identify the fixture.
Both new flags remain off by default. Keep PR #408 at its corrected production
head until a justified combined result and the remaining quality review.
