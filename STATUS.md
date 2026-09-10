# STATUS — prefill optimization (GDN chunkwise + HC up+mix + MoE down+reduce + MoE gate/up+GeGLU + grouped-QSA gather)

- **Branch:** `arena/01a08be3-mlx-serve` (pushed to origin `nikolai-vysotskyi/mlx-serve`)
- **HEAD:** see `git log -1` (after each push); ancestry …d8f4b77 → fe2de89 →
  5e4a159 (GDN wired) → (MoE gate/up+GeGLU commit).
- **Date:** 2026-09-11 (UTC). Evidence levels: hypothesis / reference test /
  component result on named hardware / full-model M5 A/B.
- **Handoff for the M5 agent:** `NEXT.md` (build/test/A-B commands + honest
  status). Keep it current alongside this file.

## Goal (handoff README, d40683f)
Accelerate prefill of the whole `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit` model
>1.5× vs actual upstream mlx-serve, no quality loss, then 2×/3×+. Baseline =
`bench.contextScaling[].prefillTokPerSec` (llmprobe 0.6.6, rungs 64k, chunk8192,
prefix-cache 0, MTP off). Target M5 Max 128 GB; agent runs cloud Linux
(x86_64, no Zig/Metal; Zig 0.16.0 via PyPI `ziglang` for ast/type-check).

## Done (evidence: CPU reference + index simulation — validated in-cloud)

### GDN chunkwise (~25% of the S=8192 block profile)
- rank-1 fold / tiled boundary scan / chunked replay in `src/transformer.zig`,
  gated `MLX_SERVE_GDN_CHUNKED=1`; parity + continuity tests.
- Two Metal bugs found & fixed (`62eac76`): fold A_c transpose; fold grid
  under-sized (MLX `set_grid` counts THREADS — re-confirmed this session against
  the shipping blocked kernel's grid math).
- `research/gdn_chunked_reference.py` + `research/gdn_chunked_kernel_sim.py`
  PASS vs f64-validated `chunk_scan`.

### HC up-projection + 4-stream mix (~15%)
- `mlxserve_hc_up_mix` kernel + `hcUpMixFused` + `hcRead` wiring, gated
  `MLX_SERVE_HC_UP_MIX=1` (4-bit/gs64/H%64/R%64); parity test.
- `research/hc_up_mix_reference.py` PASS (kernel == composed chain at every
  rounding point; LUT sigmoid bit-identical; tails + 2/4/8-bit).

### MoE down-projection + score-weight + top-K reduce (small, part of ~35%)
- `mlxserve_moe_down_reduce` kernel + `moeDownReduceFused` + `moeMLP2`
  sorted-path wiring, gated `MLX_SERVE_MOE_DOWN_REDUCE=1`; parity test.
- `research/moe_down_reduce_reference.py` PASS: kernel shares every rounding
  point with the composed chain (bf16 product; fp32 sum — MLX Reduce widens
  bfloat16 accumulators to float32, verified from mlx/backend/cpu/reduce.cpp
  `ReductionAccumulator::widen_to_float`; bf16 out); kernel-vs-composed == 0 on
  CPU, << 1 bf16 ULP on-device (fp32 sum order only).

### MoE gate/up + GeGLU fusion (sorted prefill path; larger MoE item)
- `mlxserve_moe_gateup_sched` + `mlxserve_moe_gateup` kernels (schedule pass +
  tiled plain-SIMD GEMM) + `moeGateUpFused` + `moeMLP2` `do_sort`-branch wiring,
  gated `MLX_SERVE_MOE_GATEUP_FUSED=1`; parity test. Replaces the two
  `gather_qmm` gate/up calls AND `fusedSwiGLU` with a faithful port of the
  handoff `work/grouped_qmm_tiles.metal` schedule (512 threads, per-expert
  binary search → `{start,count,block}` tiles) plus a per-(tile, col-block)
  GEMM that dequants the packed uint32 4-bit banks in **fp32** (matching MLX
  gather_qmm; no intermediate bf16 weight rounding — verified against the
  bit-exact decode `gatherQmv` dequant), accumulates fp32, rounds gate/up to
  bf16, then the same LUT silu·up as `fusedSwiGLU`.
- Gated to 4-bit affine / gs64 / K%64==0 / N%64==0 / E≤512 / no expert bias /
  silu (non-gpt-oss); falls back to the composed chain otherwise.
- `research/moe_gateup_reference.py` PASS: schedule covers every slot exactly
  once (asserted); kernel == composed modulo fp32 dot order (exact 0 at small
  configs, ~1e-3 at reduced-prod config; both ~bf16-precision vs f64).

### GDN prework + norm-gate epilogue fusion extended to prefill widths (Lever E)
- `src/transformer.zig`: `gdnPrefillFusedEnabled` + `GDN_PREFILL_MAX_ROWS`
  (=8192) + relaxed width gates in `gdnPreworkFused` / `gdnNormGateFused` +
  `prework_width_ok` in the `gatedDeltaNet` dispatch. Gated
  `MLX_SERVE_GDN_PREFILL_FUSED=1` (default off). No new kernel — the two decode
  fusion kernels (already bit-identical-tested at S 1..9) are seq-agnostic and
  per-row, so the extension is a width gate. Composes with
  `MLX_SERVE_GDN_CHUNKED=1` (prework → chunked recurrence → norm-gate).
- Test: `test "gdn packed prework: prefill widths (S 10..64) bit-identical to
  the composed chain (+ prefill gate)"` — q/k/v/conv_state/g/beta == composed
  chain at S=10/64; S>9 declines without the opt-in. (Not run on Metal.)

### MoE gate/up correctness guard (this session)
- The `do_sort` wiring now fuses only when `hidden_act == silu` **and**
  `swiglu_limit <= 0` **and** no per-expert gate/up bias. The kernel bakes in
  `silu(gate)*up` (fusedSwiGLU LUT) and cannot add the expert biases the
  composed chain adds, so the previous unconditional call could silently drop
  them for gpt_oss-clamp / biased / non-silu archs.

### HC write + group-norm fusion (Lever F, the HC write side)
- `mlxserve_hc_write_norm` kernel + `hcWriteNormFused` (write folded when
  `out`/`inj` set, else pure norm), gated `MLX_SERVE_HC_WRITE_NORM=1`; wired
  into `hcRead` (pure-norm arm) + `hcReadPending` (deferred-write arm) +
  `hcWriteOrDefer` (defer at prefill), with `hcRead` split into `hcReadTail`.
  Cap `HC_WRITE_NORM_MAX_ROWS=8192`; H%256==0 / hc 1..8 / bf16|f16.
- Folds the write's two [B,S,hc,H] intermediates AND the next read's norm into
  one per-(row, stream) dispatch. Write arm shares the decode fused-read N
  kernel's exact roundings (T(out·inj), then T(stream+that)) so the written
  stream is bit-identical; norm arm is the accepted few-bf16-ulp class.
- Test: `test "fused HC prefill write+norm matches the composed
  write+group-norm chain"` (write exact, norm within the ulp bar, WR=0 arm,
  gate-off + H%256 declines). (Not run on Metal.)

### Grouped-query QSA gather (Lever G, cross-token block reuse)
- `msv_attn_qsa256_grp` kernel (`ATTN_QSA256_GROUP_SOURCE` inline in
  `src/transformer.zig`) + `getAttnQsa256GroupKernel` + `qsaGroupEnabled/G` +
  `qsaGroupEligible` + the `use_group` dispatch arm and `QsaGroupCfgKey` cache
  in `gatherQsa256`. Gated `MLX_SERVE_QSA_GROUP=1` (default off),
  `MLX_SERVE_QSA_GROUP_G` (default 4); `BK % RATIO == 0` + `32*NSG*G <= 256`.
- One threadgroup per (group of G adjacent tokens, kv-head, batch): redundant
  per-thread k-way merge of the G sorted selections into a union; each distinct
  block staged once with a per-token row count (RATIO / tail_rows / 0); every
  token reads the same staged K/V tile; per-token online softmax stays
  per-row/per-simdgroup. Uses `SENTINEL = 2147483647` (no bare `INT_MAX`, like
  the QSA select kernel) + the stock `ATTN256_KERNEL_HEADER` mma primitives +
  `static_assert`s for the two gates. Grid `⌈qL/G⌉·32 × Hkv·NSG·G × B`.
- `research/qsa_group_reference.py` PASS: union→per-token sequences EXACT for
  every group (incl. partial last group + tail-block/selected-block overlap);
  fp32 attention grouped==stock==exact (max 8.2e-08, grouped−stock 0.0).
- Tests added: `gatherQsa256 grouped: matches the single-token gather and the
  composed reference` (grp vs `attn256Reference` within the stock bar; grp vs
  stock < 1e-2) + `gatherQsa256 grouped: geometry gate`. (Not run on Metal.)

### Zig validation (runs here)
- Whole-file `zig ast-check` clean apart from the pre-existing
  `@backingInt`/`@fromBackingInt` invalid-builtin notes (24 hits, unchanged);
  isolated semantic type-checks EXIT=0 for the new GDN prefill gate + the MoE
  gateup wiring guard + the HC write+norm kernel/wiring/test + the grouped-QSA
  gate/helpers/dispatch (`gatherQsa256` `use_group` arm, `QsaGroupCfgKey`,
  `getAttnQsa256GroupKernel`) against the `/tmp/zchk` mlx stub.

## Not done / blocked
- **NOT built / run on Metal.** MSL compile + `zig build test` need Apple
  Silicon (CONTRIBUTING bar). All four kernels are UNBUILT; the GDN fold bugs
  and the HC tid bug found by review are exactly what a first Mac build
  surfaces — more may remain. See `NEXT.md`.
- **No M5 tok/s claim.** All speedups are hypothesis until measured.
- **GDN production dispatch now WIRED** behind `gdnChunkedEnabled()` +
  `gdnChunkedEligible()` + bf16-state + staging-budget guards, with the
  blocked/stock kernel as fallback (`MLX_SERVE_GDN_CHUNKED=1` engages it).
  Still needs the parity sweep + continuity test + A/B on Mac.
- **Issue #366 comment blocked:** bot token read-only on `ddalcu/mlx-serve`
  (HTTP 403 on POST). Draft committed at
  `research/issue366-comment-2026-09-10.md` — post it from a write-capable
  account. Fork push works.

## Open levers still to pursue (user: keep stacking optimizations)
Claimed so far: GDN chunkwise (25%) + GDN prework/norm-gate prefill fusion
(part of the GDN 25% + its proj/epilogue) + HC up-mix (part of 15%) + HC
write+group-norm (part of 15%) + MoE down+reduce (small) + MoE gate/up+GeGLU
fusion (part of ~35%, plain-SIMD) + grouped-query QSA gather (part of ~23%,
block reuse).
Remaining: attention/QSA grouped-query block-reuse is now implemented (Lever
G); still open is whether the gather is HBM-bound at all (measure per-block
reads on M5 before betting on the ~G× staging win), a NAX perf pass for the
MoE gate/up and HC up-mix GEMMs (the current ports are plain-SIMD and will
need it to matter at prefill scale), and possibly a G>4 / NAX-form grouped
gather if the profile calls for it. None of the individual levers reaches 1.5×
alone.
