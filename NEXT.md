# NEXT.md — Apple Silicon (M5 Max, 128 GB) validation handoff

**For:** the agent with a real Mac (M5-class GPU, macOS, Metal, bundled MLX with NAX).
**Prepared by:** the cloud Linux agent (no Metal; all kernels below are UNBUILT
on Metal, CPU-validated only). Date: 2026-09-10.

Goal (unchanged from the task): prefill of the **whole**
`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit` model **>1.5×** vs upstream mlx-serve,
no quality loss. These four levers are the in-cloud-built candidates; **none of
them has a measured tok/s yet, and none reaches 1.5× alone** — the target is the
compatible set.

---

## 0. Repo state

- Fork: `https://github.com/nikolai-vysotskyi/mlx-serve`, branch
  `arena/01a08be3-mlx-serve` (session-fixed). Latest commit at handoff time:
  `d8f4b77` (plus the commit that lands this file + the MoE down+reduce lever).
- Upstream to compare against: `https://github.com/ddalcu/mlx-serve` (current
  main; older `cc7dea1` was the prior local baseline).
- Give EACH checkout its own real `.zig-cache` (a shared symlink has produced
  stale binaries and false attribution before). Do not reuse old cached builds.

Evidence levels used here: **reference test (CPU)** and **type-check**. Anything
noted "CPU-validated" means the math/indexing was reproduced in NumPy vs an f64
ground truth; it does NOT mean the Metal kernel compiles or is correct on-device.
The first `zig build test` on Mac is expected to surface MSL compile errors.

---

## 1. The eight implemented levers (all opt-in, all OFF by default)

### Lever A — chunkwise GDN prefill (≈25% of the S=8192 block profile)
- Files: `src/transformer.zig` (`GDN_CHUNK_KERNEL_{FOLD,SCAN,REPLAY}_BODY`,
  `gdnChunkedEnabled/Eligible`, `gdnRunYStateChunked`, `gdnChunkedParityCase`).
- Env: `MLX_SERVE_GDN_CHUNKED=1` to engage (default off), `=0` kill switch,
  `MLX_SERVE_GDN_CHUNK_C=<chunk>` to override the chunk size.
- What it changes: replaces the per-token sequential state recurrence with a
  rank-1 fold per chunk (A_c/B_c), a sequential boundary scan over chunks, and a
  parallel replay. Wired into production dispatch (`gatedDeltaNet`) behind
  `gdnChunkedEnabled()` + `gdnChunkedEligible()` + bf16-state + staging-budget
  guards, with the blocked/stock kernel as fallback.
- CPU evidence: `research/gdn_chunked_reference.py` (f64 golden; f32 chunk error
  == stock) and `research/gdn_chunked_kernel_sim.py` (index-exact sim vs
  f64-validated `chunk_scan`, PASS at T=128/500/1024/2050).
- Two Metal bugs already found + fixed in `62eac76` (fold A_c transpose; fold
  grid counted wrong) — expect the first real build to find more of this class.

### Lever B — fused HC up-projection + stream mix (≈15% of the profile)
- File: `src/transformer.zig` (`HC_UP_MIX_SOURCE`, `hcUpMixFused`), wired into
  `hcRead` (fused first, composed chain fallback).
- Env: `MLX_SERVE_HC_UP_MIX=1` (default off, `=0` kill switch). Gated to
  4-bit affine / group 64 / H%64==0 / R%64==0 (the named model's geometry).
- What it changes: folds `sigmoid(up)·n4` and the mean over the 4 streams into
  the up-projection GEMM epilogue, so the `[B,S,hc*H]` (~168 MB bf16 at S=8192)
  up intermediate is never materialized.
- CPU evidence: `research/hc_up_mix_reference.py` (kernel rounding points ==
  composed chain; LUT sigmoid bit-identical; tails + 2/4/8-bit all PASS).
- Test: `test "fused HC prefill up+mix kernel matches the composed up/sigmoid/mean chain"`.

### Lever C — fused MoE down-projection + score-weight + top-K reduce (small,
part of the ≈35% MoE block; removes ~1.3 GB/layer of intermediate traffic)
- File: `src/transformer.zig` (`MOE_DOWN_REDUCE_SOURCE`,
  `moeDownReduceFused`), wired into `moeMLP2`'s sorted prefill path.
- Env: `MLX_SERVE_MOE_DOWN_REDUCE=1` (default off, `=0` kill switch).
- What it changes: folds the inverse permute + score multiply + sum-over-K into
  one gather-form kernel, so the `[T*K, hidden]` (~419 MB bf16) down tensor is
  never permuted back nor re-read. Gather form, no atomics; coalesced reads.
- CPU evidence: `research/moe_down_reduce_reference.py` (kernel shares every
  rounding point with the composed chain — bf16 product, fp32 sum via MLX's
  bfloat16-widened Reduce, bf16 out — PASS vs f64).
- Test: `test "fused MoE down+score+reduce matches the composed take/multiply/sum chain"`.

### Lever D — fused MoE gate/up + GeGLU (sorted prefill path; the bigger MoE
item — drops two gather_qmm launches + the activation per layer)
- File: `src/transformer.zig` (`MOE_GATEUP_SCHEDULE_SOURCE`,
  `MOE_GATEUP_SOURCE`, `moeGateUpFused`), wired into `moeMLP2`'s `do_sort`
  branch before the separate gate/up `gather_qmm` calls.
- Env: `MLX_SERVE_MOE_GATEUP_FUSED=1` (default off, `=0` kill switch).
- What it changes: a 512-thread schedule pass (faithful port of the handoff
  `work/grouped_qmm_tiles.metal` — per-expert binary search over the sorted
  ids emits `{start,count,block}` tiles, `simd_prefix_inclusive_sum` for the
  offset, NO early return so every lane reaches the warp-collective) feeds a
  tiled plain-SIMD GEMM: one threadgroup per (tile, column-block), every slot
  in a tile shares ONE expert, dequant in fp32 (same model as MLX gather_qmm /
  the bit-exact decode gatherQmv — no bf16 rounding of the weight), fp32 dot
  rounded to bf16 gate/up, then the same LUT silu·up as `fusedSwiGLU`.
- Gated to 4-bit affine / group 64 / K%64==0 / N%64==0 / E≤512 / no expert
  bias / silu (non-gpt-oss); the composed chain stays as fallback. The GEMM is
  plain-SIMD (no NAX/steel header), so expect it to need a perf pass (§4)
  before it wins at prefill scale.
- CPU evidence: `research/moe_gateup_reference.py` (schedule covers every slot
  exactly once; kernel == composed modulo fp32 dot order — exact 0 at small
  configs, ~1e-3 at reduced-prod config; both ~bf16-precision vs f64).
- Test: `test "fused prefill gate+up+GeGLU matches gather_qmm + fusedSwiGLU (sorted path)"`.
- Correctness guard (this session): the `do_sort` wiring now engages the fused
  kernel only when `hidden_act == silu` **and** `swiglu_limit <= 0` **and** no
  per-expert gate/up bias — the kernel bakes in `silu(gate)*up` via the
  `fusedSwiGLU` LUT and has nowhere to add the expert biases the composed chain
  adds, so fusing under those conditions would silently drop them.

### Lever E — GDN prework + norm-gate epilogue fusion extended to prefill widths
(the decode fusions already on main are S 1..9 only; this widens them)
- File: `src/transformer.zig` (`gdnPrefillFusedEnabled`, `GDN_PREFILL_MAX_ROWS`,
  relaxed width gates in `gdnPreworkFused` / `gdnNormGateFused`, and the
  `prework_width_ok` production dispatch in `gatedDeltaNet`).
- Env: `MLX_SERVE_GDN_PREFILL_FUSED=1` (default off, `=0` kill switch). Composes
  with `MLX_SERVE_GDN_CHUNKED=1`: prework covers the input side (conv+SiLU+
  split+Q/K norm-and-scale+next conv state+gate+beta in one launch), the
  chunked recurrence the middle, and the norm-gate epilogue
  (`rms_norm(y)·silu(z)` → flat out_proj input) the output side.
- What it changes at prefill: the composed chain launches conv1d, sigmoid,
  multiply, 3× slice/reshape, 2× rms_norm, 2× multiply, gate (exp/log1p/exp),
  and sigmoid(b) per layer; the fusions collapse all of that into two kernels
  (prework + norm-gate) — the SAME kernels already bit-identical-tested at
  S 1..9, so this is a width gate, not a new kernel. Cap: `batch*seq ≤ 8192`
  (`GDN_PREFILL_MAX_ROWS`).
- Test: `test "gdn packed prework: prefill widths (S 10..64) bit-identical to
  the composed chain (+ prefill gate)"` (pins q/k/v/conv_state/g/beta == the
  composed chain at S=10 and S=64, and that S>9 declines without the opt-in).

### Lever F — fused HC write + group-norm (the HC write side, prefill)
- File: `src/transformer.zig` (`HC_WRITE_NORM_SOURCE`, `hcWriteNormFused`,
  `hcWriteNormEnabled`, `HC_WRITE_NORM_MAX_ROWS`), wired into `hcRead` /
  `hcReadPending` / `hcWriteOrDefer` (the write is now deferred to the next
  read at prefill too, behind the opt-in, and `hcRead` was split into a
  `hcReadTail` the fused norm feeds).
- Env: `MLX_SERVE_HC_WRITE_NORM=1` (default off, `=0` kill switch). Gated to
  hc 1..8 / H%256==0 / bf16|f16 / batch*seq <= HC_WRITE_NORM_MAX_ROWS=8192.
- What it changes: the composed chain materializes the write `stream += out*inj`
  (reshape + broadcast multiply + add + reshape: two [B,S,hc,H] intermediates)
  and the NEXT read's group-norm re-reads the written stream (reshape +
  fast_rms_norm(ones) + ×norm_w). The kernel folds both into ONE per-(row,
  stream) dispatch — the write uses the chain's exact two roundings
  (T(out·inj), then T(stream+that)), so the written stream is BIT-identical;
  the norm `T(T(x·rsqrt(mean+eps))·norm_w)` differs from the stock rms_norm
  only by the sum-of-squares reduction order + rsqrt (the accepted few-bf16-ulp
  class, same bar as the fused-read parity test). Also serves the pure-norm arm
  (WR=0) so the prefill group-norm is one dispatch even without a pending write.
- Test: `test "fused HC prefill write+norm matches the composed
  write+group-norm chain"` (write bit-exact, norm within the ulp bar, WR=0 arm,
  opt-in-off + H%256 declines).

### Lever G — grouped-query QSA gather (cross-token block reuse; the ~23%
attention block, beyond the upstream gather/score win)
- File: `src/transformer.zig` (`ATTN_QSA256_GROUP_SOURCE`,
  `getAttnQsa256GroupKernel`, `qsaGroupEnabled/G`, `qsaGroupEligible`, the
  `use_group` arm + `QsaGroupCfgKey` cache in `gatherQsa256`).
- Env: `MLX_SERVE_QSA_GROUP=1` (default off, `=0` kill switch),
  `MLX_SERVE_QSA_GROUP_G=<G>` (tokens/threadgroup, default 4). Gated to
  `BK % RATIO == 0`, `32*NSG*G <= 256`, `qL >= G`; the single-token
  `msv_attn_qsa256` stays the fallback and NAX is untouched.
- What it changes: one threadgroup serves G adjacent query tokens for one
  (kv-head, batch); every thread redundantly k-way-merges the G sorted block
  selections into a union, each distinct block staged ONCE with a per-token
  row count (RATIO / tail_rows / 0), then every token reads the SAME staged
  K/V tile. Per-token online-softmax state stays per-row/per-simdgroup
  (simdgroup w → token w/NSG, row-band w%NSG), so registers don't grow. Each
  token still visits its keys in ascending absolute position (its sequence is
  a sorted subset of the union), so it matches the single-token kernel up to
  fp32 tile-boundary rounding; HBM K/V reads drop ~G× at prefill where
  adjacent tokens share ~their whole selection. Grid `⌈qL/G⌉·32 × Hkv·NSG·G × B`,
  threadgroup `{32, NSG·G, 1}`.
- CPU evidence: `research/qsa_group_reference.py` (union→per-token sequences
  EXACT for every group incl. the partial last group and the
  tail-block/selected-block overlap case; fp32 attention grouped==stock==exact
  to 8.2e-08). Uses `SENTINEL = 2147483647` (not bare `INT_MAX`), the
  `ATTN256_KERNEL_HEADER` mma/row-max/row-sum primitives, and `static_assert`s
  for `BK % RATIO == 0` and `32*NSG*G <= 256`.
- Test: `test "gatherQsa256 grouped: matches the single-token gather and the
  composed reference"` (grp vs `attn256Reference` within the stock bar, grp vs
  stock < 1e-2) + `test "gatherQsa256 grouped: geometry gate"`.

### Lever H — fused PLE gate + value modulation (qwen4_exp spec-capture PLE;
small — the PLE block's cost is its key/value qmatmuls + dilated conv, not the
~10 elementwise gate dispatches this folds)
- File: `src/transformer.zig` (`PLE_GATE_SOURCE`, `pleGateFused`,
  `pleGateFusedEnabled`, `getPleGateKernel`, `PleGateKey` cache), wired into
  `pleForward` (qwen4_exp per-layer loop, after `hcFlush`) behind an opt-in,
  composed chain as fallback.
- Env: `MLX_SERVE_PLE_GATE_FUSED=1` (default off, `=0` kill switch).
- What it changes: one per-(row, hc) dispatch (32-lane simdgroup) folds
  `kq=key4*query4` → sum → `*inv_sqrt_h` → `sqrt(max(|·|,1e-6))·sign` → LUT
  sigmoid → `*value`. fp32 reduction (MLX Reduce widens bf16 accumulators),
  every bf16 rounding point preserved (the max of two bf16 values is exactly
  bf16-representable, so the skipped intermediates are no-ops — bit-exact),
  sigmoid is the shared `swigluSigTable` LUT. `inv_sqrt_h` is a 0-dim scalar
  read as `constant T&` (MLX `write_signature`: ndim==0 → reference).
- CPU evidence: `research/ple_gate_reference.py` PASS (gate [0.3672,0.6406],
  drift 2.089e-3 vs pure-f32).
- Test: `test "fused PLE gate+value-modulation matches the composed
  kq/gate/sigmoid/mul chain"`.

---

## 2. What to do on the Mac (in order)

1. **Build + full test suite** (CONTRIBUTING bar before any PR):
   ```sh
   git clone https://github.com/nikolai-vysotskyi/mlx-serve.git && cd mlx-serve
   git checkout arena/01a08be3-mlx-serve
   zig build test          # full suite must be green on Apple Silicon
   ```
   Fix any MSL compile errors the first build surfaces (the four kernels'
   bodies are in `src/transformer.zig` as `*_SOURCE` string constants).

2. **Per-lever parity tests** (each opt-in, each must match its reference):
   ```sh
   zig build test  # run the four tests named in §1; also:
   MLX_SERVE_GDN_CHUNKED=1 zig build test   # gdnChunkedParityCase + continuity
   MLX_SERVE_HC_UP_MIX=1   zig build test
   MLX_SERVE_MOE_DOWN_REDUCE=1 zig build test
   MLX_SERVE_MOE_GATEUP_FUSED=1 zig build test
   MLX_SERVE_GDN_PREFILL_FUSED=1 zig build test   # Lever E
   MLX_SERVE_GDN_CHUNKED=1 MLX_SERVE_GDN_PREFILL_FUSED=1 zig build test  # composed set
   MLX_SERVE_HC_WRITE_NORM=1 zig build test       # Lever F
   MLX_SERVE_QSA_GROUP=1 zig build test           # Lever G (grouped QSA gather)
   MLX_SERVE_QSA_GROUP=1 MLX_SERVE_QSA_GROUP_G=8 zig build test  # G=8 variant
   MLX_SERVE_PLE_GATE_FUSED=1 zig build test      # Lever H (PLE gate fusion)
   ```
   Cross-check the fused outputs against `research/*_reference.py` /
   `*_kernel_sim.py` on a fixed seed if any tolerance looks tight.

3. **GDN production dispatch is already wired** (`gatedDeltaNet` routes through
   `gdnRunYStateChunked` behind `gdnChunkedEnabled/Eligible` + bf16-state +
   staging-budget guards, blocked/stock fallback). Just re-run the full suite +
   HTTP regression with `MLX_SERVE_GDN_CHUNKED=1` on the Mac to confirm the
   parity tests still pass through the production path.

4. **A/B benchmark** (only after the build/tests pass; same machine, same
   conditions, one A/B at a time, baseline = current upstream main with its own
   `.zig-cache`):
   ```sh
   npx --yes llmprobe@0.6.6 "$SERVER_URL" \
     -m ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit \
     --bench-only --rungs 64k --runs 1 --no-save --no-color \
     --save baseline.json --label baseline
   # then the candidate build with the intended env flags set, same command,
   # --label candidate; read bench.contextScaling[].prefillTokPerSec (the long
   # rung), record real inputTokens + ttftMs. check_speedup.py from the
   # research/qwen-prefill-cloud-handoff-20260910 branch computes the ratio.
   ```
   Target model fits ~71 GB active; do not bypass admission. No profiling
   synchronizations in the speed run. Keep chunk=8192, ctx=131072, prefix-cache
   entries=0, MTP off.

5. **Report back** (append to this file or a new `REPORT-M5.md`): the commit
   SHA tested, build log, which tests passed/failed, the exact env flags used,
   the log lines showing each kernel engaged (e.g. `[hc] fused prefill up+mix
   kernel engaged`, `[moe] fused down+score+reduce kernel engaged`,
   `[moe] fused gate+up+GeGLU kernel engaged`), and the
   baseline vs candidate `prefillTokPerSec` per run.

---

## 3. Honest status (do not overstate)

- All eight new kernels (HC up-mix, HC write+norm, MoE down+reduce, MoE
  gateup×2, the chunked-GDN three, the grouped-QSA gather, and the PLE gate
  fusion) are **CPU-validated and type-checked, but NOT Metal-built**. Lever E adds no new
  kernel — it widens the two decode GDN fusion kernels to prefill widths behind
  an opt-in env gate. Lever F's write arm is bit-exact by construction (it
  shares the fused-read N kernel's rounding); its norm arm is the same few-ulp
  class as that N kernel. Lever G's per-token key order is the stock kernel's
  order exactly (only tile boundaries move), so it inherits the stock gather's
  correctness up to fp32 rescale rounding.
- **No M5 tok/s has been measured.** Depth-reduction (GDN) is the only
  potentially large algorithmic win; HC/MoE/GDN-fusions are fusion micro-wins.
  The 1.5× whole-model target is NOT yet demonstrated and likely needs the set
  plus the still-open levers below.

## 4. Still open (for whoever continues after M5 validation)

- **NAX perf pass** for the two plain-SIMD GEMMs (MoE gate/up and HC up-mix):
  the handoff prototype
  `research/qwen-prefill-cloud-handoff-20260910:patches/moe-gateup.patch` +
  `work/moe_prefill_gateup.metal` (NAX + Apple MLX steel header) compiled on
  Mac but had no whole-model proof — swap the plain-SIMD inner loop for the
  NAX outer-product form and re-measure if Lever D / B under-deliver.
- **Attention/QSA** (~23%): the grouped-query block-reuse gather is now
  implemented (Lever G, `MLX_SERVE_QSA_GROUP=1`). Still open on the attention
  side: whether the gather is HBM-bound at all (if it's compute/occupancy-bound
  the ~G× staging win won't show) — measure per-block HBM reads before betting
  on it, and consider a G>4 or NAX-form variant only if the M5 profile says so.
