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

## 1. The five implemented levers (all opt-in, all OFF by default)

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

- All four new kernels (HC up-mix, MoE down+reduce, MoE gateup×2) plus the
  chunked-GDN three are **CPU-validated and type-checked, but NOT Metal-built**.
  Lever E adds no new kernel — it widens the two decode GDN fusion kernels to
  prefill widths behind an opt-in env gate.
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
- **Attention/QSA** (~23%): QSA gather/score already upstream (#388); look for
  block-reuse / query-grouping, not re-counting the upstream win.
- **HC write side**.
