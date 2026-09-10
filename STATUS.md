# STATUS — prefill optimization (GDN chunkwise + HC up+mix; cloud Linux agent)

- **Branch:** `arena/01a08be3-mlx-serve` (pushed to origin `nikolai-vysotskyi/mlx-serve`)
- **HEAD:** `4b15430` — "feat: opt-in fused prefill HC up+mix kernel + CPU reference"
  (ancestry: …62eac76 → 3fb11fc → f3067e0 → 4b15430)
- **Date:** 2026-09-10 (UTC). Evidence levels: hypothesis / reference test /
  component result on named hardware / full-model M5 A/B.

## Goal (handoff README, d40683f)
Accelerate prefill of the whole `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit` model
>1.5× vs actual upstream mlx-serve, no quality loss, then 2×/3×+. Baseline =
`bench.contextScaling[].prefillTokPerSec` (llmprobe 0.6.6, rungs 64k, chunk8192,
prefix-cache 0, MTP off). Target M5 Max 128 GB; agent runs cloud Linux
(x86_64, no Zig/Metal; Zig 0.16.0 installed via PyPI `ziglang` for ast-check).

## Done (evidence: CPU reference + index simulation — validated)

### GDN chunkwise (~25% of the S=8192 block profile)
- `research/gdn_chunked_reference.py` — NumPy f32/f64 golden (seq, chunk_scan,
  chunk_scan_wy, chunk_scan_hier). All PASS vs f64; f32 chunk error EQUALS
  stock; NOT bit-identical (~1–2 bf16 ULP, inside repo bar).
- `src/transformer.zig` — unbuilt implementation (rank-1 fold, tiled boundary
  scan, chunked replay) + Zig plumbing + `gdnChunkedParityCase`/continuity
  tests, gated OFF by default (`MLX_SERVE_GDN_CHUNKED=1`).
- **Two Metal bugs found & fixed** (review + sim, committed `62eac76`):
  1. fold stored `A_c` TRANSPOSED; fixed to stride-Dk scatter (scan reads
     row-major `A_c[dk_out][dk']` — re-derived and re-confirmed this session).
  2. fold grid.x under-sized by 256× (MLX `set_grid` counts THREADS); fixed.
- `research/gdn_chunked_kernel_sim.py` — index-exact sim vs f64-validated
  `chunk_scan`: PASS at T=128/500/1024/2050.

### HC up-projection + 4-stream mix (~15% of the profile)
- `research/hc_up_mix_reference.py` — CPU reference (RUNS here, PASS):
  fused kernel's rounding points (bf16 up via LUT, bf16 product, fp32 stream
  sum, 1/hc scale) match the composed chain exactly; LUT sigmoid bit-identical
  to fp32 sigmoid of the bf16 input; tails (M%32, H%64) + 2/4/8-bit affine all
  tail-exact; both paths ~bf16-precision vs f64.
- `src/transformer.zig` — `mlxserve_hc_up_mix` Metal kernel (BM32×BN64×BK64,
  threadgroup {32,2,2}, epilogue folds sigmoid×n4+mean so the [B,S,hc*H]
  ~168 MB intermediate is never written) + `hcUpMixFused` plumbing + `hcRead`
  wiring (fused first, composed chain fallback) + parity test (clean/tail/
  declined). Opt-in `MLX_SERVE_HC_UP_MIX=1`, gated 4-bit/gs64/H%64/R%64.
- **One Metal bug found & fixed this session** (manual review): staging loops
  used `thread_position_in_threadgroup.x` (0..31) as the linear tid over 128
  threads — switched to `thread_index_in_threadgroup`.

### Zig validation (runs here)
- Whole-file `zig ast-check` clean (0.17-only builtins parenthesized for the
  0.16 parser).
- Isolated semantic type-checks EXIT=0: GDN kernels+plumbing (`chk.zig`),
  GDN tests (`chk2.zig`), HC up+mix production (`chk3.zig`), HC up+mix test
  (`chk4.zig`) — all against the `/tmp/zchk` mlx stub.

## Not done / blocked
- **NOT built / run on Metal.** MSL compile + `zig build test` need Apple
  Silicon (CONTRIBUTING bar before any PR). Both levers' kernels are UNBUILT;
  the two GDN bugs + the HC tid bug above are exactly what a first Mac build
  surfaces — more may remain.
- **No M5 tok/s claim.** All speedups are hypothesis until measured.
- **Issue #366 comment blocked:** bot token is read-only on `ddalcu/mlx-serve`
  (HTTP 403 on POST). Draft at `/tmp/issue366-comment.md`. Fork push works.

## Next actions (need Apple Silicon)
1. On M5: `zig build test` → fix any MSL compile errors → run `gdnChunkedParityCase`
   + continuity tests and the HC up+mix parity test; cross-check vs the CPU
   references on fixed seeds.
2. Wire production dispatch (GDN `gdnForward` behind `gdnChunkedEnabled/Eligible`
   with blocked fallback; HC up+mix already wired behind `MLX_SERVE_HC_UP_MIX=1`),
   then A/B `bench.contextScaling[].prefillTokPerSec`.
3. Publish issue #366 update / PR with a write-capable token.

## Open levers still to pursue (user: keep stacking optimizations)
Claimed so far: GDN chunkwise (25%) + HC up-mix (part of 15%). Remaining:
MoE grouped execution (~35%), attention/QSA (~23%), HC write side + the NAX
perf pass for the up-mix GEMM. None of the individual levers reaches 1.5×
alone; the target is the compatible set.
