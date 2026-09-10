# STATUS — chunkwise/WY GDN prefill

- **Branch:** `arena/01a08be3-mlx-serve` (pushed to origin `nikolai-vysotskyi/mlx-serve`)
- **HEAD:** `5147e25` (research design contract) on top of `82e239d` (CPU reference),
  base `fb15a8d` = upstream `main`.
- **Date:** 2026-09-10 (UTC). Evidence levels follow the task rule: hypothesis /
  reference test / component result on named hardware / full-model M5 A/B.

## Goal (from the handoff README, d40683f)
Accelerate prefill of the whole `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`
model >1.5× vs actual upstream mlx-serve, no quality loss, then 2×/3×+.
Baseline metric = `bench.contextScaling[].prefillTokPerSec` (llmprobe 0.6.6,
rungs 64k, chunk8192, prefix-cache 0, MTP off). Target hardware M5 Max 128 GB;
this agent runs on cloud Linux (x86_64, 2 vCPU, no Zig, no Metal).

## Done this session (evidence: reference test)
- **`research/gdn_chunked_reference.py`** — NumPy f32/f64 golden reference with
  three equivalent implementations (`seq` = shipped per-token recurrence,
  `chunk-scan` naive O(Dk³) fold, `chunk-scan-wy` fla WY O(C·Dk²) form), judged
  against f64 ground truth over bf16-truncated inputs under the repo house rule
  (`err_new ≤ 1.5·err_stock + 0.02`, mirroring `gdnBlockedParityCase`).
  - WY identities proven exact in f64: `A_c` 6.9e-17, `B_c` 2.4e-17.
  - f32 chunk forms PASS the repo bar at T=1024/2048, C=128/256 (y_err
    2.59–2.75e-2, state_err 3.84–3.89e-3, EQUAL to stock's error vs f64).
    Note: they are NOT bit-identical to stock — the chunk-boundary compose
    rounds differently by ~1–2 bf16 ULP (max|Δ| 3.9e-3 y / 9.8e-4 state),
    which is inside the repo's "no worse than stock" bar and the existing
    chunk-boundary continuity tolerance.
  - f64 block-vs-sequential: y ~2e-15, state ~6e-16.
- **`research/gdn_chunked_design.md`** — 3-phase Metal kernel contract
  (per-chunk WY fold / boundary scan / output replay), threadgroup budget →
  C=64 (default; C≤64 to fit 32 KiB), cost table (WY +1.75× @C=64 … +4.99×
  @C=256 vs +85× naive), depth T→~2C+T/C (8192→192), fallback = reuse
  `gdnBlockedEligible` + `MLX_SERVE_GDN_CHUNKED=0` kill switch, test plan
  mirroring the existing parity/continuity tests.

## Key facts driving the choice
- Issue #366 corrected authority: SDPA 36%, MoE `gather_qmm`×3 22%, GDN 8% of a
  4096-token chunk at kL≈32768; whole-model 1.5× needs a combined lift.
- Block profile (S=8192): GDN recurrence 1115→1298 ms as KV grows (24–28% of
  the chunk), running ~2800× above its ~20 GFLOP floor → latency-bound serial
  chain. Existing `GDN_KERNEL_BLOCKED_BODY` fixes coalescing (~2× per oMLX) but
  keeps the T-step chain; the chunkwise form attacks the chain directly.

## Not done / blocked
- **Metal kernel + Zig plumbing NOT written.** No Zig/Metal on this Linux box;
  upstream CONTRIBUTING requires a real Apple Silicon build + `zig build test`
  before any PR. Writing ~400 lines of unbuildable Metal here is rejected as
  unverifiable rather than shipped as a half-tested patch.
- **No M5 tok/s claim.** The depth-reduction speedup is a hypothesis until the
  kernel exists and is benched on M5 against the named baseline.
- **Issue #366 comment blocked:** the `arena-ai-coding-agent[bot]` token can
  READ `ddalcu/mlx-serve` issue #366 (13 comments) but cannot POST comments
  (HTTP 403 "Resource not accessible by integration"). The update drafted at
  `/tmp/issue366-comment.md` could not be published. GitHub write access to the
  fork `nikolai-vysotskyi/mlx-serve` works (branch pushed).

## Next action (needs Apple Silicon)
1. Implement the 3-phase Metal kernel + Zig glue in `src/transformer.zig`
   (mirror `GDN_KERNEL_BLOCKED_BODY` conventions; C=64; log-space gate cumsum).
2. Add `gdnChunkedParityCase` (no-worse-than-stock vs f64) + chunk-boundary
   continuity test; cross-check vs `gdn_chunked_reference.py` on a fixed seed.
3. `zig build test` on M5, then bench prefillTokPerSec vs the named baseline.
4. Publish the issue #366 update / PR from a token with write access to
   `ddalcu/mlx-serve` (the current bot token is read-only there).
