# STATUS — chunkwise GDN prefill (rank-1 fold / scan / replay)

- **Branch:** `arena/01a08be3-mlx-serve` (pushed to origin `nikolai-vysotskyi/mlx-serve`)
- **HEAD:** `c47728f` — "wip: GDN chunked-prefill kernels (fold/scan/replay) + parity tests — UNBUILT"
  (base `fb15a8d` = upstream `main`; ancestry c9abf8c → c877bc7 → c47728f)
- **Date:** 2026-09-10 (UTC). Evidence levels follow the task rule: hypothesis /
  reference test / component result on named hardware / full-model M5 A/B.

## Goal (from the handoff README, d40683f)
Accelerate prefill of the whole `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`
model >1.5× vs actual upstream mlx-serve, no quality loss, then 2×/3×+.
Baseline metric = `bench.contextScaling[].prefillTokPerSec` (llmprobe 0.6.6,
rungs 64k, chunk8192, prefix-cache 0, MTP off). Target hardware M5 Max 128 GB;
this agent runs on cloud Linux (x86_64, 2 vCPU, no Zig, no Metal).

## Done (evidence: CPU reference test — validated)
- **`research/gdn_chunked_reference.py`** — NumPy f32/f64 golden reference
  (`seq`, `chunk_scan`, `chunk_scan_wy`, `chunk_scan_hier`). All arms PASS vs
  f64; f32 chunk error EQUALS stock (y 2.59–2.75e-2, state 3.84–3.89e-3);
  NOT bit-identical (max|Δ| ≈ 3.9e-3 y / 9.8e-4 state, ~1–2 bf16 ULP), inside
  the repo "no worse than stock" bar. Two-level hier depth 8192/C=64/S=16 →
  168 serial steps (32768 → 192).
- **`research/gdn_chunked_design.md`** — 3-phase contract, WY vs rank-1 fold
  (rank-1 cheaper at C=128/256, no C×C solve), cost/depth tables, kill switch,
  test plan. Corrected non-bit-identity claim.
- **`src/transformer.zig`** — unbuilt implementation landed (HEAD `c47728f`):
  - Metal bodies `GDN_CHUNK_KERNEL_{FOLD,SCAN,REPLAY}_BODY` (rank-1 augmented-
    basis fold → A_c/B_c; short serial boundary scan; per-chunk exact replay).
  - Zig glue `gdnChunkedEnabled/Eligible/C`, `gdn_chunk_c_override`,
    `gdnChunkStagingTbFor`, `getGdnChunk{Fold,Scan,Replay}`, `gdnRunYStateChunked`.
  - Tests `gdnChunkedParityCase` + parity sweep and chunk-boundary continuity
    (split run == full run).

## Not done / blocked
- **NOT built / NOT run.** No Zig/Metal on this Linux box; upstream CONTRIBUTING
  requires a real Apple Silicon build + `zig build test` before any PR. The
  Metal compile WILL likely surface first-build fixes (Metal language quirk
  mismatches are expected and are NOT evidence of a failed approach).
- **Production dispatch not wired.** `gdnForward` still calls the blocked path;
  the chunked path is a test seam only. Intentional: wire it only after the
  kernels pass `zig build test` on Mac.
- **No M5 tok/s claim.** The depth-reduction speedup is a hypothesis until the
  kernel exists and is benched on M5 against the named baseline.
- **Issue #366 comment blocked:** the `arena-ai-coding-agent[bot]` token can
  READ `ddalcu/mlx-serve` issue #366 but cannot POST comments (HTTP 403
  "Resource not accessible by integration"). Draft at `/tmp/issue366-comment.md`.
  GitHub write access to the fork `nikolai-vysotskyi/mlx-serve` works (pushed).

## Next action (needs Apple Silicon)
1. On M5: `zig build test` → fix Metal compile errors → run the new parity and
   continuity tests (cross-check vs `gdn_chunked_reference.py` on a fixed seed).
2. Wire production dispatch (`gdnForward` behind `gdnChunkedEnabled/Eligible`,
   blocked fallback), then A/B `bench.contextScaling[].prefillTokPerSec` vs the
   named baseline.
3. Publish the issue #366 update / PR from a token with write access to
   `ddalcu/mlx-serve` (the current bot token is read-only there).
