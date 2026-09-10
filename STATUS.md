# STATUS — chunkwise GDN prefill (rank-1 fold / scan / replay)

- **Branch:** `arena/01a08be3-mlx-serve` (pushed to origin `nikolai-vysotskyi/mlx-serve`)
- **HEAD:** `62eac76` — "fix: chunked-GDN fold A_c transpose + grid threads; add kernel index sim"
  (ancestry: c9abf8c → c877bc7 → c47728f → f41d2f0 → 62eac76)
- **Date:** 2026-09-10 (UTC). Evidence levels: hypothesis / reference test /
  component result on named hardware / full-model M5 A/B.

## Goal (handoff README, d40683f)
Accelerate prefill of the whole `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit` model
>1.5× vs actual upstream mlx-serve, no quality loss, then 2×/3×+. Baseline =
`bench.contextScaling[].prefillTokPerSec` (llmprobe 0.6.6, rungs 64k, chunk8192,
prefix-cache 0, MTP off). Target M5 Max 128 GB; agent runs cloud Linux
(x86_64, no Zig/Metal; Zig 0.16.0 installed via PyPI `ziglang` for ast-check).

## Done (evidence: CPU reference + index simulation — validated)
- `research/gdn_chunked_reference.py` — NumPy f32/f64 golden (seq, chunk_scan,
  chunk_scan_wy, chunk_scan_hier). All PASS vs f64; f32 chunk error EQUALS
  stock; NOT bit-identical (~1–2 bf16 ULP, inside repo bar).
- `src/transformer.zig` — unbuilt implementation (rank-1 fold, tiled boundary
  scan, chunked replay) + Zig plumbing + `gdnChunkedParityCase`/continuity
  tests, gated OFF by default (`MLX_SERVE_GDN_CHUNKED=1`).
- **Two Metal bugs found & fixed this session** (review + sim):
  1. fold stored `A_c` TRANSPOSED (`A_out[g*Dk+d0+i]` = column g at row d0+i);
     scan reads row-major `[dk_out][dk']`. Fixed to stride-Dk scatter.
  2. fold grid.x used `(Dk+Dv)/32` but MLX `set_grid` counts THREADS (blocked
     kernel's own comment: "Grid: (256*(Dv/32), Hv, B) threads"). Fixed to
     `256*(Dk+Dv)/32`.
- **New CPU checks that RUN here:**
  - `zig ast-check` on the whole `src/transformer.zig` (with the 0.17-only
    `@backingInt`/`@fromBackingInt` parenthesized for the 0.16 parser): clean.
  - Isolated `zig test -fno-emit-bin` type-check of the new code (kernels +
    getters + `gdnRunYStateChunked`, then parity/continuity tests) against a
    signature-exact `mlx` stub: EXIT=0.
  - `research/gdn_chunked_kernel_sim.py` — NumPy reproduction of the kernels'
    exact index math vs the f64-validated `chunk_scan`: PASS at T=128/500/1024/
    2050 (mine-vs-ref rel y 2.0e-4…4.0e-4, state 4.2e-7; corrected fold write
    exact, pre-fix transpose O(1) off).

## Not done / blocked
- **NOT built / run on Metal.** MSL compile + `zig build test` still need Apple
  Silicon (CONTRIBUTING bar before any PR). The two bugs above are exactly the
  kind this first build surfaces; more may remain.
- **Production wiring deferred.** `gdnForward` still runs the blocked path; the
  chunked path is test-seam-only until it passes on Mac (avoids breaking a hot
  path with unverifiable code).
- **No M5 tok/s claim.** Depth-reduction speedup is hypothesis until measured.
- **Issue #366 comment blocked:** bot token is read-only on `ddalcu/mlx-serve`
  (HTTP 403 on POST). Draft at `/tmp/issue366-comment.md`. Fork push works.

## Next actions (need Apple Silicon)
1. On M5: `zig build test` → fix MSL compile errors → run `gdnChunkedParityCase`
   + continuity tests; cross-check vs `gdn_chunked_kernel_sim.py` on a fixed seed.
2. Wire production dispatch (`gdnForward` behind `gdnChunkedEnabled/Eligible`,
   blocked fallback), then A/B `bench.contextScaling[].prefillTokPerSec`.
3. Publish issue #366 update / PR with a write-capable token.

## Open levers still to pursue (user: keep stacking optimizations)
GDN chunking is ONE lever (~25% of the S=8192 profile). Others: MoE grouped
execution (~35%), attention (~23%), HC projections/epilogue (~15%). No other
lever is started yet on this branch; GDN does not reach 1.5× alone.
