# REPORT-M5.md — Apple Silicon validation of `arena/01a08be3-mlx-serve`

**Machine:** Apple M5 Max, 128 GB, macOS 26.5, server MLX 0.32.3 (bundled NAX build),
Zig 0.17.0-dev.1818. **Model:** `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit` (68.4 GB resident).
**Date:** 2026-09-11. **Validator:** Nikolai V. (M5 agent), following `NEXT.md` §2.

**Branch tested:** `arena/01a08be3-mlx-serve` @ `9bcc59b` rebased cleanly onto upstream
`main` @ `fa76a4b` (29 commits, no conflicts). Every checkout has its own real `.zig-cache`.
Baseline = clean upstream `fa76a4b`, same deps, same machine, same session.

## TL;DR

- The branch **builds** (ReleaseFast 7/7) and the Metal kernels **compile** (MLX JIT); no MSL
  compile errors surfaced. 11 new tests run on-device.
- **Full suite: 2334 pass / 154 skip / 4 FAIL** as pushed. Failing: Lever A (chunked GDN, final
  state wrong), Lever C (down+reduce, not bit-exact: 1.4× tol), Lever E (prework beta, 7.6e-6 vs
  MLX sigmoid), Lever H (PLE gate, gross 0.166 error). After my fixes (§2, §3, and the bit-exact
  down+reduce order): **2336 pass / 154 skip / 2 FAIL** (E and H remain, log `work/arena-tests2.log`);
  down+reduce now reports `worst diff/tol=0.000` on all three shapes.
- **`MLX_SERVE_PREFILL_TURBO=1` on the real model: prefill fails (MlxError) on the first
  request** — two production-wiring bugs the unit tests could not see (shape/config-cache).
  After fixing both, turbo runs but is **~3.8× SLOWER than upstream** (472–486 vs
  1785–1868 tok/s on the same 65,836-token prompt) **and produces wrong output** (needle
  test fails). None of the eight levers is a whole-model win as implemented; see the
  per-lever table below.
- The cloud agent's docs were honest ("no M5 tok/s measured, all speedups are hypothesis")
  — the "~2×" figure in circulation is a research *budget*, not a measurement. Decode is not
  touched by any lever.

## 1. Build + suite (NEXT.md §2 step 1–2)

| Step | Result |
|---|---|
| `zig build -Doptimize=ReleaseFast` (arena, rebased on fa76a4b) | 7/7 succeeded |
| `zig build -Doptimize=ReleaseFast` (baseline fa76a4b) | 7/7 succeeded |
| `zig build test` (arena, default env; new tests use their override seams) | **2334 pass, 154 skip, 4 fail** (log: `work/arena-tests.log`) |

Failures, with the on-device evidence:

| Lever | Test | Evidence |
|---|---|---|
| A chunked GDN | `GDN chunked pipeline: no worse than stock vs f64…` | `B=1 T=130 C=64: y 0.00718 vs stock 0.00718, state 0.27525 vs stock 0.00097`. y correct, final state wrong. Root cause found + fixed (below). |
| C down+reduce | `fused MoE down+score+reduce matches the composed…` | `worst diff/tol=1.407`. Metal MLX's bf16 `sum` does NOT accumulate in fp32 (the arena's reference reads the *CPU* backend `reduce.cpp`); the Metal `col_reduce_small` keeps 8 bf16 partial sums in a fixed order. Serial fp32 accumulation is *not* the composed chain's rounding. |
| E prework prefill widths | `gdn packed prework: prefill widths (S 10..64) bit-identical…` | `expected 0, found 0.0000076293945` on `beta` at ±16 inputs (same mismatch I hit on 2026-09-10 on my own widening). The kernel's `sigmoid(b)` is not bit-identical to `mlx_sigmoid` for wide b; do not relax the bar, fix the sigmoid (LUT, as `fusedSwiGLU`). |
| H PLE gate | `fused PLE gate+value-modulation matches the composed…` | `max|fused-composed|=1.6601562e-1` — gross semantic mismatch, not rounding. |

## 2. Production-wiring bugs found on the real model (fixed in this branch)

Both were invisible to the unit tests (which compare values on synthetic shapes) and both killed
the very first request under `MLX_SERVE_PREFILL_TURBO=1`:

1. **Lever B `hcUpMixFused` returned a 2-D `[M,H]` array** where the composed chain returns
   `[B,S,H]`; the GDN conv `concatenate` downstream failed with
   `All the input arrays must have the same number of dimensions … 3 and 2`. Fixed: output
   declared `[batch, seq, H]` (kernel indexes rows flat, layout identical), `batch` added to the
   config-cache key.
2. **Lever C `MoeDownReduceKey` lacked `T`**: the cached config (output `[T,hidden]`, grid over
   `T`) from the 8-token warmup was reused for a 20-token prompt →
   `Cannot reshape array of size 20480 into shape (1,20,2560)`. Fixed: `t` in the key.
   Audited the other new config caches; `PleGateKey` got `batch` for the same reason
   (`[batch,seq,hc,hidden]` output keyed by rows only).

## 3. Chunked GDN (Lever A) root cause — fixed

`GDN_CHUNK_KERNEL_SCAN_BODY` staged the `A_c` tile cooperatively as
`A_s[o][i] = A_base[(d0 + o) * Dk + …]` — but `d0 = (tid % 8) * 16` is **per-thread**, so the
shared `[16][32]` tile mixed rows from all eight segs and every reader got the wrong `A_c` rows
(only `i % 8 == seg` terms were right). The parity test hid it on full chunks because with
`g ∈ [0.5, 0.9]` and C=64 the `A_c·M_pre` term decays to ~1e-10 and `M_post ≈ B_c`; the 2-token
partial chunk exposed it (`state 0.275`). On the real model (many heads have g≈1) this produced
garbage output (needle test fails, output `!!`). Fix: stage all `Dk` rows (`A_s[Dk][16]`, 8 KiB,
BK=16 steps) and read `A_s[d0 + o][i]`. The kernel-index sim in `research/` modelled the load
per-thread and therefore could not catch it — index sims must model *shared* memory as shared.

Even fixed, this formulation is not a speed win here (see §4): fold does 2× the stock
recurrence work, replay does 1× more, and the scan is a scalar threadgroup-memory loop whose
serial chain (NC × Dk×16 FMAs per thread) is as long as the stock kernel's. A chunkwise GDN that
wins must put the intra-chunk and inter-chunk products on `simdgroup_matrix`/NAX (WY/UT-transform
form, C×C and C×Dk tiles), not scalar loops.

## 4. Whole-model prefill measurements

Harness: `work/bench_prefill.py` (own server on :11234 under the shared GPU lock; `--ctx-size
131072 --prefix-cache-entries 0 --prefill-chunk 8192`; MTP off; one identical public-source prompt
of **65,836 input tokens** with a passphrase needle at the start; `cache_read_input_tokens=0`
checked on every request). Numbers are the server's own `prefill: N tok/s`.

**Validity gate:** this machine runs other memory-heavy apps; when free memory dipped (admission
`available` < ~30 GB) *decode* — untouched by any lever — collapsed from ~61 to 5–13 tok/s and
prefill with it. Reps whose decode canary is < 55 tok/s are marked INVALID below and excluded.

| Arm | Env (MLX_SERVE_…) | prefill tok/s per rep (decode canary) | valid max | vs base | output correct |
|---|---|---|---:|---:|---|
| upstream main fa76a4b (clean) | `` | 1785 (61), 1868 (62), 1866 (62) | 1868 | 1.00× | yes |
| arena, all levers off | `` | 581 (9) INVALID, 677 (10) INVALID, 929 (13) INVALID | — | — | yes |
| arena, MLX_SERVE_PREFILL_TURBO=1 (A–H) | `TURBO=1` | 412 (9) INVALID, 375 (13) INVALID, 308 (5) INVALID | — | — | **NO** (0/3) |
| turbo minus A | `TURBO=1 GDN_CHUNKED=0` | 472 (67), 484 (67), 486 (0) INVALID | 484 | 0.26× | **NO** (0/3) |
| A chunked GDN (pre-fix binary) | `GDN_CHUNKED=1` | 1522 (40) INVALID | — | — | **NO** (0/1) |
| B HC up+mix (plain SIMD) | `HC_UP_MIX=1` | 926 (59), 979 (58) | 979 | 0.52× | yes |
| B HC up+mix (NAX) | `HC_UP_MIX=1 HC_UP_MIX_NAX=1` | 1397 (38) INVALID, 1474 (59) | 1474 | 0.79× | yes |
| C MoE down+score+reduce (arena fp32 kernel) | `MOE_DOWN_REDUCE=1` | 1682 (58), 1836 (59) | 1836 | 0.98× | yes |
| D MoE gate/up+GeGLU (plain SIMD) | `MOE_GATEUP_FUSED=1` | 583 (58), 639 (58) | 639 | 0.34× | yes |
| D MoE gate/up+GeGLU (NAX) | `MOE_GATEUP_FUSED=1 MOE_GATEUP_NAX=1` | 1298 (58), 1461 (58) | 1461 | 0.78× | yes |
| E GDN prework/norm-gate at prefill widths | `GDN_PREFILL_FUSED=1` | 1299 (54) INVALID, 1812 (58) | 1812 | 0.97× | yes |
| F HC write+group-norm | `HC_WRITE_NORM=1` | 1562 (58), 1645 (37) INVALID | 1562 | 0.84× | yes |
| G grouped QSA gather (NAX inner product) | `QSA_GROUP=1` | 611 (12) INVALID, 538 (57) | 538 | 0.29× | yes |
| G grouped QSA gather (SIMD inner product) | `QSA_GROUP=1 QSA_GROUP_NAX=0` | 1480 (40) INVALID, 1906 (44) INVALID | — | — | **NO** (0/2) |
| H PLE gate+value fusion | `PLE_GATE_FUSED=1` | 1521 (0) INVALID, 1671 (0) INVALID | — | — | **NO** (0/2) |

Base (valid max): 1868 tok/s. Canary: decode ≥ 55 tok/s (base decode ≈ 61).

Notes on the table: "INVALID" reps are the ones where the decode canary shows the machine was
being shared (a PyTorch/MPS job ran on this Mac for part of the session — its timing collapsed
decode to 5–13 tok/s and prefill with it; those reps say nothing about the lever). `output
correct` = the 65,836-token prompt returned the passphrase placed at its start on every rep.
The C row measured the arena's original fp32-accumulate kernel (binary built before I replaced it
with the bit-exact order); at 0.98× it is within noise of baseline either way. E is 0.97× —
neutral, as its 83 ms/chunk budget predicted. Turbo (all levers) could not complete a request
before the two wiring fixes; after them it is 0.26× and wrong.

**Whole-model verdict:** no lever is a measurable win on M5 as implemented. The plain-SIMD
GEMMs (B, D) are 2–3× slower than MLX's own quantized GEMMs, the NAX versions are still 20% slower,
the grouped QSA NAX kernel is 3.5× slower than the upstream `msv_qsa_nax_precise`, and the
grouped SIMD variant produces wrong output at production geometry while passing its small-shape
parity test. A, G-SIMD and H corrupt the model output.


## 5. Review notes per lever (code)

- **Env-name collision (D):** `MLX_SERVE_MOE_GATEUP_FUSED` is already upstream's kill switch for
  the *decode* gate/up fusion (`gatherQmvGateUpEnabled`, default ON). Lever D reuses the same
  name for the prefill kernel (default OFF), so `=0` also disables the decode fusion. Rename
  (e.g. `MLX_SERVE_MOE_PREFILL_GATEUP`).
- **Dequant in the NAX GEMMs (B-NAX, D-NAX):** per-element scalar nibble extraction with
  `scales`/`biases` re-loaded from global memory **per weight element** (4 loads per element in
  D-NAX), BK=32 → 80 k-iterations with 2 barriers and only 2 MMAs per simdgroup each. MLX's
  `QuantizedBlockLoader` loads scales once per group per row and vectorises the packed reads;
  reuse it (my 2026-09-10 prototype `work/grouped_gateup.metal` did, and was exact + 20–25%
  faster than stock at S=8192 on the isolated chain).
- **Lever G grouped QSA:** the union tile is processed by *all* G tokens' simdgroups with `-inf`
  masks for rows a token did not select, so compute is `G × |union|` ≥ `Σ|per-token|`; the HBM
  saving only materialises if the single-token kernel was not already served by L2/SLC (adjacent
  threadgroups read the same blocks back-to-back). The redundant per-thread k-way merge carries
  6·G scalars + `TB·G` ints per thread — register pressure at G=4, NSG=2 (256-thread groups vs 64
  in the single-token kernel). Measure before assuming the ~G× win.
- **Lever C numerics:** see §1 — the "MLX Reduce widens bf16 to fp32" claim is true for the CPU
  backend only. To be bit-exact on Metal you must reproduce `col_reduce_small`'s 8-partial-sum
  order (I did exactly that on 2026-09-07: 2.31→0.67 ms at S=4096, 10.5 M outputs, zero diff).
- **Turbo composition:** `leverOptIn` freezes the turbo decision on first query per lever; fine,
  but the master switch currently arms two levers that fail their own parity tests (A before
  the fix, H) and one that fails on the first real request (B/C before the fixes). Turbo should
  never arm a lever whose suite test is red.
- **Tests vs production:** every new kernel's test compares values on shapes ≤ a few hundred rows
  with `B=1`; none exercises the *production wiring* (return shapes, config caches across
  warmup→request, S=8192). A single HTTP smoke test (`tests/test_*.py` pattern already in the
  repo) with two consecutive prompts of different length would have caught bugs §2.1 and §2.2.

## 6. Recommendation

1. **Do not open a PR from this branch as-is.** CONTRIBUTING requires a green suite and an
   llmprobe-backed perf claim; the branch has 4 red tests (1 after my fixes, see below), no perf
   win, and three levers that break outputs. I am attaching this report + the fixes to the fork
   branch and posting the numbers on #366 instead.
2. **Keep (after review):** the four fixes in this report (wiring shapes, config-cache keys, the
   chunked-GDN scan staging, the bit-exact down+reduce order). They make the opt-in levers *safe*;
   they do not make them fast.
3. **Drop or rewrite:** B/D plain-SIMD GEMMs (never faster than MLX's `qmm`/`gather_qmm` on NAX
   hardware); H (semantic mismatch, needs a real reference on the production `pleForward` chain);
   G-NAX (register/exchange design; 3.5× slower). G's *idea* (cross-token block reuse) is the one
   attention lever worth keeping: the SIMD variant already matches the NAX-precise kernel's speed
   despite a scalar inner product, so a correct grouped kernel with the NAX inner product could beat
   upstream — but its correctness bug at qL=8192/kv≤65k must be found first (test at production
   geometry against `msv_qsa_nax_precise`, not only at S=15).
4. **The only ≥1.5× candidate is still GDN chunkwise on tensor cores.** The recurrence is ~20 of the
   33 ms/layer GDN block (36 layers → ~0.7 s of the ~4.7 s chunk) and is latency-bound (8192
   dependent steps × 192 threadgroups). A WY/UT-transform chunkwise form with C=64: intra-chunk
   `[64×128]·[128×64]` and inter-chunk `[128×64]·[64×128]` products on `simdgroup_matrix`/NAX, one
   sequential pass over 128 chunk boundaries per (b, head). Budget: 20 → ~3 ms/layer ≈ −0.6 s/chunk
   ≈ +15% whole-model — the largest single item; combined with an exact grouped-QSA (−0.4–0.6 s)
   and the exact MoE reduce (−0.15 s) it reaches the ~1.35–1.45× region. Nothing in this branch is
   that kernel yet: the fold/scan/replay here does 3× the stock arithmetic with scalar loops.
