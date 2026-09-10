# Chunkwise/WY GDN prefill kernel — design & evidence

Status: **reference-test validated, kernel unbuilt**. The math is proven exact
and passes the repo's exactness bar on CPU (f32/f64 NumPy). The Metal kernel
and any M5 tok/s win are NOT yet implemented or measured — this document is the
precise contract for that step (which needs Apple Silicon: no Zig/Metal on the
cloud box).

Evidence levels (per the task's labeling rule):
- **Reference test** (this doc + `gdn_chunked_reference.py`): the chunkwise/WY
  reformulation is the *same recurrence* (f64 max-diff ~1e-15) and the f32 form
  passes the repo's parity bar with error *equal to* the stock kernel (the
  chunk forms differ from stock by ~1–2 bf16 ULP after the chunk-boundary
  re-association, but their distance to the f64 ground truth is the same —
  which is exactly what the repo bar measures, "no worse than stock").
- **Hypothesis**: that shortening the serial dependency chain `T → ~C + T/C`
  yields an M5 prefill speedup. Not yet measured.

## 1. The recurrence (shipped, src/transformer.zig)

Per `(b, hv)` and per dv output row, state `S ∈ R^{Dk}` (fp32 registers):

```
S'  = g_t · S            # forget
kv  = S' · k_t           # readout
d   = β_t · (v_t[dv] − kv)
S'' = S' + d · k_t       # write
y   = S'' · q_t
```

Collecting the Dv rows into `M = Sᵀ ∈ R^{Dk×Dv}` (row form, as fla uses):

```
A_t = g_t (I − β_t k_t k_tᵀ)      B_t = β_t k_t v_tᵀ
M_t = A_t M_{t−1} + B_t            y_t = M_tᵀ q_t
```

This is a first-order affine recurrence in `M`; it factors into per-token
operators (parallel) + a prefix scan + per-token outputs.

## 2. The WY chunk product (proven, `check_wy_identity`)

Within a chunk of `C` tokens, let `expg_t = Π_{s≤t} g_s` (running gate product,
computed in log-space to avoid fp32 underflow — fla uses a log₂ cumsum and
`exp2` of differences for exactly this reason), and `G = expg_{C−1}`. Then:

```
L[i,j] = β_i · (k_i · k_j) · expg_i / expg_j        (i > j, strictly lower)
w      = (I + L)⁻¹ · (β · expg · k)                [C × Dk]
u      = (I + L)⁻¹ · (β · v)                       [C × Dv]

A_c = G·I − Σ_t (G/expg_t) · k_t w_tᵀ              [Dk × Dk]  (chunk transition)
B_c = Σ_t (G/expg_t) · k_t u_tᵀ                    [Dk × Dv]  (chunk forcing)

M_c = A_c M_{c−1} + B_c
```

Verified identities (f64, `gdn_chunked_reference.py`): `A_c` matches the direct
fold to 6.9e-17, `B_c` to 2.4e-17. These are exactly fla's
`chunk_gated_delta_rule_fwd_intra` (kkt + solve_tril + `recompute_w_u`) +
`fwd_h` state update, transcribed for the per-token multiplicative gate.

## 3. Measured numerics (f32 vs f64, repo bar)

Repo house rule (`gdnBlockedParityCase`): judge vs f64 ground truth, never
kernel-vs-kernel; pass iff `err_new ≤ 1.5·err_stock + 0.02` on max-abs-diff of
the bf16-truncated outputs. Results (Dk=Dv=128, Hk=16, Hv=48, GQA=3):

| T     | C   | stock y_err | chunk-scan | chunk-scan-wy | bar (y)      | verdict |
|-------|-----|-------------|------------|---------------|--------------|---------|
| 1024  | 128 | 2.75e-2     | 2.75e-2    | 2.75e-2       | 6.13e-2      | PASS    |
| 2048  | 128 | 2.59e-2     | 2.59e-2    | 2.59e-2       | 5.89e-2      | PASS    |
| 2048  | 256 | 2.59e-2     | 2.59e-2    | 2.59e-2       | 5.89e-2      | PASS    |

State errors are likewise equal (3.89e-3 / 3.84e-3) across all arms. The naive
O(Dk³) fold and the WY O(C·Dk²) form produce the *same error magnitude* as the
stock kernel after bf16 truncation, but not bit-identical values: the
chunk-boundary re-association (one `A_c M + B_c` compose vs T per-token
updates) rounds differently by ~1–2 bf16 ULP (measured max|Δ| = 3.9e-3 in y,
9.8e-4 in state at magnitude ~1). This is the same tolerance class the existing
"chunk-boundary continuity" test already accepts (`0.02·max|state| + 0.02`),
and the repo parity bar is "no worse than stock vs f64" — not bit-equality.

f64 exactness of the block form vs sequential: y 2.0e-15 / state 6.1e-16
(chunk-scan), y 6.3e-15 / state 4.1e-15 (WY).

## 4. Cost / depth budget (per GDN layer)

Sequential recurrence FLOP: `T · Hv · (2·Dk·Dv + Dk·Dv + Dk)`.

| T    | C   | recurrence | + WY chunk products | depth T → (scan+replay) |
|------|-----|-----------:|--------------------:|--------------------------|
| 4096 | 64  |  9.69 GFLOP | +16.91 GFLOP (1.75×) | 64 + 64                  |
| 4096 | 128 |  9.69 GFLOP | +25.77 GFLOP (2.66×) | 32 + 128                 |
| 8192 | 64  | 19.38 GFLOP | +33.82 GFLOP (1.75×) | 128 + 64                 |
| 8192 | 128 | 19.38 GFLOP | +51.54 GFLOP (2.66×) | 64 + 128                 |
| 8192 | 256 | 19.38 GFLOP | +96.64 GFLOP (4.99×) | 32 + 256                 |

**C=64 is the default**: matches fla's chunk size and gives the best depth
(≈192 = scan 128 + replay 64, plus the ≤64-step solve) at the lowest added
FLOP (+1.75×).

**Two ways to form the chunk product, both validated:**
1. *WY (fla)* — the C×C solve `(I+L)⁻¹`. Needs a triangular solve in Metal
   (forward substitution or nilpotent repeated squaring); lowest FLOP
   (+1.75× @C=64), no A_c materialization.
2. *Rank-1 fold* — since `A_t = g_t(I − β_t k kᵀ)` is a scaled rank-1
   perturbation of I, the chunk product is `O(C·(Dk² + Dk·Dv))` ≈ 1.3× the WY
   FLOP and needs **no solve at all**: it is literally the shipped per-token
   recurrence run on `(Dk + Dv)` basis vectors (A_c's columns from the Dk unit
   vectors, B_c's columns from zero state fed by β k vᵀ). This makes the fold
   kernel a near-copy of `GDN_KERNEL_BLOCKED_BODY`'s inner loop, at the cost of
   materializing A_c/B_c (~0.8 GB transient per layer at T=8192). A dense
   Dk³ fold would be +85× — never do that; the rank-1 structure is what makes
   the fold cheap.

The win is purely the dependency-depth reduction (T → ~`2C + T/C`). FLOPs go
*up*, acceptable only because the kernel is latency-bound (see §6).

## 5. Kernel structure (three phases)

All phases keyed by `(b, hv)`; GQA maps `hv → hk = hv/(Hv/Hk)` exactly as the
shipped kernels do. Chunk size `C` is a compile-time constant (16/32/64) —
see the threadgroup budget below.

**Phase A — per-chunk rank-1 fold (fully parallel over `(b, hv, chunk)`):**
The IMPLEMENTED form (chosen over WY for this first cut — no C×C solve, and
the fold kernel is a near-copy of the blocked kernel's inner loop). Because
`A_t = g_t(I − β_t k_t k_tᵀ)` is a scaled rank-1 perturbation of I, the chunk
product `A_c` (Dk×Dk) and forcing `B_c` (Dk×Dv) are the shipped per-token
recurrence run on the `Dk + Dv` augmented basis columns: A-columns start at
`e_j` with v≡0; B-columns start at 0 with the real v. No solve at all.

```
for col in 0..Dk+Dv-1:          # parallel over 32-column tiles
    st = (col < Dk) ? e_col : 0
    for t in chunk:
        st = g_t·st − g_t·β_t·(st·k_t)·k_t + β_t·v_t[col−Dk]·k_t   # v≡0 if col<Dk
    A_c[:, col]     = st         (col < Dk)      # [dk_out][dk']
    B_c[:, col−Dk]  = st         (col ≥ Dk)      # [dk_out][dv]
```

Cost O(C·(Dk² + Dk·Dv)) ≈ 1.3× the WY FLOP at C=64, but it materializes
A_c/B_c (403/201 MB at 8192/C=64 — see §4). WY remains the follow-up for
larger C (≤64 budget-free) once the scan/rank-1 path is measured.

**Phase B — boundary scan (sequential over chunks, NC = T/C steps):**
Per `(b, hv)`, advance the running `M` (Dk×Dv) via the materialized product:
`M ← A_c @ M + B_c`, tiled 32×32 over dk, with `M` held transposed `[dv][dk]`
in threadgroup memory (the shipped state-buffer convention). Store the
per-chunk POST boundary `M_c` (as `[dv][dk]` f32) for Phase C. NC sequential
matmul steps vs T vector steps today.

**Phase C — output replay (parallel over `(b, hv, chunk)`):**
From each chunk's boundary `M_{c−1}`, replay the shipped per-token recurrence
inside the chunk (exact same op order as `GDN_KERNEL_BLOCKED_BODY`) to emit
`y_t` and the final state. This is the existing blocked kernel's inner loop
run over `C` tokens instead of `T`.

**Threadgroup budget (Metal, 32 KiB):** Phase A needs `L` (C×C fp32) resident;
C=64 → 16 KiB fits, C=128 → 64 KiB does NOT. Phase C reuses the blocked
kernel's staging (`k_s`/`q_s`/`v_s`, TB-token blocks — its `gdnBlockedTgBytes`
clamp applies unchanged). So **C ≤ 64** for a self-contained Phase A, or spill
`L` to device and tile the solve. C=64 (default) gives depth `8192 → 192`
(≈43×) at +1.75× added FLOP.

## 6. Why this is the right lever (hypothesis, unmeasured)

The block profile (`reports/block-profile.json`, S=8192) puts the GDN bucket at
1115–1298 ms/layer. **That profile already includes the blocked-seq kernel**
(cc7dea1 contains `GDN_KERNEL_BLOCKED_BODY`, default-on): it fixes
coalescing (~2× per oMLX: 14.9 vs 29.7 ms @16K) but still loops per-token
serially inside each TB block (confirmed in LOCAL_HISTORY: "blocked-seq kernel
лишь staging блоков, внутри всё ещё идёт цикл по токенам"). The chunked form
attacks that remaining serial chain directly, on top of the blocked kernel.

**Open question the kernel must resolve, not the cost model:** how much of the
1115 ms bucket is the recurrence's serial chain vs. the GDN projections
(36 dense matmuls — issue #366 puts them at 8% at 4096 tokens) vs. conv/prework.
The recurrence's FLOP-equivalent is ~20 GFLOP (~0.4 ms at 50 TFLOP/s), so it is
~2800× above its FLOP floor — latency-bound — but its share of the bucket is
unmeasured. **Whether the depth win shows up in tok/s must be measured on M5.**
No claim of tok/s is made from this box.

## 7. Fallback / geometry contract

Reuse `gdnBlockedEligible` (Dk==128, Dv%32==0, GQA-aligned, T ≥ 64) and add a
chunk gate: enable only when `T ≥ C` and the phase-A solve fits the budget.
Everything else (decode, spec-verify, capture-seq, off-geometry) stays on the
existing stock/blocked kernels, exactly as today. Kill switch pattern
(`MLX_SERVE_GDN_CHUNKED=0`) mirrors `MLX_SERVE_GDN_BLOCKED`.

## 8. Test plan (mirrors the existing harness)

- Extend `gdnRunYState` with a `chunked` mode and add a
  `gdnChunkedParityCase` asserting no-less-accurate-than-stock vs the f64 host
  ref across the `GdnCase` sweep (bf16/f32/f16 activations, GQA, Dv 32–128,
  T non-multiple of C).
- Chunk-boundary continuity: split run at a non-multiple of C must match the
  full run within `0.02·max|state| + 0.02`.
- Golden cross-check vs `research/gdn_chunked_reference.py` (NumPy f64) for a
  fixed seed, asserting `max|diff| < 1e-3`-class agreement.
- **`research/gdn_chunked_kernel_sim.py`** (runs here, CPU): reproduces the
  kernels' EXACT index arithmetic (rank-1 fold → A_c/B_c, sequential scan with
  POST-per-chunk M_seq, per-chunk replay) in NumPy and checks end-to-end vs the
  f64-validated `chunk_scan`. PASS at T=128/500/1024/2050: mine-vs-reference
  rel y err 2.0e-4…4.0e-4 (the reference's own re-association vs seq is
  1.7e-4…2.4e-4), state 4.2e-7. It also pins the fold write orientation: the
  corrected `A_out[dk_out*Dk+dk']` scatter is exact (0.0) while the pre-fix
  transpose is O(1) off (A_c 3-step asym 0.018–0.064). Two Metal bugs were
  found and fixed by this review: (1) fold stored A_c transposed; (2) fold
  grid.x lacked the ×256 thread factor (`set_grid` counts threads).

## 9. Next steps (need Apple Silicon)

1. ✅ Implemented (UNBUILT): the three phases are in src/transformer.zig —
   `GDN_CHUNK_KERNEL_{FOLD,SCAN,REPLAY}_BODY` (rank-1 fold, tiled boundary
   scan, chunked replay), Zig plumbing (`gdnChunkedEnabled/C/Eligible`,
   `gdnRunYStateChunked`, per-(C,TB) kernel caches), and
   `gdnChunkedParityCase` + parity sweep + chunk-boundary continuity tests.
   NOT compiled or run here (no Zig/Metal); two index bugs already found and
   fixed via the CPU sim (§8). Production wiring into `gdnForward` is
   intentionally deferred until the kernels pass on Mac.
2. `zig build test` + the new parity tests on real Apple Silicon; fix any MSL
   compile issues this first build surfaces.
3. Bench prefill at chunk 8192 on the named model vs the named baseline
   (`bench.contextScaling[].prefillTokPerSec`, llmprobe 0.6.6, rungs 64k,
   chunk8192, prefix-cache 0, MTP off).
4. Only then claim any tok/s number; report per the evidence-level rules.
