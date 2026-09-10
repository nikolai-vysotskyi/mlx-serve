# Chunkwise/WY GDN prefill kernel — design & evidence

Status: **reference-test validated, kernel unbuilt**. The math is proven exact
and passes the repo's exactness bar on CPU (f32/f64 NumPy). The Metal kernel
and any M5 tok/s win are NOT yet implemented or measured — this document is the
precise contract for that step (which needs Apple Silicon: no Zig/Metal on the
cloud box).

Evidence levels (per the task's labeling rule):
- **Reference test** (this doc + `gdn_chunked_reference.py`): the chunkwise/WY
  reformulation is the *same recurrence* (f64 max-diff ~1e-15) and the f32 form
  passes the repo's parity bar with error bit-identical to the stock kernel.
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

State errors are likewise bit-identical (3.89e-3 / 3.84e-3) across all arms.
Both the naive O(Dk³) fold and the WY O(C·Dk²) form produce *bit-identical*
outputs to the stock kernel after bf16 truncation — the chunk-boundary
re-association is absorbed by the bf16 state hand-off (same tolerance class as
the existing "chunk-boundary continuity" test: `0.02·max|state| + 0.02`).

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

**C=64 is the default**: matches fla's chunk size, fits the 32 KiB threadgroup
budget for the C×C `L` matrix, and gives the best depth (≈192 = scan 128 +
replay 64, plus the ≤64-step solve) at the lowest added FLOP (+1.75×). Naive
Dk³ fold would be +85.1× — WY is the only viable form. The win is purely the
dependency-depth reduction (T → ~`2C + T/C`: chunk solve `C`, chunk scan `T/C`,
intra-chunk replay `C`). FLOPs go *up*, acceptable only because the kernel is
latency-bound (see §6).

## 5. Kernel structure (three phases)

All phases keyed by `(b, hv)`; GQA maps `hv → hk = hv/(Hv/Hk)` exactly as the
shipped kernels do. Chunk size `C` is a compile-time constant (16/32/64) —
see the threadgroup budget below.

**Phase A — per-chunk WY (fully parallel over `(b, hv, chunk)`):**
1. Load k,v,β,g for the chunk; compute `expg` via a log-domain prefix scan.
2. Build `L` (C×C strictly-lower) = β_i·(k_i·k_j)·expg_i/expg_j — a C×C matrix
   of rank-Dk inner products, parallel over (i,j).
3. Solve `(I+L)⁻¹ x` for x = (β·expg·k) and x = (β·v) → `w`, `u`. Use forward
   substitution (depth C) or the nilpotent repeated-squaring
   `(I+L)⁻¹ = Π_j (I + L^{2^j})` (depth log₂C); correctness pinned by §2.
4. Emit `w` [C×Dk], `u` [C×Dv], and the chunk's `G` and `expg` tail (for the
   scan), to device.

**Phase B — boundary scan (sequential over chunks, NC = T/C steps):**
Per `(b, hv)`, advance the running `M` (Dk×Dv) without materializing `A_c`:

```
M ← G·M − Σ_t (G/expg_t) k_t (w_tᵀ M) + Σ_t (G/expg_t) k_t u_tᵀ
```

i.e. two `O(C·Dk·Dv)` matmuls per chunk (K̃ @ (w @ M) and K̃ @ u) — the same
form as fla `fwd_h` (`b_v = w·h; b_h += k·(u − b_v)`). Store the per-chunk
boundary `M_c` (Dk×Dv per chunk) for Phase C. The scan is NC sequential steps
vs T today.

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

The block profile (`reports/block-profile.json`, S=8192) puts the GDN
recurrence at 1115–1298 ms/layer. Its FLOP-equivalent is ~20 GFLOP (~0.4 ms at
50 TFLOP/s), i.e. the stock kernel runs ~2800× above its FLOP floor — the
signature of a latency-bound serial chain, not a bandwidth- or FLOP-bound one.
The existing `GDN_KERNEL_BLOCKED_BODY` fixes the memory-coalescing side
(~2× per oMLX: 14.9 vs 29.7 ms @16K) but leaves the T-step serial chain. The
chunked form attacks the chain directly. **Whether the post-blocked kernel is
still latency-bound — and therefore whether the depth win shows up — must be
measured on M5.** No claim of tok/s is made from this box.

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

## 9. Next steps (need Apple Silicon)

1. Implement the three phases in Metal + Zig plumbing in src/transformer.zig
   (mirror `GDN_KERNEL_BLOCKED_BODY` conventions: template dtypes InT/StT/OutT,
   simdgroup reductions, threadgroup staging, per-TB cached kernel objects).
2. `zig build test` + the new parity tests on real Apple Silicon.
3. Bench prefill at chunk 8192 on the named model vs the named baseline
   (`bench.contextScaling[].prefillTokPerSec`, llmprobe 0.6.6, rungs 64k,
   chunk8192, prefix-cache 0, MTP off).
4. Only then claim any tok/s number; report per the evidence-level rules.
