"""Index-level simulation of the chunked GDN Metal kernels.

Reproduces the index arithmetic of `GDN_CHUNK_KERNEL_{FOLD,SCAN,REPLAY}_BODY`
in `src/transformer.zig` (fold rank-1 augmented-basis recurrence, sequential
boundary scan, per-chunk replay) and checks the result end-to-end against the
f32/f64 references in `gdn_chunked_reference.py`. It is a CPU *reference* check
of kernel INDEXING and the fold/scan orientation convention — NOT a Metal speed
measurement.

The fold is run BOTH ways:
  * vectorized (matrix recurrence over all basis columns at once — fast), and
  * per-column scalar (the Metal's actual per-thread evolution) for ONE
    (hv, chunk), whose write is then scattered with the Metal's EXACT index
    expression `A_out[dk_out*Dk + g] = st[dk_out]` (and, for contrast, the
    pre-fix transpose `A_out[g*Dk + dk_out]`, which must NOT match A_c).

Conventions locked in here (must match the Metal):
  * state buffer is TRANSPOSED [Hv, Dv, Dk] (row dv, col dk) — M[dv][dk].
  * A_c is [dk_out][dk'] row-major; scan does M_new[dk_out][dv] =
    sum_dk' A_c[dk_out][dk'] * Mpre[dv][dk'] + B_c[dk_out][dv].
  * fold thread owns column g (= dk') and output rows d0..d0+16 (= dk_out);
    A-part write is A_out[dk_out*Dk + g] (NOT g*Dk + dk_out).
  * replay reproduces the shipped per-token recurrence inside each chunk from
    the pre-chunk boundary state M_seq[nc-1] (or state_in for nc==0).

The one thing it cannot check is the fp32 reduction tree of the in-kernel
simd_shuffle_down dot products; NumPy's dot sum order differs by a few fp32
ulp, so the tolerances below are set well above that but far below O(1) errors
an index bug would produce.
"""
import numpy as np

from gdn_chunked_reference import (
    DK, DV, HK, HV, GROUP,
    bf16, gen, seq, chunk_scan,
)

C = 64          # chunk size (matches gdnChunkC() default)
DB = 32         # dv columns per threadgroup (constexpr in the Metal)


def fold_vec(k, v, g, beta, hv, t0, tt):
    """Vectorized rank-1 fold for one (hv, chunk) -> (A_c [Dk,Dk], B_c [Dk,Dv]).

    A_c[:,g] = evolve e_g ; B_c[:,dv] = evolve 0 with v forcing — the same
    per-token op order as the Metal fold (state *= gt; kv = state.k; delta =
    (v - kv)*bt; state += k*delta)."""
    hk = hv // GROUP
    A = np.eye(DK, dtype=np.float32)
    B = np.zeros((DK, DV), dtype=np.float32)
    for t in range(tt):
        gt = np.float32(g[t0 + t, hv])
        bt = np.float32(beta[t0 + t, hv])
        kvec = k[t0 + t, hk].astype(np.float32)          # [Dk]
        vrow = v[t0 + t, hv].astype(np.float32)          # [Dv]
        # A: state *= gt ; kv = state.k ; state -= (bt*gt*kv)*k
        kvA = A.T @ kvec                                 # [Dk] per column g
        A = gt * A - (gt * bt) * np.outer(kvec, kvA)
        # B: state *= gt ; kv = state.k ; state += bt*(v - kv)*k
        kvB = B.T @ kvec                                 # [Dv] per column dv
        delta = (vrow - gt * kvB) * bt                   # [Dv]
        B = gt * B + np.outer(kvec, delta)
    return A, B


def fold_scalar_write_check(k, v, g, beta, hv, t0, tt):
    """Per-column scalar fold (the Metal's per-thread evolution) for ONE
    (hv, chunk); scatters with the Metal's exact write expression and returns
    (A_fixed_flat, A_buggy_flat) to prove the orientation."""
    hk = hv // GROUP
    A = np.eye(DK, dtype=np.float32)
    for g_ in range(DK):                                 # thread column = dk'
        state = np.zeros(DK, dtype=np.float32)
        state[g_] = 1.0
        for t in range(tt):
            gt = np.float32(g[t0 + t, hv])
            bt = np.float32(beta[t0 + t, hv])
            kvec = k[t0 + t, hk].astype(np.float32)
            state *= gt
            kv = np.float32(state @ kvec)               # already includes gt
            state -= (bt * kv) * kvec                   # A column: vcol == 0
        A[:, g_] = state
    fixed = np.empty(DK * DK, dtype=np.float32)
    buggy = np.empty(DK * DK, dtype=np.float32)
    for g_ in range(DK):
        for dk_out in range(DK):
            fixed[dk_out * DK + g_] = A[dk_out, g_]      # the corrected Metal write
            buggy[g_ * DK + dk_out] = A[dk_out, g_]      # the pre-fix transpose
    return fixed, buggy, A


def scan_sim(Ac, Bc, state_in_hv):
    """Metal scan for one hv -> (M_seq [NC,Dv,Dk], state_out [Dv,Dk]).

    Mirrors the Metal: M_seq[nc] holds the POST-chunk state (after chunk nc),
    so the replay of chunk nc reads M_seq[nc-1] (or state_in for nc==0)."""
    NC = Ac.shape[0]
    state = state_in_hv.astype(np.float32).copy()        # [Dv, Dk] transposed
    M_seq = np.empty((NC, DV, DK), dtype=np.float32)
    for nc in range(NC):
        state = (Ac[nc] @ state.T + Bc[nc]).T            # POST state [Dv, Dk]
        M_seq[nc] = state
    return M_seq, state


def replay_sim(k, q, v, g, beta, hv, t0, tt, boundary):
    """Metal replay for one (hv, chunk) from boundary [Dv,Dk]; -> y[tt,Dv]."""
    hk = hv // GROUP
    state = boundary.astype(np.float32).copy()
    y = np.empty((tt, DV), dtype=np.float32)
    for t in range(tt):
        gt = np.float32(g[t0 + t, hv])
        bt = np.float32(beta[t0 + t, hv])
        kvec = k[t0 + t, hk].astype(np.float32)
        qvec = q[t0 + t, hk].astype(np.float32)
        vrow = v[t0 + t, hv].astype(np.float32)
        state *= gt
        kv = state @ kvec
        delta = (vrow - kv) * bt
        state += np.outer(delta, kvec)
        y[t] = state @ qvec
    return y, state


def run(seed=0x5EED, T=1024):
    inp = gen(seed, T)
    k, q, v, g, beta = (inp[x].astype(np.float32) for x in ("k", "q", "v", "g", "beta"))
    S0 = np.zeros((HV, DV, DK), dtype=np.float32)
    NC = (T + C - 1) // C

    # f64-validated chunked reference (naive fold + per-token replay) and the
    # per-token sequential reference, both fp32.
    y_ref, S_ref = chunk_scan(np.float32, inp, S0, C, wy=False)
    y_seq, S_seq = seq(np.float32, inp, S0)

    y = np.empty((T, HV, DV), dtype=np.float32)
    state_out = np.empty((HV, DV, DK), dtype=np.float32)
    for hv in range(HV):
        Ac = np.empty((NC, DK, DK), dtype=np.float32)
        Bc = np.empty((NC, DK, DV), dtype=np.float32)
        for nc in range(NC):
            t0 = nc * C
            tt = min(C, T - t0)
            Ac[nc], Bc[nc] = fold_vec(k, v, g, beta, hv, t0, tt)
        M_seq, s_out = scan_sim(Ac, Bc, S0[hv])
        state_out[hv] = s_out
        for nc in range(NC):
            t0 = nc * C
            tt = min(C, T - t0)
            boundary = S0[hv] if nc == 0 else M_seq[nc - 1]
            y[t0:t0 + tt, hv], _ = replay_sim(k, q, v, g, beta, hv, t0, tt, boundary)

    # mine vs the f64-validated chunked reference: same algorithm, only the
    # fold's fp32 summation order differs -> ~1e-4 rel. An index bug (transpose,
    # off-by-a-stride, wrong boundary) is O(1), three orders above the tol.
    mask = np.abs(y_ref) > 1e-3
    dy = float((np.abs(y - y_ref)[mask] / np.abs(y_ref)[mask]).max()) if mask.any() else 0.0
    dS = float(np.abs(state_out - S_ref).max())
    # context: how far the chunked reference itself sits from the per-token seq
    dy_ref_vs_seq = float((np.abs(y_ref - y_seq)[mask] / np.abs(y_seq)[mask]).max()) if mask.any() else 0.0

    # --- one (hv, chunk) per-column scalar fold + exact Metal write expression.
    # Use a SHORT chunk so the gate product has not collapsed: A_c is then O(1)
    # and non-symmetric, and the transpose-vs-correct write differ by O(1). ---
    fixed, buggy, A0 = fold_scalar_write_check(k, v, g, beta, 0, 0, 3)
    asym = float(np.abs(A0 - A0.T).max())
    dfixed = float(np.abs(fixed - A0.flatten()).max())
    dbuggy = float(np.abs(buggy - A0.flatten()).max())

    print(f"T={T} C={C} NC={NC}")
    print(f"  mine vs chunk_scan(f64-validated) rel y err = {dy:.3e}")
    print(f"  mine vs chunk_scan          max|dstate|     = {dS:.3e}")
    print(f"  [ctx] chunk_scan vs seq(fp32) rel y err    = {dy_ref_vs_seq:.3e}")
    print(f"  A_c asym (3-step, must be >> tol)          = {asym:.3f}")
    print(f"  fold write FIXED  max|diff| vs A           = {dfixed:.3e}")
    print(f"  fold write BUGGY  max|diff| vs A           = {dbuggy:.3f}  (must be ~O(1))")
    ok = (dy < 1e-3 and dS < 1e-3 and dfixed < 1e-6 and dbuggy > 1e-2 and asym > 1e-2)
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ok = True
    for T in (128, 500, 1024, 2050):   # 2050: non-multiple of C, ragged tail
        ok &= run(seed=0x5EED, T=T)
    raise SystemExit(0 if ok else 1)
