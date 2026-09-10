#!/usr/bin/env python3
"""
GDN prefill: chunkwise (block) reformulation of the GatedDeltaNet recurrence.

Evidence level: CPU reference test (NumPy f32/f64). NOT an M5/Metal tok/s
measurement. It answers three gating questions about the chunk-parallel
reformulation of the prefill recurrence (the #1 candidate from
ddalcu/mlx-serve#366):

  Q1  Is the block form the same recurrence?        (f64 block vs seq ~ eps)
  Q2  Does the f32 block form clear the repo's exactness bar?
  Q3  What does the realistic WY form cost / buy in dependency depth?

Shipped kernel (src/transformer.zig `gdnKernelSource`, `GDN_KERNEL_BLOCKED_BODY`),
per (b, hv) and per dv output row, state S in R^{Dk} (fp32 registers):

    S'  = g_t * S            (forget)
    kv  = S' . k_t           (readout)
    d   = beta_t * (v_t[dv] - kv)
    S'' = S' + d * k_t       (write)
    y   = S'' . q_t

Collecting the Dv rows into M = S^T (R^{Dk x Dv}):

    A_t = g_t (I - beta_t k_t k_t^T)     B_t = beta_t k_t v_t^T
    M_t = A_t M_{t-1} + B_t              y_t = M_t^T q_t

Two block variants are measured, both leave the INTRA-chunk recurrence in the
shipped kernel's exact per-token order and only reformulate the chunk->chunk
boundary state:

  chunk_scan     naive fold: A_c = prod A_t, B_c = sum (prod_{s>t} A_s) B_t via
                 O(Dk^3) matmuls. Numerically the cleanest baseline; proves the
                 boundary-hand-off is exact. NOT the shipable kernel.

  chunk_scan_wy  fla's WY form (O(C*Dk^2) + C^3 solve), the realistic kernel.
                 Proven identities (verified to ~1e-17 in f64, see check_wy):
                   expg[t] = prod_{s<=t} g_s            G = expg[C-1]
                   L[i,j]  = beta_i (k_i . k_j) expg_i/expg_j     (i > j)
                   w = (I+L)^{-1} (beta * expg * k)     [C x Dk]
                   u = (I+L)^{-1} (beta * v)            [C x Dv]
                   A_c = G*I - sum_t (G/expg[t]) k_t w_t^T
                   B_c = sum_t (G/expg[t]) k_t u_t^T
                   M_c = A_c M_{c-1} + B_c
                 These are exactly fla's chunk_gated_delta_rule_fwd_intra +
                 fwd_h state update (g gate folded as exp2 of the local cumsum).

Exactness bar (mirrors the repo's `gdnBlockedParityCase` house rule): judge vs
the f64 host ground truth, never kernel-vs-kernel; a new kernel passes iff
    err_new  <=  1.5 * err_stock  +  0.02
on max-abs-diff of the bf16-truncated outputs (both upcast to f32).

NOTE: the chunk forms are NOT bit-identical to the stock kernel — the
chunk-boundary compose (`A_c M + B_c` vs T per-token updates) rounds
differently by ~1-2 bf16 ULP (max|diff| ~3.9e-3 y / ~9.8e-4 state). They pass
because their distance to f64 is EQUAL to the stock kernel's, which is exactly
what the repo bar ("no worse than stock") measures.

Run:  python3 research/gdn_chunked_reference.py
"""

import numpy as np

DK = DV = 128     # head dim (Flash-Next GDN)
HK = 16           # key heads
HV = 48           # value heads (Hv/Hk = 3)
GROUP = HV // HK


def bf16(x: np.ndarray) -> np.ndarray:
    """Truncate f32 to bf16-representable values (top 16 bits), mirroring the
    repo's `bf16Trunc` so every arm consumes identical inputs."""
    return (x.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def gen(seed: int, T: int) -> dict:
    rng = np.random.default_rng(seed)
    g = bf16((0.5 + 0.4 * rng.random((T, HV))).astype(np.float32))
    beta = bf16(rng.random((T, HV)).astype(np.float32))
    k = bf16((rng.random((T, HK, DK)) - 0.5).astype(np.float32))
    q = bf16((rng.random((T, HK, DK)) - 0.5).astype(np.float32))
    v = bf16((rng.random((T, HV, DV)) - 0.5).astype(np.float32))
    return {"k": k, "q": q, "v": v, "g": g, "beta": beta}


def seq(dtype, inp, S0):
    """Faithful per-token recurrence over (t, hv, dv-row). State S[Hv, Dv, Dk].
    With dtype=f64 this is exactly the repo's f64 host ground truth."""
    k, q, v, g, beta = (a.astype(dtype) for a in
                        (inp["k"], inp["q"], inp["v"], inp["g"], inp["beta"]))
    T = k.shape[0]
    hk = np.arange(HV) // GROUP
    S = S0.astype(dtype).copy()
    y = np.empty((T, HV, DV), dtype=dtype)
    for t in range(T):
        S *= g[t][:, None, None]
        kv = np.einsum("hid,hd->hi", S, k[t][hk])
        delta = (v[t] - kv) * beta[t][:, None]
        S += delta[:, :, None] * k[t][hk][:, None, :]
        y[t] = np.einsum("hid,hd->hi", S, q[t][hk])
    return y, S


def chunk_scan(dtype, inp, S0, C, wy=False):
    """Chunk-scan: intra-chunk in the shipped kernel's exact per-token order;
    chunk boundary state advanced via a chunk product A_c / forcing B_c.
    wy=False uses the naive O(Dk^3) fold; wy=True uses the WY O(C*Dk^2) form.
    Both leave y produced by an exact per-token replay inside each chunk."""
    k, q, v, g, beta = (a.astype(dtype) for a in
                        (inp["k"], inp["q"], inp["v"], inp["g"], inp["beta"]))
    T = k.shape[0]
    hk = np.arange(HV) // GROUP
    M0 = S0.astype(dtype).transpose(0, 2, 1)                  # [Hv, Dk, Dv]
    I = np.eye(DK, dtype=dtype)
    NC = (T + C - 1) // C

    def fold(t0, t1):
        """Return (A_c, B_c) for the chunk, either naive or WY."""
        if not wy:
            a = np.broadcast_to(I, (HV, DK, DK)).copy()
            b = np.zeros((HV, DK, DV), dtype=dtype)
            for t in range(t0, t1):
                gt = g[t][:, None, None]
                bt = beta[t][:, None, None]
                kt = k[t][hk]
                at = gt * (I[None] - bt * np.einsum("hi,hj->hij", kt, kt))
                btt = bt * np.einsum("hi,hj->hij", kt, v[t])
                b = at @ b + btt
                a = at @ a
            return a, b
        # ---- WY form (fla), per hv head
        Ac = np.empty((HV, DK, DK), dtype=dtype)
        Bc = np.empty((HV, DK, DV), dtype=dtype)
        for h in range(HV):
            kk = k[t0:t1, hk[h]]                              # [C, Dk]
            vv = v[t0:t1, h]                                  # [C, Dv]
            bb = beta[t0:t1, h]                               # [C]
            gg = g[t0:t1, h]                                  # [C]
            # log-space running gate (fla: g is a log2 cumsum; exp2 of
            # differences). Avoids cumprod underflow for large C in fp32.
            lg = np.cumsum(np.log(gg.astype(np.float64)))     # [C] (f64 log)
            lg = lg.astype(dtype)
            G = np.exp(lg[-1])
            # log-diff masked to the lower triangle BEFORE exp: exp(+large) in
            # the upper triangle would overflow fp32 and 0*inf -> NaN.
            logdiff = np.where(np.tril(np.ones((t1 - t0, t1 - t0)), -1).astype(bool),
                               lg[:, None] - lg[None, :], -np.inf)
            ratio = np.exp(logdiff)                           # expg_i/expg_j, i>j
            expg = np.exp(lg)                                 # running product
            KK = kk @ kk.T                                    # [C, C]
            L = KK * bb[:, None] * ratio                      # [C, C] (i>j only)
            Ainv = np.linalg.inv(np.eye(t1 - t0, dtype=dtype) + L)
            w = Ainv @ (bb[:, None] * expg[:, None] * kk)     # [C, Dk]
            u = Ainv @ (bb[:, None] * vv)                     # [C, Dv]
            scale = np.exp(lg[-1] - lg)[:, None]              # G/expg[t]
            Ac[h] = G * I - (kk * scale).T @ w                # [Dk, Dk]
            Bc[h] = (kk * scale).T @ u                        # [Dk, Dv]
        return Ac, Bc

    # fold chunks (parallel-able), then scan boundary states (sequential over NC)
    y = np.empty((T, HV, DV), dtype=dtype)
    M = M0
    for c in range(NC):
        t0, t1 = c * C, min(T, (c + 1) * C)
        Ac, Bc = fold(t0, t1)
        Mpre = M                                              # state at t0 - 1
        M = Ac @ Mpre + Bc                                    # state at t1 - 1
        Scur = Mpre.transpose(0, 2, 1).copy()                 # [Hv, Dv, Dk]
        for t in range(t0, t1):                               # exact per-token replay
            Scur *= g[t][:, None, None]
            kv = np.einsum("hid,hd->hi", Scur, k[t][hk])
            delta = (v[t] - kv) * beta[t][:, None]
            Scur += delta[:, :, None] * k[t][hk][:, None, :]
            y[t] = np.einsum("hid,hd->hi", Scur, q[t][hk])
    return y, M.transpose(0, 2, 1)


def err(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(bf16(x).astype(np.float64) -
                               ref.astype(np.float64))))


def run_case(T, C, seed=0x5EED):
    inp = gen(seed, T)
    S0 = bf16((np.random.default_rng(seed + 1).random((HV, DV, DK)) - 0.5)
              .astype(np.float32))

    ref_y, ref_S = seq(np.float64, inp, S0)              # f64 ground truth
    seq_y, seq_S = seq(np.float32, inp, S0)              # shipped kernel (stock)
    cs_y, cs_S = chunk_scan(np.float32, inp, S0, C)      # naive fold
    wy_y, wy_S = chunk_scan(np.float32, inp, S0, C, wy=True)  # WY fold

    def row(name, ey, eS):
        print(f"  {name:18s} y_err={ey:9.2e}  state_err={eS:9.2e}")

    print(f"[T={T}, C={C}]")
    row("stock", err(seq_y, ref_y), err(seq_S, ref_S))
    row("chunk-scan", err(cs_y, ref_y), err(cs_S, ref_S))
    row("chunk-scan-wy", err(wy_y, ref_y), err(wy_S, ref_S))

    se_y, se_S = err(seq_y, ref_y), err(seq_S, ref_S)
    bar_y, bar_S = 1.5 * se_y + 0.02, 1.5 * se_S + 0.02
    ok = {n: (err(y, ref_y) <= bar_y) and (err(s, ref_S) <= bar_S)
          for n, y, s in (("chunk-scan", cs_y, cs_S),
                          ("chunk-scan-wy", wy_y, wy_S))}
    print(f"  repo bar y <= {bar_y:.2e}, state <= {bar_S:.2e}  ->  "
          f"chunk-scan {'PASS' if ok['chunk-scan'] else 'FAIL'}, "
          f"chunk-scan-wy {'PASS' if ok['chunk-scan-wy'] else 'FAIL'}\n")
    return se_y, se_S, bar_y, bar_S, ok


def check_wy_identity(C=8, Dk=8, Dv=6, seed=3):
    """Prove the WY identities used by chunk_scan_wy in f64."""
    rng = np.random.default_rng(seed)
    k = rng.random((C, Dk)) - 0.5
    v = rng.random((C, Dv)) - 0.5
    beta = rng.random((C,))
    g = 0.5 + 0.4 * rng.random((C,))
    I = np.eye(Dk)
    lg = np.cumsum(np.log(g))
    expg = np.exp(lg)
    G = expg[-1]
    L = np.tril(k @ k.T, -1) * beta[:, None] * \
        np.exp(lg[:, None] - lg[None, :])
    Afla = np.linalg.inv(np.eye(C) + L)
    w = Afla @ (beta[:, None] * expg[:, None] * k)
    u = Afla @ (beta[:, None] * v)
    scale = np.exp(lg[-1] - lg)
    Ac = G * I - (k * scale[:, None]).T @ w
    Bc = (k * scale[:, None]).T @ u

    Pdirect = np.eye(Dk)
    for t in range(C):
        Pdirect = g[t] * (I - beta[t] * np.outer(k[t], k[t])) @ Pdirect
    Bdirect = np.zeros((Dk, Dv))
    for t in range(C):
        prod = np.eye(Dk)
        for s in range(t + 1, C):
            prod = g[s] * (I - beta[s] * np.outer(k[s], k[s])) @ prod
        Bdirect += prod @ (beta[t] * np.outer(k[t], v[t]))
    print(f"[Q1 WY identity, f64] A_c err {np.max(np.abs(Ac - Pdirect)):.2e}, "
          f"B_c err {np.max(np.abs(Bc - Bdirect)):.2e} (expect ~1e-16)\n")


def budget(T, C):
    nch = T // C
    base = T * HV * (2 * DK * DV + DK * DV + DK)
    # WY per chunk: KK [C*Dk^2], solve C^3, w/u recompute 2*C^2*(Dk+Dv),
    # A_c/B_c compose C*Dk*(Dk+Dv)
    wy = nch * HV * (C * DK**2 + C**3 + 2 * C**2 * (DK + DV)
                     + C * DK * (DK + DV))
    naive = nch * HV * C * 2 * DK**3
    print(f"  per-layer FLOP budget (T={T}, C={C}):")
    print(f"    sequential recurrence : {base/1e9:7.2f} GFLOP, depth {T}")
    print(f"    + WY chunk products   : +{wy/1e9:7.2f} GFLOP ({wy/base:6.2f}x "
          f"the recurrence); depth {T} -> {nch} + {C}")
    print(f"    (naive Dk^3 fold would be +{naive/1e9:7.1f} GFLOP "
          f"= {naive/base:5.1f}x)  -> WY is the only viable form\n")


def main():
    print(f"GDN chunkwise reference  (Dk={DK}, Dv={DV}, Hk={HK}, Hv={HV}, "
          f"Hv/Hk={GROUP})\n")

    check_wy_identity()

    for T, C in ((1024, 128), (2048, 128), (2048, 256)):
        run_case(T, C)

    # f64 exactness (Q1): block form is the same recurrence.
    T, C = 512, 128
    inp = gen(0, T)
    S0 = bf16((np.random.default_rng(1).random((HV, DV, DK)) - 0.5)
              .astype(np.float32))
    r_y, r_S = seq(np.float64, inp, S0)
    for name, fn in (("chunk-scan", chunk_scan),
                     ("chunk-scan-wy", lambda d, i, s, c: chunk_scan(d, i, s, c, True))):
        c_y, c_S = fn(np.float64, inp, S0, C)
        d_y = float(np.max(np.abs(c_y - r_y)))
        d_S = float(np.max(np.abs(c_S - r_S)))
        print(f"[Q1 f64 exactness {name}] max |block - seq| = y {d_y:.3e}, "
              f"state {d_S:.3e} (expect ~1e-15)")

    print()
    budget(4096, 128)
    budget(8192, 256)
    print("  NOTE: the win comes from shortening the T-step dependency chain,")
    print("        not FLOPs. WY adds ~O(C*Dk^2) per chunk vs O(Dk*Dv) per token.")
    print("  On M5 the GDN recurrence is ~2800x above its FLOP floor at S=8192")
    print("  (block-profile 1115 ms vs ~0.4 ms FLOP-equivalent), i.e. latency-")
    print("  bound, so depth reduction is the right lever. M5 tok/s must still be")
    print("  measured on hardware; this file is a CPU correctness gate only.")


if __name__ == "__main__":
    main()
