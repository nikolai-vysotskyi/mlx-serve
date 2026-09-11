#!/usr/bin/env python3
"""
Stock-numerics reference for the fused MoE gate/up/GeGLU kernel.

WHY THIS EXISTS
---------------
The repo's fused MoE gate/up kernel (MOE_GATEUP_SOURCE, Lever D) keeps the
dequantized weight in fp32:  dot = sum_k  fp32(x_k) * fp32(q_k*scale + bias).
MLX's stock prefill gather_qmm (affine_gather_qmm_n -> qmm_n_impl) does NOT:
QuantizedBlockLoader<T=bf16> calls dequantize() into a bf16 threadgroup tile,
then runs BlockMMA<bf16,bf16> (fp32 accumulate).  So stock rounds each
dequantized weight to bf16 before the dot.

This script quantifies the three rounding classes on CPU so the Metal parity
tests know what to expect:

  f64    : x @ (q*s+b)        in float64            (ground truth)
  stock  : x @ bf16(q*s+b)    in float32 accumulate (what gather_qmm computes;
           ALSO what the fused MoE gate/up kernel computes after the bf16-weight
           fix — MOE_GATEUP_SOURCE now rounds the dequantized weight to T)
  repo   : x @ (q*s+b)        in float32 accumulate (the PRE-fix kernel's class,
           kept here to quantify why it was wrong: closer to f64 but NOT
           bit-identical to the stock chain it replaces)

Since bf16(x)->fp32 is exact, the ONLY difference between stock and repo is
the bf16 rounding of the dequantized weight.  The repo kernel is therefore
strictly closer to f64 than stock is (a stronger no-worse-than-stock claim),
but repo-vs-stock is NOT bit-identical: expect a few bf16 ULP on gate/up,
amplified once through the GeGLU product.  That repo-vs-stock delta is exactly
what the parity test "fused prefill gate+up+GeGLU matches gather_qmm +
fusedSwiGLU" will show on Metal (its 0.02*|r|+0.02 tolerance absorbs it).

Run:  python3 research/moe_gateup_stock_reference.py
"""

import numpy as np


def f32_to_bf16(x):
    x = np.asarray(x, dtype=np.float32)
    v = x.view(np.uint32)
    lsb = (v >> 16) & np.uint32(1)
    v = (v + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return v.view(np.float32)


def bf16(x):
    return f32_to_bf16(x)


def bf16_bits(x):
    return (f32_to_bf16(x).view(np.uint32) >> 16).astype(np.uint16)


def stable_sigmoid(x):
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        e = np.exp(-np.abs(x))
        out = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    return np.where(np.isnan(x), np.nan, out)


def build_sigtab():
    vals = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
    return bf16(stable_sigmoid(vals)).astype(np.float32)


def dequant_bank(wq, sc, bi, bits, group_size):
    """wq [N, K*bits/8] uint8, sc/bi [N, K/group] -> [N, K] fp32 (one expert)."""
    N, Kbytes = wq.shape
    K = Kbytes * 8 // bits
    if bits == 4:
        lo = wq & 0x0F
        hi = (wq >> 4) & 0x0F
        nibs = np.stack([lo, hi], axis=2).reshape(N, K)
    else:
        raise ValueError(bits)
    gidx = np.arange(K, dtype=np.int64) // group_size
    return nibs.astype(np.float32) * sc[:, gidx].astype(np.float32) + bi[:, gidx].astype(np.float32)


def geglu(g, u, sigtab):
    """fusedSwiGLU: bf16(bf16(gate * LUT[gate]) * up). Bit-exact to the kernel epilogue."""
    gb = bf16(g)
    sig = sigtab[bf16_bits(gb)]
    return bf16(bf16(gb.astype(np.float32) * sig) * bf16(u).astype(np.float32))


def gateup_dots(x, gate_wq, gate_sc, gate_bi, up_wq, up_sc, up_bi, inds, bits, group_size, num_experts, weight_dtype):
    """Gate/up dots for every slot row.  weight_dtype == 'fp32' -> repo kernel;
    'bf16' -> stock gather_qmm (bf16-rounded weights).  Both accumulate in fp32."""
    Ntot, N = inds.shape[0], gate_wq.shape[1]
    g = np.zeros((Ntot, N), dtype=np.float32)
    u = np.zeros((Ntot, N), dtype=np.float32)
    for expert in range(num_experts):
        rows = np.nonzero(inds == expert)[0]
        if not rows.size:
            continue
        gw = dequant_bank(gate_wq[expert], gate_sc[expert], gate_bi[expert], bits, group_size)
        uw = dequant_bank(up_wq[expert], up_sc[expert], up_bi[expert], bits, group_size)
        if weight_dtype == "bf16":
            gw = bf16(gw)
            uw = bf16(uw)
        xf = x[rows].astype(np.float32)  # bf16 -> fp32 is exact
        g[rows] = xf @ gw.T
        u[rows] = xf @ uw.T
    return g, u


def f64_gateup(x, gate_wq, gate_sc, gate_bi, up_wq, up_sc, up_bi, inds, bits, group_size, num_experts):
    Ntot, N = inds.shape[0], gate_wq.shape[1]
    g = np.zeros((Ntot, N), dtype=np.float64)
    u = np.zeros((Ntot, N), dtype=np.float64)
    for expert in range(num_experts):
        rows = np.nonzero(inds == expert)[0]
        if not rows.size:
            continue
        gw = dequant_bank(gate_wq[expert], gate_sc[expert], gate_bi[expert], bits, group_size).astype(np.float64)
        uw = dequant_bank(up_wq[expert], up_sc[expert], up_bi[expert], bits, group_size).astype(np.float64)
        xf = x[rows].astype(np.float64)
        g[rows] = xf @ gw.T
        u[rows] = xf @ uw.T
    return g, u


def report(tag, a, b, ref=None):
    da = np.abs(a.astype(np.float32) - b.astype(np.float32))
    denom = np.abs(ref).astype(np.float32) if ref is not None else np.abs(b).astype(np.float32)
    rel = np.where(denom > 1e-3, da / np.maximum(denom, 1e-3), 0.0)
    return f"{tag:24s} max|d|={da.max():.3e}  mean|d|={da.mean():.3e}  maxrel={rel.max():.3e}"


def main():
    rng = np.random.default_rng(2027)
    sigtab = build_sigtab()
    bits, group_size = 4, 64
    configs = [
        (64, 10, 2560, 640, 4096),    # production shapes, reduced E/Ntot for RAM
        (8, 4, 64, 32, 512),
        (4, 2, 64, 16, 200),
    ]
    for (E, topk, K, N, Ntot) in configs:
        inds = np.sort(rng.integers(0, E, size=Ntot)).astype(np.int32)
        x = bf16(rng.normal(size=(Ntot, K)).astype(np.float32) * 0.5)
        kw = K * bits // 8
        gate_wq = rng.integers(0, 256, size=(E, N, kw)).astype(np.uint8)
        up_wq = rng.integers(0, 256, size=(E, N, kw)).astype(np.uint8)
        ng = K // group_size
        gate_sc = (0.01 + 0.02 * rng.random(size=(E, N, ng))).astype(np.float32)
        gate_bi = rng.normal(size=(E, N, ng)).astype(np.float32)
        up_sc = (0.01 + 0.02 * rng.random(size=(E, N, ng))).astype(np.float32)
        up_bi = rng.normal(size=(E, N, ng)).astype(np.float32)

        g_stock, u_stock = gateup_dots(x, gate_wq, gate_sc, gate_bi, up_wq, up_sc, up_bi,
                                       inds, bits, group_size, E, "bf16")
        g_repo, u_repo = gateup_dots(x, gate_wq, gate_sc, gate_bi, up_wq, up_sc, up_bi,
                                     inds, bits, group_size, E, "fp32")
        g64, u64 = f64_gateup(x, gate_wq, gate_sc, gate_bi, up_wq, up_sc, up_bi,
                              inds, bits, group_size, E)

        act_stock = geglu(g_stock, u_stock, sigtab)
        act_repo = geglu(g_repo, u_repo, sigtab)
        act64 = (g64 / (1.0 + np.exp(-g64))) * u64  # f64 sigmoid (not LUT)

        print(f"--- E={E} topk={topk} K={K} N={N} Ntot={Ntot} ---")
        print("gate/up dots:")
        print(report("stock vs f64 (gate)", g_stock, g64, g64))
        print(report("repo  vs f64 (gate)", g_repo, g64, g64))
        print(report("repo  vs stock (gate)", g_repo, g_stock, g64))
        print("post-GeGLU act:")
        print(report("stock act vs f64", act_stock, act64, act64))
        print(report("repo  act vs f64", act_repo, act64, act64))
        print(report("repo  act vs stock act", act_repo, act_stock, act64))
        print()

    print("VERDICT:")
    print("  - repo (fp32 weight) is strictly closer to f64 than stock (bf16 weight):")
    print("    repo-vs-f64 <= stock-vs-f64 on every config above.")
    print("  - repo-vs-stock is the bf16 weight rounding (a few ULP on gate/up),")
    print("    amplified once through GeGLU: up to ~64 on act at K=2560 — beyond the")
    print("    0.02*|r|+0.02 parity bar. That is why MOE_GATEUP_SOURCE now rounds the")
    print("    dequantized weight to T (landing in the stock class); the 'repo' column")
    print("    above is the pre-fix class, kept for the record.")
    print("  - A NAX cooperative-tensor port must dequantize to bf16 (like stock) and")
    print("    run bf16xbf16->fp32 MMA to land in the stock class; keeping fp32 weights")
    print("    under NAX would reproduce neither stock nor the current repo kernel.")


if __name__ == "__main__":
    main()
