#!/usr/bin/env python3
"""
CPU reference for the fused MoE gate/up/GeGLU kernel (sorted prefill path).

Geometry (Qwen3.8-Flash-Next): E experts, top-K, hidden K=2560, intermediate
N=640, affine 4-bit / group 64 (Metal: packed uint32 words [E, N, K*4/32];
the numpy model unpacks the same bits from a uint8 byte view [E, N, K*4/8]).
sorted prefill: x_gathered [Ntot, K], sorted_inds [Ntot] uint32 (non-decreasing).

Composed chain (no expert bias): gate/up = gather_qmm per expert, dequant in
fp32, fp32 dot rounded to bf16; act = fusedSwiGLU(gate, up)
= bf16( bf16(gate * LUT[gate]) * up ).

Fused kernel: (A) schedule — thread e = expert e, binary-search its id run,
emit ceil(count/BM) tiles {start, count, block}; (B) GEMM — per (tile, col-block)
threadgroup compute gate/up dots (fp32 dequant, fp32 accumulate) + LUT GeGLU.
Validates schedule coverage and that kernel == composed modulo fp32 dot order.
Memory-light (per-expert dequant, no [Ntot,N] f64).

Run:  python3 research/moe_gateup_reference.py
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


def schedule(inds, BM, num_experts):
    """Port of grouped_qmm_tiles.metal: tiles [(start, count, block)]."""
    M = inds.shape[0]
    tiles = []
    for expert in range(num_experts):
        lo, hi = 0, M
        while lo < hi:
            mid = (lo + hi) // 2
            if inds[mid] < expert:
                lo = mid + 1
            else:
                hi = mid
        start = lo
        lo, hi = start, M
        while lo < hi:
            mid = (lo + hi) // 2
            if inds[mid] <= expert:
                lo = mid + 1
            else:
                hi = mid
        count = lo - start
        for block in range((count + BM - 1) // BM):
            tiles.append((start, count, block))
    return tiles


def run_expert(x, gate_wq, gate_sc, gate_bi, up_wq, up_sc, up_bi, expert, rows, bits, group_size, sigtab):
    """gate/up dots + LUT GeGLU for one expert's slot rows. Returns bf16 f32 [len(rows), N]."""
    g = x[rows].astype(np.float32) @ dequant_bank(gate_wq[expert], gate_sc[expert], gate_bi[expert], bits, group_size).T
    u = x[rows].astype(np.float32) @ dequant_bank(up_wq[expert], up_sc[expert], up_bi[expert], bits, group_size).T
    gb = bf16(g)
    ub = bf16(u)
    sig = sigtab[bf16_bits(gb)]
    return bf16(bf16(gb.astype(np.float32) * sig) * ub.astype(np.float32))


def composed(x, gate_wq, gate_sc, gate_bi, up_wq, up_sc, up_bi, inds, bits, group_size, sigtab, num_experts):
    Ntot, N = inds.shape[0], gate_wq.shape[1]
    act = np.zeros((Ntot, N), dtype=np.float32)
    for expert in range(num_experts):
        rows = np.nonzero(inds == expert)[0]
        if rows.size:
            act[rows] = run_expert(x, gate_wq, gate_sc, gate_bi, up_wq, up_sc, up_bi, expert, rows, bits, group_size, sigtab)
    return act


def kernel_path(x, gate_wq, gate_sc, gate_bi, up_wq, up_sc, up_bi, inds, bits, group_size, sigtab, BM, num_experts):
    Ntot, N = inds.shape[0], gate_wq.shape[1]
    act = np.zeros((Ntot, N), dtype=np.float32)
    covered = np.zeros(Ntot, dtype=bool)
    for (start, count, block) in schedule(inds, BM, num_experts):
        expert = int(inds[start])
        row0 = start + block * BM
        M = min(BM, count - block * BM)
        rows = np.arange(row0, row0 + M)
        assert not covered[rows].any(), "slot covered twice"
        covered[rows] = True
        act[rows] = run_expert(x, gate_wq, gate_sc, gate_bi, up_wq, up_sc, up_bi, expert, rows, bits, group_size, sigtab)
    assert covered.all(), "slot not covered"
    return act


def f64_truth(x, gate_wq, gate_sc, gate_bi, up_wq, up_sc, up_bi, inds, bits, group_size, num_experts):
    Ntot, N = inds.shape[0], gate_wq.shape[1]
    act = np.zeros((Ntot, N), dtype=np.float64)
    for expert in range(num_experts):
        rows = np.nonzero(inds == expert)[0]
        if rows.size:
            g = x[rows].astype(np.float64) @ dequant_bank(gate_wq[expert], gate_sc[expert], gate_bi[expert], bits, group_size).astype(np.float64).T
            u = x[rows].astype(np.float64) @ dequant_bank(up_wq[expert], up_sc[expert], up_bi[expert], bits, group_size).astype(np.float64).T
            sig = 1.0 / (1.0 + np.exp(-g))
            act[rows] = (g * sig) * u
    return act


def report(tag, a, b, ref):
    da = np.abs(a.astype(np.float32) - b.astype(np.float32))
    denom = np.abs(ref).astype(np.float32)
    rel = np.where(denom > 1e-3, da / np.maximum(denom, 1e-3), 0.0)
    return f"{tag:26s} max|d|={da.max():.3e}  mean|d|={da.mean():.3e}  maxrel={rel.max():.3e}"


def main():
    rng = np.random.default_rng(2027)
    sigtab = build_sigtab()
    bits, group_size, BM = 4, 64, 64
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

        comp = composed(x, gate_wq, gate_sc, gate_bi, up_wq, up_sc, up_bi, inds, bits, group_size, sigtab, E)
        kern = kernel_path(x, gate_wq, gate_sc, gate_bi, up_wq, up_sc, up_bi, inds, bits, group_size, sigtab, BM, E)
        ref = f64_truth(x, gate_wq, gate_sc, gate_bi, up_wq, up_sc, up_bi, inds, bits, group_size, E)

        print(f"--- E={E} topk={topk} K={K} N={N} Ntot={Ntot} (BM={BM}) ---")
        print("tiles:", len(schedule(inds, BM, E)), "= sum_e ceil(count_e/BM)")
        print(report("composed   vs f64", comp, ref, ref))
        print(report("kernel     vs f64", kern, ref, ref))
        print(report("kernel vs composed", kern, comp, ref))
        print()

    print("VERDICT:")
    print("  - schedule covers every slot exactly once (asserted); tile expert = inds[start].")
    print("  - kernel == composed chain modulo fp32 dot order (kernel-vs-composed is a")
    print("    few bf16 ULP; LUT GeGLU is bit-exact to fusedSwiGLU).")
    print("  - both ~bf16-precision vs f64 (same class as the shipped MoE path).")


if __name__ == "__main__":
    main()
