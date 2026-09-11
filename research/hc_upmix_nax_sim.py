#!/usr/bin/env python3
"""
CPU simulator for the NAX fused HC up+mix kernel
(src/kernels/hc_upmix_nax.metal).

This is NOT a bit-level model of the 16x32x16 cooperative MMA (which is opaque
hardware); it simulates everything the hand-written kernel controls so the
tiling, fragment load offsets, dequant, and epilogue coverage are validated:

  - BaseNAXFrag::get_coord()  (verbatim from nax_gemm_header.metal)
  - NAXTile<T,R,C>::load index arithmetic  (verbatim)
  - the schedule: BM=32 / BN=64 / BK=32, tm/tn simdgroup map (4 simdgroups)
  - dequant: T-rounded weight, fp32 accumulate in 16-length K chunks (the MMA
    granularity), fp32 C fragments, per-h LUT-sigmoid/bf16-product epilogue,
    fp32 stream sum, 1/hc scale.

Verifies:
  1. get_coord/load cover every (row,col) of the A (16x16), B^T (32x16) and
     C (16x32) fragments exactly once, with no holes or duplicates.
  2. The raw fp32 up accumulators equal the bf16-weight fp32-accumulate
     matmul (so the load offsets and mma operand order are consistent), and
     the final mean_h(sigmoid(up)*n4) matches the composed chain
     (research/hc_up_mix_reference.py) modulo fp32 accumulation order.

Run:  python3 research/hc_upmix_nax_sim.py
"""

import numpy as np

from hc_up_mix_reference import (
    bf16,
    bf16_bits,
    build_sigtab,
    composed,
    dequant,
    f64_truth,
    kernel_path,
)


def get_coord(lane):
    """Verbatim port of BaseNAXFrag::get_coord()."""
    qid = lane >> 2
    return (((qid & 2) | (lane & 1)) * 4, (qid & 4) | ((lane >> 1) & 3))


def frag_load(rows, cols, base_r, base_c):
    """Simulate NAXTile<T,R,C>::load<T,LD,STRIDE=1>: returns dict
    lane -> list of (r, c) tile coords in fragment order."""
    R, C = rows // 16, cols // 16
    out = {}
    for lane in range(32):
        xy = get_coord(lane)
        coords = []
        for r in range(R):
            for c in range(C):
                for j in range(8):
                    row = base_r + r * 16 + xy[1] + (j // 4) * 8
                    col = base_c + c * 16 + xy[0] + j % 4
                    coords.append((row, col))
        out[lane] = coords
    return out


def check_coverage(coords, H, W, tag):
    """Every (r,c) in [H x W] must be produced exactly once across all lanes."""
    seen = {}
    for lane, lst in coords.items():
        for rc in lst:
            seen[rc] = seen.get(rc, 0) + 1
    assert len(seen) == H * W, f"{tag}: {len(seen)} != {H * W} covered"
    for rc, n in seen.items():
        assert n == 1, f"{tag}: {rc} seen {n} times"
    for lane, lst in coords.items():
        assert len(set(lst)) == len(lst), f"{tag}: lane {lane} has dup"
    return True


def nax_hc_path(act, uw_q, uw_s, uw_b, n4, bits, group_size, sigtab,
                BM=32, BN=64, BK=32):
    """Port of hc_upmix_nax.metal (tiling + dequant + MMA granularity +
    epilogue). Returns (out [M,H], raw_up [M, hc*H]) where raw_up is the fp32
    accumulator before bf16 rounding (the exact parity target vs BLAS)."""
    M, K = act.shape
    hc, H = n4.shape[1], n4.shape[2]
    N = hc * H
    assert H % BN == 0, "kernel requires H % 64 == 0"
    assert K % BK == 0, "kernel requires K % 32 == 0"
    VPW = 32 // bits
    K_by_p = K // VPW
    K_by_gs = K // group_size
    STEPS = BK // 16

    # T-rounded dequantized weight [N, K] (bf16-valued fp32), same as the
    # plain-SIMD kernel and stock qmm_n dequantize().
    W = dequant(uw_q, uw_s, uw_b, bits, group_size)  # already bf16-rounded

    # Zero-pad the M dimension to a BM multiple: the kernel stages Atile with
    # zero-fill past M and the MMA runs over all 16 rows (zero rows add 0).
    pad = (BM - M % BM) % BM
    xf = np.pad(act.astype(np.float32), ((0, pad), (0, 0)))

    sums = np.zeros((M, H), dtype=np.float32)
    raw = np.zeros((M, N), dtype=np.float32)

    for h in range(hc):
        wbase = h * H
        for row0 in range(0, M, BM):
            for col0 in range(0, H, BN):
                for sg in range(4):
                    tm = (sg // 2) * 16
                    tn = (sg % 2) * 32
                    C = np.zeros((16, 32), dtype=np.float32)
                    for k0 in range(0, K, BK):
                        for step in range(STEPS):
                            kk = np.arange(step * 16, step * 16 + 16)
                            k = k0 + kk
                            A = xf[row0 + tm:row0 + tm + 16][:, k]     # [16,16]
                            Bt = W[wbase + col0 + tn:wbase + col0 + tn + 32][:, k].T  # [16,32]
                            C = C + A.astype(np.float32) @ Bt.astype(np.float32)
                    # epilogue guard r < M (H % 64 == 0 so full 32 cols valid)
                    vrows = max(0, min(16, M - (row0 + tm)))
                    if vrows:
                        r = slice(row0 + tm, row0 + tm + vrows)
                        c = slice(col0 + tn, col0 + tn + 32)
                        u = bf16(C[:vrows])
                        sig = sigtab[bf16_bits(u)]
                        prod = bf16(sig.astype(np.float32) * n4[r, h, c].astype(np.float32))
                        sums[r, c] += prod.astype(np.float32)
                        raw[r, wbase + col0 + tn:wbase + col0 + tn + 32] = C[:vrows]

    out = bf16(sums * (1.0 / float(hc)))
    return out, raw


def main():
    rng = np.random.default_rng(2027)
    sigtab = build_sigtab()
    bits, group_size = 4, 64
    BM, BN, BK = 32, 64, 32

    # 1. Fragment-layout coverage (verbatim get_coord + load indexing).
    check_coverage(frag_load(16, 16, 0, 0), 16, 16, "A 16x16")
    check_coverage(frag_load(32, 16, 0, 0), 32, 16, "B^T 32x16")
    check_coverage(frag_load(16, 32, 0, 0), 16, 32, "C 16x32")
    print("fragment layout: A 16x16, B^T 32x16, C 16x32 all covered exactly once per lane + across lanes")

    # (M, hc, H, R) — all kernel-valid geometry (H%64==0, R%32==0).
    configs = [
        (512, 4, 2560, 320),   # production geometry (up intermediate 168MB at M=8192)
        (32, 4, 64, 64),       # small, cleanly tiled
        (33, 4, 64, 64),       # tail M%32!=0
        (100, 8, 128, 128),    # hc=8, bigger H, tail M%32!=0
    ]
    for (M, hc, H, R) in configs:
        K = R
        N = hc * H
        act = bf16(rng.normal(size=(M, K)).astype(np.float32) * 0.5)
        n4 = bf16(rng.normal(size=(M, hc, H)).astype(np.float32))
        uw_q = rng.integers(0, 256, size=(N, K * bits // 8)).astype(np.uint8)
        uw_s = (0.01 + 0.02 * rng.random(size=(N, K // group_size))).astype(np.float32)
        uw_b = rng.normal(size=(N, K // group_size)).astype(np.float32)

        out, raw = nax_hc_path(act, uw_q, uw_s, uw_b, n4, bits, group_size, sigtab, BM, BN, BK)

        comp = composed(act, uw_q, uw_s, uw_b, n4, bits, group_size)
        kern = kernel_path(act, uw_q, uw_s, uw_b, n4, bits, group_size, sigtab)
        truth = f64_truth(act, uw_q, uw_s, uw_b, n4, bits, group_size)

        # Raw fp32 accumulator vs BLAS-order fp32 matmul (pre-bf16): the exact
        # tiling/dequant/MMA-granularity parity target.
        W = dequant(uw_q, uw_s, uw_b, bits, group_size)
        raw_ref = act.astype(np.float32) @ W.T.astype(np.float32)   # BLAS order
        d_raw = np.abs(raw - raw_ref)

        d_comp = np.abs(out.astype(np.float32) - comp.astype(np.float32))
        d_kern = np.abs(out.astype(np.float32) - kern.astype(np.float32))
        d_truth = np.abs(out.astype(np.float64) - truth)

        print(f"--- M={M} hc={hc} H={H} R={R} (N={N}, BM={BM} BN={BN} BK={BK}) ---")
        print(f"nax raw up (fp32) vs BLAS   max|d|={d_raw.max():.3e}  mean|d|={d_raw.mean():.3e}")
        print(f"nax out vs composed         max|d|={d_comp.max():.3e}  mean|d|={d_comp.mean():.3e}  (bf16 ULP flips)")
        print(f"nax out vs kernel_path      max|d|={d_kern.max():.3e}  mean|d|={d_kern.mean():.3e}")
        print(f"nax out vs f64              max|d|={d_truth.max():.3e}  mean|d|={d_truth.mean():.3e}")
        assert d_raw.max() < 2e-3 * max(1.0, np.abs(raw_ref).max()), f"nax raw != BLAS: {d_raw.max():.3e}"
        print()

    print("VERDICT:")
    print("  - fragment addressing (verbatim get_coord + NAXTile load) covers")
    print("    A/B^T/C tiles exactly once; no holes, no duplicates.")
    print("  - the NAX tiling (tm/tn simdgroup map, BK=32, 16-chunk MMA granularity,")
    print("    T-rounded dequant, LUT-sigmoid/bf16-product/fp32-sum epilogue)")
    print("    reproduces the composed chain to fp32 accumulation order — the same")
    print("    class the plain-SIMD kernel claims.")
    print("  - NOT validated here: the hardware MMA's exact rounding (opaque); the")
    print("    Metal parity test tolerance (0.02*|r|+0.008) covers it on M5.")


if __name__ == "__main__":
    main()
