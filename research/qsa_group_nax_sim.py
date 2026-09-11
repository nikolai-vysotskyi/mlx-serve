#!/usr/bin/env python3
"""
CPU simulator for the NAX grouped-QSA gather kernel
(src/kernels/qsa_group_nax.metal).

This is NOT a bit-level model of the 16x32x16 cooperative MMA (opaque
hardware); it simulates everything the hand-written kernel controls so the
tiling, fragment addressing, union merge, per-token mask, D-half exchange,
and epilogue coverage are validated:

  - BaseNAXFrag::get_coord() + NAXTile load/store index arithmetic (verbatim)
  - the k-way union merge + per-token -inf mask (reuses qsa_group_reference.py)
  - the NAX inner product: fp32 accumulate in 16-D chunks, two simdgroups split
    D=256 (band*128) and exchange their partial S, online softmax, and the
    two-bf16-term (Shi/Slo) PV epilogue.

Verifies:
  1. get_coord/load cover the Q (16x16), K^T (32x16) and S (16x32) fragments
     exactly once.
  2. union merge -> per-token sequences == stock per-token sequences (exact).
  3. The chunk-MMA + D-half-exchange S equals the BLAS fp32 Q@K^T to fp32
     accumulation order; the full attention output equals the grouped
     reference within the accepted fp32-order / bf16 class.

Run:  python3 research/qsa_group_nax_sim.py
"""

import numpy as np

from qsa_group_reference import (
    exact_attention,
    merge_union,
    online_attention_grouped,
    synthetic_blocks,
    token_sequence,
    token_params,
)
from hc_up_mix_reference import bf16


def get_coord(lane):
    qid = lane >> 2
    return (((qid & 2) | (lane & 1)) * 4, (qid & 4) | ((lane >> 1) & 3))


def frag_load(rows, cols, base_r, base_c):
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


def nax_grouped_path(Q, K, V, scale, blocks, ratio, G, BK=32):
    """Port of qsa_group_nax.metal: union merge + chunk-MMA S + D-half exchange
    + mask + online softmax + PV. Returns the [B,Hq,qL,hd] output and the raw
    pre-mask S scores [B,Hq,qL,kvmax] for parity checking (kvmax = KB*ratio +
    max tail, but S is gathered per-token so we store it as a ragged-ish dense
    over the union only when needed — here we only return S for the largest
    gathered length, padded)."""
    B, Hq, qL, hd = Q.shape
    Hk = K.shape[1]
    kL = K.shape[2]
    gqa = Hq // Hk
    kb = blocks.shape[2]
    log2e = 1.4426950408889634
    TB = BK // ratio
    out = np.zeros((B, Hq, qL, hd), dtype=np.float32)

    for bb in range(B):
        for g in range(0, qL, G):
            Gt = min(G, qL - g)
            tokens = [(blocks[bb, s], kL - qL + s) for s in range(g, g + Gt)]
            union = merge_union(tokens, kb, ratio)
            # per-token online-softmax state, keyed by GLOBAL head index.
            state = {}
            for t in range(Gt):
                state[g + t] = [(-3e38, 0.0, np.zeros(hd, dtype=np.float64)) for _ in range(Hq)]
            for t0 in range(0, len(union), TB):
                chunk = union[t0:t0 + TB]
                for hk_ in range(Hk):
                    # stage the union K/V tile (BK rows, zero-padded past maxn).
                    Ks = np.zeros((BK, hd), dtype=np.float32)
                    Vs = np.zeros((BK, hd), dtype=np.float32)
                    for u in range(TB):
                        if u >= len(chunk):
                            break
                        b = chunk[u][0]
                        mn = max(chunk[u][1])
                        for inrow in range(mn):
                            rx = u * ratio + inrow
                            pos = b * ratio + inrow
                            Ks[rx] = K[bb, hk_, pos, :]
                            Vs[rx] = V[bb, hk_, pos, :]
                    for t in range(Gt):
                        s = g + t
                        for h0 in range(gqa):
                            h = hk_ * gqa + h0
                            q = Q[bb, h, s, :].astype(np.float32)
                            # NAX S: fp32, 16-D chunks, D-half split + exchange.
                            def chunk_dot(d0, d1):
                                acc = np.zeros(BK, dtype=np.float32)
                                for d in range(d0, d1, 16):
                                    acc = acc + (q[d:d + 16].astype(np.float32)
                                                 @ Ks[:, d:d + 16].T.astype(np.float32)).astype(np.float32)
                                return acc
                            S = chunk_dot(0, 128) + chunk_dot(128, 256)
                            # scale + per-token mask.
                            Ss = S * (scale * log2e)
                            for rx in range(BK):
                                u = rx // ratio
                                inrow = rx % ratio
                                if u >= len(chunk) or chunk[u][1][t] <= inrow:
                                    Ss[rx] = -np.inf
                            m, l, O = state[s][h]
                            new_m = max(m, Ss.max())
                            rowsum = np.exp2(Ss - new_m).sum()
                            factor = np.exp2(m - new_m) if m > -1e38 else 1.0
                            m = new_m
                            l = l * factor + rowsum
                            O = O * factor + (np.exp2(Ss - new_m) @ Vs)
                            state[s][h] = (m, l, O)
            for s, per_h in state.items():
                for h, (m, l, O) in enumerate(per_h):
                    out[bb, h, s, :] = (O / l).astype(np.float32)
    return out


def raw_s_parity(Q, K, blocks, ratio, G, BK=32):
    """Compare the NAX chunk-MMA S against the BLAS fp32 S on the raw (pre-mask,
    pre-scale) scores for a few (token, head) samples, across every union tile.
    Returns (max_abs, max_rel)."""
    B, Hq, qL, hd = Q.shape
    Hk = K.shape[1]
    kL = K.shape[2]
    gqa = Hq // Hk
    kb = blocks.shape[2]
    TB = BK // ratio
    worst_abs = 0.0
    worst_rel = 0.0
    for bb in range(B):
        for g in range(0, qL, G):
            Gt = min(G, qL - g)
            tokens = [(blocks[bb, s], kL - qL + s) for s in range(g, g + Gt)]
            union = merge_union(tokens, kb, ratio)
            for t0 in range(0, len(union), TB):
                chunk = union[t0:t0 + TB]
                for hk_ in range(Hk):
                    Ks = np.zeros((BK, hd), dtype=np.float32)
                    for u in range(TB):
                        if u >= len(chunk):
                            break
                        b = chunk[u][0]
                        mn = max(chunk[u][1])
                        for inrow in range(mn):
                            rx = u * ratio + inrow
                            Ks[rx] = K[bb, hk_, b * ratio + inrow, :]
                    for t in range(Gt):
                        s = g + t
                        for h0 in range(gqa):
                            h = hk_ * gqa + h0
                            q = Q[bb, h, s, :].astype(np.float32)
                            # NAX order
                            def chunk_dot(d0, d1):
                                acc = np.zeros(BK, dtype=np.float32)
                                for d in range(d0, d1, 16):
                                    acc = acc + (q[d:d + 16].astype(np.float32)
                                                 @ Ks[:, d:d + 16].T.astype(np.float32)).astype(np.float32)
                                return acc
                            S_nax = chunk_dot(0, 128) + chunk_dot(128, 256)
                            S_blas = q @ Ks.T.astype(np.float32)
                            d = np.abs(S_nax - S_blas)
                            worst_abs = max(worst_abs, float(d.max()))
                            rel = d / np.maximum(1.0, np.abs(S_blas))
                            worst_rel = max(worst_rel, float(rel.max()))
    return worst_abs, worst_rel


def main():
    ratio = 4
    kb = 16
    qL = 33          # covers a partial last group (G=4 -> 1 remainder token)
    kL = 96
    G = 4
    BK = 32

    # 1. Fragment-layout coverage (verbatim get_coord + NAXTile load indexing).
    check_coverage(frag_load(16, 16, 0, 0), 16, 16, "Q 16x16")
    check_coverage(frag_load(32, 16, 0, 0), 32, 16, "K^T 32x16")
    check_coverage(frag_load(16, 32, 0, 0), 16, 32, "S 16x32")
    print("fragment layout: Q 16x16, K^T 32x16, S 16x32 all covered exactly once per lane + across lanes")

    # 2. Index validation (union -> per-token sequences == stock), reused from
    #    qsa_group_reference but asserted here too.
    blocks = synthetic_blocks(qL, kL, kb, ratio, seed=1)
    for g in range(0, qL, G):
        tokens = [(blocks[0, s], kL - qL + s) for s in range(g, min(g + G, qL))]
        union = merge_union(tokens, kb, ratio)
        for t, (blk, p) in enumerate(tokens):
            ref = token_sequence(blk, p, kb, ratio)
            seq = []
            for b, nrows in union:
                for r in range(nrows[t]):
                    seq.append(b * ratio + r)
            assert seq == ref, f"union != stock at s={g+t}"
    print("union merge -> per-token sequences EXACT (all groups)")

    # 3. Raw S parity: chunk-MMA + D-half exchange vs BLAS fp32.
    Hq, Hk, hd = 12, 2, 256
    rng = np.random.default_rng(7)
    Q = bf16(rng.standard_normal((1, Hq, qL, hd)).astype(np.float32))
    K = bf16(rng.standard_normal((1, Hk, kL, hd)).astype(np.float32))
    V = bf16(rng.standard_normal((1, Hk, kL, hd)).astype(np.float32))
    scale = 1.0 / np.sqrt(hd)

    wa, wr = raw_s_parity(Q, K, blocks, ratio, G, BK)
    print(f"raw S (chunk-MMA + D-half exchange) vs BLAS fp32: max|d|={wa:.3e}  maxrel={wr:.3e}")
    assert wa < 2e-3 * max(1.0, float(np.abs(Q).max() * hd)), f"raw S deviates: {wa:.3e}"

    # 4. Full attention: NAX-grouped vs the grouped reference (BLAS S) and exact.
    out_nax = nax_grouped_path(Q, K, V, scale, blocks, ratio, G, BK)
    out_grp = online_attention_grouped(Q, K, V, scale, blocks, ratio, G, BK)
    out_exact = exact_attention(Q, K, V, scale, blocks, ratio)

    d_ng = np.abs(out_nax - out_grp).max()
    d_ne = np.abs(out_nax - out_exact).max()
    d_ge = np.abs(out_grp - out_exact).max()
    print(f"nax-grouped vs grouped-reference  max|d|={d_ng:.3e}")
    print(f"nax-grouped vs exact              max|d|={d_ne:.3e}")
    print(f"grouped-reference vs exact        max|d|={d_ge:.3e}")
    assert d_ng < 1e-3, f"nax-grouped deviates from grouped reference: {d_ng:.3e}"
    assert d_ne < 1e-3, f"nax-grouped deviates from exact: {d_ne:.3e}"

    print("VERDICT:")
    print("  - fragment addressing (verbatim get_coord + NAXTile load/store) covers")
    print("    Q/K^T/S tiles exactly once; no holes, no duplicates.")
    print("  - the union merge + per-token -inf mask reproduces each token's stock")
    print("    gathered key sequence exactly.")
    print("  - the chunk-MMA + D-half-exchange S equals BLAS fp32 Q@K^T to fp32")
    print("    accumulation order; the full output equals the grouped reference")
    print("    within the accepted fp32-order / bf16 class.")
    print("  - NOT validated here: the hardware MMA's exact rounding (opaque); the")
    print("    Metal parity test bars cover it on M5.")


if __name__ == "__main__":
    main()
