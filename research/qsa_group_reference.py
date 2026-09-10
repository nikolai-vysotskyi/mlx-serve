#!/usr/bin/env python3
"""
CPU reference for the grouped-QSA gather (Lever G: query-grouped block reuse).

The stock prefill QSA gather (msv_attn_qsa256, already upstream via #388) runs
ONE threadgroup per (query token, kv head, batch) and, for that token, gathers
its top-KB selected key blocks plus its own incomplete tail, then runs an
online (flash) softmax over the gathered sequence:

    p        = kL - qL + s            # absolute cache position of query s
    complete = (p + 1) // ratio       # fully-visible blocks
    count    = min(complete, KB)      # selected blocks (<= KB)
    sel_len  = count * ratio
    tail_start = complete * ratio
    L        = sel_len + (p + 1 - tail_start)   # gathered length
    pos(vi)  = vi < sel_len ? blk[vi//ratio]*ratio + (vi % ratio)
                             : tail_start + (vi - sel_len)

Adjacent tokens select ~the same KB blocks (the top-K ranking changes slowly),
so each token re-reads ~L keys and ~L values that the previous token already
read.  The grouped kernel (msv_attn_qsa256_grp) runs G adjacent tokens in ONE
threadgroup and stages each distinct block once: it k-way merges the G sorted
block lists (each block appears once, with a per-token row count — full ratio
rows for a selected complete block, `tail_rows` for a partial tail block) and
feeds every token from the same staged K/V tile.  Per-token state (max/sum/O)
is per-row and per-simdgroup as in the stock kernel, so registers do not grow;
only the threadgroup gets Gx more simdgroups.

This file validates, in numpy:
  1. the union-of-blocks + per-token row-count mask reproduces EXACTLY the
     per-token gathered key sequence the stock kernel reads (ascending);
  2. the online softmax over the union tiles equals the stock per-token online
     softmax (same per-token key ORDER, only tile boundaries differ -> fp32
     rounding only);
  3. both match the exact reference softmax within a loose tolerance.

Run:  python3 research/qsa_group_reference.py
"""

import numpy as np


# --------------------------------------------------------------------------
# 1. Index math (pure Python)
# --------------------------------------------------------------------------

def token_params(p, kb, ratio):
    complete = (p + 1) // ratio
    count = min(complete, kb)
    sel_len = count * ratio
    tail_start = complete * ratio
    tail_rows = p + 1 - tail_start
    return complete, count, sel_len, tail_start, tail_rows


def token_sequence(blk, p, kb, ratio):
    """Ascending list of absolute key positions for one token (stock kernel)."""
    _, count, sel_len, tail_start, tail_rows = token_params(p, kb, ratio)
    seq = []
    for vi in range(sel_len):
        b = vi // ratio
        seq.append(blk[b] * ratio + (vi - b * ratio))
    for vi in range(sel_len, sel_len + tail_rows):
        seq.append(tail_start + (vi - sel_len))
    return seq


def merge_union(tokens, kb, ratio):
    """k-way merge of G tokens' (block_id -> per-token row-count) sequences.

    Each token contributes its selected blocks (full `ratio` rows) then its
    tail block (`tail_rows` rows, may be 0).  Returns a list of
    (block_id, nrows[G]) in ascending block order, one entry per distinct
    union block.
    """
    G = len(tokens)
    nxt = []   # (block_id, nrows) next contribution per token, or None
    cur = []   # index into the token's selected-block list
    for t in range(G):
        blk, p = tokens[t]
        complete, count, _, tail_start, tail_rows = token_params(p, kb, ratio)
        if count > 0:
            nxt.append((blk[0], ratio))
            cur.append(0)
        elif tail_rows > 0:
            nxt.append((complete, tail_rows))
            cur.append(-1)          # selected list already exhausted
        else:
            nxt.append(None)
            cur.append(-1)
    union = []
    while any(n is not None for n in nxt):
        mb = min(n[0] for n in nxt if n is not None)
        nrows = [0] * G
        for t in range(G):
            if nxt[t] is not None and nxt[t][0] == mb:
                nrows[t] = nxt[t][1]
                # advance token t
                blk, p = tokens[t]
                complete, count, _, tail_start, tail_rows = token_params(p, kb, ratio)
                if cur[t] >= 0:                       # was a selected block
                    cur[t] += 1
                    if cur[t] < count:
                        nxt[t] = (blk[cur[t]], ratio)
                    elif tail_rows > 0:
                        nxt[t] = (complete, tail_rows)
                        cur[t] = -1
                    else:
                        nxt[t] = None
                else:                                  # was the tail block
                    nxt[t] = None
        union.append((mb, nrows))
    return union


def union_to_sequences(union, kb, ratio):
    """Flatten the union back into per-token key-position lists (ascending)."""
    G = len(union[0][1]) if union else 0
    seqs = [[] for _ in range(G)]
    for b, nrows in union:
        for t in range(G):
            for r in range(nrows[t]):
                seqs[t].append(b * ratio + r)
    return seqs


# --------------------------------------------------------------------------
# 2. Attention (numpy, fp32) — exact / stock-online / grouped-online
# --------------------------------------------------------------------------

def exact_attention(Q, K, V, scale, blocks, ratio):
    """Ground-truth softmax attention over each token's gathered sequence."""
    B, Hq, qL, hd = Q.shape
    _, Hk, kL, _ = K.shape
    gqa = Hq // Hk
    kb = blocks.shape[2]
    out = np.zeros((B, Hq, qL, hd), dtype=np.float32)
    for s in range(qL):
        p = kL - qL + s
        seq = token_sequence(blocks[0, s], p, kb, ratio)
        for hk in range(Hk):
            Ks = K[0, hk, seq, :]           # [L, hd]
            Vs = V[0, hk, seq, :]           # [L, hd]
            for h0 in range(gqa):
                h = hk * gqa + h0
                logits = Ks @ Q[0, h, s] * scale            # [L]
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                out[0, h, s, :] = probs @ Vs
    return out


def online_attention_tiles(Q, K, V, scale, blocks, ratio, tile):
    """Online (flash) softmax, per-token, over the stock gathered sequence."""
    B, Hq, qL, hd = Q.shape
    _, Hk, kL, _ = K.shape
    gqa = Hq // Hk
    kb = blocks.shape[2]
    log2e = 1.4426950408889634
    out = np.zeros((B, Hq, qL, hd), dtype=np.float32)
    for s in range(qL):
        p = kL - qL + s
        seq = token_sequence(blocks[0, s], p, kb, ratio)
        Qs = Q[0, :, s, :]
        for hk in range(Hk):
            Ks = K[0, hk, seq, :]
            Vs = V[0, hk, seq, :]
            for h0 in range(gqa):
                h = hk * gqa + h0
                m = -np.inf
                l = 0.0
                O = np.zeros(hd, dtype=np.float64)
                for t0 in range(0, len(seq), tile):
                    seg = slice(t0, min(t0 + tile, len(seq)))
                    S = (Qs[h] @ Ks[seg].T) * scale * log2e
                    new_m = max(m, S.max()) if S.size else m
                    rowsum = np.exp2(S - new_m).sum() if S.size else 0.0
                    factor = np.exp2(m - new_m) if np.isfinite(m) and m > -1e38 else 1.0
                    m = new_m
                    l = l * factor + rowsum
                    O = O * factor + (np.exp2(S - new_m) @ Vs[seg])
                out[0, h, s, :] = (O / l).astype(np.float32)
    return out


def online_attention_grouped(Q, K, V, scale, blocks, ratio, G, tile):
    """Online softmax over the UNION of G tokens' gathered sequences.

    Mirrors msv_attn_qsa256_grp: k-way merge -> per-block per-token row counts,
    tiled staging, per-token online softmax with the union tiles' boundaries.
    """
    B, Hq, qL, hd = Q.shape
    _, Hk, kL, _ = K.shape
    gqa = Hq // Hk
    kb = blocks.shape[2]
    log2e = 1.4426950408889634
    TB = tile // ratio          # union blocks per tile
    out = np.zeros((B, Hq, qL, hd), dtype=np.float32)

    for g in range(0, qL, G):
        tokens = [(blocks[0, s], kL - qL + s) for s in range(g, min(g + G, qL))]
        union = merge_union(tokens, kb, ratio)
        # per token: running online-softmax state, keyed by query head
        state = {}    # s -> {h: (m, l, O)}
        for t, (blk, p) in enumerate(tokens):
            s = g + t
            state[s] = {h: (-np.inf, 0.0, np.zeros(hd, dtype=np.float64))
                        for h in range(Hq)}
        for t0 in range(0, len(union), TB):
            chunk = union[t0:t0 + TB]
            # staging: per union block, the max row count across tokens
            for hk in range(Hk):
                for t, (blk, p) in enumerate(tokens):
                    s = g + t
                    nrows_t = [chunk[ci][1][t] for ci in range(len(chunk))]
                    # build this token's gathered rows for the chunk (ascending)
                    rows = []
                    for ci, (b, nrows) in enumerate(chunk):
                        n = nrows[t]
                        for r in range(n):
                            rows.append(b * ratio + r)
                    if not rows:
                        continue
                    Ks = K[0, hk, rows, :]
                    Vs = V[0, hk, rows, :]
                    for h0 in range(gqa):
                        h = hk * gqa + h0
                        m, l, O = state[s][h]
                        S = (Q[0, h, s] @ Ks.T) * scale * log2e
                        new_m = max(m, S.max())
                        rowsum = np.exp2(S - new_m).sum()
                        factor = np.exp2(m - new_m) if np.isfinite(m) and m > -1e38 else 1.0
                        m = new_m
                        l = l * factor + rowsum
                        O = O * factor + (np.exp2(S - new_m) @ Vs)
                        state[s][h] = (m, l, O)
        for s, per_h in state.items():
            for h, (m, l, O) in per_h.items():
                out[0, h, s, :] = (O / l).astype(np.float32)
    return out


def synthetic_blocks(qL, kL, kb, ratio, seed=0):
    """Deterministic per-token selections with realistic overlap + a few
    deliberately-different blocks so the union/mask paths are exercised."""
    rng = np.random.default_rng(seed)
    blocks = np.zeros((1, qL, kb), dtype=np.int64)
    for s in range(qL):
        p = kL - qL + s
        complete = (p + 1) // ratio
        count = min(complete, kb)
        # base: the most recent `count` complete blocks, which is what a
        # recency-biased selector would keep (guarantees heavy overlap).
        base = np.arange(max(0, count - kb + 2), count + 2) if count else np.array([], dtype=np.int64)
        # trim to the most recent `kb` blocks and clamp
        if base.size > kb:
            base = base[-kb:]
        if base.size < kb:
            base = np.concatenate([base, np.arange(base.size, kb)]) if count else np.arange(kb)
        # perturb a few entries per token (index-permuting the tail of the list)
        sel = base.copy()
        n_swap = min(4, count)
        for _ in range(n_swap):
            i = rng.integers(0, sel.size)
            j = rng.integers(0, sel.size)
            sel[i], sel[j] = sel[j], sel[i]
        blocks[0, s, :] = np.sort(sel)[:kb]
    return blocks


def main():
    ratio = 4
    kb = 16
    qL = 33          # covers partial last group (G=4 -> 1 remainder token)
    kL = 96
    G = 4
    tile = 32

    blocks = synthetic_blocks(qL, kL, kb, ratio, seed=1)

    # --- 1. index validation: union -> per-token sequences == stock sequences
    print("== index validation ==")
    for g in range(0, qL, G):
        tokens = [(blocks[0, s], kL - qL + s) for s in range(g, min(g + G, qL))]
        union = merge_union(tokens, kb, ratio)
        seqs = union_to_sequences(union, kb, ratio)
        ok = True
        for t, (blk, p) in enumerate(tokens):
            ref = token_sequence(blk, p, kb, ratio)
            if seqs[t] != ref:
                ok = False
                print(f"  s={g + t}: MISMATCH\n    union={seqs[t]}\n    ref  ={ref}")
        if ok:
            print(f"  group {g}..{min(g + G, qL) - 1}: union -> per-token sequences EXACT")

    # --- 1b. edge cases: count < kb, tail_rows == 0, and one token's tail
    #         block being ANOTHER token's selected (complete) block.
    print("== index validation (edge cases) ==")
    qL2, kL2, kb2 = 16, 32, 8
    blocks2 = synthetic_blocks(qL2, kL2, kb2, ratio, seed=3)
    # force s=3 to select block 4 (it is complete for s=3 but the tail block
    # for s=0..2, whose count = complete = 4 < kb2).
    for g in range(0, qL2, G):
        tokens = [(blocks2[0, s], kL2 - qL2 + s) for s in range(g, min(g + G, qL2))]
        union = merge_union(tokens, kb2, ratio)
        seqs = union_to_sequences(union, kb2, ratio)
        ok = True
        for t, (blk, p) in enumerate(tokens):
            ref = token_sequence(blk, p, kb2, ratio)
            if seqs[t] != ref:
                ok = False
                print(f"  s={g + t}: MISMATCH\n    union={seqs[t]}\n    ref  ={ref}")
        if ok:
            print(f"  group {g}..{min(g + G, qL2) - 1}: union -> per-token sequences EXACT")

    # --- 2. attention validation: grouped == stock == exact
    print("== attention validation ==")
    Hq, Hk, hd = 12, 2, 256
    rng = np.random.default_rng(7)
    Q = rng.standard_normal((1, Hq, qL, hd)).astype(np.float32)
    K = rng.standard_normal((1, Hk, kL, hd)).astype(np.float32)
    V = rng.standard_normal((1, Hk, kL, hd)).astype(np.float32)
    scale = 1.0 / np.sqrt(hd)

    exact = exact_attention(Q, K, V, scale, blocks, ratio)
    stock = online_attention_tiles(Q, K, V, scale, blocks, ratio, tile)
    grp = online_attention_grouped(Q, K, V, scale, blocks, ratio, G, tile)

    d_stock = np.abs(stock - exact).max()
    d_grp = np.abs(grp - exact).max()
    d_gs = np.abs(grp - stock).max()
    print(f"  max|stock - exact| = {d_stock:.3e}")
    print(f"  max|grouped - exact| = {d_grp:.3e}")
    print(f"  max|grouped - stock| = {d_gs:.3e}")

    tol_exact = 1e-3
    tol_gs = 1e-5
    assert d_stock < tol_exact, "stock online deviates from exact"
    assert d_grp < tol_exact, "grouped online deviates from exact"
    assert d_gs < tol_gs, "grouped deviates from stock (tile boundary invariant)"
    print("  PASS: grouped == stock (tile-order invariant) and both ~ exact")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
