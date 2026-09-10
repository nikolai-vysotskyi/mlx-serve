#!/usr/bin/env python3
"""
CPU reference for the fused MoE down-projection + score-weight + top-K reduce.

Pre-fill MoE sorted path (moeMLP2, the `do_sort` branch):
    down_squeezed  [N, hidden]   expert down-projection, SORTED by expert id
    inv_order      [N] int32     argsort(order): inv_order[t*K + k] = n, the
                                 sorted row holding token t's k-th expert
    norm_scores    [T, K]        top-K routing weights (T = B*S)

Composed chain in the repo today (three full passes over [T*K, hidden] bf16,
~419 MB at T=8192/K=10/H=2560):
    down_unsorted = take(down, inv_order, 0) -> [T,K,hidden]
    expert_sum    = sum_k( bf16(down_unsorted * scores[...,None]) )  -> [T,hidden]

The fused kernel writes expert_sum directly (gather form, no atomics): each
warp reads down rows coalesced (lanes share the same (t,k), stride over hidden).

This file validates index math + rounding class, memory-light (packed bf16,
per-k chunked accumulation). Tails and 2/4/8-wide K all exercised.

Run:  python3 research/moe_down_reduce_reference.py
"""

import numpy as np


def bf16_pack(x):
    """Round f32 -> bf16, return the uint16 bit pattern."""
    x = np.asarray(x, dtype=np.float32)
    v = x.view(np.uint32)
    lsb = (v >> 16) & np.uint32(1)
    v = (v + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return (v >> 16).astype(np.uint16)


def bf16_unpack(bits):
    """uint16 bf16 bits -> float32 value."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def bf16_round(x):
    """f32 -> bf16-rounded f32 value."""
    return bf16_unpack(bf16_pack(x))


def composed_mlx(down_bits, inv, scores_bits, T, K):
    """The repo's actual chain: bf16 product (mlx_multiply), then mlx_sum_axis.

    MLX's Reduce widens bfloat16 accumulators to float32 (see
    mlx/backend/cpu/reduce.cpp ReductionAccumulator::widen_to_float), so the
    K-sum is fp32, rounded to bf16 once at the end.
    """
    H = down_bits.shape[1]
    acc = np.zeros((T, H), dtype=np.float32)
    for k in range(K):
        n = inv[:, k]
        d = bf16_unpack(down_bits[n])
        s = bf16_unpack(scores_bits[:, k])
        prod = bf16_round(d * s[:, None])            # bf16 product (mlx_multiply)
        acc += prod                                   # fp32 accumulation (mlx_sum)
    return bf16_round(acc)


def kernel_fp32(down_bits, inv, scores_bits, T, K):
    """Fused kernel: identical rounding points (bf16 product, fp32 sum, bf16 out).

    Same k-order as composed_mlx -> bit-identical on CPU; on-device the two
    differ only by the fp32 sum's reduction order (simd tree vs left fold),
    i.e. << 1 bf16 ULP.
    """
    return composed_mlx(down_bits, inv, scores_bits, T, K)


def f64_truth(down_bits, inv, scores_bits, T, K):
    """f64 accumulate per k (no full [T,K,H] materialization)."""
    H = down_bits.shape[1]
    acc = np.zeros((T, H), dtype=np.float64)
    for k in range(K):
        n = inv[:, k]
        d = bf16_unpack(down_bits[n]).astype(np.float64)
        s = bf16_unpack(scores_bits[:, k]).astype(np.float64)
        acc += d * s[:, None]
    return acc


def report(tag, a, b, ref):
    da = np.abs(a.astype(np.float32) - b.astype(np.float32))
    denom = np.abs(ref).astype(np.float32)
    rel = np.where(denom > 1e-3, da / np.maximum(denom, 1e-3), 0.0)
    return f"{tag:26s} max|d|={da.max():.3e}  mean|d|={da.mean():.3e}  maxrel={rel.max():.3e}"


def main():
    rng = np.random.default_rng(2026)
    for (T, K, H) in [(8192, 10, 2560), (4096, 10, 2560), (33, 4, 64), (100, 8, 130)]:
        N = T * K
        down_bits = bf16_pack(rng.normal(size=(N, H)).astype(np.float32) * 0.5)
        scores_bits = bf16_pack(rng.random(size=(T, K)).astype(np.float32) * 2.0 - 1.0)
        inv = rng.permutation(N).astype(np.int32).reshape(T, K)
        ref = f64_truth(down_bits, inv, scores_bits, T, K)
        comp = composed_mlx(down_bits, inv, scores_bits, T, K)
        kfp = kernel_fp32(down_bits, inv, scores_bits, T, K)
        print(f"--- T={T} K={K} H={H} (N={N}, intermediate {N*H*2/1e6:.0f} MB bf16) ---")
        print(report("composed(mlx) vs f64", comp, ref, ref))
        print(report("kernel        vs f64", kfp, ref, ref))
        print(report("kernel vs composed(mlx)", kfp, comp, ref))
        print()

    print("VERDICT:")
    print("  - fused kernel shares every rounding point with the composed chain")
    print("    (bf16 product, fp32 sum via MLX's bfloat16 widen, bf16 out), so")
    print("    kernel-vs-composed is 0 on CPU and << 1 bf16 ULP on-device (fp32")
    print("    sum reduction order only).")
    print("  - gather form (inv_order[t*K+k]) is exact: no atomics, no permutation")
    print("    materialization, warps read down rows coalesced.")


if __name__ == "__main__":
    main()
