// Query-grouped (block-reuse) sparse QSA with the NAX cooperative-tensor
// inner product. G adjacent query tokens share one threadgroup and stage each
// distinct selected K/V block once (the Lever G gather); the QK^T / PV inner
// products use the 16x32x16 bf16 MMA (mpp::tensor_ops::matmul2d) instead of
// the per-lane 8x8 simdgroup mma. Two simdgroups split D=256 per token
// (band*128) and exchange their fp32 partial S through the KV staging buffer
// (K is dead by then, so no extra threadgroup memory), exactly like
// msv_qsa_nax_precise; the k-way union merge, per-token -inf mask and the
// Shi/Slo PV epilogue are ported from msv_attn_qsa256_grp / msv_qsa_nax_precise.
//
// Same parity class as msv_qsa_nax_precise (bf16 operands, fp32 accumulate,
// fp32 partial-S exchange, two-bf16-term PV): identical to the composed
// attention modulo the MMA's fp32 accumulation order + softmax rescale order.
using namespace mlx::steel;

static_assert(NSG == 2 && BK == 32, "msv_qsa_group_nax is written for 2 simdgroups and BK 32");
constexpr int BD = 256, LD = 264, TDH = 8, TK = BK / 16;   // TDH=8 => 128 D per warp
constexpr int TB = BK / RATIO;      // union blocks per tile
constexpr int NT = 32 * NSG * G;    // <= 256
constexpr int SENTINEL = 2147483647;

const int qL = q_shape[2], kL = k_shape[2], Hq = q_shape[1], Hk = k_shape[1];
const int gqa = Hq / Hk;
const int KB = blocks_shape[2];

const int g = int(threadgroup_position_in_grid.x);
const int hk = int(threadgroup_position_in_grid.y);
const int bb = int(threadgroup_position_in_grid.z);
const ushort lane = ushort(thread_index_in_simdgroup);
const ushort warp = ushort(simdgroup_index_in_threadgroup);
const int tix = int(thread_index_in_threadgroup);

const int t_in = int(warp) / NSG;   // token within the group (0..G-1)
const int band = int(warp) % NSG;   // D-half (0..1)
const int s = g * G + t_in;         // global query token
const bool s_ok = s < qL;

const float scale_log2e = scl[0] * 1.44269504088896340736f;

const device T* Kp = k + bb * k_strides[0] + hk * k_strides[1];
const device T* Vp = v + bb * v_strides[0] + hk * v_strides[1];

threadgroup T KV[BK * LD];   // staged K tile, then partial-S exchange, then V tile

// Q: this token's 16 head rows x this warp's D-half (128), in 16-D fragments.
const device T* Qp = q + bb * q_strides[0] + (long)(hk * gqa) * q_strides[1] + (long)s * q_strides[2] + band * 128;
NAXTile<T,1,1> Q[TDH];
if (s_ok) {
    for (short d = 0; d < TDH; ++d) Q[d].load_rows(Qp + d * 16, int(q_strides[1]), short(gqa));
} else {
    for (short d = 0; d < TDH; ++d) Q[d].clear();
}

using ST = NAXTile<float,1,TK>;   // 16 head rows x 32 keys
using OT = NAXTile<float,1,TDH>;  // 16 head rows x 128 D (this warp's half)
OT O; O.clear();
const short2 coord = BaseNAXFrag::get_coord();
float2 max_score = float2(-3e38f), sum_score = float2(0.0f);

// k-way merge state (every thread computes the same union redundantly).
int cnt[G]; int cpl[G]; int tlr[G]; int bi[G]; int nxt[G]; int nxtn[G];
for (int t = 0; t < G; ++t) {
    const int st = g * G + t;
    if (st >= qL) { cnt[t] = 0; cpl[t] = 0; tlr[t] = 0; bi[t] = -1; nxt[t] = SENTINEL; nxtn[t] = 0; continue; }
    const int p = (kL - qL) + st;
    const int complete = (p + 1) / RATIO;
    const int count = metal::min(complete, KB);
    const int tail_start = complete * RATIO;
    const int tail_rows = p + 1 - tail_start;
    cnt[t] = count; cpl[t] = complete; tlr[t] = tail_rows;
    const device int* bp = blocks + (long)bb * blocks_strides[0] + (long)st * blocks_strides[1];
    if (count > 0) { bi[t] = 0; nxt[t] = bp[0]; nxtn[t] = RATIO; }
    else if (tail_rows > 0) { bi[t] = -1; nxt[t] = complete; nxtn[t] = tail_rows; }
    else { bi[t] = -1; nxt[t] = SENTINEL; nxtn[t] = 0; }
}

int tile_blk[TB];
int tile_nrows[TB * G];

for (;;) {
    int produced = 0;
    for (int u = 0; u < TB; ++u) {
        int mb = SENTINEL;
        for (int t = 0; t < G; ++t) mb = metal::min(mb, nxt[t]);
        if (mb == SENTINEL) break;
        tile_blk[u] = mb;
        for (int t = 0; t < G; ++t) tile_nrows[u * G + t] = (nxt[t] == mb) ? nxtn[t] : 0;
        for (int t = 0; t < G; ++t) {
            if (nxt[t] != mb) continue;
            if (bi[t] >= 0) {
                bi[t] += 1;
                if (bi[t] < cnt[t]) {
                    const device int* bp = blocks + (long)bb * blocks_strides[0] + (long)(g * G + t) * blocks_strides[1];
                    nxt[t] = bp[bi[t]]; nxtn[t] = RATIO;
                } else if (tlr[t] > 0) { nxt[t] = cpl[t]; nxtn[t] = tlr[t]; bi[t] = -1; }
                else { nxt[t] = SENTINEL; nxtn[t] = 0; }
            } else { nxt[t] = SENTINEL; nxtn[t] = 0; }
        }
        produced += 1;
    }
    if (produced == 0) break;
    for (int u = produced; u < TB; ++u) {
        tile_blk[u] = SENTINEL;
        for (int t = 0; t < G; ++t) tile_nrows[u * G + t] = 0;
    }
    int maxn[TB];
    for (int u = 0; u < TB; ++u) {
        int mn = 0;
        for (int t = 0; t < G; ++t) mn = metal::max(mn, tile_nrows[u * G + t]);
        maxn[u] = mn;
    }

    // Stage the union K tile (row-major, LD-padded, 8-bf16 uint4 chunks).
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tix; i < BK * 32; i += NT) {
        int r = i >> 5, c8 = i & 31;
        uint4 w = uint4(0);
        int u = r / RATIO, inrow = r - u * RATIO;
        if (tile_blk[u] != SENTINEL && inrow < maxn[u]) {
            int pos = tile_blk[u] * RATIO + inrow;
            w = *((const device uint4*)(Kp + (long)pos * k_strides[2]) + c8);
        }
        *((threadgroup uint4*)(KV + r * LD) + c8) = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S = Q @ K^T (this warp's D-half), fp32 accumulate.
    ST S; S.clear();
    for (short d = 0; d < TDH; ++d) {
        NAXTile<T,2,1> Kf;
        Kf.template load<T,LD,1>(KV + band * 128 + d * 16);
        BaseNAXFrag::mma(S.frag_at(0,0), S.frag_at(0,1), Q[d].frag_at(0,0),
            metal::false_type{}, Kf.frag_at(0,0), Kf.frag_at(1,0), metal::true_type{});
    }
    // Exchange partials with the other warp of this token (K is dead now, so
    // the KV buffer is reused as fp32 scratch: G tokens x 2 warps x 512).
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* X = reinterpret_cast<threadgroup float*>(KV);
    for (short ik = 0; ik < TK; ++ik)
        for (short i = 0; i < 8; ++i)
            X[t_in * 1024 + band * 512 + lane * 16 + ik * 8 + i] = S.frag_at(0, ik)[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short ik = 0; ik < TK; ++ik)
        for (short i = 0; i < 8; ++i)
            S.frag_at(0, ik)[i] += X[t_in * 1024 + (1 - band) * 512 + lane * 16 + ik * 8 + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scale + per-token mask (rows outside this token's selection -> -inf).
    for (short ik = 0; ik < TK; ++ik) {
        for (short i = 0; i < 8; ++i) {
            S.frag_at(0, ik)[i] *= scale_log2e;
            const int rx = ik * 16 + coord.x + (i % 4);
            const int ux = rx / RATIO, ix = rx - ux * RATIO;
            if (tile_blk[ux] == SENTINEL || tile_nrows[ux * G + t_in] <= ix)
                S.frag_at(0, ik)[i] = -INFINITY;
        }
    }

    // Online softmax (per row, over this warp's two head rows).
    float2 new_max = max_score;
    S.template row_reduce<QsaMax>(new_max);
    S.template row_bin_op<QsaExpSub>(new_max);
    float2 factor = metal::exp2(max_score - new_max);
    max_score = new_max;
    sum_score *= factor;
    S.template row_reduce<QsaSum>(sum_score);
    O.template row_bin_op<QsaMul>(factor);

    // Stage the union V tile over the dead exchange data.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tix; i < BK * 32; i += NT) {
        int r = i >> 5, c8 = i & 31;
        uint4 w = uint4(0);
        int u = r / RATIO, inrow = r - u * RATIO;
        if (tile_blk[u] != SENTINEL && inrow < maxn[u]) {
            int pos = tile_blk[u] * RATIO + inrow;
            w = *((const device uint4*)(Vp + (long)pos * v_strides[2]) + c8);
        }
        *((threadgroup uint4*)(KV + r * LD) + c8) = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O += P @ V (this warp's D-half); two-bf16-term (Shi/Slo) P for precision.
    for (short d = 0; d < TDH; d += 2) {
        for (short ik = 0; ik < TK; ++ik) {
            NAXTile<T,1,2> Vf;
            Vf.template load<T,LD,1>(KV + ik * 16 * LD + band * 128 + d * 16);
            NAXTile<T,1,1> Shi, Slo;
            for (short j = 0; j < 8; ++j) {
                Shi.frag_at(0,0)[j] = T(S.frag_at(0, ik)[j]);
                const float residual = S.frag_at(0, ik)[j] - float(Shi.frag_at(0,0)[j]);
                Slo.frag_at(0,0)[j] = T(residual);
            }
            BaseNAXFrag::mma(O.frag_at(0,d), O.frag_at(0,d+1), Shi.frag_at(0,0),
                metal::false_type{}, Vf.frag_at(0,0), Vf.frag_at(0,1), metal::false_type{});
            BaseNAXFrag::mma(O.frag_at(0,d), O.frag_at(0,d+1), Slo.frag_at(0,0),
                metal::false_type{}, Vf.frag_at(0,0), Vf.frag_at(0,1), metal::false_type{});
        }
    }
}

// Normalize and store this warp's D-half of the token's head rows.
float2 inv = 1.0f / sum_score;
O.template row_bin_op<QsaMul>(inv);
if (s_ok) {
    device T* Op = out + (((long)bb * Hq + hk * gqa) * qL + s) * BD + band * 128;
    O.store_rows(Op, qL * BD, short(gqa));
}
