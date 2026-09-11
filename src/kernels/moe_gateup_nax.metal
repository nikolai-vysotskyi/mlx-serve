// Fused MoE gate/up + GeGLU on the sorted prefill path, NAX cooperative-tensor
// GEMM. Same schedule/geometry/envelope as the plain-SIMD MOE_GATEUP_SOURCE
// (BM=64, BN=64, one threadgroup of 8 simdgroups per (tile, col-block), fp32
// accumulate, bf16 gate/up once at the end, LUT GeGLU), but the inner product
// is a 16x32x16 bf16 MMA (mpp::tensor_ops::matmul2d, multiply_accumulate)
// instead of per-lane fp32 FMA. BK=32 keeps the three staged tiles at 12 KiB,
// well under Metal's 32 KiB threadgroup budget (METAL_TG_BYTES).
//
// Parity: the dequantized weight rounds to T (bf16/f16) exactly like stock
// prefill gather_qmm (affine_gather_qmm_n -> qmm_n_impl dequantize() into a T
// tile + BlockMMA<T,T,AccumType=fp32>). A is x (bf16), B is the T-rounded
// weight, the MMA destination accumulates in fp32 (CType=float), so this is
// bit-identical to the composed gather_qmm chain modulo fp32 accumulation
// order (the accepted ~2 bf16 ULP class). No Shi/Slo bf16 term split is
// needed: both operands are genuinely bf16 here (unlike attention, which
// splits to recover precision lost by rounding Q to bf16).
//
// GS == 64 spans two 32-blocks; a 16-long MMA K-step never crosses a group
// boundary (64 % 32 == 0, 32 % 16 == 0).
using namespace mlx::steel;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int VPW = 32 / BITS;   // 8 for BITS=4
constexpr int STEPS = BK / 16;   // 2 MMA K-steps per block
constexpr uint MASK = (1u << BITS) - 1u;

const int tg_col = threadgroup_position_in_grid.x;
const int tile = threadgroup_position_in_grid.y;
const int start = tiles[tile * 3];
const int count = tiles[tile * 3 + 1];
const int block = tiles[tile * 3 + 2];
if (count <= 0) return;
const int row0 = start + block * BM;
const int M = min(BM, count - block * BM);
if (M <= 0) return;
const uint expert = inds[start];
const int col0 = tg_col * BN;

const int sg = simdgroup_index_in_threadgroup;   // 0..7
const int tid = thread_index_in_threadgroup;     // 0..255
// 8 simdgroups tile 64x64 as 4(row) x 2(col); each computes a 16x32 tile.
const int tm = (sg / 2) * 16;
const int tn = (sg % 2) * 32;

const int K = int(K_size);
const int N = int(N_size);
const int K_by_p = K / VPW;
const int K_by_gs = K / GS;

const size_t wbase = (size_t)expert * (size_t)N * (size_t)K_by_p;
const size_t sbase = (size_t)expert * (size_t)N * (size_t)K_by_gs;

threadgroup T Atile[BM * BK];    // staged x tile [BM][BK]
threadgroup T Wgtile[BN * BK];   // dequantized gate weights [BN][BK]
threadgroup T Wutile[BN * BK];   // dequantized up weights [BN][BK]

using AccFrag = NAXTile<float, 1, 2>;   // 16 M-rows x 32 N-cols, fp32
AccFrag Cg;
AccFrag Cu;
Cg.clear();
Cu.clear();

for (int k0 = 0; k0 < K; k0 += BK) {
    // Stage x tile (guard M tail; K % 32 == 0 so k < K always).
    for (int p = tid; p < BM * BK; p += 256) {
        int r = p / BK, kk = p % BK;
        int k = k0 + kk;
        Atile[p] = (r < M) ? x[(size_t)(row0 + r) * K + k] : T(0.0f);
    }
    // Dequantize gate/up weights into T tiles (one group spans two blocks).
    const int gi = k0 / GS;
    for (int p = tid; p < BN * BK; p += 256) {
        int c = p / BK, kk = p % BK;
        int n = col0 + c;
        int k = k0 + kk;
        const float sg_n = float(scales[sbase + (size_t)n * K_by_gs + gi]);
        const float bg_n = float(biases[sbase + (size_t)n * K_by_gs + gi]);
        const float su_n = float(up_scales[sbase + (size_t)n * K_by_gs + gi]);
        const float bu_n = float(up_biases[sbase + (size_t)n * K_by_gs + gi]);
        const uint32_t sh = (k % VPW) * BITS;
        const float qg = float((w_q[wbase + (size_t)n * K_by_p + (k / VPW)] >> sh) & MASK);
        const float qu = float((up_w_q[wbase + (size_t)n * K_by_p + (k / VPW)] >> sh) & MASK);
        Wgtile[p] = T(qg * sg_n + bg_n);
        Wutile[p] = T(qu * su_n + bu_n);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Two 16x32x16 MMAs per block: Cg += A @ Bg^T, Cu += A @ Bu^T.
    for (int step = 0; step < STEPS; ++step) {
        NAXTile<T, 1, 1> A;
        A.template load<T, BK, 1>(Atile + tm * BK + step * 16);
        NAXTile<T, 2, 1> Bg;
        Bg.template load<T, BK, 1>(Wgtile + tn * BK + step * 16);
        NAXTile<T, 2, 1> Bu;
        Bu.template load<T, BK, 1>(Wutile + tn * BK + step * 16);

        BaseNAXFrag::mma(
            Cg.frag_at(0, 0),
            Cg.frag_at(0, 1),
            A.frag_at(0, 0),
            metal::false_type{},
            Bg.frag_at(0, 0),
            Bg.frag_at(1, 0),
            metal::true_type{});
        BaseNAXFrag::mma(
            Cu.frag_at(0, 0),
            Cu.frag_at(0, 1),
            A.frag_at(0, 0),
            metal::false_type{},
            Bu.frag_at(0, 0),
            Bu.frag_at(1, 0),
            metal::true_type{});
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// GeGLU epilogue, bit-identical to the plain-SIMD kernel:
//   out = bf16( bf16( bf16(gate) * sigtab[gate] ) * bf16(up) )
const short2 xy = BaseNAXFrag::get_coord();
for (short c = 0; c < 2; ++c) {
    for (short j = 0; j < 8; ++j) {
        const int row = xy.y + (j / 4) * 8;      // 0..15
        const int ncol = c * 16 + xy.x + j % 4;  // 0..31
        const int r = row0 + tm + row;
        const int col = col0 + tn + ncol;
        if (r < row0 + M && col < N) {
            const T gv = T(Cg.data[c][j]);
            const T uv = T(Cu.data[c][j]);
            const T sv = sigtab[as_type<ushort>(gv)];
            const T act = T(float(gv) * float(sv));
            out[(size_t)r * N + col] = T(float(act) * float(uv));
        }
    }
}
