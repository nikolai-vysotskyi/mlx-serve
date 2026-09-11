// Fused HC up+mix on the prefill path, NAX cooperative-tensor GEMM:
//   mixed[r, col] = T( (1/hc) * sum_h  T( sigmoid(T(up_h)) * n4[r,h,col] ) )
// where up_h = act @ up_w_h^T. Same geometry/envelope as the plain-SIMD
// HC_UP_MIX_SOURCE (BM=32, BN=64, one threadgroup of 4 simdgroups per
// (row-block, col-block), fp32 accumulate, bf16 up once per stream, LUT
// sigmoid, bf16 product, fp32 stream sum, 1/hc scale), but the inner product
// is a 16x32x16 bf16 MMA (mpp::tensor_ops::matmul2d, multiply_accumulate)
// instead of per-lane fp32 FMA. BK=32 keeps the two staged T tiles at 6 KiB.
//
// Parity: the dequantized weight rounds to T (bf16/f16) exactly like the
// plain-SIMD kernel (and stock qmm_n dequantize()); the MMA destination
// accumulates in fp32 (CType=float). A is act (already T) and B is the
// T-rounded weight, so each MMA product is bit-identical to the plain-SIMD
// float(float(A))*float(T(B)) product; only the fp32 accumulation order
// differs (the accepted ~2 bf16 ULP class the HC reference already claims).
//
// GS == 64 spans two 32-blocks; a 16-long MMA K-step never crosses a group
// boundary (64 % 32 == 0, 32 % 16 == 0).
using namespace mlx::steel;

constexpr int BM = 32;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int VPW = 32 / BITS;   // 8 for BITS=4
constexpr int STEPS = BK / 16;   // 2 MMA K-steps per block
constexpr uint MASK = (1u << BITS) - 1u;

const int tg_col = threadgroup_position_in_grid.x;
const int tg_row = threadgroup_position_in_grid.y;
const int row0 = tg_row * BM;
const int col0 = tg_col * BN;

const int sg = simdgroup_index_in_threadgroup;   // 0..3
const int tid = thread_index_in_threadgroup;     // 0..127
// 4 simdgroups tile 32x64 as 2(row) x 2(col); each computes a 16x32 tile.
const int tm = (sg / 2) * 16;
const int tn = (sg % 2) * 32;

const int M = int(M_size);
const int K = int(K_size);
const int K_by_p = K / VPW;
const int K_by_gs = K / GS;

threadgroup T Atile[BM * BK];   // staged act tile [BM][BK]   (2 KiB)
threadgroup T Wtile[BN * BK];   // dequantized up weights [BN][BK] (4 KiB)

const short2 xy = BaseNAXFrag::get_coord();

// Per-(row,col) partial products summed across streams (h): one fp32 per
// fragment position, laid out exactly like a NAXTile<float,1,2>.
NAXTile<float, 1, 2> sums;
sums.clear();

for (int h = 0; h < HC; ++h) {
    const int wrow0 = h * H + col0;
    NAXTile<float, 1, 2> C;   // 16 M-rows x 32 N-cols, fp32
    C.clear();

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Stage act tile (guard M tail; K % 32 == 0 so k < K always).
        for (int p = tid; p < BM * BK; p += 128) {
            int r = p / BK, kk = p % BK;
            int row = row0 + r, k = k0 + kk;
            Atile[p] = (row < M) ? act[(size_t)row * K + k] : T(0.0f);
        }
        // Dequantize up weights for stream h into the T tile (one group spans
        // two 32-blocks: gi = k0 / GS).
        const int gi = k0 / GS;
        for (int p = tid; p < BN * BK; p += 128) {
            int c = p / BK, kk = p % BK;
            int n = wrow0 + c, k = k0 + kk;
            const float s = float(uw_s[(size_t)n * K_by_gs + gi]);
            const float b = float(uw_b[(size_t)n * K_by_gs + gi]);
            const uint32_t sh = (k % VPW) * BITS;
            const float q = float((uw_q[(size_t)n * K_by_p + (k / VPW)] >> sh) & MASK);
            Wtile[p] = T(q * s + b);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Two 16x32x16 MMAs per block: C += A @ B^T.
        for (int step = 0; step < STEPS; ++step) {
            NAXTile<T, 1, 1> A;
            A.template load<T, BK, 1>(Atile + tm * BK + step * 16);
            NAXTile<T, 2, 1> B;
            B.template load<T, BK, 1>(Wtile + tn * BK + step * 16);

            BaseNAXFrag::mma(
                C.frag_at(0, 0),
                C.frag_at(0, 1),
                A.frag_at(0, 0),
                metal::false_type{},
                B.frag_at(0, 0),
                B.frag_at(1, 0),
                metal::true_type{});
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Per-h epilogue, bit-identical to the plain-SIMD kernel:
    //   sums += T( sigmoid(T(up)) * n4 )
    for (short c = 0; c < 2; ++c) {
        for (short j = 0; j < 8; ++j) {
            const int row = xy.y + (j / 4) * 8;      // 0..15
            const int ncol = c * 16 + xy.x + j % 4;  // 0..31
            const int r = row0 + tm + row;
            const int col = col0 + tn + ncol;
            if (r < M && col < H) {
                const T u = T(C.data[c][j]);
                const T sg = sigtab[as_type<ushort>(u)];
                const T prod = T(float(sg) * float(n4[((size_t)r * HC + h) * H + col]));
                sums.data[c][j] += float(prod);
            }
        }
    }
}

// Final store: mixed = T( sums / hc ).
for (short c = 0; c < 2; ++c) {
    for (short j = 0; j < 8; ++j) {
        const int row = xy.y + (j / 4) * 8;
        const int ncol = c * 16 + xy.x + j % 4;
        const int r = row0 + tm + row;
        const int col = col0 + tn + ncol;
        if (r < M && col < H)
            mixed[(size_t)r * H + col] = T(sums.data[c][j] * (1.0f / float(HC)));
    }
}
