// Inter-chunk half of the GDN chunkwise (WY) prefill: one threadgroup per
// (32-row dv slice, hv, b) walks the NC chunks in order, keeping its 32 x Dk
// state slice in simdgroup-matrix registers. Per chunk it does two MMAs --
// [W;Qeff]·S^T and Vnew^T·Kd -- so the only sequential work left is NC steps
// of small matrix products instead of T dependent scalar steps.
constexpr int C = 64;
constexpr int DB = 32;   // dv rows per threadgroup
constexpr int SLD = Dk + 4;
constexpr int VLD = DB + 4;

static_assert(Dk == 128, "gdn wy state splits Dk 16 columns per simdgroup");
static_assert(Dv % DB == 0, "gdn wy state needs whole 32-row dv slices");

const ushort sg = ushort(simdgroup_index_in_threadgroup);
const ushort lane = ushort(thread_index_in_simdgroup);
const int blk = int(threadgroup_position_in_grid.x);
const int hv = int(threadgroup_position_in_grid.y);
const int b = int(threadgroup_position_in_grid.z);
const int dv0 = blk * DB;
const int NC = (T + C - 1) / C;

// One arena: the 32 x Dk state slice while the R product runs, then the
// C x 32 Vnew tile for the state update. 4224 floats covers both.
threadgroup float arena[DB * SLD];

simdgroup_float8x8 S[DB / 8][2];
{
    const device StT* si = state_in + ((size_t)(b * Hv + hv) * Dv + dv0) * Dk;
    for (int rf = 0; rf < DB / 8; ++rf) {
        for (int cf = 0; cf < 2; ++cf) {
            S[rf][cf] = gdn_wy_load<StT>(si, Dk, ulong2(16 * sg + 8 * cf, 8 * rf));
        }
    }
}

const short2 co = gdn_wy_coord(lane);
const int arow = 16 * (sg & 3);

for (int c = 0; c < NC; ++c) {
    const int t0 = c * C;
    const int tt = min(C, T - t0);
    const size_t tile = ((size_t)(b * Hv + hv) * NC + c) * C;
    const float gc = Gc[(size_t)(b * Hv + hv) * NC + c];

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int rf = 0; rf < DB / 8; ++rf) {
        for (int cf = 0; cf < 2; ++cf) {
            simdgroup_store(S[rf][cf], arena, SLD, ulong2(16 * sg + 8 * cf, 8 * rf), false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // R = [W; Qeff] · S_blk^T, 16 rows per simdgroup: the low four hold W
    // rows (-> Vnew), the high four Qeff rows (-> y).
    const device MidT* Ad = (sg < 4) ? (W + tile * Dk) : (Qeff + tile * Dk);
    simdgroup_float8x8 R[2][DB / 8];
    for (int rr = 0; rr < 2; ++rr) {
        for (int cf = 0; cf < DB / 8; ++cf) R[rr][cf] = gdn_wy_zero();
    }
    for (int kk = 0; kk < Dk; kk += 8) {
        simdgroup_float8x8 a0 = gdn_wy_load<MidT>(Ad, Dk, ulong2(kk, arow));
        simdgroup_float8x8 a1 = gdn_wy_load<MidT>(Ad, Dk, ulong2(kk, arow + 8));
        for (int cf = 0; cf < DB / 8; ++cf) {
            simdgroup_float8x8 bfr;
            simdgroup_load(bfr, arena, SLD, ulong2(kk, 8 * cf), true);
            simdgroup_multiply_accumulate(R[0][cf], a0, bfr, R[0][cf]);
            simdgroup_multiply_accumulate(R[1][cf], a1, bfr, R[1][cf]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg >= 4) {
        const device MidT* Yl = Yloc + tile * Dv;
        device OutT* yb = y + ((size_t)(b * T + t0) * Hv + hv) * Dv + dv0;
        for (int rr = 0; rr < 2; ++rr) {
            const int row = arow + 8 * rr + co.y;
            if (row >= tt) continue;
            for (int cf = 0; cf < DB / 8; ++cf) {
                const int d0 = 8 * cf + co.x;
                thread float2& e = gdn_wy_el(R[rr][cf]);
                yb[(size_t)row * Hv * Dv + d0] = static_cast<OutT>(float(Yl[row * Dv + dv0 + d0]) + e[0]);
                yb[(size_t)row * Hv * Dv + d0 + 1] = static_cast<OutT>(float(Yl[row * Dv + dv0 + d0 + 1]) + e[1]);
            }
        }
    } else {
        const device MidT* Ud = U + tile * Dv;
        for (int rr = 0; rr < 2; ++rr) {
            const int row = arow + 8 * rr + co.y;
            for (int cf = 0; cf < DB / 8; ++cf) {
                const int d0 = 8 * cf + co.x;
                thread float2& e = gdn_wy_el(R[rr][cf]);
                e[0] = float(Ud[row * Dv + dv0 + d0]) - e[0];
                e[1] = float(Ud[row * Dv + dv0 + d0 + 1]) - e[1];
            }
        }
        for (int rr = 0; rr < 2; ++rr) {
            for (int cf = 0; cf < DB / 8; ++cf) {
                simdgroup_store(R[rr][cf], arena, VLD, ulong2(8 * cf, arow + 8 * rr), false);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device MidT* Kdd = Kd + tile * Dk;
    for (int rf = 0; rf < DB / 8; ++rf) {
        for (int cf = 0; cf < 2; ++cf) {
            thread float2& e = gdn_wy_el(S[rf][cf]);
            e[0] *= gc;
            e[1] *= gc;
        }
    }
    for (int kk = 0; kk < C; kk += 8) {
        simdgroup_float8x8 b0 = gdn_wy_load<MidT>(Kdd, Dk, ulong2(16 * sg, kk));
        simdgroup_float8x8 b1 = gdn_wy_load<MidT>(Kdd, Dk, ulong2(16 * sg + 8, kk));
        for (int rf = 0; rf < DB / 8; ++rf) {
            simdgroup_float8x8 a;
            simdgroup_load(a, arena, VLD, ulong2(8 * rf, kk), true);
            simdgroup_multiply_accumulate(S[rf][0], a, b0, S[rf][0]);
            simdgroup_multiply_accumulate(S[rf][1], a, b1, S[rf][1]);
        }
    }
}

device StT* so = state_out + ((size_t)(b * Hv + hv) * Dv + dv0) * Dk;
for (int rf = 0; rf < DB / 8; ++rf) {
    for (int cf = 0; cf < 2; ++cf) {
        gdn_wy_store<StT>(S[rf][cf], so, Dk, ulong2(16 * sg + 8 * cf, 8 * rf));
    }
}
