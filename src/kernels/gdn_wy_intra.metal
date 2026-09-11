// Chunk-local half of the GDN chunkwise (WY) prefill: one threadgroup (256
// threads = 8 simdgroups) per (chunk, hv, b) computes W, U, Kd, Qeff, Yloc
// and the chunk decay Gc for its C = 64 tokens. Everything the inter-chunk
// pass needs, nothing that depends on the incoming state.
constexpr int C = 64;
constexpr int SLD = 36; // staged K/Q row pitch (32 columns + padding)

static_assert(Dk % 32 == 0, "gdn wy intra stages Dk in 32-column steps");
static_assert(Dv % 8 == 0, "gdn wy intra writes Yloc in 8-column fragments");

const int tid = int(thread_index_in_threadgroup);
const ushort sg = ushort(simdgroup_index_in_threadgroup);
const ushort lane = ushort(thread_index_in_simdgroup);
const int cc = int(threadgroup_position_in_grid.x);
const int hv = int(threadgroup_position_in_grid.y);
const int b = int(threadgroup_position_in_grid.z);
const int hk = hv / (Hv / Hk);
const int NC = (T + C - 1) / C;
const int t0 = cc * C;
const int tt = min(C, T - t0);

// One arena, two lives: the staged K/Q columns of phase 1, then the masked
// C x C matrix (A, later P) that phases 2-5 read. 4608 floats covers both.
threadgroup float arena[2 * C * SLD];
threadgroup float lg[C];
threadgroup float bet[C];
threadgroup float* Ksh = arena;
threadgroup float* Qsh = arena + C * SLD;
threadgroup float* M = arena;

const device InT* kbase = k + ((size_t)b * T * Hk + hk) * Dk;
const device InT* qbase = q + ((size_t)b * T * Hk + hk) * Dk;
const device InT* vbase = v + ((size_t)b * T * Hv + hv) * Dv;
const size_t krow = (size_t)Hk * Dk;
const size_t vrow = (size_t)Hv * Dv;

if (tid == 0) {
    float cum = 0.0f;
    for (int i = 0; i < C; ++i) {
        // Padded rows contribute log g = 0; the clamp guards g == 0.
        if (i < tt) {
            cum += max(log(float(g[((size_t)(b * T + t0 + i)) * Hv + hv])), -80.0f);
        }
        lg[i] = cum;
    }
    Gc[(size_t)(b * Hv + hv) * NC + cc] = exp(cum);
}
for (int p = tid; p < C; p += 256) {
    bet[p] = (p < tt) ? float(beta[((size_t)(b * T + t0 + p)) * Hv + hv]) : 0.0f;
}

simdgroup_float8x8 Araw[C / 8];
simdgroup_float8x8 Praw[C / 8];
for (int j = 0; j < C / 8; ++j) {
    Araw[j] = gdn_wy_zero();
    Praw[j] = gdn_wy_zero();
}

for (int dc = 0; dc < Dk; dc += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = tid; p < C * 32; p += 256) {
        const int r = p >> 5, d = p & 31;
        const bool live = r < tt;
        Ksh[r * SLD + d] = live ? float(kbase[(size_t)(t0 + r) * krow + dc + d]) : 0.0f;
        Qsh[r * SLD + d] = live ? float(qbase[(size_t)(t0 + r) * krow + dc + d]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < 32; kk += 8) {
        simdgroup_float8x8 ak, aq;
        simdgroup_load(ak, Ksh, SLD, ulong2(kk, 8 * sg), false);
        simdgroup_load(aq, Qsh, SLD, ulong2(kk, 8 * sg), false);
        for (int jf = 0; jf < C / 8; ++jf) {
            simdgroup_float8x8 bfr;
            simdgroup_load(bfr, Ksh, SLD, ulong2(kk, 8 * jf), true);
            simdgroup_multiply_accumulate(Araw[jf], ak, bfr, Araw[jf]);
            simdgroup_multiply_accumulate(Praw[jf], aq, bfr, Praw[jf]);
        }
    }
}

threadgroup_barrier(mem_flags::mem_threadgroup);
for (int jf = 0; jf < C / 8; ++jf) {
    simdgroup_store(Araw[jf], M, C, ulong2(8 * jf, 8 * sg), false);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int p = tid; p < C * C; p += 256) {
    const int i = p / C, j = p % C;
    M[p] = (j < i) ? bet[i] * M[p] * exp(lg[i] - lg[j]) : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// (I + A) W = Kg and (I + A) U = Vb by forward substitution: thread col < Dk
// owns a column of W, thread col - Dk < Dv a column of U. The 64-entry column
// lives in registers; A[i][j] is a threadgroup broadcast.
const int col = tid;
const bool do_w = col < Dk;
const bool do_u = (col >= Dk) && (col < Dk + Dv);
const size_t tile = ((size_t)(b * Hv + hv) * NC + cc) * C;
// The column and its device writes stay in ONE loop: a `float x[C]` that
// outlives the solve block reads back as zeros here.
float x[C];
if (do_w || do_u) {
    device MidT* Wd = W + tile * Dk;
    device MidT* Kdd = Kd + tile * Dk;
    device MidT* Ud = U + tile * Dv;
    const float lgend = lg[C - 1];
    for (int i = 0; i < C; ++i) {
        float acc = 0.0f;
        if (i < tt) {
            acc = do_w ? bet[i] * exp(lg[i]) * float(kbase[(size_t)(t0 + i) * krow + col])
                       : bet[i] * float(vbase[(size_t)(t0 + i) * vrow + (col - Dk)]);
        }
        for (int j = 0; j < i; ++j) acc -= M[i * C + j] * x[j];
        x[i] = acc;
        if (do_w) {
            Wd[i * Dk + col] = static_cast<MidT>(acc);
            const float kv = (i < tt) ? exp(lgend - lg[i]) * float(kbase[(size_t)(t0 + i) * krow + col]) : 0.0f;
            Kdd[i * Dk + col] = static_cast<MidT>(kv);
        } else {
            Ud[i * Dv + (col - Dk)] = static_cast<MidT>(acc);
        }
    }
}

// W and U are read back as MMA operands below, so the device writes above must
// be visible to the whole threadgroup before the arena is recycled for P.
threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
for (int jf = 0; jf < C / 8; ++jf) {
    simdgroup_store(Praw[jf], M, C, ulong2(8 * jf, 8 * sg), false);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int p = tid; p < C * C; p += 256) {
    const int i = p / C, j = p % C;
    M[p] = (j <= i) ? M[p] * exp(lg[i] - lg[j]) : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const short2 co = gdn_wy_coord(lane);
const int row = 8 * sg + co.y;
{
    simdgroup_float8x8 acc[Dk / 8];
    for (int n = 0; n < Dk / 8; ++n) acc[n] = gdn_wy_zero();
    const device MidT* Wd = W + tile * Dk;
    for (int kk = 0; kk < C; kk += 8) {
        simdgroup_float8x8 a;
        simdgroup_load(a, M, C, ulong2(kk, 8 * sg), false);
        for (int n = 0; n < Dk / 8; ++n) {
            simdgroup_multiply_accumulate(acc[n], a, gdn_wy_load<MidT>(Wd, Dk, ulong2(8 * n, kk)), acc[n]);
        }
    }
    device MidT* Qe = Qeff + tile * Dk;
    const float gi = exp(lg[row]);
    for (int n = 0; n < Dk / 8; ++n) {
        const int c0 = 8 * n + co.x;
        thread float2& e = gdn_wy_el(acc[n]);
        const float q0 = (row < tt) ? gi * float(qbase[(size_t)(t0 + row) * krow + c0]) : 0.0f;
        const float q1 = (row < tt) ? gi * float(qbase[(size_t)(t0 + row) * krow + c0 + 1]) : 0.0f;
        Qe[row * Dk + c0] = static_cast<MidT>(q0 - e[0]);
        Qe[row * Dk + c0 + 1] = static_cast<MidT>(q1 - e[1]);
    }
}
{
    simdgroup_float8x8 acc[Dv / 8];
    for (int n = 0; n < Dv / 8; ++n) acc[n] = gdn_wy_zero();
    const device MidT* Ud = U + tile * Dv;
    for (int kk = 0; kk < C; kk += 8) {
        simdgroup_float8x8 a;
        simdgroup_load(a, M, C, ulong2(kk, 8 * sg), false);
        for (int n = 0; n < Dv / 8; ++n) {
            simdgroup_multiply_accumulate(acc[n], a, gdn_wy_load<MidT>(Ud, Dv, ulong2(8 * n, kk)), acc[n]);
        }
    }
    device MidT* Yl = Yloc + tile * Dv;
    for (int n = 0; n < Dv / 8; ++n) gdn_wy_store<MidT>(acc[n], Yl, Dv, ulong2(8 * n, 8 * sg));
}
