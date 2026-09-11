// Chunk-local half of the GDN chunkwise (WY) prefill: one threadgroup (256
// threads = 8 simdgroups) per (chunk, hv, b) computes W, U, Kd, Qeff, Yloc
// and the chunk decay Gc for its C = 64 tokens. Everything the inter-chunk
// pass needs, nothing that depends on the incoming state.
constexpr int C = 64;
constexpr int SLD = 36; // staged column-block row pitch (32 columns + padding)
constexpr int TLD = C;  // pitch of the C x C matrix (A, then T, then P)

static_assert(C == 64, "the blocked triangular inverse doubles 8 -> 16 -> 32 -> 64");
static_assert(Dk % 32 == 0, "gdn wy intra stages Dk in 32-column steps");
static_assert(Dv % 32 == 0, "gdn wy intra stages Dv in 32-column steps");

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

// One arena, three lives: the staged K/Q columns of phase 1, then the C x C
// matrix (A -> T, later P) plus one 32-column staging block beside it.
threadgroup float arena[C * TLD + C * SLD];
threadgroup float lg[C];
threadgroup float bet[C];
threadgroup float bg[C]; // beta_i * exp(lg_i)
threadgroup float de[C]; // exp(lg_{C-1} - lg_i)
threadgroup float* Ksh = arena;
threadgroup float* Qsh = arena + C * SLD;
threadgroup float* M = arena;
threadgroup float* Ssh = arena + C * TLD; // 32-column staging beside M
threadgroup float* Z = arena + C * TLD;   // 32x32 scratch of the doubling steps

const device InT* kbase = k + ((size_t)b * T * Hk + hk) * Dk;
const device InT* qbase = q + ((size_t)b * T * Hk + hk) * Dk;
const device InT* vbase = v + ((size_t)b * T * Hv + hv) * Dv;
const size_t krow = (size_t)Hk * Dk;
const size_t vrow = (size_t)Hv * Dv;

const short2 co = gdn_wy_coord(lane);
const int row = 8 * sg + co.y;

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
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int p = tid; p < C; p += 256) {
    bg[p] = bet[p] * exp(lg[p]);
    de[p] = exp(lg[C - 1] - lg[p]);
}

simdgroup_float8x8 Araw[C / 8];
simdgroup_float8x8 Praw[C / 8];
for (int j = 0; j < C / 8; ++j) {
    Araw[j] = gdn_wy_zero();
    Praw[j] = gdn_wy_zero();
}

// A and P are lower triangular, so column blocks past this simdgroup's row
// block are structurally zero and never worth an MMA.
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
            if (jf > int(sg)) continue;
            simdgroup_float8x8 bfr;
            simdgroup_load(bfr, Ksh, SLD, ulong2(kk, 8 * jf), true);
            simdgroup_multiply_accumulate(Araw[jf], ak, bfr, Araw[jf]);
            simdgroup_multiply_accumulate(Praw[jf], aq, bfr, Praw[jf]);
        }
    }
}

threadgroup_barrier(mem_flags::mem_threadgroup);
for (int jf = 0; jf < C / 8; ++jf) {
    if (jf > int(sg)) continue;
    simdgroup_store(Araw[jf], M, TLD, ulong2(8 * jf, 8 * sg), false);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int p = tid; p < C * C; p += 256) {
    const int i = p / C, j = p % C;
    M[p] = (j < i) ? bet[i] * M[p] * exp(lg[i] - lg[j]) : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// (I + A)^-1 in place. A is nilpotent inside each 8x8 diagonal block, so
// N = -A_dd gives T_dd = (I+N)(I+N^2)(I+N^4) in four MMAs; the off-diagonal
// halves then fall out of T21 = -T22 A21 T11 at 16, 32 and 64 rows.
{
    simdgroup_float8x8 N;
    simdgroup_load(N, M, TLD, ulong2(8 * sg, 8 * sg), false);
    {
        thread float2& e = gdn_wy_el(N);
        e[0] = -e[0];
        e[1] = -e[1];
    }
    simdgroup_float8x8 n2 = gdn_wy_zero();
    simdgroup_multiply_accumulate(n2, N, N, n2);
    simdgroup_float8x8 n4 = gdn_wy_zero();
    simdgroup_multiply_accumulate(n4, n2, n2, n4);
    simdgroup_float8x8 p2 = n2;
    {
        thread float2& e = gdn_wy_el(p2);
        if (co.y == co.x) e[0] += 1.0f;
        if (co.y == co.x + 1) e[1] += 1.0f;
    }
    simdgroup_float8x8 x = p2;
    simdgroup_multiply_accumulate(x, N, p2, x);
    simdgroup_float8x8 td = x;
    simdgroup_multiply_accumulate(td, x, n4, td);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(td, M, TLD, ulong2(8 * sg, 8 * sg), false);
}
for (int lev = 0; lev < 3; ++lev) {
    const int nb = 1 << lev;            // 8x8 blocks per triangular half
    const int nsuper = 4 >> lev;        // independent super-blocks at this level
    const int nfrag = nb * nb * nsuper; // output fragments of T21
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int f = int(sg); f < nfrag; f += 8) {
        const int super = f / (nb * nb);
        const int rf = (f % (nb * nb)) / nb;
        const int cf = f % nb;
        const int bb = super * 2 * nb;
        simdgroup_float8x8 acc = gdn_wy_zero();
        for (int kb = 0; kb < nb; ++kb) {
            simdgroup_float8x8 a, bfr;
            simdgroup_load(a, M, TLD, ulong2(8 * (bb + kb), 8 * (bb + nb + rf)), false);
            simdgroup_load(bfr, M, TLD, ulong2(8 * (bb + cf), 8 * (bb + kb)), false);
            simdgroup_multiply_accumulate(acc, a, bfr, acc);
        }
        simdgroup_store(acc, Z, SLD, ulong2(8 * cf, 8 * (super * nb + rf)), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int f = int(sg); f < nfrag; f += 8) {
        const int super = f / (nb * nb);
        const int rf = (f % (nb * nb)) / nb;
        const int cf = f % nb;
        const int bb = super * 2 * nb;
        simdgroup_float8x8 acc = gdn_wy_zero();
        for (int kb = 0; kb < nb; ++kb) {
            simdgroup_float8x8 a, bfr;
            simdgroup_load(a, M, TLD, ulong2(8 * (bb + nb + kb), 8 * (bb + nb + rf)), false);
            simdgroup_load(bfr, Z, SLD, ulong2(8 * cf, 8 * (super * nb + kb)), false);
            simdgroup_multiply_accumulate(acc, a, bfr, acc);
        }
        {
            thread float2& e = gdn_wy_el(acc);
            e[0] = -e[0];
            e[1] = -e[1];
        }
        simdgroup_store(acc, M, TLD, ulong2(8 * (bb + cf), 8 * (bb + nb + rf)), false);
    }
}

// W = T Kg, U = T Vb and Kd, one staged 32-column block at a time.
const size_t tile = ((size_t)(b * Hv + hv) * NC + cc) * C;
{
    device MidT* Wd = W + tile * Dk;
    device MidT* Kdd = Kd + tile * Dk;
    for (int dc = 0; dc < Dk; dc += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int p = tid; p < C * 32; p += 256) {
            const int r = p >> 5, d = p & 31;
            const float kv = (r < tt) ? float(kbase[(size_t)(t0 + r) * krow + dc + d]) : 0.0f;
            Ssh[r * SLD + d] = bg[r] * kv;
            Kdd[r * Dk + dc + d] = static_cast<MidT>(de[r] * kv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 acc[4];
        for (int n = 0; n < 4; ++n) acc[n] = gdn_wy_zero();
        for (int kk = 0; kk <= 8 * int(sg); kk += 8) {
            simdgroup_float8x8 a;
            simdgroup_load(a, M, TLD, ulong2(kk, 8 * sg), false);
            for (int n = 0; n < 4; ++n) {
                simdgroup_float8x8 bfr;
                simdgroup_load(bfr, Ssh, SLD, ulong2(8 * n, kk), false);
                simdgroup_multiply_accumulate(acc[n], a, bfr, acc[n]);
            }
        }
        for (int n = 0; n < 4; ++n) gdn_wy_store<MidT>(acc[n], Wd, Dk, ulong2(dc + 8 * n, 8 * sg));
    }
}
{
    device MidT* Ud = U + tile * Dv;
    for (int dc = 0; dc < Dv; dc += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int p = tid; p < C * 32; p += 256) {
            const int r = p >> 5, d = p & 31;
            Ssh[r * SLD + d] = (r < tt) ? bet[r] * float(vbase[(size_t)(t0 + r) * vrow + dc + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 acc[4];
        for (int n = 0; n < 4; ++n) acc[n] = gdn_wy_zero();
        for (int kk = 0; kk <= 8 * int(sg); kk += 8) {
            simdgroup_float8x8 a;
            simdgroup_load(a, M, TLD, ulong2(kk, 8 * sg), false);
            for (int n = 0; n < 4; ++n) {
                simdgroup_float8x8 bfr;
                simdgroup_load(bfr, Ssh, SLD, ulong2(8 * n, kk), false);
                simdgroup_multiply_accumulate(acc[n], a, bfr, acc[n]);
            }
        }
        for (int n = 0; n < 4; ++n) gdn_wy_store<MidT>(acc[n], Ud, Dv, ulong2(dc + 8 * n, 8 * sg));
    }
}

// W and U are read back as MMA operands below, so the device writes above must
// be visible to the whole threadgroup before the arena is recycled for P.
threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
for (int jf = 0; jf < C / 8; ++jf) {
    if (jf > int(sg)) continue;
    simdgroup_store(Praw[jf], M, TLD, ulong2(8 * jf, 8 * sg), false);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int p = tid; p < C * C; p += 256) {
    const int i = p / C, j = p % C;
    M[p] = (j <= i) ? M[p] * exp(lg[i] - lg[j]) : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

{
    simdgroup_float8x8 acc[Dk / 8];
    for (int n = 0; n < Dk / 8; ++n) acc[n] = gdn_wy_zero();
    const device MidT* Wd = W + tile * Dk;
    for (int kk = 0; kk <= 8 * int(sg); kk += 8) {
        simdgroup_float8x8 a;
        simdgroup_load(a, M, TLD, ulong2(kk, 8 * sg), false);
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
    for (int kk = 0; kk <= 8 * int(sg); kk += 8) {
        simdgroup_float8x8 a;
        simdgroup_load(a, M, TLD, ulong2(kk, 8 * sg), false);
        for (int n = 0; n < Dv / 8; ++n) {
            simdgroup_multiply_accumulate(acc[n], a, gdn_wy_load<MidT>(Ud, Dv, ulong2(8 * n, kk)), acc[n]);
        }
    }
    device MidT* Yl = Yloc + tile * Dv;
    for (int n = 0; n < Dv / 8; ++n) gdn_wy_store<MidT>(acc[n], Yl, Dv, ulong2(8 * n, 8 * sg));
}
