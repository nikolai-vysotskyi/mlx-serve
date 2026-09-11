// Shared fragment helpers for the GatedDeltaNet chunkwise (WY) kernels.
// MLX's custom-kernel prelude pulls in metal_stdlib but not the simdgroup
// matrix types, so they come in here.
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

// The 2 elements an 8x8 fragment keeps in this lane. simdgroup_matrix declares
// its storage as vec<T,64> (an opaque compiler type); only the first 2 slots
// are real, which is how MLX steel reads and writes fragments as well.
inline thread float2& gdn_wy_el(thread simdgroup_float8x8& m) {
    return reinterpret_cast<thread float2&>(m.thread_elements());
}

// Lane -> (col, row) of an 8x8 fragment; mirrors steel's BaseMMAFrag<T,8,8>.
// The lane holds (row, col) and (row, col + 1).
inline short2 gdn_wy_coord(ushort lane) {
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    return short2(fn, fm);
}

// simdgroup_multiply_accumulate wants one element type for all four operands,
// so narrow tiles are widened on load and narrowed on store.
template <typename E>
inline simdgroup_float8x8 gdn_wy_load(const device E* src, ulong ld, ulong2 org) {
    simdgroup_matrix<E, 8, 8> m;
    simdgroup_load(m, src, ld, org, false);
    simdgroup_float8x8 f;
    thread auto& s = reinterpret_cast<thread vec<E, 2>&>(m.thread_elements());
    thread float2& d = gdn_wy_el(f);
    d[0] = static_cast<float>(s[0]);
    d[1] = static_cast<float>(s[1]);
    return f;
}

template <typename E>
inline void gdn_wy_store(simdgroup_float8x8 f, device E* dst, ulong ld, ulong2 org) {
    simdgroup_matrix<E, 8, 8> m;
    thread float2& s = gdn_wy_el(f);
    thread auto& d = reinterpret_cast<thread vec<E, 2>&>(m.thread_elements());
    d[0] = static_cast<E>(s[0]);
    d[1] = static_cast<E>(s[1]);
    simdgroup_store(m, dst, ld, org, false);
}

inline simdgroup_float8x8 gdn_wy_zero() {
    return make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}
