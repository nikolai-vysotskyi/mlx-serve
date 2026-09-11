// Fused hyper-connection group-norm: rms_norm(x, ones[hidden]) * w[hc,hidden]
// in ONE pass. The two stock ops exist only because mlx_fast_rms_norm's
// weight arg is fixed at shape [hidden] and can't carry the real per-stream
// [hc,hidden] weight, so the chain norms with an all-ones vector and applies
// the real weight as a separate multiply. One threadgroup per (b,s,h) row of
// HID elements; the reduction mirrors rms_single_row (rms_norm.metal):
// per-lane float32 partials, simd_sum, a threadgroup tree over simdgroups,
// precise rsqrt — so the fused result differs from the composed chain only
// by reduction order, not by formula.
constexpr int NR = HID / 256;
constexpr uint SIMD_SIZE = 32;

uint row = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint simd_lane = thread_index_in_simdgroup;
uint simd_group = simdgroup_index_in_threadgroup;

threadgroup float local_inv[1];
threadgroup float local_sums[SIMD_SIZE];

const device T* xr = x + (size_t)row * uint(HID) + lid * uint(NR);
uint hc_idx = row % uint(HC);
const device T* wr = w + (size_t)hc_idx * uint(HID) + lid * uint(NR);
device T* yr = out + (size_t)row * uint(HID) + lid * uint(NR);

float xs[NR];
float acc = 0.0f;
for (int i = 0; i < NR; ++i) {
    xs[i] = float(xr[i]);
    acc += xs[i] * xs[i];
}
acc = simd_sum(acc);
if (simd_group == 0) {
    local_sums[simd_lane] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_lane == 0) {
    local_sums[simd_group] = acc;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_group == 0) {
    acc = simd_sum(local_sums[simd_lane]);
    if (simd_lane == 0) {
        local_inv[0] = metal::precise::rsqrt(acc / float(HID) + eps);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float r = local_inv[0];
for (int i = 0; i < NR; ++i) {
    yr[i] = T(float(wr[i]) * (xs[i] * r));
}
