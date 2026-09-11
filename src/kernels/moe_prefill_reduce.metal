// Fused MoE prefill tail: inverse permutation + router-score weight + K-reduce.
// One thread per (token, column vec4); `down` is read once and the unsorted
// [N, H] intermediate is never written.
//
// The accumulation order is load-bearing. MLX's Metal `sum` over a bf16 axis
// accumulates in the INPUT dtype (the fp32 widening lives in the CPU backend),
// and col_reduce_small keeps GROUPS = min(K, 8) partial sums: partial[g] folds
// slots g, g+8, g+16, ... ascending, then partial[0..GROUPS-1] combine in
// order. Reproducing that grouping is what makes this bit-identical to the
// composed chain; a plain fp32 left fold is not.
uint j0 = thread_position_in_grid.x * 4;
uint t = thread_position_in_grid.y;
if (j0 >= uint(H)) return;
constexpr int GROUPS = K < 8 ? K : 8;
T acc0 = T(0.0f), acc1 = T(0.0f), acc2 = T(0.0f), acc3 = T(0.0f);
for (int g = 0; g < GROUPS; ++g) {
    T q0 = T(0.0f), q1 = T(0.0f), q2 = T(0.0f), q3 = T(0.0f);
    for (int k = g; k < K; k += GROUPS) {
        uint n = inv[(size_t)t * K + k];
        T sf = scores[(size_t)t * K + k];
        const device T* dp = down + (size_t)n * H + j0;
        // T-rounded product (mlx_multiply), then a T add in ascending slot order.
        q0 = T(float(dp[0]) * float(sf)) + q0;
        q1 = T(float(dp[1]) * float(sf)) + q1;
        q2 = T(float(dp[2]) * float(sf)) + q2;
        q3 = T(float(dp[3]) * float(sf)) + q3;
    }
    acc0 = (g == 0) ? q0 : q0 + acc0;
    acc1 = (g == 0) ? q1 : q1 + acc1;
    acc2 = (g == 0) ? q2 : q2 + acc2;
    acc3 = (g == 0) ? q3 : q3 + acc3;
}
device T* op = out + (size_t)t * H + j0;
uint lanes = uint(H) - j0;
op[0] = acc0;
if (lanes > 1) op[1] = acc1;
if (lanes > 2) op[2] = acc2;
if (lanes > 3) op[3] = acc3;
