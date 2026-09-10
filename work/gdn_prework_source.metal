uint lane = thread_position_in_threadgroup.x;
// Grid row = b*S + r over the batch: q/k/v/g/beta/b/a are [B,S,..] flat
// so `row` indexes them directly; the conv taps + next state are per batch.
uint row = threadgroup_position_in_grid.y;
uint b = row / uint(S);
uint r = row - b * uint(S);
uint logical_head = threadgroup_position_in_grid.z;
constexpr uint q_heads = uint(HK);
constexpr uint k_head_base = uint(HK);
constexpr uint v_head_base = 2 * uint(HK);
bool is_q = logical_head < q_heads;
bool is_k = logical_head >= k_head_base && logical_head < v_head_base;
uint head = is_q ? logical_head
           : (is_k ? logical_head - k_head_base : logical_head - v_head_base);
uint channel_base = is_q ? head * uint(DK)
                   : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                           : 2 * uint(HK) * uint(DK) + head * uint(DV));
T activated[4];
float sumsq = 0.0f;
for (uint i = 0; i < 4; ++i) {
    uint channel = channel_base + lane * 4 + i;
    float acc = 0.0f;
    for (uint tap = 0; tap < 4; ++tap) {
        uint input_row = r + tap;
        const T xv = input_row < uint(NKEEP)
            ? conv_state[(b * uint(NKEEP) + input_row) * uint(C) + channel]
            : qkv[(row + tap - uint(NKEEP)) * uint(QSTRIDE) + uint(QOFF) + channel];
        acc += float(xv) * float(conv_w[channel * 4 + tap]);
    }
    const T conv = T(acc);
    // MLX's unary Sigmoid formula (unary_ops.h), verbatim, in the tensor dtype.
    T sy = T(1) / (T(1) + metal::exp(metal::abs(conv)));
    const T act = conv * ((conv < T(0)) ? sy : T(1) - sy);
    activated[i] = act;
    float value = float(act);
    sumsq += value * value;
}
if (is_q || is_k) {
    sumsq = simd_sum(sumsq);
    float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
    const T scale = is_q ? q_scale : k_scale;
    uint out_base = (row * uint(HK) + head) * uint(DK) + lane * 4;
    for (uint i = 0; i < 4; ++i) {
        // ones-weight rms_norm rounding (T(x*inv)), then the separate
        // scalar multiply's rounding — the composed chain's two casts.
        const T rms = T(1) * T(float(activated[i]) * inv);
        const T value = scale * rms;
        if (is_q) {
            q_out[out_base + i] = value;
        } else {
            k_out[out_base + i] = value;
        }
    }
} else {
    uint out_base = (row * uint(HV) + head) * uint(DV) + lane * 4;
    for (uint i = 0; i < 4; ++i) {
        v_out[out_base + i] = activated[i];
    }
    if (lane == 0) {
        // beta = sigmoid(b) (MLX unary formula); g = exp(-exp(A_log) *
        // softplus(a + dt_bias)) with the compiled chain's own casts:
        // bf16 add, f32 precise exp / log1p / exp, bf16 store.
        const T bv = b_in[row * uint(BSTRIDE) + uint(BOFF) + head];
        T by = T(1) / (T(1) + metal::exp(metal::abs(bv)));
        beta_out[row * uint(HV) + head] = (bv < T(0)) ? by : T(1) - by;
        const T apd = T(float(a_in[row * uint(ASTRIDE) + uint(AOFF) + head]) + float(dt_bias[head]));
        float sp = msv_log1p(metal::precise::exp(float(apd)));
        float ea = metal::precise::exp(float(A_log[head]));
        g_out[row * uint(HV) + head] = T(metal::precise::exp(-(ea * sp)));
    }
}
// Next conv state = rows [S, S+NKEEP) of concat(conv_state, qkv).
if (r + uint(NKEEP) >= uint(S)) {
    uint state_row = r + uint(NKEEP) - uint(S);
    uint raw_base = row * uint(QSTRIDE) + uint(QOFF) + channel_base + lane * 4;
    uint state_base = (b * uint(NKEEP) + state_row) * uint(C) + channel_base + lane * 4;
    for (uint i = 0; i < 4; ++i) {
        conv_out[state_base + i] = qkv[raw_base + i];
    }
}
if (r == 0) {
    for (uint rr = 0; rr + uint(S) < uint(NKEEP); ++rr) {
        uint src_base = (b * uint(NKEEP) + rr + uint(S)) * uint(C) + channel_base + lane * 4;
        uint dst_base = (b * uint(NKEEP) + rr) * uint(C) + channel_base + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            conv_out[dst_base + i] = conv_state[src_base + i];
        }
    }
}
