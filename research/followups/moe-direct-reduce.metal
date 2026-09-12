const uint col = thread_position_in_grid.x;
const uint token = thread_position_in_grid.y;
if (col >= D) return;
float acc_f = 0.0f;
T acc_t = T(0.0f);
if (GROUPED) {
    constexpr uint GROUPS = K < 8 ? K : 8;
    for (uint group = 0; group < GROUPS; ++group) {
        T partial = T(0.0f);
        for (uint slot = group; slot < K; slot += GROUPS) {
            const uint original = token * K + slot;
            const T product = down[(size_t)inverse[original] * D + col] * scores[original];
            partial = product + partial;
        }
        acc_t = group == 0 ? partial : partial + acc_t;
    }
    out[(size_t)token * D + col] = acc_t;
    return;
}
for (uint slot = 0; slot < K; ++slot) {
    const uint original = token * K + slot;
    const uint sorted = inverse[original];
    const T product = down[(size_t)sorted * D + col] * scores[original];
    if (BFACC) acc_t = acc_t + product;
    else acc_f += float(product);
}
out[(size_t)token * D + col] = BFACC ? acc_t : T(acc_f);
