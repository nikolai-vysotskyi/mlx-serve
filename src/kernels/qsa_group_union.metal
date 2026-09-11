// Per-group block union for the grouped QSA prefill gather: one simdgroup per
// (query group, batch) merges the G ascending, INT_MAX-padded `blocks` rows of
// one group into a single ascending, de-duplicated list plus a per-query
// membership bitmask, so the gather can stage each selected K/V tile once for
// all G queries instead of once per query.
//
// Lane t < G walks query s0+t's stream: its `count` selected block ids, then its
// tail block `complete` when that block still holds rows (<= p). Selected ids
// are all < complete, so a lane's stream is strictly ascending and the k-way
// merge below is a plain repeated simd_min. Lanes without a query sit at
// SENTINEL, which never wins the min, so simd_min/simd_shuffle stay uniform.
static_assert(G >= 1 && G <= 8, "member is a uint8 bitmask, one bit per query");
constexpr int SENTINEL = 2147483647;
const int qL = blocks_shape[1];
const int KB = blocks_shape[2];
const int NG = (qL + G - 1) / G;
const int NUMAX = G * KB + 4;
const int gi = int(threadgroup_position_in_grid.x);
const int bb = int(threadgroup_position_in_grid.z);
const ushort lane = ushort(thread_index_in_simdgroup);
if (gi < NG) {
  const int kL = kvlen[0];
  const int s0 = gi * G;
  const long bstep = blocks_strides[2];
  long roff = 0;
  int idx = 0, count = 0, tail = SENTINEL, head = SENTINEL;
  if (lane < ushort(G) && s0 + int(lane) < qL) {
    const int s = s0 + int(lane);
    const int p = kL - qL + s;
    const int complete = (p + 1) / RATIO;
    count = min(complete, KB);
    tail = (p + 1 - complete * RATIO > 0) ? complete : SENTINEL;
    roff = bb * blocks_strides[0] + (long)s * blocks_strides[1];
    head = (count > 0) ? blocks[roff] : tail;
  }
  device int* ub = union_blocks + (long)(bb * NG + gi) * NUMAX;
  device uchar* mbits = member + (long)(bb * NG + gi) * NUMAX;
  int u = 0;
  while (true) {
    const int best = simd_min(head);
    if (best == SENTINEL) { break; }
    uint mask = 0;
    for (ushort t = 0; t < ushort(G); ++t) {
      mask |= uint(simd_shuffle(head, t) == best) << t;
    }
    if (lane == 0 && u < NUMAX) {
      ub[u] = best;
      mbits[u] = uchar(mask);
    }
    if (head == best) {
      ++idx;
      head = (idx < count) ? blocks[roff + (long)idx * bstep]
                           : ((idx == count) ? tail : SENTINEL);
    }
    ++u;
  }
  for (int i = u + int(lane); i < NUMAX; i += 32) {
    ub[i] = SENTINEL;
    mbits[i] = 0;
  }
  if (lane == 0) { union_len[bb * NG + gi] = min(u, NUMAX); }
}
