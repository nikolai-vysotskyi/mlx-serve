
constexpr int group_size=64, bits=4, BN=64, BK=64;
constexpr bool transpose=true;
const int K=x_input_shape[2], N=w_shape[1];
const int tile=threadgroup_position_in_grid.y;
const int start=tiles[tile*3], M=tiles[tile*3+1], block=tiles[tile*3+2];
if(M==0)return;
const uint3 tid=uint3(threadgroup_position_in_grid.x,block,0);
const uint simd_group_id=simdgroup_index_in_threadgroup;
const uint simd_lane_id=thread_index_in_simdgroup;
const device T* x=x_input;
const device uint32_t* indices=indices_input+start;
device T* y=out+(long)start*N;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Ws[transpose ? BN * BK_padded : BK * BN_padded];
  threadgroup T Us[transpose ? BN * BK_padded : BK * BN_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  auto ul = (const device uint8_t*)up_w;
  // Input rows are gathered directly into NAX fragments.
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  ul += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  up_scales += transpose ? y_col_long * K_g : y_col / group_size;
  up_biases += transpose ? y_col_long * K_g : y_col / group_size;
  scales += transpose ? y_col_long * K_g : y_col / group_size;
  biases += transpose ? y_col_long * K_g : y_col / group_size;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm =
      align_M ? SM : min(SM, short(max(0, (M - (y_row + tm)))));
  const short sgp_sn =
      align_N ? SN : min(SN, short(max(0, (N - (y_col + tn)))));

  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = transpose ? TN : TK;
  constexpr short BC = transpose ? TK : TN;

  using AccumType = float;

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
    const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));
    const bool sg_active = m_hi_lim > m_lo_lim;

    NAXTile<AccumType, TM, TN> Dtile;
    Dtile.clear();
    NAXTile<AccumType, TM, TN> Utile;
    Utile.clear();

    const device T* xn = x + tm * K;

    // Prepare threadgroup loading operations
    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        biases + index * stride_s,
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id);
    thread loader_w_t loader_u(ul + index * stride_w,
        up_scales + index * stride_s, up_biases + index * stride_s,
        transpose ? K : N, Us, simd_group_id, simd_lane_id);

    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
        for (int k = 0; k < K_it; k++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            loader_w.load_unsafe();
            loader_u.load_unsafe();
          } else {
            loader_w.load_safe(
                transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
            loader_u.load_safe(transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            if (sg_active) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

              volatile int compiler_barrier;

              STEEL_PRAGMA_UNROLL
              for (short i=0;i<TM;++i) {
                STEEL_PRAGMA_UNROLL
                for(short j=0;j<TK;++j) {
                  STEEL_PRAGMA_UNROLL
                  for(short e=0;e<Atile.kElemsPerFrag;++e) {
                    short2 c=BaseNAXFrag::get_coord(e);
                    int r=y_row+tm+i*16+c.y;
                    Atile.val_frags[i*TK+j][e] = r<M ?
                      x_input[(long)lhs_indices[start+r]*K+k*BK+kk1+j*16+c.x] : T(0);
                  }
                }
              }

              if constexpr (transpose) {
                Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
              } else {
                Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
              }

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<transpose>{});
              NAXTile<T, BR, BC> Bup;
              if constexpr (transpose) Bup.template load<T, BK_padded, 1>(Us + tn * BK_padded + kk1);
              else Bup.template load<T, BN_padded, 1>(Us + tn + kk1 * BN_padded);
              tile_matmad_nax(Utile, Atile, metal::bool_constant<false>{}, Bup,
                  metal::bool_constant<transpose>{});

              (void)compiler_barrier;
            }
          }

          xn += BK;
          loader_w.next();
          loader_u.next();
        }

        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_w.load_safe(tile_w);
          loader_u.load_safe(tile_w);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            if (sg_active) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

              volatile int compiler_barrier;

              const short psk = min(int(SK), max(0, (BK - kk1)));
              Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

              if constexpr (transpose) {
                Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
              } else {
                Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
              }

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<transpose>{});
              NAXTile<T, BR, BC> Bup;
              if constexpr (transpose) Bup.template load<T, BK_padded, 1>(Us + tn * BK_padded + kk1);
              else Bup.template load<T, BN_padded, 1>(Us + tn + kk1 * BN_padded);
              tile_matmad_nax(Utile, Atile, metal::bool_constant<false>{}, Bup,
                  metal::bool_constant<transpose>{});

              (void)compiler_barrier;
            }
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for(short f=0;f<Dtile.kNumFrags;++f) {
          for(short e=0;e<Dtile.kElemsPerFrag;++e) {
            T gv=T(Dtile.val_frags[f][e]);
            T uv=T(Utile.val_frags[f][e]);
            T sv=sigtab[as_type<ushort>(gv)];
            T act=T(float(gv)*float(sv));
            Dtile.val_frags[f][e]=float(T(float(act)*float(uv)));
          }
        }
        // Store results to device memory
        if constexpr (kAlignedN.value) {
          if (m_lo_lim == 0 && m_hi_lim == SM) {
            Dtile.store(y + tm * N + tn, N);
          } else {
            Dtile.store_slice(
                y + tm * N + tn, N, short2(0, m_lo_lim), short2(SN, m_hi_lim));
          }
        } else {
          Dtile.store_slice(
              y + tm * N + tn,
              N,
              short2(0, m_lo_lim),
              short2(sgp_sn, m_hi_lim));
        }
      });
    });
  }
