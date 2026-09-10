constexpr int BN=64,BK=64,WM=2,WN=2,SM=BM/WM,SN=BN/WN;
constexpr int TM=SM/16,TN=SN/16,TK=2, BKS=72;
const int M=a_shape[0], H=normed_shape[2], K=a_shape[1];
const int row0=threadgroup_position_in_grid.y*BM, col0=threadgroup_position_in_grid.x*BN;
const int sg=simdgroup_index_in_threadgroup;
const int lane=thread_index_in_simdgroup;
const int tm=(sg/WN)*SM,tn=(sg%WN)*SN;
threadgroup T ws[BN*BKS];
using Loader=QuantizedBlockLoader<T,BN,BK,BKS,true,128,64,4>;
NAXTile<float,TM,TN> sums;
sums.clear();
for(int h=0;h<4;++h) {
  NAXTile<float,TM,TN> acc;acc.clear();
  Loader loader((const device uint8_t*)w+(long)(h*H+col0)*K/2,
    scales+(long)(h*H+col0)*(K/64),biases+(long)(h*H+col0)*(K/64),K,ws,sg,lane);
  for(int k=0;k<K;k+=BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(int kk=0;kk<BK;kk+=32) {
      NAXTile<T,TM,TK> av;
      NAXTile<T,TN,TK> bv;
      av.load(a+(long)(row0+tm)*K+k+kk,K);
      bv.template load<T,BKS,1>(ws+tn*BKS+kk);
      tile_matmad_nax(acc,av,metal::bool_constant<false>{},bv,metal::bool_constant<true>{});
    }
    loader.next();
  }
  for(short i=0;i<TM;++i)for(short j=0;j<TN;++j) {
    for(short e=0;e<acc.kElemsPerFrag;++e) {
      short2 c=BaseNAXFrag::get_coord(e);
      int r=row0+tm+i*16+c.y,col=col0+tn+j*16+c.x;
      T val=T(acc.val_frags[i*TN+j][e]);
      T sig=sigtab[as_type<ushort>(val)];
      T prod=T(float(sig)*float(normed[((long)r*4+h)*H+col]));
      sums.val_frags[i*TN+j][e]+=float(prod);
    }
  }
}
for(short f=0;f<sums.kNumFrags;++f)for(short e=0;e<sums.kElemsPerFrag;++e)
  sums.val_frags[f][e]*=0.25f;
sums.store(out+(long)(row0+tm)*H+col0+tn,H);
