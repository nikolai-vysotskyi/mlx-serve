// Dense BF16 HC up projection, with four interleaved output channels per feature.
using namespace mlx::steel;
constexpr int BM=64,BN=64,BK=64,LD=72;
const int M=x_shape[0],K=x_shape[1],H=normed_shape[2];
const int tid=thread_index_in_threadgroup,sg=simdgroup_index_in_threadgroup;
const int ym=threadgroup_position_in_grid.y*BM,yn=threadgroup_position_in_grid.x*BN;
const int tm=(sg/2)*32,tn=(sg%2)*32;
threadgroup T As[BM*LD],Bs[BN*LD];
NAXTile<float,2,2> D;D.clear();
for(int k=0;k<K;k+=BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=tid;i<BM*BK;i+=128) {
    const int row=i/BK,col=i%BK;
    As[row*LD+col]=ym+row<M ? x[(long)(ym+row)*K+k+col] : T(0);
    Bs[row*LD+col]=w[(long)(yn+row)*K+k+col];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  STEEL_PRAGMA_UNROLL
  for(short kk=0;kk<BK;kk+=32) {
    NAXTile<T,2,2> A,B;
    A.template load<T,LD,1>(As+tm*LD+kk);
    B.template load<T,LD,1>(Bs+tn*LD+kk);
    STEEL_PRAGMA_UNROLL
    for(short ik=0;ik<2;ik++) {
      STEEL_PRAGMA_UNROLL
      for(short im=0;im<2;im++)
        BaseNAXFrag::mma(D.frag_at(im,0),D.frag_at(im,1),A.frag_at(im,ik),metal::false_type{},B.frag_at(0,ik),B.frag_at(1,ik),metal::true_type{});
    }
  }
}
const short2 coord=BaseNAXFrag::get_coord();
STEEL_PRAGMA_UNROLL
for(short fm=0;fm<2;fm++) {
  STEEL_PRAGMA_UNROLL
  for(short fn=0;fn<2;fn++) {
    auto frag=D.frag_at(fm,fn);
    STEEL_PRAGMA_UNROLL
    for(short r=0;r<2;r++) {
      const int row=ym+tm+fm*16+coord.y+r*8;
      const int col=(yn+tn+fn*16+coord.x)/4;
      if(row<M) {
        T total=T(0);
        STEEL_PRAGMA_UNROLL
        for(short h=0;h<4;h++) {
          T up=T(frag[r*4+h]);
          T product=T(float(sigtab[as_type<ushort>(up)])*float(normed[((long)row*4+h)*H+col]));
          total=h==0?product:T(float(product)+float(total));
        }
        out[(long)row*H+col]=T(float(total)*0.25f);
      }
    }
  }
}
