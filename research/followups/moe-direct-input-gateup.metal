using namespace mlx::steel;
constexpr int BN=64,BK=64,LD=72,SM=32,SN=32;
const int K=x_input_shape[2],N=w_shape[1];
const int tile=threadgroup_position_in_grid.y;
const int start=tiles[tile*3],M=tiles[tile*3+1],block=tiles[tile*3+2];
if(M==0)return;
const int expert=indices_input[start];
const int tid=thread_index_in_threadgroup;
const int sg=simdgroup_index_in_threadgroup;
const int ym=block*BM,yn=threadgroup_position_in_grid.x*64;
const int tm=(sg/2)*32,tn=(sg%2)*32;
const int valid=min(32,max(0,M-ym-tm));

// Resolve each original token once, outside the 40 K tiles. Invalid rows
// may read token zero because their independent output rows are discarded.
const short2 ac=BaseNAXFrag::get_coord();
long rowbase[4];
STEEL_PRAGMA_UNROLL
for(short ar=0;ar<2;ar++) {
  STEEL_PRAGMA_UNROLL
  for(short jr=0;jr<2;jr++) {
    int rr=ym+tm+ar*16+ac.y+jr*8;
    rowbase[ar*2+jr]=(rr<M)?long(order_input[start+rr]/TOPK)*K:0l;
  }
}
const long wb=(long)expert*N*(K/8),sb=(long)expert*N*(K/64);
threadgroup T Ws[BN*LD];
NAXTile<float,2,2> D;D.clear();
for(int k=0;k<K;k+=64) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int index=tid;index<512;index+=WM*WN*32) {
    int row=index/8,col=index%8*8;
    int rn=(yn+row)/2;
    bool is_up=((yn+row)&1)!=0;
    uint codes=is_up?up_w[wb+(long)rn*(K/8)+(k+col)/8]:w[wb+(long)rn*(K/8)+(k+col)/8];
    float sc=float(is_up?up_scales[sb+(long)rn*(K/64)+k/64]:scales[sb+(long)rn*(K/64)+k/64]);
    float bi=float(is_up?up_biases[sb+(long)rn*(K/64)+k/64]:biases[sb+(long)rn*(K/64)+k/64]);
    STEEL_PRAGMA_UNROLL
    for(short j=0;j<8;j++)Ws[row*LD+col+j]=T(sc*float((codes>>(4*j))&15)+bi);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(disable)")
  for(int kk=0;kk<64;kk+=32) {
    NAXTile<T,2,2> A,B;
    STEEL_PRAGMA_UNROLL
    for(short ar=0;ar<2;ar++) {
      STEEL_PRAGMA_UNROLL
      for(short ak=0;ak<2;ak++) {
        STEEL_PRAGMA_UNROLL
        for(short j=0;j<8;j++) {
          int cc=k+kk+ak*16+ac.x+j%4;
          A.frag_at(ar,ak)[j]=x_input[rowbase[ar*2+j/4]+cc];
        }
      }
    }
    B.template load<T,LD,1>(Ws+tn*LD+kk);
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
      int rm=ym+tm+fm*16+coord.y+r*8;
      STEEL_PRAGMA_UNROLL
      for(short cp=0;cp<2;cp++) {
        int col=(yn+tn+fn*16+coord.x+cp*2)/2;
        T g=T(frag[r*4+cp*2]),u=T(frag[r*4+cp*2+1]);
        T sig=sigtab[as_type<ushort>(g)];
        T act=T(float(g)*float(sig));
        if(rm<M && col<N)out[(long)(start+rm)*N+col]=T(float(act)*float(u));
      }
    }
  }
}
