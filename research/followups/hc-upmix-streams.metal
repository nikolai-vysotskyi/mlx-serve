// One SIMD group per HC stream; reduce rounded products in threadgroup memory.
using namespace mlx::steel;
constexpr int BM=32,BH=32,BK=64,LD=72;
const int M=x_shape[0],K=x_shape[1],H=normed_shape[2];
const int tid=thread_index_in_threadgroup,sg=simdgroup_index_in_threadgroup;
const int ym=threadgroup_position_in_grid.y*BM,yn=threadgroup_position_in_grid.x*BH;
threadgroup T shared[(BM+4*BH)*LD];
threadgroup T* As=shared;
threadgroup T* Bs=shared+BM*LD;
NAXTile<float,2,2> D;D.clear();
for(int k=0;k<K;k+=BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=tid;i<BM*BK;i+=128) {
    int row=i/BK,col=i%BK;
    As[row*LD+col]=ym+row<M ? x[(long)(ym+row)*K+k+col] : T(0);
  }
  for(int i=tid;i<4*BH*BK;i+=128) {
    int row=i/BK,col=i%BK;
    Bs[row*LD+col]=w[(long)((row/BH)*H+yn+row%BH)*K+k+col];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  STEEL_PRAGMA_UNROLL
  for(short kk=0;kk<BK;kk+=32) {
    NAXTile<T,2,2> A,B;
    A.template load<T,LD,1>(As+kk);
    B.template load<T,LD,1>(Bs+sg*BH*LD+kk);
    STEEL_PRAGMA_UNROLL
    for(short ik=0;ik<2;ik++) {
      STEEL_PRAGMA_UNROLL
      for(short im=0;im<2;im++)
        BaseNAXFrag::mma(D.frag_at(im,0),D.frag_at(im,1),A.frag_at(im,ik),metal::false_type{},B.frag_at(0,ik),B.frag_at(1,ik),metal::true_type{});
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const short2 coord=BaseNAXFrag::get_coord();
STEEL_PRAGMA_UNROLL
for(short fm=0;fm<2;fm++) {
  STEEL_PRAGMA_UNROLL
  for(short fn=0;fn<2;fn++) {
    auto frag=D.frag_at(fm,fn);
    STEEL_PRAGMA_UNROLL
    for(short j=0;j<8;j++) {
      int row=fm*16+coord.y+(j/4)*8,col=fn*16+coord.x+j%4;
      T up=T(frag[j]);
      T n=ym+row<M?normed[((long)(ym+row)*4+sg)*H+yn+col]:T(0);
      shared[(sg*BM+row)*BH+col]=T(float(sigtab[as_type<ushort>(up)])*float(n));
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(int i=tid;i<BM*BH;i+=128) {
  int row=i/BH,col=i%BH;T value=shared[i];
  STEEL_PRAGMA_UNROLL
  for(int h=1;h<4;h++)value=T(float(shared[h*BM*BH+i])+float(value));
  if(ym+row<M)out[(long)(ym+row)*H+yn+col]=T(float(value)*0.25f);
}
