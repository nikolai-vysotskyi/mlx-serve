using namespace metal;
using namespace mpp::tensor_ops;
const int M=x_shape[0],K=x_shape[1],H=normed_shape[2],N=H*4;
const int sg=simdgroup_index_in_threadgroup,tid=thread_index_in_threadgroup;
const int n0=threadgroup_position_in_grid.x*64+(sg%2)*32;
const int m0=threadgroup_position_in_grid.y*64+(sg/2)*32;
auto A=tensor<device T,dextents<int,2>,tensor_inline>((device T*)x,dextents<int,2>(K,M));
auto W=tensor<device T,dextents<int,2>,tensor_inline>((device T*)w,dextents<int,2>(K,N));
constexpr auto desc=matmul2d_descriptor(32,32,64,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc,execution_simdgroup> op;
auto total=op.template get_destination_cooperative_tensor<decltype(A),decltype(W),float>();
_Pragma("clang loop unroll(full)")
for(int i=0;i<32;i++)total[i]=0;
for(int k=0;k<K;k+=64) {
  auto a=A.template slice<64,32>(k,m0);
  auto w=W.template slice<64,32>(k,n0);
  op.run(a,w,total);
}
threadgroup T tmp[64*64];
auto dst=tensor<threadgroup T,dextents<int,2>,tensor_inline>(tmp,dextents<int,2>(64,64));
auto result=op.template get_destination_cooperative_tensor<decltype(A),decltype(W),T>();
_Pragma("clang loop unroll(full)")
for(int i=0;i<32;i++)result[i]=T(total[i]);
result.store(dst.template slice<32,32>((sg%2)*32,(sg/2)*32));
threadgroup_barrier(mem_flags::mem_threadgroup);
const int mb=threadgroup_position_in_grid.y*64,nb=threadgroup_position_in_grid.x*16;
for(int i=tid;i<64*16;i+=128) {
  int row=i/16,col=i%16;
  T value=T(0);
  _Pragma("clang loop unroll(full)")
  for(int h=0;h<4;h++) {
    T up=tmp[row*64+col*4+h];
    T product=T(float(sigtab[as_type<ushort>(up)])*float(normed[((long)(mb+row)*4+h)*H+nb+col]));
    value=h==0?product:T(float(product)+float(value));
  }
  out[(long)(mb+row)*H+nb+col]=T(float(value)*0.25f);
}
