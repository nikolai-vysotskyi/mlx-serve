using namespace metal;
using namespace mpp::tensor_ops;
const int sg=simdgroup_index_in_threadgroup;
const int n0=threadgroup_position_in_grid.x*64+(sg%2)*32;
const int m0=threadgroup_position_in_grid.y*64+(sg/2)*32;
auto A=tensor<device T,dextents<int,2>,tensor_inline>((device T*)x,dextents<int,2>(K,M));
auto W=tensor<device T,dextents<int,2>,tensor_inline>((device T*)w,dextents<int,2>(K,N));
constexpr auto desc=matmul2d_descriptor(32,32,128,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc,execution_simdgroup> op;
auto total=op.template get_destination_cooperative_tensor<decltype(A),decltype(W),float>();
_Pragma("clang loop unroll(full)")
for(int i=0;i<32;i++)total[i]=0;
for(int k=0;k<K;k+=128) {
  threadgroup_barrier(mem_flags::mem_none);
  if(m0<M && n0<N) {
    auto a=A.template slice<128,32>(k,m0);
    auto w=W.template slice<128,32>(k,n0);
    op.run(a,w,total);
  }
}
if(m0<M && n0<N) {
  auto dst=tensor<device T,dextents<int,2>,tensor_inline>(out,dextents<int,2>(N,M));
  auto result=op.template get_destination_cooperative_tensor<decltype(A),decltype(W),T>();
  _Pragma("clang loop unroll(full)")
  for(int i=0;i<32;i++)result[i]=T(total[i]);
  result.store(dst.template slice<32,32>(n0,m0));
}
