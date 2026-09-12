using namespace metal;
using namespace mpp::tensor_ops;
const int sg=simdgroup_index_in_threadgroup;
const int n0=threadgroup_position_in_grid.x*64+(sg%2)*32;
const int m0=threadgroup_position_in_grid.y*64+(sg/2)*32;
const int e=threadgroup_position_in_grid.z;
const int G=K/64;
auto A=tensor<device T,dextents<int,2>,tensor_inline>((device T*)x+(long)e*M*K,dextents<int,2>(K,M));
auto W=tensor<device uint4b_format,dextents<int,2>,tensor_inline>((device uchar*)w+(long)e*N*K/2,dextents<int,2>(K,N));
constexpr auto desc=matmul2d_descriptor(32,32,64,false,true,true);
matmul2d<desc,execution_simdgroup> op;
auto total=op.template get_destination_cooperative_tensor<decltype(A),decltype(W),float>();
_Pragma("clang loop unroll(full)")
for(int i=0;i<32;i++)total[i]=0;
for(int g=0;g<G;g++) {
  threadgroup_barrier(mem_flags::mem_none);
  if(m0>=M || n0>=N)continue;
  auto a=A.template slice<64,32>(g*64,m0);
  auto b=W.template slice<64,32>(g*64,n0);
  auto part=op.template get_destination_cooperative_tensor<decltype(A),decltype(W),float>();
  op.run(a,b,part);
  _Pragma("clang loop unroll(full)")
  for(int i=0;i<32;i++) {
    auto c=part.get_multidimensional_index(i);
    float scale=float(sc[((long)e*N+n0+c[0])*G+g]);
    float bias=float(bi[((long)e*N+n0+c[0])*G+g]);
    float asum=xs[((long)e*M+m0+c[1])*G+g];
    total[i]+=part[i]*scale+asum*bias;
  }
}
if(m0<M && n0<N) {
  auto dst=tensor<device T,dextents<int,2>,tensor_inline>(out+(long)e*M*N,dextents<int,2>(N,M));
  auto result=op.template get_destination_cooperative_tensor<decltype(A),decltype(W),T>();
  _Pragma("clang loop unroll(full)")
  for(int i=0;i<32;i++)result[i]=T(total[i]);
  result.store(dst.template slice<32,32>(n0,m0));
}
