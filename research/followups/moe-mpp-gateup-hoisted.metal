using namespace metal;
using namespace mpp::tensor_ops;
constexpr int BN=128,BK=64,LD=72;
static_assert(BM==64 && WM==2,"fixed 128-thread tile");
const int K=x_rep_shape[2],N=w_shape[1];
const int ti=threadgroup_position_in_grid.y;
const int start=tiles[ti*3],rows=tiles[ti*3+1],block=tiles[ti*3+2];
if(rows==0)return;
const int expert=indices_input[start];
const int sg=simdgroup_index_in_threadgroup,tid=thread_index_in_threadgroup;
const int m0=start+block*BM+(sg/2)*32;
const int n0=threadgroup_position_in_grid.x*BN+(sg%2)*64;
const bool up_lane=((tid/8)&1)!=0;
const long row0=(long)expert*N+threadgroup_position_in_grid.x*64+tid/16;
const device uint* qp=(up_lane?up_w:w)+row0*(K/8)+(tid%8);
const device T* sp=(up_lane?up_scales:scales)+row0*(K/64);
const device T* bp=(up_lane?up_biases:biases)+row0*(K/64);
threadgroup T weights[BN*LD];
auto X=tensor<device T,dextents<int,2>,tensor_inline>((device T*)x_rep,dextents<int,2>(K,x_rep_shape[0]));
auto W=tensor<threadgroup T,dextents<int,2>,tensor_inline>(weights,dextents<int,2>(LD,BN));
constexpr auto desc=matmul2d_descriptor(32,64,64,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc,execution_simdgroup> op;
auto total=op.template get_destination_cooperative_tensor<decltype(X),decltype(W),float>();
_Pragma("clang loop unroll(full)")
for(int i=0;i<64;i++)total[i]=0;
for(int k=0;k<K;k+=BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for(int q=0;q<8;q++) {
    int row=tid/8+q*16,col=(tid%8)*8;
    uint codes=qp[(long)q*K+k/8];
    float sc=float(sp[(long)q*(K/8)+k/64]);
    float bi=float(bp[(long)q*(K/8)+k/64]);
    _Pragma("clang loop unroll(full)")
    for(int j=0;j<8;j++)weights[row*LD+col+j]=T(sc*float((codes>>(4*j))&15)+bi);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto x=X.template slice<64,32>(k,m0);
  auto weight=W.template slice<64,64>(0,(sg%2)*64);
  if(m0<start+rows)op.run(x,weight,total);
}
_Pragma("clang loop unroll(full)")
for(int i=0;i<64;i+=2) {
  auto coord=total.get_multidimensional_index(i),next=total.get_multidimensional_index(i+1);
  const int row=m0+coord[1],col=(n0+coord[0])/2;
  if(row<start+rows) {
    if(next[0]!=coord[0]+1 || next[1]!=coord[1]){out[(long)row*N+col]=T(NAN);return;}
    T gate=T(total[i]),up=T(total[i+1]);
    T act=T(float(gate)*float(sigtab[as_type<ushort>(gate)]));
    out[(long)row*N+col]=T(float(act)*float(up));
  }
}
