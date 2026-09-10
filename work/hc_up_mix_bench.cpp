#define main old_qsa_main
#include "qsa_bench.cpp"
#undef main
#include <cstring>
int main() {
 const int M=8192,K=320,H=2560,N=4*H;
 std::mt19937 rng(887);
 auto a=random_array({M,K},rng),norm=random_array({M,4,H},rng);
 std::vector<uint32_t> data(N*K/8);for(auto&v:data)v=rng();
 auto w=mx::array(data.data(),{N,K/8});
 auto sc=mx::abs(random_array({N,K/64},rng))*mx::array(.02f,mx::bfloat16)+mx::array(.002f,mx::bfloat16),bi=random_array({N,K/64},rng);
 std::vector<float> bits(65536);for(uint32_t i=0;i<65536;i++){uint32_t u=i<<16;std::memcpy(&bits[i],&u,4);}
 auto tab=mx::sigmoid(mx::astype(mx::array(bits.data(),{65536}),mx::bfloat16));
 mx::eval(a,norm,w,sc,bi,tab);
 auto mix=mx::compile([](const std::vector<mx::array>& x){return std::vector<mx::array>{mx::mean(mx::sigmoid(x[0])*x[1],-2)};});
 auto kernel=mx::fast::metal_kernel("hc_up_mix",{"a","w","scales","biases","normed","sigtab"},{"out"},read_file("work/hc_up_mix.metal"),read_file("work/grouped_qmm_header.metal"),true);
 auto run=[&](bool fused) {
  if(fused)return kernel({a,w,sc,bi,norm,tab},{{M,H}},{mx::bfloat16},{(H/64)*32,(M/32)*2,2},{32,2,2},{{"T",mx::bfloat16},{"BM",32}},{},false,mx::Device::gpu)[0];
  auto up=mx::quantized_matmul(a,w,sc,bi,true,64,4,"affine");
  return mix({mx::reshape(up,{M,4,H}),norm})[0];
 };
 auto ref=run(false);mx::eval(ref);
 for(bool f:{false,true}) {
  auto y=run(f);mx::eval(y);
  float diff=mx::max(mx::abs(mx::astype(y,mx::float32)-mx::astype(ref,mx::float32))).item<float>();
  std::vector<double> ms;for(int i=0;i<3;i++){auto t=std::chrono::steady_clock::now();auto y=run(f);mx::eval(y);ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}
  std::sort(ms.begin(),ms.end());
  std::cout<<"{\"fused\":"<<f<<",\"median_ms\":"<<ms[1]<<",\"max_diff\":"<<diff<<"}"<<std::endl;
 }
}
