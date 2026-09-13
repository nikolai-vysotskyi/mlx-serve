#include "../qsa_pair/bench_common.h"
#include <cstring>

int main(int argc,char**argv) {
  const int M=argc>1?8192:128,H=2560,K=320,HC=4,N=HC*H;
  std::mt19937 rng(772391);
  auto x=random_array({M,K},rng),normed=random_array({M,HC,H},rng);
  std::vector<uint32_t> codes(N*K/4);for(auto& v:codes)v=rng();
  auto wq=mx::array(codes.data(),{N,K/4});
  auto sc=mx::abs(random_array({N,K/64},rng))*mx::array(.0003f,mx::bfloat16)+mx::array(.0001f,mx::bfloat16);
  auto bi=sc*mx::array(-127.5f,mx::bfloat16);
  if(argc>3) {
    auto arrays=mx::load_safetensors(argv[2]).first;
    std::string prefix=argv[3];
    wq=arrays.at(prefix+".weight");sc=arrays.at(prefix+".scales");bi=arrays.at(prefix+".biases");
    if(wq.shape()!=mx::Shape{N,K/4} || sc.shape()!=mx::Shape{N,K/64} || bi.shape()!=sc.shape() || sc.dtype()!=mx::bfloat16 || bi.dtype()!=mx::bfloat16)throw std::runtime_error("actual weight geometry/dtype unsupported");
    std::cerr<<"actual_weights="<<prefix<<" synthetic_activations=true"<<std::endl;
  }
  std::vector<float> bits(65536);for(uint32_t i=0;i<65536;i++){uint32_t z=i<<16;std::memcpy(&bits[i],&z,4);}
  auto table=mx::sigmoid(mx::astype(mx::array(bits.data(),{65536}),mx::bfloat16));
  auto dequant=[&](){return mx::dequantize(wq,sc,bi,64,8,"affine",{},mx::bfloat16);};
  auto interleave=[&](mx::array w){return mx::contiguous(mx::reshape(mx::transpose(mx::reshape(w,{HC,H,K}),{1,0,2}),{N,K}));};
  auto cached=dequant(),packed=interleave(cached);mx::eval(x,normed,wq,sc,bi,table,cached,packed);
  auto mix=mx::fast::metal_kernel("hc_upmix_reference",{"up","normed","sigtab"},{"out"},read_file("src/kernels/hc_prefill_mix.metal"));
  auto fused=mx::fast::metal_kernel("hc_upmix_nax",{"x","w","normed","sigtab"},{"out"},read_file("research/followups/hc-upmix-nax.metal"),read_file("src/kernels/qsa_nax_header.metal"));
  auto streams=mx::fast::metal_kernel("hc_upmix_streams",{"x","w","normed","sigtab"},{"out"},read_file("research/followups/hc-upmix-streams.metal"),read_file("src/kernels/qsa_nax_header.metal"));
  auto mpp=mx::fast::metal_kernel("hc_upmix_mpp",{"x","w","normed","sigtab"},{"out"},read_file("research/followups/hc-upmix-mpp.metal"),read_file("src/kernels/qsa_nax_header.metal"));
  auto direct=mx::fast::metal_kernel("hc_upmix_mpp_direct",{"x","w","normed","sigtab"},{"out"},read_file("research/followups/hc-upmix-mpp-direct.metal"),read_file("src/kernels/qsa_nax_header.metal"));
  auto run=[&](int mode) {
    if(mode==8 || mode==9)return direct({x,mode==8?packed:interleave(dequant()),normed,table},{{M,H}},{mx::bfloat16},{(N/64)*32,(M/64)*4,1},{32,4,1},{{"T",mx::bfloat16}},{},false,mx::Device::gpu)[0];
    if(mode==6 || mode==7)return mpp({x,mode==6?packed:interleave(dequant()),normed,table},{{M,H}},{mx::bfloat16},{(N/64)*32,(M/64)*4,1},{32,4,1},{{"T",mx::bfloat16}},{},false,mx::Device::gpu)[0];
    if(mode==4 || mode==5) return streams({x,mode==4?cached:dequant(),normed,table},{{M,H}},{mx::bfloat16},{(H/32)*32,((M+31)/32)*4,1},{32,4,1},{{"T",mx::bfloat16}},{},false,mx::Device::gpu)[0];
    if(mode==2 || mode==3) {
      auto w=mode==2?packed:interleave(dequant());
      return fused({x,w,normed,table},{{M,H}},{mx::bfloat16},{(N/64)*32,((M+63)/64)*4,1},{32,4,1},{{"T",mx::bfloat16}},{},false,mx::Device::gpu)[0];
    }
    auto up=mx::matmul(x,mx::transpose(mode==1?cached:dequant()));
    return mix({up,normed,table},{{M,H}},{mx::bfloat16},{M*H,1,1},{256,1,1},{{"T",mx::bfloat16},{"M",M},{"H",H},{"HC",HC}},{},false,mx::Device::gpu)[0];
  };
  auto ref=run(0);mx::eval(ref);
  for(int mode:{0,8,9}) {auto y=run(mode);mx::eval(y);}
  for(int mode:(argc>3?std::vector<int>{0,8,9,8,0}:std::vector<int>{0,6,8,9,8,0})) {
    auto y=run(mode);mx::eval(y);
    float maxdiff=mx::max(mx::abs(mx::astype(y,mx::float32)-mx::astype(ref,mx::float32))).item<float>();
    float fraction=mx::mean(mx::astype(y!=ref,mx::float32)).item<float>();
    std::cout<<"{\"mode\":"<<mode<<",\"M\":"<<M<<",\"H\":"<<H<<",\"max_diff\":"<<maxdiff<<",\"different_fraction\":"<<fraction;
    if(!mx::all(mx::isfinite(y)).item<bool>() || maxdiff!=0){std::cout<<",\"passed\":false}"<<std::endl;return 2;}
    std::vector<double> times;
    for(int i=0;i<3;i++){auto start=std::chrono::steady_clock::now();auto z=run(mode);mx::eval(z);times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());}
    std::sort(times.begin(),times.end());std::cout<<",\"median_ms\":"<<times[1]<<",\"passed\":true}"<<std::endl;
  }
}
