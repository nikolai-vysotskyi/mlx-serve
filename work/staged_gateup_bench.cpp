#define main benchmark_main
#include "qsa_bench.cpp"
#undef main
#include <cstring>
int main(int argc,char**argv) {
  int M=(argc>1?std::stoi(argv[1]):8192)*10;
  constexpr int E=512,K=2560,N=640;
  std::mt19937 rng(argc>2?std::stoi(argv[2]):41);
  auto base=random_array({M/10,1,K},rng);
  std::vector<uint32_t> lhs(M);for(auto&v:lhs)v=rng()%(M/10);
  auto lhs_idx=mx::array(lhs.data(),{M});
  auto x=mx::take(base,lhs_idx,0);
  std::vector<uint32_t> weights(size_t(E)*N*K/8);for(auto&v:weights)v=rng();
  auto w=mx::array(weights.data(),{E,N,K/8});weights.clear();weights.shrink_to_fit();
  auto wu=mx::bitwise_xor(w,mx::array(uint32_t(0x963c87e1)));
  auto sc=mx::abs(random_array({E,N,K/64},rng))*mx::array(0.02f,mx::bfloat16)+mx::array(0.002f,mx::bfloat16), bi=random_array({E,N,K/64},rng);
  auto usc=mx::abs(random_array({E,N,K/64},rng))*mx::array(0.02f,mx::bfloat16)+mx::array(0.002f,mx::bfloat16), ubi=random_array({E,N,K/64},rng);
  std::vector<uint32_t> ids(M);for(auto&v:ids)v=rng()%E;std::sort(ids.begin(),ids.end());
  auto ind=mx::array(ids.data(),{M});
  std::vector<float> bits(65536);for(uint32_t i=0;i<65536;i++){uint32_t u=i<<16;std::memcpy(&bits[i],&u,4);}
  auto tab=mx::sigmoid(mx::astype(mx::array(bits.data(),{65536}),mx::bfloat16));
  mx::eval(x,w,wu,sc,bi,usc,ubi,ind,tab);
  auto swiglu=mx::fast::metal_kernel("stock_swiglu",{"gate","up","sigtab","N_size"},{"y"},
    "uint i=thread_position_in_grid.x; if(i>=uint(N_size))return; T g=gate[i]; T sig=sigtab[as_type<ushort>(g)]; T act=g*sig; y[i]=act*up[i];");
  auto stock=[&]() {
    auto x=mx::take(base,lhs_idx,0);
    auto g=mx::gather_qmm(x,w,sc,bi,{},ind,true,64,4,"affine",true);
    auto u=mx::gather_qmm(x,wu,usc,ubi,{},ind,true,64,4,"affine",true);
    return swiglu({g,u,tab,mx::array(M*N)},{{M,1,N}},{mx::bfloat16},{M*N,1,1},{256,1,1},{{"T",mx::bfloat16}},{},false,mx::Device::gpu)[0];
  };
  auto ref=stock();mx::eval(ref);
  auto schedule=mx::fast::metal_kernel("moe_schedule",{"indices"},{"tiles"},read_file("work/grouped_qmm_tiles.metal"),"",true);
  auto kernel=mx::fast::metal_kernel("grouped_gateup",{"x_input","w","scales","biases","indices_input","tiles","up_w","up_scales","up_biases","sigtab"},{"out"},read_file("work/grouped_gateup.metal"),read_file("work/grouped_qmm_header.metal"),true);
  auto direct=mx::fast::metal_kernel("staged_gateup",{"x_input","w","scales","biases","indices_input","tiles","up_w","up_scales","up_biases","sigtab","lhs_indices"},{"out"},read_file("work/staged_gateup.metal"),read_file("work/grouped_qmm_header.metal"),true);
  for(int variant:{0,164,1164}) {
    int BM=variant%100,WM=variant%1000==164?4:2,WN=variant==264?4:2;
    auto run=[&]() {
      if(!BM)return stock();
      auto tiles=schedule({ind},{{M/BM+E,3}},{mx::int32},{512,1,1},{512,1,1},{{"BM",BM}},{},false,mx::Device::gpu)[0];
      if(variant==1164) return direct({base,w,sc,bi,ind,tiles,wu,usc,ubi,tab,lhs_idx},{{M,1,N}},{mx::bfloat16},{((N+63)/64)*32,(M/BM+E)*WM,WN},{32,WM,WN},{{"T",mx::bfloat16},{"BM",BM},{"WM",WM},{"WN",WN}},{},false,mx::Device::gpu)[0];
      auto x=mx::take(base,lhs_idx,0);
      return kernel({x,w,sc,bi,ind,tiles,wu,usc,ubi,tab},{{M,1,N}},{mx::bfloat16},{((N+63)/64)*32,(M/BM+E)*WM,WN},{32,WM,WN},{{"T",mx::bfloat16},{"BM",BM},{"WM",WM},{"WN",WN}},{},false,mx::Device::gpu)[0];
    };
    auto y=run();mx::eval(y);
    float diff=mx::max(mx::abs(mx::astype(y,mx::float32)-mx::astype(ref,mx::float32))).item<float>();
    std::vector<double> ms;
    for(int i=0;i<3;i++){auto t=std::chrono::steady_clock::now();auto r=run();mx::eval(r);ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}
    std::sort(ms.begin(),ms.end());
    std::cout<<"{\"tokens\":"<<M/10<<",\"variant\":"<<variant<<",\"BM\":"<<BM<<",\"WM\":"<<WM<<",\"WN\":"<<WN<<",\"median_ms\":"<<ms[1]<<",\"max_diff\":"<<diff<<"}"<<std::endl;
    if(diff!=0)throw std::runtime_error("gate/up numerical parity failed");
  }
}
