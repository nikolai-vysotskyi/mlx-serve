#include "../qsa_pair/bench_common.h"
#include <cstring>
int main(int argc,char**argv) {
  const int E=argc>1?512:3,M=argc>1?81920:53,H=2560,I=640;
  std::mt19937 rng(197231);
  auto x=random_array({M,1,H},rng);
  auto weight=[&](int n,int k) {
    std::vector<uint32_t> data(size_t(E)*n*k/8);for(auto& z:data)z=rng();
    auto w=mx::array(data.data(),{E,n,k/8});
    auto s=mx::abs(random_array({E,n,k/64},rng))*mx::array(.004f,mx::bfloat16)+mx::array(.001f,mx::bfloat16);
    auto b=s*mx::array(-7.5f,mx::bfloat16);
    mx::eval(w,s,b);return std::vector<mx::array>{w,s,b};
  };
  auto g=weight(I,H),u=weight(I,H),d=weight(H,I);
  std::vector<uint32_t> ids(M);for(auto& z:ids)z=rng()%E;std::sort(ids.begin(),ids.end());
  auto ind=mx::array(ids.data(),{M});
  std::vector<float> bits(65536);for(uint32_t i=0;i<65536;i++){uint32_t z=i<<16;std::memcpy(&bits[i],&z,4);}
  auto tab=mx::sigmoid(mx::astype(mx::array(bits.data(),{65536}),mx::bfloat16));
  mx::eval(x,ind,tab);
  auto act=mx::fast::metal_kernel("resident_reference_act",{"gate","up","sigtab"},{"y"},
    "uint i=thread_position_in_grid.x; if(i>=uint(gate_shape[0])*640u)return; T a=gate[i];T s=sigtab[as_type<ushort>(a)];T b=T(float(a)*float(s));y[i]=T(float(b)*float(up[i]));");
  auto stock=[&]() {
    auto gg=mx::gather_qmm(x,g[0],g[1],g[2],{},ind,true,64,4,"affine",true);
    auto uu=mx::gather_qmm(x,u[0],u[1],u[2],{},ind,true,64,4,"affine",true);
    auto a=act({gg,uu,tab},{{M,1,I}},{mx::bfloat16},{M*I,1,1},{256,1,1},{{"T",mx::bfloat16}},{},false,mx::Device::gpu)[0];
    return mx::gather_qmm(a,d[0],d[1],d[2],{},ind,true,64,4,"affine",true);
  };
  auto schedule=mx::fast::metal_kernel("resident_schedule",{"indices"},{"tiles"},read_file("research/followups/moe-resident-schedule.metal"),"",true);
  auto kernel=mx::fast::metal_kernel("resident_expert",{"x","gw","gs","gb","uw","us","ub","dw","ds","db","indices","tiles","sigtab"},{"out"},read_file("research/followups/moe-resident.metal"),read_file("src/kernels/qsa_nax_header.metal"),true);
  auto run=[&](bool fused) {
    if(!fused)return stock();
    auto tiles=schedule({ind},{{M/16+512,3}},{mx::int32},{512,1,1},{512,1,1},{{"BM",16}},{},false,mx::Device::gpu)[0];
    return kernel({x,g[0],g[1],g[2],u[0],u[1],u[2],d[0],d[1],d[2],ind,tiles,tab},{{M,1,H}},{mx::bfloat16},{(M/16+512)*1024,1,1},{1024,1,1},{{"T",mx::bfloat16}},{},false,mx::Device::gpu)[0];
  };
  auto ref=stock();mx::eval(ref);
  for(bool fused:{false,true,false}) {
    auto y=run(fused);mx::eval(y);
    auto delta=mx::astype(y,mx::float32)-mx::astype(ref,mx::float32);
    float diff=mx::max(mx::abs(delta)).item<float>();
    float rmse=mx::sqrt(mx::mean(mx::square(delta))).item<float>();
    bool finite=mx::all(mx::isfinite(y)).item<bool>();
    std::cout<<"{\"fused\":"<<fused<<",\"E\":"<<E<<",\"rows\":"<<M<<",\"max_diff\":"<<diff<<",\"rmse\":"<<rmse<<",\"finite\":"<<finite<<std::flush;
    if(diff!=0 || !finite){std::cout<<",\"parity_failed\":true}"<<std::endl;return 2;}
    std::vector<double> ts;for(int i=0;i<3;i++){auto t=std::chrono::steady_clock::now();auto z=run(fused);mx::eval(z);ts.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}
    std::sort(ts.begin(),ts.end());std::cout<<",\"median_ms\":"<<ts[1]<<",\"gpu_schedule_included\":true}"<<std::endl;
  }
}
