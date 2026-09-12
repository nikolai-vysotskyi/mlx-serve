#include "../qsa_pair/bench_common.h"
int main(int argc,char** argv) {
  const int E=argc>1?512:1,M=argc>1?160:32,K=argc>1?2560:256,N=argc>1?640:64;
  std::mt19937 rng(9981);
  auto x=random_array({E*M,1,K},rng);
  std::vector<uint32_t> data(size_t(E)*N*K/8);for(auto& z:data)z=rng();
  auto w=mx::array(data.data(),{E,N,K/8});data.clear();data.shrink_to_fit();
  auto sc=mx::abs(random_array({E,N,K/64},rng))*mx::array(.02f,mx::bfloat16)+mx::array(.002f,mx::bfloat16);
  auto bi=random_array({E,N,K/64},rng);
  std::vector<uint32_t> ids(E*M);for(int i=0;i<E*M;i++)ids[i]=i/M;
  auto ind=mx::array(ids.data(),{E*M});mx::eval(x,w,sc,bi,ind);
  auto kernel=mx::fast::metal_kernel("native_int4_factor",{"x","w","sc","bi","xs"},{"out"},read_file("research/native_int4/factor.metal"),"#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",true);
  auto run=[&](bool factor){
    if(!factor)return mx::reshape(mx::gather_qmm(x,w,sc,bi,{},ind,true,64,4,"affine",true),{E,M,N});
    auto xs=mx::sum(mx::reshape(mx::astype(x,mx::float32),{E,M,K/64,64}),-1);
    return kernel({x,w,sc,bi,xs},{{E,M,N}},{mx::bfloat16},{((N+63)/64)*32,((M+63)/64)*4,E},{32,4,1},{{"T",mx::bfloat16},{"K",K},{"N",N},{"M",M}},{},false,mx::Device::gpu)[0];
  };
  auto ref=run(false);mx::eval(ref);
  for(bool factor:{false,true,false}) {
    auto y=run(factor);mx::eval(y);
    float diff=mx::max(mx::abs(mx::astype(y,mx::float32)-mx::astype(ref,mx::float32))).item<float>();
    float rel=mx::sqrt(mx::mean(mx::square(mx::astype(y,mx::float32)-mx::astype(ref,mx::float32)))/mx::mean(mx::square(mx::astype(ref,mx::float32)))).item<float>();
    std::vector<double> ts;for(int i=0;i<3;i++){auto t=std::chrono::steady_clock::now();auto z=run(factor);mx::eval(z);ts.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}
    std::sort(ts.begin(),ts.end());std::cout<<"{\"factor\":"<<factor<<",\"E\":"<<E<<",\"M\":"<<M<<",\"K\":"<<K<<",\"N\":"<<N<<",\"ms\":"<<ts[1]<<",\"max_diff\":"<<diff<<",\"rel_rmse\":"<<rel<<"}"<<std::endl;
  }
}
