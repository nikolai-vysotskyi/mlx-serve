#include "../qsa_pair/bench_common.h"
#include <cstring>
#include <set>
float bf16(float x) {
  uint32_t u;std::memcpy(&u,&x,4);u+=0x7fff+((u>>16)&1);u&=0xffff0000;std::memcpy(&x,&u,4);return x;
}
int main() {
  const int M=8192,H=2560,HC=4,K=HC*H,R=320;
  std::mt19937 rng(9987);
  auto x=random_array({M,HC,H},rng),w=random_array({HC,H},rng);
  auto wo=random_array({M,1,H},rng), wi=random_array({M,HC,1},rng);
  auto iw=random_array({K,HC},rng)*mx::array(.015625f,mx::bfloat16);
  auto makew=[&](int n,int k){
    std::vector<uint32_t> data(size_t(n)*k/8);for(auto& z:data)z=rng();
    return mx::array(data.data(),{n,k/8});
  };
  auto dw=makew(R,K),uw=makew(K,R);
  auto ds=mx::full({R,K/64},.003f,mx::bfloat16),db=mx::full({R,K/64},-.0225f,mx::bfloat16);
  auto us=mx::full({K,R/64},.01f,mx::bfloat16),ub=mx::full({K,R/64},-.075f,mx::bfloat16);
  auto ones=mx::ones({H},mx::bfloat16),eps=mx::array(1e-6f);
  std::vector<float> values(65536);for(uint32_t i=0;i<65536;i++){uint32_t b=i<<16;std::memcpy(&values[i],&b,4);}
  auto tab=mx::sigmoid(mx::astype(mx::array(values.data(),{65536}),mx::bfloat16));
  mx::eval(wo,wi,x,w,iw,dw,uw,ds,db,us,ub,ones,eps,tab);
  auto nk=mx::fast::metal_kernel("hc_norm_inject",{"x","w","iw","eps","wo","wi"},{"normed","ipart","stream"},read_file("research/hc_prefill/norm_inject.metal"),"",true);
  auto mk=mx::fast::metal_kernel("hc_mix_direct",{"up","normed","sigtab"},{"out"},read_file("research/hc_prefill/mix.metal"),"",true);
  auto silu=mx::compile([](const std::vector<mx::array>& a){return std::vector<mx::array>{a[0]*mx::sigmoid(a[0])};});
  auto mix=mx::compile([](const std::vector<mx::array>& a){return std::vector<mx::array>{mx::mean(mx::sigmoid(a[0])*a[1],-2)};});
  auto inj=mx::compile([](const std::vector<mx::array>& a){return std::vector<mx::array>{mx::sigmoid(a[0])*mx::array(2.f,mx::bfloat16)};});
  auto write=mx::compile([](const std::vector<mx::array>& a){return std::vector<mx::array>{a[0]+a[1]*a[2]};});
  auto run=[&](int mode){
    mx::array stream=x,n=x,raw=mx::zeros({M,HC},mx::bfloat16);
    if(mode&1) {
      auto p=nk({x,w,iw,eps,wo,wi},{{M,HC,H},{M,HC,HC},{M,HC,H}},{mx::bfloat16,mx::float32,mx::bfloat16},{M*HC*(H/4),1,1},{H/4,1,1},{{"T",mx::bfloat16},{"H",H},{"HC",HC},{"WR",true}},{},false,mx::Device::gpu);
      n=p[0];stream=p[2];raw=mx::astype(mx::sum(p[1],1),mx::bfloat16);
    } else {stream=write({x,wo,wi})[0];n=mx::fast::rms_norm(stream,ones,1e-6)*w;raw=mx::matmul(mx::reshape(n,{M,K}),iw);}
    auto down=mx::quantized_matmul(mx::reshape(n,{M,K}),dw,ds,db,true,64,4,"affine");
    auto act=silu({down})[0];
    auto up=mx::reshape(mx::quantized_matmul(act,uw,us,ub,true,64,4,"affine"),{M,HC,H});
    auto mixed=(mode&2)?mk({up,n,tab},{{M,H}},{mx::bfloat16},{M*H,1,1},{256,1,1},{{"T",mx::bfloat16},{"M",M},{"H",H}},{},false,mx::Device::gpu)[0]:mix({up,n})[0];
    return std::vector<mx::array>{mixed,inj({raw})[0],n,raw,stream};
  };
  auto ref=run(0);mx::eval(ref);
  for(int mode:{0,3,0}) {
    auto y=run(mode);mx::eval(y);
    std::cout<<"{\"mode\":"<<mode<<",\"diffs\":[";
    for(int j=0;j<5;j++)std::cout<<(j?",":"")<<mx::max(mx::abs(mx::astype(y[j],mx::float32)-mx::astype(ref[j],mx::float32))).item<float>();
    std::cout<<"]";
    if(mode) {
      auto nc=mx::astype(y[2],mx::float32),wc=mx::astype(iw,mx::float32);
      auto nr=mx::astype(y[3],mx::float32),sr=mx::astype(ref[3],mx::float32);
      auto ni=mx::astype(y[1],mx::float32),si=mx::astype(ref[1],mx::float32);
      auto tc=mx::astype(tab,mx::float32);mx::eval(nc,wc,nr,sr,ni,si,tc);
      std::set<int> rows{0,1,2,M-1};for(int i=0;i<32;i++)rows.insert(i*M/32);
      int worst=mx::argmax(mx::abs(nr-sr)).item<uint32_t>();rows.insert(worst/4);
      double ne=0,se=0;int wrong_new=0,wrong_stock=0,checked=0;
      for(int row:rows)for(int c=0;c<4;c++) {
        double sum=0;for(int j=0;j<K;j++)sum+=double(nc.data<float>()[row*K+j])*wc.data<float>()[j*4+c];
        float raw=bf16(float(sum));uint32_t bits;std::memcpy(&bits,&raw,4);
        float gate=2*tc.data<float>()[bits>>16];
        ne=std::max(ne,std::abs(double(nr.data<float>()[row*4+c])-sum));
        se=std::max(se,std::abs(double(sr.data<float>()[row*4+c])-sum));
        wrong_new+=ni.data<float>()[row*4+c]!=gate;wrong_stock+=si.data<float>()[row*4+c]!=gate;checked++;
      }
      std::cout<<",\"f64_raw_stock_max\":"<<se<<",\"f64_raw_new_max\":"<<ne<<",\"gate_mismatches_new\":"<<wrong_new<<",\"gate_mismatches_stock\":"<<wrong_stock<<",\"gates_checked\":"<<checked;
    }
    std::vector<double> ts;for(int i=0;i<3;i++) {auto t=std::chrono::steady_clock::now();auto z=run(mode);mx::eval(z[0],z[1],z[4]);ts.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}
    std::sort(ts.begin(),ts.end());std::cout<<",\"ms\":"<<ts[1]<<"}"<<std::endl;
  }
}
