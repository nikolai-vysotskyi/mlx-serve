#include "../qsa_pair/bench_common.h"
#include <stdexcept>

// Actual mixed-checkpoint projection weights; deterministic BF16 activation fixture.
// No model generation or aggregate throughput is measured here.
int main(int argc, char** argv) {
  if (argc != 2) throw std::runtime_error("usage: mixed-dense-probe MODEL_DIRECTORY");
  std::mt19937 rng(129831);
  const std::string root=argv[1];
  auto bank=mx::load_safetensors(root+"/model-00001.safetensors",mx::Device::cpu).first;
  auto attn=mx::load_safetensors(root+"/model-00051.safetensors",mx::Device::cpu).first;
  auto get=[](const auto& b,const std::string& p) {
    return std::vector<mx::array>{b.at(p+".weight"),b.at(p+".scales"),b.at(p+".biases")};
  };
  const std::string p="language_model.model.layers.0.linear_attn.";
  auto qkv=get(bank,p+"in_proj_qkv"),z=get(bank,p+"in_proj_z"),out=get(bank,p+"out_proj");
  auto q=get(attn,"language_model.model.layers.3.self_attn.q_proj");
  std::vector<mx::array> in;
  for(int j=0;j<3;j++)in.push_back(mx::concatenate({qkv[j],z[j]},0));
  struct Fixture {const char* name; int k; std::vector<mx::array> w;};
  for(auto f:std::vector<Fixture>{{"gdn_in",2560,in},{"gdn_out",6144,out},{"attn_q",2560,q}}) {
    const int m=8192,n=f.w[0].shape(0);
    const int bits=f.w[0].shape(1)*32/f.k,gs=f.k/f.w[1].shape(1);
    if(bits!=8 || gs!=64)throw std::runtime_error("unexpected checkpoint geometry");
    auto x=random_array({m,f.k},rng);
    auto dense=mx::transpose(mx::dequantize(f.w[0],f.w[1],f.w[2],gs,bits));
    mx::eval(x,dense);
    auto run=[&](int mode) {
      if(mode==0)return mx::quantized_matmul(x,f.w[0],f.w[1],f.w[2],true,gs,bits);
      if(mode==1)return mx::matmul(x,mx::transpose(mx::dequantize(f.w[0],f.w[1],f.w[2],gs,bits)));
      return mx::matmul(x,dense);
    };
    auto ref=run(0);mx::eval(ref);
    for(int mode:{0,1,2,0}) {
      auto y=run(mode);mx::eval(y);
      auto delta=mx::astype(y,mx::float32)-mx::astype(ref,mx::float32);
      float diff=mx::max(mx::abs(delta)).item<float>();
      float neq=mx::mean(mx::astype(y!=ref,mx::float32)).item<float>();
      std::vector<double> ts;
      for(int i=0;i<3;i++){
        auto t=std::chrono::steady_clock::now();auto v=run(mode);mx::eval(v);
        ts.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
      }
      std::sort(ts.begin(),ts.end());
      std::cout<<"{\"shape\":\""<<f.name<<"\",\"M\":"<<m<<",\"K\":"<<f.k<<",\"N\":"<<n
        <<",\"bits\":"<<bits<<",\"group_size\":"<<gs<<",\"mode\":"<<mode<<",\"median_ms\":"<<ts[1]
        <<",\"max_diff\":"<<diff<<",\"different_fraction\":"<<neq<<",\"actual_checkpoint_weights\":true}"<<std::endl;
    }
  }
}
