#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
namespace mx = mlx::core;
std::string read_file(const std::string& path) {
    std::ifstream f(path); std::stringstream s; s << f.rdbuf();
    if (!f) throw std::runtime_error("Cannot read " + path);
    return s.str();
}
mx::array random_array(mx::Shape shape, std::mt19937& rng) {
    size_t n=1; for(int d:shape)n*=d;
    std::vector<float> v(n); std::uniform_real_distribution<float> dist(-1,1);
    for(auto& x:v)x=dist(rng);
    return mx::astype(mx::array(v.data(),shape),mx::bfloat16);
}
int main(int argc,char** argv) {
    const std::string capture=argc>1?argv[1]:"work/qsa_blocks.bin";
    std::ifstream f(capture,std::ios::binary); int meta[5]; f.read((char*)meta,sizeof(meta));
    int S=meta[0],KB=meta[1],KV=meta[3]+S,RATIO=meta[4];
    std::vector<int> ids(size_t(S)*KB); f.read((char*)ids.data(),ids.size()*sizeof(int));
    if(!f)throw std::runtime_error("Invalid capture");
    auto blocks=mx::array(ids.data(),{1,S,KB});
    std::mt19937 rng(4321);
    auto q=random_array({1,24,S,256},rng);
    auto k=random_array({1,2,KV,256},rng);
    auto v=random_array({1,2,KV,256},rng);
    std::vector<mx::array> inputs={q,k,v,mx::array({0.0625f}),blocks}; mx::eval(inputs);
    std::vector<mx::array> reference;
    for(int variant:{0,3,0}) {
        int BK=32;
        auto prefix=variant==3 ? "work/qsa_nax_reg" : (variant==2 ? "work/qsa_nax" : (variant ? "work/qsa_bf16" : "work/qsa"));
        auto kernel=mx::fast::metal_kernel("codex_qsa_"+std::to_string(variant),{"q","k","v","scl","blocks"},{"out"},
            read_file(std::string(prefix)+"_source.metal"),read_file(std::string(prefix)+"_header.metal"),false);
        auto run=[&]() {return kernel(inputs,{{1,24,S,256}},{mx::bfloat16},
            {S*32,2*2,1},{32,2,1},{{"T",mx::bfloat16},{"NSG",2},{"BK",BK},{"RATIO",RATIO}},
            {},false,mx::Device::gpu);};
        auto out=run();mx::eval(out);if(reference.empty())reference=out;
        auto delta=mx::astype(out[0],mx::float32)-mx::astype(reference[0],mx::float32);
        float diff=mx::max(mx::abs(delta)).item<float>();
        float rmse=mx::sqrt(mx::mean(mx::square(delta))).item<float>();
        std::vector<double> ms;
        for(int i=0;i<9;++i) {
            auto t=std::chrono::steady_clock::now();auto y=run();mx::eval(y);
            ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
        }
        std::sort(ms.begin(),ms.end());
        std::cout<<"{\"variant\":"<<variant<<",\"bk\":"<<BK<<",\"s\":"<<S<<",\"kv\":"<<KV<<",\"median_ms\":"<<ms[4]
            <<",\"min_ms\":"<<ms.front()<<",\"max_ms\":"<<ms.back()<<",\"max_diff\":"<<diff
            <<",\"rmse\":"<<rmse<<"}"<<std::endl;
    }
    return 0;
}
