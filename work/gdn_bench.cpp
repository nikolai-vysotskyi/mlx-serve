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
mx::array random_array(mx::Shape shape, float low, float high, std::mt19937& rng) {
    size_t n = 1; for (int d : shape) n *= d;
    std::vector<float> v(n); std::uniform_real_distribution<float> dist(low, high);
    for (auto& x : v) x = dist(rng);
    return mx::astype(mx::array(v.data(), shape), mx::bfloat16);
}
int main(int argc, char** argv) {
    int T = argc > 1 ? std::stoi(argv[1]) : 4096;
    std::mt19937 rng(1234);
    auto q = random_array({1,T,16,128}, -0.09,0.09,rng);
    auto k = random_array({1,T,16,128}, -0.15,0.15,rng);
    auto v = random_array({1,T,48,128}, -1,1,rng);
    auto g = random_array({1,T,48}, 0.90,0.999,rng);
    auto beta = random_array({1,T,48}, 0,1,rng);
    auto st = random_array({1,48,128,128}, -0.1,0.1,rng);
    std::vector<mx::array> inputs = {q,k,v,g,beta,st,mx::array(T)};
    mx::eval(inputs);
    std::vector<mx::array> reference;
    for (int lanes : {8,4,16,2,32}) {
        auto kernel = mx::fast::metal_kernel("codex_gdn_"+std::to_string(lanes),
            {"q","k","v","g","beta","state_in","T"}, {"y","state_out"},
            read_file("work/gdn_"+std::to_string(lanes)+".metal"));
        auto run = [&]() { return kernel(inputs, {{1,T,48,128},{1,48,128,128}},
            {mx::bfloat16,mx::bfloat16}, {256*(128/(256/lanes)),48,1}, {256,1,1},
            {{"InT",mx::bfloat16},{"StT",mx::bfloat16},{"OutT",mx::bfloat16},
             {"Dk",128},{"Dv",128},{"Hk",16},{"Hv",48}}, {}, false, mx::Device::gpu); };
        auto output=run(); mx::eval(output);
        if(reference.empty()) reference=output;
        float diff_y=mx::max(mx::abs(mx::astype(output[0],mx::float32)-mx::astype(reference[0],mx::float32))).item<float>();
        float diff_s=mx::max(mx::abs(mx::astype(output[1],mx::float32)-mx::astype(reference[1],mx::float32))).item<float>();
        float rmse=mx::sqrt(mx::mean(mx::square(mx::astype(output[0],mx::float32)-mx::astype(reference[0],mx::float32)))).item<float>();
        std::vector<double> ms;
        for (int i=0;i<7;++i) {
            auto t=std::chrono::steady_clock::now();
            auto out=run(); mx::eval(out);
            ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
        }
        std::sort(ms.begin(),ms.end());
        std::cout << "{\"tokens\":"<<T<<",\"lanes\":"<<lanes<<",\"median_ms\":"<<ms[3]
            <<",\"min_ms\":"<<ms[0]<<",\"max_ms\":"<<ms.back()<<",\"max_y\":"<<diff_y
            <<",\"max_state\":"<<diff_s<<",\"rmse_y\":"<<rmse<<"}"<<std::endl;
    }
}
