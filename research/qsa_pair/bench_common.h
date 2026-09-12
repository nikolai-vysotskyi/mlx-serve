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
