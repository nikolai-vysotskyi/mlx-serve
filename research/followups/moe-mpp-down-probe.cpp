#include "../qsa_pair/bench_common.h"
#include <cstring>
#include <set>
int main(int argc,char**argv) {
  const int S=argc>1?8192:257,E=512,TOPK=10,M=S*TOPK,H=2560,I=640;
  const int BM=argc>3?std::stoi(argv[3]):64,WM=BM/32;
  if(BM!=64 && BM!=192)throw std::runtime_error("supported BM=64 or192");
  const int groups=(M+1023)/1024,capacity=M/BM+E;
  // MLX uses a different quantized-vector arithmetic when M/E < 4.
  // This candidate only targets the sorted matrix prefill dispatch.
  if(M/E<4)throw std::runtime_error("requires sorted matrix prefill dispatch M/E >= 4");
  std::mt19937 rng(1964231);
  auto x=random_array({S,1,H},rng);
  auto weight=[&](int n,int k) {
    std::vector<uint32_t> data(size_t(E)*n*k/8);for(auto& z:data)z=rng();
    auto w=mx::array(data.data(),{E,n,k/8});
    auto sc=mx::abs(random_array({E,n,k/64},rng))*mx::array(.004f,mx::bfloat16)+mx::array(.001f,mx::bfloat16);
    auto bi=sc*mx::array(-7.5f,mx::bfloat16);mx::eval(w,sc,bi);return std::vector<mx::array>{w,sc,bi};
  };
  auto g=weight(I,H),u=weight(I,H),d=weight(H,I);
  std::vector<uint32_t> ids(M);
  for(int t=0;t<S;t++){std::set<uint32_t> used;for(int j=0;j<TOPK;j++){uint32_t e;do{e=rng()%E;}while(used.contains(e));used.insert(e);ids[t*TOPK+j]=e;}}
  auto ind=mx::array(ids.data(),{M});
  if(argc>2 && std::string(argv[2])!="-") {
    ind=mx::astype(mx::load_safetensors(std::string(argv[2])).first.at("indices"),mx::uint32);
    if(ind.size()!=size_t(M))throw std::runtime_error("routing fixture shape differs");
    mx::eval(ind);std::copy_n(ind.data<uint32_t>(),M,ids.begin());
  }
  std::vector<int> hist(E);for(auto e:ids){if(e>=E)throw std::runtime_error("expert ID out of range");hist[e]++;}
  int tile64=0;for(int n:hist)tile64+=(n+63)/64;
  std::cerr<<"routing fixture="<<(argc>2?argv[2]:"uniform random")<<" max_expert_rows="<<*std::max_element(hist.begin(),hist.end())<<" nonempty_experts="<<std::count_if(hist.begin(),hist.end(),[](int n){return n>0;})<<" aligned_64_tiles="<<tile64<<std::endl;
  auto scores=mx::abs(random_array({S,TOPK},rng))*mx::array(.2f,mx::bfloat16);
  std::vector<float> bits(65536);for(uint32_t i=0;i<65536;i++){uint32_t z=i<<16;std::memcpy(&bits[i],&z,4);}
  auto tab=mx::sigmoid(mx::astype(mx::array(bits.data(),{65536}),mx::bfloat16));mx::eval(x,ind,scores,tab);
  auto count=mx::fast::metal_kernel("direct_count",{"indices"},{"counts"},read_file("research/followups/moe-count-sort-count.metal"));
  auto prefix=mx::fast::metal_kernel("direct_prefix",{"counts"},{"offsets","tiles"},read_file("research/followups/moe-count-sort-prefix.metal"));
  auto scatter=mx::fast::metal_kernel("direct_scatter",{"indices","offsets"},{"order","inverse","sorted_ids"},read_file("research/followups/moe-count-sort-scatter.metal"));
  auto pack=[&]() {
    auto c=count({ind},{{groups,E}},{mx::uint32},{groups*512,1,1},{512,1,1},{},{},false,mx::Device::gpu)[0];
    auto p=prefix({c},{{groups,E},{capacity,3}},{mx::uint32,mx::int32},{512,1,1},{512,1,1},{{"BM",BM},{"CAPACITY",capacity}},{},false,mx::Device::gpu);
    auto r=scatter({ind,p[0]},{{M},{M},{M}},{mx::uint32,mx::uint32,mx::uint32},{groups*512,1,1},{512,1,1},{},{},false,mx::Device::gpu);
    r.push_back(p[1]);return r;
  };
  // Verify a bijection, its inverse, ascending expert IDs and tile coverage.
  auto packed=pack();mx::eval(packed);
  auto ar=mx::arange(M,mx::uint32);
  if(!mx::all(mx::take(packed[0],packed[1],0)==ar).item<bool>() ||
     !mx::all(mx::take(ind,packed[0],0)==packed[2]).item<bool>() ||
     !mx::all(packed[2]==mx::sort(ind)).item<bool>())throw std::runtime_error("grouping permutation failed");
  std::vector<int> seen(M);auto tt=packed[3].data<int>();auto si=packed[2].data<uint32_t>();
  for(int t=0;t<capacity;t++)if(tt[t*3+1]){
    int start=tt[t*3],n=tt[t*3+1],b=tt[t*3+2];
    if(start<0 || n<0 || start+n>M)throw std::runtime_error("tile bounds");
    for(int j=b*BM;j<std::min(n,(b+1)*BM);j++){if(si[start+j]!=si[start])throw std::runtime_error("expert crossing");seen[start+j]++;}
  }
  if(!std::all_of(seen.begin(),seen.end(),[](int n){return n==1;}))throw std::runtime_error("tile coverage");
  auto act=mx::fast::metal_kernel("direct_reference_act",{"gate","up","sigtab"},{"y"},
    "uint i=thread_position_in_grid.x;if(i>=uint(gate_shape[0])*640u)return;T g=gate[i];T a=T(float(g)*float(sigtab[as_type<ushort>(g)]));y[i]=T(float(a)*float(up[i]));");
  auto gateup=mx::fast::metal_kernel("direct_gateup",{"x_input","w","scales","biases","indices_input","tiles","up_w","up_scales","up_biases","sigtab","order_input"},{"out"},read_file("research/followups/moe-direct-input-gateup.metal"),read_file("src/kernels/qsa_nax_header.metal"));
  auto wide=mx::fast::metal_kernel("wide_register_gateup",{"x_input","w","scales","biases","indices_input","tiles","up_w","up_scales","up_biases","sigtab","order_input"},{"out"},read_file("research/followups/moe-wide-register-gateup.metal"),read_file("src/kernels/qsa_nax_header.metal"));
  auto reduce=mx::fast::metal_kernel("direct_reduce",{"down","scores","inverse"},{"out"},read_file("research/followups/moe-direct-reduce.metal"));
  auto prod_reduce=mx::fast::metal_kernel("actual_production_reduce",{"down","scores","inverse"},{"out"},read_file("src/kernels/moe_prefill_reduce.metal"));
  auto mpp=mx::fast::metal_kernel("moe_mpp_gateup",{"x_rep","w","scales","biases","indices_input","tiles","up_w","up_scales","up_biases","sigtab"},{"out"},read_file("research/followups/moe-mpp-gateup.metal"),read_file("src/kernels/qsa_nax_header.metal"));
  auto mpp_down=mx::fast::metal_kernel("moe_mpp_down",{"x_rep","w","scales","biases","indices_input","tiles"},{"out"},read_file("research/followups/moe-mpp-down.metal"),read_file("src/kernels/qsa_nax_header.metal"));
  auto run=[&](int mode) {
    std::vector<mx::array> p=mode?pack():std::vector<mx::array>{};
    auto order=mode?p[0]:mx::argsort(ind);
    auto inv=mode?p[1]:mx::argsort(order);
    auto sorted=mode?p[2]:mx::take(ind,order,0);
    mx::array a=mx::zeros({1},mx::bfloat16);
    if(mode>=5) {
      // Padding is part of the original row gather, not a second wide copy.
      auto padded_order=mx::concatenate({order,mx::zeros({BM},mx::uint32)},0);
      auto rep=mx::take(x,mx::floor_divide(padded_order,mx::array(uint32_t(TOPK))),0);
      a=mpp({rep,g[0],g[1],g[2],sorted,p[3],u[0],u[1],u[2],tab},{{M,1,I}},{mx::bfloat16},{(I*2/128)*32,capacity*WM*2,1},{32,WM*2,1},{{"T",mx::bfloat16},{"BM",BM},{"WM",WM}},{},false,mx::Device::gpu)[0];
    }else if(mode==2 || mode==3) {
      const int wm=mode==2?6:2;
      a=(mode==2?gateup:wide)({x,g[0],g[1],g[2],sorted,p[3],u[0],u[1],u[2],tab,order},{{M,1,I}},{mx::bfloat16},{(I*2/64)*32,capacity*wm,2},{32,wm,2},{{"T",mx::bfloat16},{"BM",BM},{"WM",wm},{"WN",2},{"TOPK",TOPK}},{},false,mx::Device::gpu)[0];
    }else {
      auto rep=mx::take(x,mx::floor_divide(order,mx::array(uint32_t(TOPK))),0);
      auto gg=mx::gather_qmm(rep,g[0],g[1],g[2],{},sorted,true,64,4,"affine",true);
      auto uu=mx::gather_qmm(rep,u[0],u[1],u[2],{},sorted,true,64,4,"affine",true);
      a=act({gg,uu,tab},{{M,1,I}},{mx::bfloat16},{M*I,1,1},{256,1,1},{{"T",mx::bfloat16}},{},false,mx::Device::gpu)[0];
    }
    mx::array down=mx::zeros({1},mx::bfloat16);
    if(mode>=6) {
      int bn=mode==6?128:256;
      auto padded=mx::concatenate({a,mx::zeros({64,1,I},mx::bfloat16)},0);
      down=mpp_down({padded,d[0],d[1],d[2],sorted,p[3]},{{M,1,H}},{mx::bfloat16},{(H/bn)*32,capacity*4,1},{32,4,1},{{"T",mx::bfloat16},{"BN",bn}},{},false,mx::Device::gpu)[0];
    }else down=mx::gather_qmm(a,d[0],d[1],d[2],{},sorted,true,64,4,"affine",true);
    if(mode==4)return prod_reduce({down,scores,inv},{{S,H}},{mx::bfloat16},{H,S,1},{256,1,1},{},{},false,mx::Device::gpu)[0];
    if(mode)return reduce({down,scores,inv},{{S,H}},{mx::bfloat16},{H,S,1},{256,1,1},{{"T",mx::bfloat16},{"K",TOPK},{"D",H},{"BFACC",false},{"GROUPED",true}},{},false,mx::Device::gpu)[0];
    return mx::sum(mx::reshape(mx::take(down,inv,0),{S,TOPK,H})*mx::expand_dims(scores,-1),1);
  };
  auto ref=run(0);mx::eval(ref);
  for(int mode:std::vector<int>{1,5,6,7,5,1}) {
    auto y=run(mode);mx::eval(y);auto delta=mx::astype(y,mx::float32)-mx::astype(ref,mx::float32);
    float diff=mx::max(mx::abs(delta)).item<float>();float neq=mx::mean(mx::astype(y!=ref,mx::float32)).item<float>();
    std::cout<<"{\"mode\":"<<mode<<",\"tokens\":"<<S<<",\"BM\":"<<BM<<",\"max_diff\":"<<diff<<",\"different_fraction\":"<<neq<<std::flush;
    if(!mx::all(mx::isfinite(y)).item<bool>() || diff!=0){std::cout<<",\"parity_failed\":true}"<<std::endl;return 2;}
    std::vector<double> ts;for(int i=0;i<3;i++){auto t=std::chrono::steady_clock::now();auto z=run(mode);mx::eval(z);ts.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}
    std::sort(ts.begin(),ts.end());std::cout<<",\"median_ms\":"<<ts[1]<<",\"grouping_gather_expert_reduce_included\":true}"<<std::endl;
  }
}
