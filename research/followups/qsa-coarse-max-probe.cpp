#include "../qsa_pair/bench_common.h"
#include <map>
#include <set>
#include <numeric>
int main(int argc,char** argv) {
  std::ifstream f(argc>1?argv[1]:"research/qsa_pair/blocks-4096.bin",std::ios::binary);
  int meta[5]; f.read((char*)meta,sizeof(meta));
  int S=meta[0],KB=meta[1],KV=meta[3]+S,RATIO=meta[4];
  std::vector<int> ids(size_t(S)*KB);f.read((char*)ids.data(),ids.size()*4);
  if(!f)throw std::runtime_error("capture invalid");
  auto blocks=mx::array(ids.data(),{1,S,KB});
  std::mt19937 rng(92831);
  auto q=random_array({1,24,S,256},rng),k=random_array({1,2,KV,256},rng),v=random_array({1,2,KV,256},rng);
  auto scl=mx::array({.0625f});mx::eval(q,k,v,blocks,scl);
  auto header=read_file("src/kernels/qsa_nax_header.metal");
  auto base=mx::fast::metal_kernel("qsa_base",{"q","k","v","scl","blocks"},{"out"},read_file("src/kernels/qsa_nax.metal"),header,false);
  auto stock=[&](){return base({q,k,v,scl,blocks},{{1,24,S,256}},{mx::bfloat16},{S*32,4,1},{32,2,1},{{"T",mx::bfloat16},{"BK",32},{"NSG",2},{"RATIO",RATIO}},{},false,mx::Device::gpu)[0];};
  auto ref=stock();mx::eval(ref);
  for(int arm:{0,1,2,1,0}) {
    int G=arm ? 2 : 0;
    int group=std::max(G,1),NG=(S+group-1)/group,NT=0;
    long work_tiles=0,active_tiles=0;
    std::vector<std::vector<int>> positions(NG),masks(NG);
    for(int gi=0;G && gi<NG;gi++) {
      std::map<int,int> members;
      for(int t=0;t<G && gi*G+t<S;t++) {
        int s=gi*G+t,p=KV-S+s,complete=(p+1)/RATIO,count=std::min(complete,KB);
        for(int i=0;i<count;i++)members[ids[s*KB+i]]|=1<<t;
        if((p+1)%RATIO)members[complete]|=1<<t;
      }
      std::vector<std::vector<int>> buckets(1<<G);
      for(auto [b,mask]:members)for(int r=0;r<RATIO;r++)buckets[mask].push_back(b*RATIO+r);
      for(int mask=1;mask<(1<<G);mask++) {
        auto& b=buckets[mask];
        for(size_t off=0;off<b.size();off+=32) {
          masks[gi].push_back(mask);work_tiles++;
          active_tiles+=__builtin_popcount(unsigned(mask));
          for(int j=0;j<32;j++)positions[gi].push_back(off+j<b.size()?b[off+j]:-1);
        }
      }
      NT=std::max(NT,int(masks[gi].size()));
      for(int t=0;t<G && gi*G+t<S;t++) {
        int s=gi*G+t,p=KV-S+s,complete=(p+1)/RATIO;
        std::vector<int> expect,got;
        for(int i=0;i<std::min(complete,KB);i++)for(int r=0;r<RATIO;r++)expect.push_back(ids[s*KB+i]*RATIO+r);
        for(int r=complete*RATIO;r<=p;r++)expect.push_back(r);
        for(size_t it=0;it<masks[gi].size();it++)if((masks[gi][it]>>t)&1)
          for(int r=0;r<32;r++){int p0=positions[gi][it*32+r];if(p0>=0&&p0<=p)got.push_back(p0);}
        std::sort(expect.begin(),expect.end());std::sort(got.begin(),got.end());
        if(expect!=got)throw std::runtime_error("coverage mismatch");
      }
    }
    if(G)NT=2*((KB+1)*RATIO+31)/32+3;
    std::vector<int> pp(size_t(NG)*std::max(NT,1)*32,-1),mm(size_t(NG)*std::max(NT,1),0);
    if(G)for(int gi=0;gi<NG;gi++) {
      std::copy(positions[gi].begin(),positions[gi].end(),pp.begin()+size_t(gi)*NT*32);
      std::copy(masks[gi].begin(),masks[gi].end(),mm.begin()+size_t(gi)*NT);
    }
    auto pos=mx::array(pp.data(),{NG,std::max(NT,1),32});
    auto mask=mx::array(mm.data(),{NG,std::max(NT,1)});mx::eval(pos,mask);
    auto fn=mx::fast::metal_kernel("qsa_bucketed",{"q","k","v","scl","blocks","tilepos","tilemask"},{"out"},read_file(arm==2?"research/followups/qsa_pair_coarse_max.metal":"src/kernels/qsa_pair.metal"),header,false);
    auto planner=mx::fast::metal_kernel("qsa_pair_plan",{"blocks","kvlen"},{"tilepos","tilemask"},read_file("src/kernels/qsa_pair_plan.metal"),"",false);
    auto plan=[&](){return planner({blocks,mx::array(KV)},{{NG,NT,32},{NG,NT}},{mx::int32,mx::int32},{NG*32,1,1},{32,1,1},{{"KB",KB},{"RATIO",RATIO},{"NT",NT}},{},false,mx::Device::gpu);};
    if(G){auto got=plan();mx::eval(got);if(!mx::all(got[0]==pos).item<bool>() || !mx::all(got[1]==mask).item<bool>())throw std::runtime_error("GPU plan differs from exact CPU plan");}
    auto run=[&](){
      if(!G)return stock();
      auto packed=plan();
      return fn({q,k,v,scl,blocks,packed[0],packed[1]},{{1,24,S,256}},{mx::bfloat16},{NG*32,2*G*2,1},{32,2*G,1},{{"T",mx::bfloat16},{"BK",32},{"NSG",2*G},{"G",G},{"RATIO",RATIO}},{},false,mx::Device::gpu)[0];
    };
    auto out=run();mx::eval(out);
    float diff=mx::max(mx::abs(mx::astype(out,mx::float32)-mx::astype(ref,mx::float32))).item<float>();
    float rmse=mx::sqrt(mx::mean(mx::square(mx::astype(out,mx::float32)-mx::astype(ref,mx::float32)))).item<float>();
    if(!mx::all(mx::isfinite(out)).item<bool>())throw std::runtime_error("nonfinite output");
    std::vector<double> ts;
    for(int i=0;i<3;i++){auto t=std::chrono::steady_clock::now();auto y=run();mx::eval(y);ts.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}
    std::sort(ts.begin(),ts.end());
    std::cout<<"{\"coarse_max\":"<<(arm==2)<<",\"G\":"<<G<<",\"S\":"<<S<<",\"KV\":"<<KV<<",\"max_tiles\":"<<NT<<",\"tiles\":"<<work_tiles<<",\"active_tiles\":"<<active_tiles<<",\"ms\":"<<ts[1]<<",\"max_diff\":"<<diff<<",\"rmse\":"<<rmse<<",\"plan_included\":true}"<<std::endl;
  }
}
