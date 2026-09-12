#include "../qsa_pair/bench_common.h"
int main(int argc,char**argv) {
  const int S=argc>1?8192:128,NB=argc>1?16384:700,KB=512;
  std::mt19937 rng(784932);
  auto q=random_array({1,4,S,128},rng),k=random_array({1,NB,128},rng);
  std::vector<int> bb(S);for(int r=0;r<S;r++)bb[r]=(NB*4-S+r+1)/4;
  auto bounds=mx::array(bb.data(),{S});mx::eval(q,k,bounds);
  auto header=read_file("research/followups/qsa-score-header.metal");
  auto stock=mx::fast::metal_kernel("qsa_score_existing",{"Q","POOL"},{"OUT"},read_file("research/followups/qsa-score-stock.metal"),header,false);
  auto reload=mx::fast::metal_kernel("qsa_score_reload",{"Q","POOL"},{"OUT"},read_file("research/followups/qsa-score-reload.metal"),header,false);
  auto select=mx::fast::metal_kernel("qsa_select_existing",{"scores","bounds"},{"ids"},read_file("research/followups/qsa-select-stock.metal"),read_file("research/followups/qsa-select-header.metal"),false);
  auto score=[&](int mode) {
    int nsg=mode==2?4:8,rows_per=mode==2?16:128,ng=(S+rows_per-1)/rows_per,slabs=(NB+31)/32;
    int nsh=std::min(slabs,std::max(slabs/(mode==2?1:32),(256+ng-1)/ng));
    return (mode==1?reload:stock)({q,k},{{1,S,NB}},{mx::float32},{nsh*nsg*32,ng,1},{nsg*32,1,1},{{"NSGVAL",nsg},{"USEH4VAL",mode!=2}},{},false,mx::Device::gpu)[0];
  };
  auto pick=[&](mx::array scores) {return select({scores,bounds},{{1,S,KB}},{mx::int32},{256,S,1},{256,1,1},{{"TGS",256},{"DIGITS",11},{"K",KB}},{},false,mx::Device::gpu)[0];};
  auto ref=score(0);mx::eval(ref);
  auto refids=pick(ref);mx::eval(refids);
  auto median=[&](auto fn){std::vector<double> ts;for(int i=0;i<3;i++){auto t=std::chrono::steady_clock::now();auto z=fn();mx::eval(z);ts.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}std::sort(ts.begin(),ts.end());return ts[1];};
  for(int mode:{0,1,2,0}) {
    auto y=score(mode);mx::eval(y);bool same=mx::all(y==ref).item<bool>();
    auto ids=pick(y);mx::eval(ids);bool picks=mx::all(ids==refids).item<bool>();
    if(!same || !picks){std::cerr<<"score/pick mismatch mode="<<mode<<std::endl;return 2;}
    double scms=median([&](){return score(mode);});
    double allms=median([&](){return pick(score(mode));});
    std::cout<<"{\"mode\":"<<mode<<",\"S\":"<<S<<",\"NB\":"<<NB<<",\"score_ms\":"<<scms<<",\"score_select_ms\":"<<allms<<",\"score_and_picks_bit_exact\":true}"<<std::endl;
  }
}
