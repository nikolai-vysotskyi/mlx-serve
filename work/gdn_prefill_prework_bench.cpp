#define main old_qsa_main
#include "qsa_bench.cpp"
#undef main
int main(int argc,char**argv) {
 const int S=argc>1?std::stoi(argv[1]):8192, B=1, HK=16,HV=48,D=128,C=(2*HK+HV)*D;
 std::mt19937 rng(661);
 auto x=random_array({B,S,C},rng), state=random_array({B,3,C},rng), w=random_array({C,4,1},rng);
 auto a=random_array({B,S,HV},rng),b=random_array({B,S,HV},rng),A=random_array({HV},rng),dt=random_array({HV},rng);
 auto qs=mx::array(1.0f/D,mx::bfloat16),ks=mx::array(1.0f/std::sqrt(float(D)),mx::bfloat16),ones=mx::ones({D},mx::bfloat16);
 mx::eval(x,state,w,a,b,A,dt,qs,ks,ones);
 auto fast=mx::fast::metal_kernel("gdn_wide_prework",{"qkv","conv_state","conv_w","q_scale","k_scale","b_in","a_in","A_log","dt_bias"},{"q_out","k_out","v_out","conv_out","g_out","beta_out"},read_file("work/gdn_prework_source.metal"),read_file("work/gdn_kernel_header.metal"),true);
 auto run=[&](bool fused) -> std::vector<mx::array> {
   if(fused)return fast({x,state,w,qs,ks,b,a,A,dt},{{B,S,HK,D},{B,S,HK,D},{B,S,HV,D},{B,3,C},{B,S,HV},{B,S,HV}},std::vector<mx::Dtype>(6,mx::bfloat16),{32,B*S,2*HK+HV},{32,1,1},{{"T",mx::bfloat16},{"HK",HK},{"HV",HV},{"DK",D},{"DV",D},{"NKEEP",3},{"C",C},{"S",S},{"QSTRIDE",C},{"QOFF",0},{"BSTRIDE",HV},{"BOFF",0},{"ASTRIDE",HV},{"AOFF",0}},{},false,mx::Device::gpu);
   auto cat=mx::concatenate({state,x},1);
   auto raw=mx::conv1d(cat,w,1,0,1,C);
   auto act=raw*mx::sigmoid(raw);
   auto q=mx::reshape(mx::slice(act,{0,0,0},{B,S,HK*D}),{B,S,HK,D});
   auto k=mx::reshape(mx::slice(act,{0,0,HK*D},{B,S,2*HK*D}),{B,S,HK,D});
   auto v=mx::reshape(mx::slice(act,{0,0,2*HK*D},{B,S,C}),{B,S,HV,D});
   auto g=mx::astype(mx::exp(-mx::exp(mx::astype(A,mx::float32))*mx::log1p(mx::exp(mx::astype(a+dt,mx::float32)))),mx::bfloat16);
   return {mx::fast::rms_norm(q,ones,1e-6f)*qs,mx::fast::rms_norm(k,ones,1e-6f)*ks,v,mx::slice(cat,{0,S,0},{B,S+3,C}),g,mx::sigmoid(b)};
 };
 auto ref=run(false);mx::eval(ref);
 for(bool fused:{false,true}) {
   auto out=run(fused);mx::eval(out);
   for(int i=0;i<6;i++) {
     auto diff=mx::max(mx::abs(mx::astype(out[i],mx::float32)-mx::astype(ref[i],mx::float32))).item<float>();
     std::cout<<"{\"fused\":"<<fused<<",\"output\":"<<i<<",\"max_diff\":"<<diff<<"}\n";
     if(diff!=0)throw std::runtime_error("prework parity failed");
   }
   std::vector<double> ms;
   for(int j=0;j<3;j++){auto t=std::chrono::steady_clock::now();auto y=run(fused);mx::eval(y);ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}
   std::sort(ms.begin(),ms.end());
   std::cout<<"{\"S\":"<<S<<",\"fused\":"<<fused<<",\"median_ms\":"<<ms[1]<<"}"<<std::endl;
 }
}
