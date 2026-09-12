// Paired sparse attention: shared K/V is staged once; private buckets skip the other query.
using namespace mlx::steel;
static_assert(NSG == G * 2 && BK == 32, "two SIMD groups per query");
constexpr int BD = 256, LD = 264, TDH = 8, TK = BK / 16;
const int qL=q_shape[2], kL=k_shape[2], Hq=q_shape[1], Hk=k_shape[1];
const int gqa=Hq/Hk;
const int gi=threadgroup_position_in_grid.x;
const int hk=threadgroup_position_in_grid.y;
const int bb=threadgroup_position_in_grid.z;
const ushort sg=simdgroup_index_in_threadgroup;
const ushort warp=sg % 2;
const int qi=sg/2, s=gi*G+qi;
const ushort lane=thread_index_in_simdgroup;
const int tid=thread_index_in_threadgroup;
const int p=kL-qL+s;
const int NT=tilepos_shape[1], NG=(qL+G-1)/G;
const device int* poslist=tilepos+((long)bb*NG+gi)*NT*BK;
const device int* masks=tilemask+((long)bb*NG+gi)*NT;
const device T* Kp=kp+((long)bb*Hk+hk)*((kL+3)/4)*1024;
const device T* Vp=vp+((long)bb*Hk+hk)*((kL+3)/4)*1024;
const device T* Qp=q+bb*q_strides[0]+(long)(hk*gqa)*q_strides[1]+(long)s*q_strides[2]+warp*128;
threadgroup float exchange[NSG*512];
using ST=NAXTile<float,1,TK>;
using OT=NAXTile<float,1,TDH>;
OT O; O.clear();
NAXTile<T,1,1> Q[TDH];
STEEL_PRAGMA_UNROLL
for (short d=0;d<TDH;++d) Q[d].load_rows(Qp+d*16, int(q_strides[1]), short(s<qL?gqa:0));
const short2 coord=BaseNAXFrag::get_coord();
float2 max_score=float2(-3e38f),sum_score=float2(0.0f);
const float scale=scl[0]*1.44269504088896340736f;
for(int it=0;it<NT;++it) {
  const int mask=masks[it];
  if(mask==0)break;
  const bool active=s<qL && ((mask>>qi)&1);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ST S; S.clear();
  if(active) {
  STEEL_PRAGMA_UNROLL
  for(short ik=0;ik<TK;ik+=2) {
    STEEL_PRAGMA_UNROLL
    for(short d=0;d<TDH;++d) {
      NAXTile<T,2,1> K;
      for(short r=0;r<2;r++)for(short j=0;j<2;j++) {
        int pos=poslist[it*BK+(ik+r)*16+coord.y+j*8];
        uint2 z=uint2(0);
        if(pos>=0 && pos<kL)z=*((const device uint2*)(Kp+(long)(pos/4)*1024+(warp*8+d)*64+(pos%4)*16+coord.x));
        ((thread uint2*)&K.frag_at(r,0))[j]=z;
      }
      BaseNAXFrag::mma(S.frag_at(0,ik),S.frag_at(0,ik+1),Q[d].frag_at(0,0),
        metal::false_type{},K.frag_at(0,0),K.frag_at(1,0),metal::true_type{});
    }
  }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  STEEL_PRAGMA_UNROLL
  for(short i=0;i<8;++i) {
    exchange[sg*512+lane*16+i]=S.frag_at(0,0)[i];
    exchange[sg*512+lane*16+8+i]=S.frag_at(0,1)[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  STEEL_PRAGMA_UNROLL
  for(short ik=0;ik<TK;++ik) {
    STEEL_PRAGMA_UNROLL
    for(short i=0;i<8;++i) {
      S.frag_at(0,ik)[i]+=exchange[(sg^1)*512+lane*16+ik*8+i];
      S.frag_at(0,ik)[i]*=scale;
      const int pos=poslist[it*BK+ik*16+coord.x+i%4];
      if(!active || pos<0 || pos>p)S.frag_at(0,ik)[i]=-INFINITY;
    }
  }
  float2 new_max=max_score;
  S.template row_reduce<QsaMax>(new_max);
  S.template row_bin_op<QsaExpSub>(new_max);
  float2 factor=metal::exp2(max_score-new_max);
  max_score=new_max;
  sum_score*=factor;
  S.template row_reduce<QsaSum>(sum_score);
  O.template row_bin_op<QsaMul>(factor);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(active) {
  STEEL_PRAGMA_UNROLL
  for(short d=0;d<TDH;d+=2) {
    STEEL_PRAGMA_UNROLL
    for(short ik=0;ik<TK;++ik) {
      NAXTile<T,1,2> V;
      for(short c=0;c<2;c++)for(short j=0;j<2;j++) {
        int pos=poslist[it*BK+ik*16+coord.y+j*8];
        uint2 z=uint2(0);
        if(pos>=0 && pos<kL)z=*((const device uint2*)(Vp+(long)(pos/4)*1024+(warp*8+d+c)*64+(pos%4)*16+coord.x));
        ((thread uint2*)&V.frag_at(0,c))[j]=z;
      }
      NAXTile<T,1,1> Shi, Slo;
      for(short j=0;j<8;++j) {
        Shi.frag_at(0,0)[j]=T(S.frag_at(0,ik)[j]);
        float residual=S.frag_at(0,ik)[j]-float(Shi.frag_at(0,0)[j]);
        Slo.frag_at(0,0)[j]=T(residual);
      }
      BaseNAXFrag::mma(O.frag_at(0,d),O.frag_at(0,d+1),Shi.frag_at(0,0),
        metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
      BaseNAXFrag::mma(O.frag_at(0,d),O.frag_at(0,d+1),Slo.frag_at(0,0),
        metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
    }
  }
  }
}
float2 inv=1.0f/sum_score;
O.template row_bin_op<QsaMul>(inv);
device T* Op=out+(((long)bb*Hq+hk*gqa)*qL+s)*BD+warp*128;
O.store_rows(Op,qL*BD,short(s<qL?gqa:0));
