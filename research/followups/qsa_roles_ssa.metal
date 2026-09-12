using namespace mlx::steel;
static_assert(NSG == G*3 && BK == 32, "one score and two value SIMD groups per query");
constexpr int BD=256,LD=264,TK=2,TDH=8;
const int qL=q_shape[2],kL=k_shape[2],Hq=q_shape[1],Hk=k_shape[1],gqa=Hq/Hk;
const int gi=threadgroup_position_in_grid.x,hk=threadgroup_position_in_grid.y,bb=threadgroup_position_in_grid.z;
const ushort sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
const int qi=sg/3,role=sg%3,s=gi*G+qi,p=kL-qL+s;
const int tid=thread_index_in_threadgroup,NT=tilepos_shape[1],NG=(qL+G-1)/G;
const device int* poslist=tilepos+((long)bb*NG+gi)*NT*BK;
const device int* masks=tilemask+((long)bb*NG+gi)*NT;
const device T* Kp=k+bb*k_strides[0]+hk*k_strides[1];
const device T* Vp=v+bb*v_strides[0]+hk*v_strides[1];
const device T* Qp=q+bb*q_strides[0]+(long)(hk*gqa)*q_strides[1]+(long)s*q_strides[2];
threadgroup T KV[BK*LD];
threadgroup T probs[G*1024];
threadgroup float factors[G*16],denoms[G*16];
NAXTile<float,1,8> bank;
if(role==0) {
  STEEL_PRAGMA_UNROLL
  for(short d=0;d<16;d++) {
    NAXTile<T,1,1> part;
    part.load_rows(Qp+d*16,int(q_strides[1]),short(s<qL?gqa:0));
    STEEL_PRAGMA_UNROLL
    for(short j=0;j<4;j++)
      bank.frag_at(0,d/2)[(d%2)*4+j]=as_type<float>(metal::vec<T,2>(part.frag_at(0,0)[j*2],part.frag_at(0,0)[j*2+1]));
  }
} else bank.clear();
const short2 coord=BaseNAXFrag::get_coord();
float2 max_score=float2(-3e38f),sum_score=float2(0.0f);
const float scale=scl[0]*1.44269504088896340736f;
for(int it=0;it<NT;it++) {
  const int mask=masks[it];
  if(mask==0)break;
  const bool active=s<qL && ((mask>>qi)&1);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=tid;i<BK*32;i+=NSG*32) {
    int r=i>>5,c=i&31,pos=poslist[it*BK+r];
    uint4 z=uint4(0);
    if(pos>=0 && pos<kL)z=*((const device uint4*)(Kp+(long)pos*k_strides[2])+c);
    *((threadgroup uint4*)(KV+r*LD)+c)=z;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(role==0) {
    NAXTile<float,1,2> S;S.clear();
    if(active) {
      STEEL_PRAGMA_UNROLL
      for(short d=0;d<16;d++) {
        NAXTile<T,2,1> K;
        K.template load<T,LD,1>(KV+d*16);
        NAXTile<T,1,1> Q;
        STEEL_PRAGMA_UNROLL
        for(short j=0;j<4;j++) {
          auto pair=as_type<metal::vec<T,2>>(bank.frag_at(0,d/2)[(d%2)*4+j]);
          Q.frag_at(0,0)[2*j]=pair.x;Q.frag_at(0,0)[2*j+1]=pair.y;
        }
        BaseNAXFrag::mma(S.frag_at(0,0),S.frag_at(0,1),Q.frag_at(0,0),
          metal::false_type{},K.frag_at(0,0),K.frag_at(1,0),metal::true_type{});
      }
    }
    STEEL_PRAGMA_UNROLL
    for(short ik=0;ik<2;ik++) {
      STEEL_PRAGMA_UNROLL
      for(short j=0;j<8;j++) {
        const int pos=poslist[it*BK+ik*16+coord.x+j%4];
        S.frag_at(0,ik)[j]*=scale;
        if(!active || pos<0 || pos>p)S.frag_at(0,ik)[j]=-INFINITY;
      }
    }
    float2 new_max=max_score;
    S.template row_reduce<QsaMax>(new_max);
    S.template row_bin_op<QsaExpSub>(new_max);
    float2 factor=metal::exp2(max_score-new_max);
    max_score=new_max;sum_score*=factor;
    S.template row_reduce<QsaSum>(sum_score);
    if(coord.x==0) {
      factors[qi*16+coord.y]=factor.x;factors[qi*16+coord.y+8]=factor.y;
      denoms[qi*16+coord.y]=sum_score.x;denoms[qi*16+coord.y+8]=sum_score.y;
    }
    STEEL_PRAGMA_UNROLL
    for(short ik=0;ik<2;ik++) {
      STEEL_PRAGMA_UNROLL
      for(short j=0;j<8;j++) {
        int at=qi*1024+(coord.y+j/4*8)*32+ik*16+coord.x+j%4;
        T hi=T(S.frag_at(0,ik)[j]);
        probs[at]=hi;probs[at+512]=T(S.frag_at(0,ik)[j]-float(hi));
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=tid;i<BK*32;i+=NSG*32) {
    int r=i>>5,c=i&31,pos=poslist[it*BK+r];
    uint4 z=uint4(0);
    if(pos>=0 && pos<kL)z=*((const device uint4*)(Vp+(long)pos*v_strides[2])+c);
    *((threadgroup uint4*)(KV+r*LD)+c)=z;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(role!=0) {
    float2 factor=float2(factors[qi*16+coord.y],factors[qi*16+coord.y+8]);
    bank.template row_bin_op<QsaMul>(factor);
    if(active) {
      STEEL_PRAGMA_UNROLL
      for(short ik=0;ik<2;ik++) {
        NAXTile<T,1,1> hi,lo;
        hi.template load<T,32,1>(probs+qi*1024+ik*16);
        lo.template load<T,32,1>(probs+qi*1024+512+ik*16);
        STEEL_PRAGMA_UNROLL
        for(short d=0;d<8;d+=2) {
          NAXTile<T,1,2> V;
          V.template load<T,LD,1>(KV+ik*16*LD+(role-1)*128+d*16);
          BaseNAXFrag::mma(bank.frag_at(0,d),bank.frag_at(0,d+1),hi.frag_at(0,0),
            metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
          BaseNAXFrag::mma(bank.frag_at(0,d),bank.frag_at(0,d+1),lo.frag_at(0,0),
            metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
        }
      }
    }
  }
}
if(role!=0) {
  float2 inv=1.0f/float2(denoms[qi*16+coord.y],denoms[qi*16+coord.y+8]);
  bank.template row_bin_op<QsaMul>(inv);
  device T* Op=out+(((long)bb*Hq+hk*gqa)*qL+s)*BD+(role-1)*128;
  bank.store_rows(Op,qL*BD,short(s<qL?gqa:0));
}
