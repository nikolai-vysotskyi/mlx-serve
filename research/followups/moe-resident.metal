// Research only: one expert / 16 routed rows per threadgroup.
// Keep the BF16 SwiGLU activation on chip through the down projection.
using namespace mlx::steel;
constexpr int H=2560,I=640,BM=16;
const int tile=threadgroup_position_in_grid.x;
const int start=tiles[tile*3],count=tiles[tile*3+1],block=tiles[tile*3+2];
if(count==0)return;
const int row0=block*BM,valid=min(BM,count-row0);
const int expert=indices[start];
const int sg=simdgroup_index_in_threadgroup;
const short2 xy=BaseNAXFrag::get_coord();
const device T* xp=x+(long)(start+row0)*H;
threadgroup T act[BM*I];
if(sg<I/32) {
  NAXTile<float,1,2> gate,up;gate.clear();up.clear();
  for(int k0=0;k0<H;k0+=16) {
    NAXTile<T,1,1> A;
    A.load_rows(xp+k0,H,short(valid));
    NAXTile<T,2,1> G,U;
    STEEL_PRAGMA_UNROLL
    for(short n=0;n<2;n++) {
      STEEL_PRAGMA_UNROLL
      for(short j=0;j<8;j++) {
        int r=sg*32+n*16+xy.y+(j/4)*8,c=k0+xy.x+j%4;
        long wi=((long)expert*I+r)*(H/8)+c/8;
        long si=((long)expert*I+r)*(H/64)+c/64;
        G.frag_at(n,0)[j]=T(float(gs[si])*float((gw[wi]>>(4*(c%8)))&15)+float(gb[si]));
        U.frag_at(n,0)[j]=T(float(us[si])*float((uw[wi]>>(4*(c%8)))&15)+float(ub[si]));
      }
    }
    BaseNAXFrag::mma(gate.frag_at(0,0),gate.frag_at(0,1),A.frag_at(0,0),metal::false_type{},G.frag_at(0,0),G.frag_at(1,0),metal::true_type{});
    BaseNAXFrag::mma(up.frag_at(0,0),up.frag_at(0,1),A.frag_at(0,0),metal::false_type{},U.frag_at(0,0),U.frag_at(1,0),metal::true_type{});
  }
  STEEL_PRAGMA_UNROLL
  for(short n=0;n<2;n++) {
    STEEL_PRAGMA_UNROLL
    for(short j=0;j<8;j++) {
      int r=xy.y+(j/4)*8,c=sg*32+n*16+xy.x+j%4;
      T g=T(gate.frag_at(0,n)[j]),u=T(up.frag_at(0,n)[j]);
      T silu=T(float(g)*float(sigtab[as_type<ushort>(g)]));
      act[r*I+c]=T(float(silu)*float(u));
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
NAXTile<float,1,2> D0,D1,D2;D0.clear();D1.clear();D2.clear();
for(int k0=0;k0<I;k0+=16) {
  NAXTile<T,1,1> A;A.template load<T,I,1>(act+k0);
  #define DOWN_STEP(OFFSET,DST) { \
    int n0=(sg+(OFFSET))*32; \
    if(n0<H) { \
      NAXTile<T,2,1> W; \
      STEEL_PRAGMA_UNROLL \
      for(short n=0;n<2;n++) { \
        STEEL_PRAGMA_UNROLL \
        for(short j=0;j<8;j++) { \
          int r=n0+n*16+xy.y+(j/4)*8,c=k0+xy.x+j%4; \
          long wi=((long)expert*H+r)*(I/8)+c/8; \
          long si=((long)expert*H+r)*(I/64)+c/64; \
          W.frag_at(n,0)[j]=T(float(ds[si])*float((dw[wi]>>(4*(c%8)))&15)+float(db[si])); \
        } \
      } \
      BaseNAXFrag::mma(DST.frag_at(0,0),DST.frag_at(0,1),A.frag_at(0,0),metal::false_type{},W.frag_at(0,0),W.frag_at(1,0),metal::true_type{}); \
    } \
  }
  DOWN_STEP(0,D0) DOWN_STEP(32,D1) DOWN_STEP(64,D2)
  #undef DOWN_STEP
}
device T* dst=out+(long)(start+row0)*H;
if(sg*32<H)D0.store_rows(dst+sg*32,H,short(valid));
if((sg+32)*32<H)D1.store_rows(dst+(sg+32)*32,H,short(valid));
if((sg+64)*32<H)D2.store_rows(dst+(sg+64)*32,H,short(valid));
