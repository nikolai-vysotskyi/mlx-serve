#include <metal_stdlib>
using namespace metal;
struct Params { ulong woff,soff,boff; uint rows,dim,wcols,scols,group_size; };
kernel void ple_mmap_gather(const device uchar* table [[buffer(0)]],
                            const device uint* ids [[buffer(1)]],
                            device ushort* out [[buffer(2)]],
                            constant Params& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if(ulong(i)>=ulong(p.rows)*p.dim)return;
  uint row=ids[i/p.dim],c=i%p.dim,g=c/p.group_size;
  const device uint* w=reinterpret_cast<const device uint*>(table+p.woff);
  const device ushort* s=reinterpret_cast<const device ushort*>(table+p.soff);
  const device ushort* b=reinterpret_cast<const device ushort*>(table+p.boff);
  uint q=(w[ulong(row)*p.wcols+c/8]>>(4*(c%8)))&15u;
  float sc=as_type<float>(uint(s[ulong(row)*p.scols+g])<<16);
  float bi=as_type<float>(uint(b[ulong(row)*p.scols+g])<<16);
  uint bits=as_type<uint>(float(q)*sc+bi);
  out[i]=ushort((bits+0x7fffu+((bits>>16)&1u))>>16);
}
