#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
#include <stdexcept>
#include <unordered_map>
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Params {uint64_t woff,soff,boff;uint32_t rows,dim,wcols,scols,group_size;};
int main(int argc,char**argv){@autoreleasepool {
  if(argc!=2)throw std::runtime_error("usage: probe /path/to/ngram_table.bin");
  int fd=open(argv[1],O_RDONLY);if(fd<0)throw std::runtime_error("table open failed");
  struct stat st;if(fstat(fd,&st))throw std::runtime_error("stat failed");
  uint64_t hn;if(pread(fd,&hn,8,0)!=8 || hn>1048576)throw std::runtime_error("bad header length");
  std::vector<char> header(hn);if(pread(fd,header.data(),hn,8)!=(ssize_t)hn)throw std::runtime_error("header read failed");
  NSError* err=nil;
  NSDictionary* h=[NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:header.data() length:hn] options:0 error:&err];
  if(!h || [h[@"__metadata__"][@"bits"] intValue]!=4)throw std::runtime_error("requires affine4 table");
  auto offset=[&](NSString* n){return hn+8+[h[n][@"data_offsets"][0] unsignedLongLongValue];};
  uint32_t total=[h[@"weight"][@"shape"][0] unsignedIntValue];
  uint32_t scols=[h[@"scales"][@"shape"][1] unsignedIntValue];
  uint32_t gs=[h[@"__metadata__"][@"group_size"] intValue];
  Params p{offset(@"weight"),offset(@"scales"),offset(@"biases"),8192*16,scols*gs,[h[@"weight"][@"shape"][1] unsignedIntValue],scols,gs};
  size_t page=sysconf(_SC_PAGESIZE),len=(st.st_size+page-1)/page*page;
  if(p.dim!=160 || p.group_size!=32 || p.wcols!=20)throw std::runtime_error("unexpected geometry");
  void* map=mmap(nullptr,len,PROT_READ,MAP_PRIVATE,fd,0);if(map==MAP_FAILED)throw std::runtime_error("mmap failed");
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();
  std::cout<<"{\"table_bytes\":"<<st.st_size<<",\"max_buffer_bytes\":"<<device.maxBufferLength<<"}"<<std::endl;
  if(len>device.maxBufferLength)throw std::runtime_error("table exceeds Metal buffer length");
  auto begin=Clock::now();
  NSString* src=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void packed_ple(const device uchar* packed [[buffer(0)]],const device uint* slots [[buffer(1)]],device ushort* out [[buffer(2)]],constant uint& n [[buffer(3)]],uint i [[thread_position_in_grid]]) {if(i>=n*160)return;uint row=slots[i/160],c=i%160,g=c/32;const device uchar* p=packed+ulong(row)*100;uint q=(reinterpret_cast<const device uint*>(p)[c/8]>>(4*(c%8)))&15u;float sc=as_type<float>(uint(reinterpret_cast<const device ushort*>(p+80)[g])<<16);float bi=as_type<float>(uint(reinterpret_cast<const device ushort*>(p+90)[g])<<16);uint b=as_type<uint>(float(q)*sc+bi);out[i]=ushort((b+0x7fffu+((b>>16)&1u))>>16);}";
  MTLCompileOptions* opts=[MTLCompileOptions new];opts.mathMode=MTLMathModeSafe;
  id<MTLLibrary> lib=[device newLibraryWithSource:src options:opts error:&err];
  if(!lib){std::cerr<<err.localizedDescription.UTF8String<<std::endl;return 2;}
  id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"packed_ple"] error:&err];
  if(!pipeline)throw std::runtime_error("pipeline failed");
  id<MTLCommandQueue> queue=[device newCommandQueue];
  std::mt19937 rng(418971);std::vector<uint32_t> ids(p.rows);for(auto& row:ids)row=rng()%total;
  id<MTLBuffer> packed=[device newBufferWithLength:size_t(p.rows)*100 options:MTLResourceStorageModeShared];
  id<MTLBuffer> ib=[device newBufferWithLength:ids.size()*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> ob=[device newBufferWithLength:size_t(p.rows)*p.dim*2 options:MTLResourceStorageModeShared];
  std::unordered_map<uint32_t,uint32_t> slots;slots.reserve(p.rows);
  auto stage=[&](bool clear){
    if(clear)slots.clear();
    auto dst=(uint8_t*)packed.contents;auto lookup=(uint32_t*)ib.contents;auto data=(const uint8_t*)map;
    for(size_t i=0;i<ids.size();i++){
      auto [it,inserted]=slots.try_emplace(ids[i],slots.size());lookup[i]=it->second;
      if(inserted){uint8_t* row=dst+size_t(it->second)*100;uint64_t id=ids[i];std::memcpy(row,data+p.woff+id*80,80);std::memcpy(row+80,data+p.soff+id*10,10);std::memcpy(row+90,data+p.boff+id*10,10);}
    }
  };
  std::vector<uint16_t> reference(size_t(p.rows)*p.dim);
  auto cpu=[&](){
    auto data=(const uint8_t*)map;auto weights=(const uint32_t*)(data+p.woff);auto scales=(const uint16_t*)(data+p.soff);auto biases=(const uint16_t*)(data+p.boff);
    for(size_t i=0;i<reference.size();i++){
      uint32_t row=ids[i/p.dim],c=i%p.dim,g=c/p.group_size;
      uint32_t q=(weights[size_t(row)*p.wcols+c/8]>>(4*(c%8)))&15;
      uint32_t sb=uint32_t(scales[size_t(row)*p.scols+g])<<16,bb=uint32_t(biases[size_t(row)*p.scols+g])<<16;float sc,bi;
      std::memcpy(&sc,&sb,4);std::memcpy(&bi,&bb,4);float v=float(q)*sc+bi;uint32_t b;std::memcpy(&b,&v,4);
      reference[i]=uint16_t((b+0x7fff+((b>>16)&1))>>16);
    }
  };
  auto gpu=[&](){
    id<MTLCommandBuffer> cmd=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];[enc setBuffer:packed offset:0 atIndex:0];[enc setBuffer:ib offset:0 atIndex:1];[enc setBuffer:ob offset:0 atIndex:2];[enc setBytes:&p.rows length:4 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(size_t(p.rows)*160,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
    if(cmd.status==MTLCommandBufferStatusError)throw std::runtime_error("GPU failed");
  };
  begin=Clock::now();cpu();std::cout<<"{\"arm\":\"cpu_first\",\"ms\":"<<ms(begin)<<"}"<<std::endl;
  begin=Clock::now();stage(true);gpu();double first=ms(begin);if(std::memcmp(reference.data(),ob.contents,reference.size()*2))throw std::runtime_error("bit parity failed");
  std::cout<<"{\"arm\":\"gpu_first\",\"ms\":"<<first<<",\"bit_exact\":true}"<<std::endl;
  for(int arm:{0,1,2,0}) {std::vector<double> times;for(int i=0;i<3;i++){begin=Clock::now();if(arm){stage(arm==1);gpu();}else cpu();times.push_back(ms(begin));}std::sort(times.begin(),times.end());
    if(std::memcmp(reference.data(),ob.contents,reference.size()*2))throw std::runtime_error("bit parity failed");
    std::cout<<"{\"arm\":\""<<(arm==1?"stage_misses_plus_gpu":arm==2?"cache_hits_plus_gpu":"cpu_warm")<<"\",\"median_ms\":"<<times[1]<<",\"bit_exact\":true,\"rows\":"<<p.rows<<",\"unique_rows\":"<<slots.size()<<",\"packed_capacity_bytes\":"<<packed.length<<",\"output_bytes\":"<<ob.length<<"}"<<std::endl;}
  // The original table remains CPU-only. Only selected 100-byte rows enter Metal.
  munmap(map,len);close(fd);
}}
