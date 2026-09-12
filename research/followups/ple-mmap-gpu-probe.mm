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
  id<MTLBuffer> table=[device newBufferWithBytesNoCopy:map length:len options:MTLResourceStorageModeShared deallocator:nil];
  if(!table)throw std::runtime_error("file mmap Metal buffer rejected");
  if(table.contents!=map)throw std::runtime_error("unexpected non-aliasing buffer");
  std::cout<<"{\"wrap_ms\":"<<ms(begin)<<",\"same_address\":true,\"device_allocated_bytes\":"<<device.currentAllocatedSize<<"}"<<std::endl;
  NSString* src=[NSString stringWithContentsOfFile:@"research/followups/ple-mmap-gpu.metal" encoding:NSUTF8StringEncoding error:&err];
  MTLCompileOptions* opts=[MTLCompileOptions new];opts.mathMode=MTLMathModeSafe;
  id<MTLLibrary> lib=[device newLibraryWithSource:src options:opts error:&err];
  if(!lib){std::cerr<<err.localizedDescription.UTF8String<<std::endl;return 2;}
  id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ple_mmap_gather"] error:&err];
  if(!pipeline)throw std::runtime_error("pipeline failed");
  id<MTLCommandQueue> queue=[device newCommandQueue];
  std::mt19937 rng(418971);std::vector<uint32_t> ids(p.rows);for(auto& row:ids)row=rng()%total;
  id<MTLBuffer> ib=[device newBufferWithBytes:ids.data() length:ids.size()*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> ob=[device newBufferWithLength:size_t(p.rows)*p.dim*2 options:MTLResourceStorageModeShared];
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
    [enc setComputePipelineState:pipeline];[enc setBuffer:table offset:0 atIndex:0];[enc setBuffer:ib offset:0 atIndex:1];[enc setBuffer:ob offset:0 atIndex:2];[enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(size_t(p.rows)*p.dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
    if(cmd.status==MTLCommandBufferStatusError){std::cerr<<cmd.error.localizedDescription.UTF8String<<std::endl;throw std::runtime_error("GPU execution failed");}
  };
  begin=Clock::now();cpu();std::cout<<"{\"arm\":\"cpu_first\",\"ms\":"<<ms(begin)<<"}"<<std::endl;
  begin=Clock::now();gpu();double first=ms(begin);if(std::memcmp(reference.data(),ob.contents,reference.size()*2))throw std::runtime_error("bit parity failed");
  std::cout<<"{\"arm\":\"gpu_first\",\"ms\":"<<first<<",\"bit_exact\":true}"<<std::endl;
  for(int arm:{0,1,0}) {std::vector<double> times;for(int i=0;i<3;i++){begin=Clock::now();if(arm)gpu();else cpu();times.push_back(ms(begin));}std::sort(times.begin(),times.end());
    if(std::memcmp(reference.data(),ob.contents,reference.size()*2))throw std::runtime_error("bit parity failed");
    std::cout<<"{\"arm\":\""<<(arm?"gpu_warm":"cpu_warm")<<"\",\"median_ms\":"<<times[1]<<",\"bit_exact\":true,\"rows\":"<<p.rows<<",\"dim\":"<<p.dim<<"}"<<std::endl;}
  // The mapping must outlive its Metal buffer. The process releases both on exit.
}}
