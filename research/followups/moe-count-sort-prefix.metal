uint e=thread_index_in_threadgroup,lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
uint groups=counts_shape[0],capacity=CAPACITY;
uint total=0;
for(uint g=0;g<groups;g++)total+=counts[g*512u+e];
uint nt=(total+BM-1)/BM;
threadgroup uint row_totals[16],tile_totals[16];
uint row_local=simd_prefix_exclusive_sum(total),tile_local=simd_prefix_exclusive_sum(nt);
uint row_sum=simd_sum(total),tile_sum=simd_sum(nt);
if(lane==0){row_totals[sg]=row_sum;tile_totals[sg]=tile_sum;}
for(uint j=e;j<capacity*3u;j+=512u)tiles[j]=0;
threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
uint row_start=row_local,tile_start=tile_local;
for(uint g=0;g<sg;g++){row_start+=row_totals[g];tile_start+=tile_totals[g];}
uint cursor=row_start;
for(uint g=0;g<groups;g++){offsets[g*512u+e]=cursor;cursor+=counts[g*512u+e];}
for(uint b=0;b<nt;b++){
  uint at=(tile_start+b)*3u;tiles[at]=row_start;tiles[at+1]=total;tiles[at+2]=b;
}
