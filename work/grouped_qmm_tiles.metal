// One 512-thread group builds a compact tile schedule from sorted expert IDs.
// No CPU readback of routing counts. Unused schedule entries have M=0.
const uint t=thread_index_in_threadgroup;
const uint lane=thread_index_in_simdgroup;
const uint warp=simdgroup_index_in_threadgroup;
const int M=indices_shape[0], capacity=M/BM+512;
for(int j=t;j<capacity*3;j+=512)tiles[j]=0;
int lo=0,hi=M;
while(lo<hi) { int mid=(lo+hi)/2;if(indices[mid]<t)lo=mid+1;else hi=mid; }
const int start=lo;
hi=M;
while(lo<hi) { int mid=(lo+hi)/2;if(indices[mid]<=t)lo=mid+1;else hi=mid; }
const int count=lo-start,n=(count+BM-1)/BM;
const int within=simd_prefix_inclusive_sum(n);
threadgroup int group_counts[16];
if(lane==31)group_counts[warp]=within;
threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
int offset=within-n;
for(int g=0;g<warp;g++)offset+=group_counts[g];
for(int block=0;block<n;block++) {
  int at=(offset+block)*3;
  tiles[at]=start;tiles[at+1]=count;tiles[at+2]=block;
}
