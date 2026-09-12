# Expert grouping and direct-input MoE pipeline

This combines three compatible ideas: GPU counting-sort grouping over 512 experts, direct token gather inside a 192-row gate/up NAX tile with fused BF16 SwiGLU, and inverse-permutation plus weighted expert reduction. The stock 8-group BF16 reduction order is retained. The route count never leaves the GPU.

The measured chain starts from already-selected expert IDs and scores, and includes grouping, input replication/gather, all three routed expert projections, activation, restoration of original slots and weighted reduction. It excludes router logits/top-K selection, shared experts and the rest of the layer/model. Weights and activations are synthetic; no model quality or whole-model throughput result is claimed.

M5 Max 128 GB, E512/top10/H2560/I640, S8192. Three timed repetitions per arm after correctness evaluation:

| Arm | Before hoisting input indices | Final run |
|---|---:|---:|
| Stock before | 29.6483 ms | 26.9177 ms |
| Counting grouping + native projections + fused reduction | 26.3597 ms | **23.5255 ms** |
| Direct-input wide gate/up + grouping/reduction | 48.0767 ms | 24.2075 ms |
| Stock after | 29.9044 ms | 27.0064 ms |

All arms match stock bit-for-bit on the final S257 and S8192 matrix-dispatch fixtures. The first direct-input version reloaded/divided each token index inside every K tile. Hoisting the four row addresses per lane outside the 40-tile loop removed that repeated dependency; compare each run with its own controls, since the controls also moved. The final native-projection arm is still faster than the direct-input gate/up arm. Its chain acceleration is **1.144–1.148×**, too small on a fraction of model time to justify a long full-model run. No runtime dispatch was added.

## Correctness boundary found during investigation

The original S53/E512 fixture dispatched MLX's quantized-vector path because routed rows/expert was below four. That path uses different arithmetic from the sorted matrix prefill path. The direct-input matrix implementation did not match it, although selected direct projections matched a CPU dequantized dot-product oracle. One-hot projections matched both paths. Source inspection located the native `GatherQMM::eval_gpu` gate: M==1, B>=16, sorted, B/E>=4. The final probe uses S257 for its small matrix fixture and explicitly refuses M/E<4. A future production integration must keep native fallback below that threshold, not loosen a tolerance or substitute one arithmetic path silently.

The low-occupancy rejected result is preserved in `moe-direct-low-occupancy-rejected.jsonl`; it is not a supported performance result. `moe-direct-pipeline-unhoisted.jsonl` records the earlier large kernel. The final small file was measured before address hoisting; the large file exercises the final kernel.

## Reproduce

```sh
clang++ -O3 -std=c++20 -mmacosx-version-min=26.3 \
  research/followups/moe-direct-pipeline-probe.cpp -I lib/mlx-src -L lib/mlx/lib \
  -lmlx -Wl,-rpath,"$PWD/lib/mlx/lib" -o /tmp/moe-direct-pipeline-probe
# Hold the exclusive GPU lock; small first, then one large component run.
/tmp/moe-direct-pipeline-probe
/tmp/moe-direct-pipeline-probe large
```

Remaining useful candidate: retain the GPU counting permutation and fused inverse/reduction as a compatible part of a larger optimization. Do not integrate the slower direct-input kernel or present the recovered 48→24 ms as acceleration over upstream.
