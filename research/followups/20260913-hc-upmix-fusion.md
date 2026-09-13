# HC up projection and mix without the wide intermediate

This is a new code optimization, independent of fan settings or power modes.
It is not a measured whole-model 1.5x gain or a 2400 tok/s result at 65K.

The existing path dequantizes the HC up weight, performs a native BF16 GEMM,
writes `[M,4,2560]`, then reads that entire array for the exact sigmoid-table
and BF16 stream mix. The new path permutes the exactly dequantized weight's
output channels to `[hidden,HC,K]`. Native Metal MPP performs the projection;
four HC columns share a lane, so the epilogue rounds to BF16, looks up the
native sigmoid, rounds each product and running sum, then multiplies by the
exact BF16 reciprocal 1/4 before storing only `[M,2560]`.

At M8192, the eliminated intermediate is **167,772,160 bytes**. Its write and
subsequent read remove **335,544,320 bytes of logical global-memory traffic**
per HC block. This is not a DRAM-counter measurement: caches can service some
traffic. Native inject matmul, HC down projection, activation, model precision,
and attention selection remain unchanged.

## Component measurements

M5 Max 128 GB, pinned local MLX, BF16 activations, affine8/group64 up weight,
M8192/HC4/H2560/K320. The control is **native dequant + GEMM plus the already
fused HC mix**, not the older unoptimized graph. Each timing is a median of
three component evaluations. All arrays are evaluated before timing; control
and candidate kernels are warmed. No full-model benchmark was spent on this
component alone.

| Weights | Control before / after, ms | Direct fused, preparation included, ms | Prepared-weight observations, ms |
|---|---:|---:|---:|
|Synthetic|1.95392 / 1.92637|**1.28808**|1.22075 / 1.24596|
|Actual layer0 attention HC|2.05888 / 2.08288|**1.29817**|1.26337 / 1.96138|
|Actual layer0 MLP HC|1.90088 / 2.05888|**1.32137**|1.26775 / 1.28638|

The inclusive synthetic comparison is 1.50–1.52x; actual attention 1.59–1.60x;
actual MLP 1.44–1.56x. The one noisy prepared-weight observation is retained.
Every tested output element matched bit-for-bit: **20,971,520 BF16 elements**
per wide comparison. Actual-weight probes still use synthetic activations;
they are not a claim of live-model parity by themselves. No model tensors are
exported: `load_safetensors` builds lazy loads and only the three selected
weight tensors are evaluated.

Earlier variants are negative controls, not runtime options: explicit
shared-memory NAX loading was 3.37–4.14 ms with prepared weights; one SIMD
group per HC stream was 8.66–8.79 ms. Switching to native device-tensor MPP
reached 1.30–1.33 ms with an intermediate threadgroup store; directly mixing
its cooperative output registers removed that store/barrier too. Early small
and first-wide observations had noticeable timing drift and are not used for
the headline ratio.

The benefit budget is bounded: approximately 0.6–0.8 ms per HC block suggests
roughly 0.4–0.5 seconds over 672 eligible HC blocks in seven full chunks of the
64,947-token workload. This is a component-based estimate, **not measured
model throughput**. Combine it with other useful work reductions; do not run
another full ladder merely to resolve this small predicted model increment.
Watts, joules/token and temperature improvement from this code remain
unmeasured.

## Research integration

`MLX_SERVE_HC_UPMIX=1`, default off, selects the new path only on a NAX-capable
GPU with the supported BF16 affine8/group64 geometry. It requires M>=2048
and M divisible by64; unsupported/tail shapes use the previous path. Weight
dequantization and permutation occur in the normal graph, with **no persistent
weight cache or added CPU worker**. The sequence driver assigns bit4 to this
option:0=current,4=upmix,7=PLE-ahead + grouping + upmix.

`QWEN4_HC_UPMIX_VERIFY=1` compares the first two eligible live outputs against
the previous HC path using all elements. It adds synchronization, so that
request is diagnostic only. The kernel also checks the cooperative layout's
four-column property; a future production version needs a startup probe and
automatic fallback rather than allowing a failed layout check into a model
graph. This experimental implementation is not added to production PR#408.

## Reproduction

From the research checkout, using the existing exclusive GPU lock:

```sh
clang++ -O3 -std=c++20 -mmacosx-version-min=26.3 \
  research/followups/hc-upmix-probe.cpp -I lib/mlx-src \
  -L lib/mlx/lib -lmlx -Wl,-rpath,"$PWD/lib/mlx/lib" -o /tmp/hc-upmix-probe
/path/to/gpu_lock.sh acquire hc-upmix
trap '/path/to/gpu_lock.sh release hc-upmix' EXIT
/tmp/hc-upmix-probe wide
/tmp/hc-upmix-probe wide /path/to/model/model-00001.safetensors \
  language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up
/tmp/hc-upmix-probe wide /path/to/model/model-00003.safetensors \
  language_model.model.layers.0.mlp_hyper_connection.input_mix_weight_up
```

Current modes:0=dequant+native GEMM+mix,6=prepared-weight MPP with threadgroup
epilogue,8=prepared-weight direct epilogue,9=direct including dequant/repack.
The probe also retains implementations of earlier modes for source analysis.
Actual shard names above were resolved from this installed pack's index;
resolve them again if the checkpoint changes.

For the actual research server, after a ReleaseFast rebuild:

```sh
PREFILL_HC_UPMIX=1 PREFILL_HC_UPMIX_VERIFY=1 PLE_BENCH_ARMS=current \
  PLE_BENCH_TAG=hc-upmix-live python3 research/followups/bench-prefill-combined.py \
  --out /tmp/hc-upmix-live --lock /path/to/gpu_lock.sh
```

This last command verifies live outputs and recall; do not quote its overall
rate as an uninstrumented speed measurement. Component JSONLs beside this
report preserve both positive and negative results.

## Runtime validation completed

ReleaseFast build passed 7/7; focused runner passed 3/3, including the new native
GEMM/compiled HC mix parity test and unsupported-tail fallback. A real 13,515-token
request engaged the new kernel at S8192. Its first two eligible HC outputs each
compared all 20,971,520 BF16 elements against the previous path, with **zero bit
differences**. Passphrase recall passed, 10 output tokens, cached_tokens=0. This
run adds explicit synchronization and its observed 2080.4 tok/s is diagnostic,
not a speedup measurement. No new full suite or long llmprobe run was made.

[Build, test, live output and fingerprints](hc-upmix-validation/).
