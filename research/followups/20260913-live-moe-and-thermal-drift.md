# Live MoE parity and prefill frequency drift

The earlier cross-boot 64,947-token screen recorded a real throughput drop,
1804.0 to 1485.6 tok/s, but attributing that entire drop to the research code is
not justified. New same-process screening exposes substantial control drift,
and IOReport/SMC telemetry measures falling GPU frequency as temperature rises.
There is no new 65K / 2400 tok/s or >1.5x model claim.

## MoE on actual model tensors

`QWEN4_MOE_LIVE_AB=1` drains the inputs to the first and fourth wide MoE calls,
warms both implementations, then evaluates OFF/ON/ON/OFF on the **same live
weights and inputs**, including the router, native expert projections,
activation, reduction and shared expert. It returns the baseline output.
No tensors are exported. This adds synchronization and replay, so the request's
overall rate is diagnostic only.

| Wide MoE call, S8192 | OFF eval ms | Grouping eval ms | Mean eval speed ratio |
|---|---:|---:|---:|
|0|33.233 /33.947|29.102 /30.489|1.127x|
|3|33.421 /33.778|30.265 /30.251|1.110x|

Every replay compared **all 20,971,520 BF16 output elements**, with zero bit
differences. Host construction took 0.045–0.059ms. This extends the previous
synthetic-weight/real-route probe to actual weights and activations, but does
not measure neighboring graph execution or a whole-model gain. The full
13,515-token diagnostic request also recalled MAGNOLIA-7731, cached_tokens=0.
Binary: `71d8b0ee54355359a19640891a812a30bb300029cb9fbabc8a501f80077c57e0`.

Source correction: this model's `hidden_act=silu`; scheduler/main only call
`compileGeglu` for `gelu_approx`. Its SwiGLU uses the existing exact sigmoid-table
kernel. The earlier suggestion that a compiled SwiGLU discrepancy remained
to be isolated was inaccurate.

## One process, identical prompt, changing arms

`QWEN4_PREFILL_ARM_SEQUENCE=030330` selects one arm per wide text request:
0 = current HC/GDN/QSA; 1 = plus grouping; 2 = plus PLE-ahead; 3 = both. It does not
change prompt contents, cache policy, tensor precision or GPU synchronization.
The first 0/3 requests warm the two implementations; the following 0/3/3/0
requests bracket the comparison. This is opt-in research for an isolated
single-model server, not a production feature/API.

All requests in both series were identical 13,515-token prompts, all returned
the passphrase and 10 output tokens, all had zero cached tokens. Same binary:
`d20dafd136d0a6dd38163ea52fc027141f79ea6cabeec611c99e6879aa4fb7dc`.
Same mixed-4-8bit model, chunk 8192, ctx 131072, KV quant off, no MTP/PLD/drafter,
AC 100%. This contains bf38063's production fixes, not a latest-main rebase.

| Request | Arm | First series tok/s | Series with telemetry tok/s | Mean GPU MHz | Last GPU Celsius |
|---:|---|---:|---:|---:|---:|
|0|current warmup|2128.8|2153.7|1618.6|77.5|
|1|combined warmup|2235.2|2206.2|1506.0|85.0|
|2|current|2046.4|2098.8|1518.2|90.9|
|3|combined|2027.8|2145.2|1439.7|96.0|
|4|combined|1944.2|2035.5|1361.4|97.0|
|5|current|1833.9|1886.9|1332.6|97.0|

The warmed control dropped 10.4% in the first series and 10.1% in the second.
Decode remained approximately 47–48 tok/s. Therefore steady decode alone did
not detect this prefill drift. GPU active residency was approximately 98–100%
during the sampled request interiors. First-to-last mean GPU frequency fell
17.7%; GPU temperature rose from 64.9 C in the first interior sample to 97.0 C.
Fan speeds were initially 1343/1463 RPM and reached 2300/2494 RPM in the last
request; the reported maximums are 5349/5777 RPM. Fan/clock/power settings were
**not changed**. This is measured thermal/frequency drift, not proof that
every part of the older cross-boot regression was thermal.

Counters come from [macmon 0.8.2](https://github.com/vladkens/macmon/releases/tag/v0.8.2),
running locally without sudo. Its release archive matched the published SHA256;
binary/archive hashes are in the manifest. Samples are 1-second windows, and
the table selects timestamps between HTTP start + 0.3 s and HTTP end − 0.3 s. These
are approximate request-interior samples, **not per-kernel hardware timing**.
Do not divide throughput by frequency and publish the extrapolation as a
measured result. Sequential bracketing in this short screen is not llmprobe
acceptance evidence for a production performance PR.

## Implication for the next optimization

The new evidence supports a faster isolated MoE and disproves the assumption
of a stationary control across this sequence. It does not establish the
remaining full-graph cost or a precise combined gain. Keep both experimental
options off by default; preserve the actual negative observations and add
the attribution caveat to them.

Before another long comparison, record GPU frequency/temperature and separate
cold preparation from sustained prefill. Do not spend another full ladder on
an unknown thermal state, and do not change cooling between comparison arms.
Cooling policy is a separate operational lever; no fan-control experiment or
2400 tok/s extrapolation is claimed here.

The next QSA arithmetic idea considered was replacing the two BF16 probability
fragments in PV with one FP16 probability. Source analysis rejects a naive
version before a model test: it removes residual precision, while FP16's
roughly 2^-11 relative rounding alone does not guarantee the existing 2.5e-6
FP32 absolute oracle floor. Converting V would also restrict exponent range.
The agreed numerical bar must stay fixed; retaining only a BF16-output test
would hide the pre-cast error. No such variant is integrated or benchmarked.

## Reproduction and artifacts

From this research checkout, with the pinned local MLX dependencies and
ReleaseFast server built:

```sh
PREFILL_MOE_LIVE_AB=1 PLE_BENCH_ARMS=current PLE_BENCH_TAG=live-moe \
  python3 research/followups/bench-prefill-combined.py \
  --out /tmp/live-moe --lock /path/to/gpu_lock.sh

PLE_BENCH_TAG=same-process \
  python3 research/followups/bench-prefill-combined.py --sequence 030330 \
  --monitor /path/to/macmon --out /tmp/prefill-sequence --lock /path/to/gpu_lock.sh
```

The monitor is optional.
The driver owns its server, monitor and GPUlock and cleans them up. It now
rejects combining a sequence with synchronizing route capture/layer replay.

[Raw screens, GPU samples, summaries, build logs and fingerprints](live-moe-thermal/).
Both ReleaseFast builds passed 7/7 and the real-model diagnostic/screens passed.
No new full Zig suite or long llmprobe run was made for this research-only
instrumentation. Production PR #408 remains at bf38063, unchanged by this work.
