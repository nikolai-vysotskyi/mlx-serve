# Current combined prefill: 2347 tok/s at 64,947 tokens

One same-process HTTP comparison on M5 Max 128 GB, mixed-4-8bit checkpoint,
frozen published research 087c210 binary
`cc7f4574620557b263212e009c7e933783a3e0fac4a00ff778d708467de59d48`:

| Input tokens | Current HC/GDN/QSA | Plus PLE-ahead, MoE grouping and HC-upmix |
|---:|---:|---:|
|13,515, warmed bracketing series|2182.1 /2163.5|2335.6 /2330.9|
|64,947, one pair|**2173.8**|**2347.1**|

Short bracketing mean improvement is 7.38%; long observed improvement is 7.97%.
The new MoE MPP gate/up prototype is **not** in this measured executable.
This is not llmprobe acceptance, sustained-session throughput or a >1.5x
whole-model claim. 2400 tok/s has not been demonstrated by this comparison.
Do not compare these HTTP numbers with the older full llmprobe ladder's 1761
as though the two were the same workload or an isolated code speedup.

The requests in each comparison use exactly the same prompt hash, tokenizer,
model and binary. All returned MAGNOLIA-7731 with 10 output tokens and
cached_tokens=0. Chunk 8192, ctx 131072, KV quant off, no MTP/PLD/drafter. Two
13.5K warmups prime both implementations before the long pair. Both screens
use the normal graph, with no route capture or live-layer replay.

## Controlling the previous thermal confound

Before each request, the driver waits **outside the timed request** for three
fresh idle telemetry samples: GPU<=50C, GPU power<=5W and both reported fan
speeds<=1600RPM. It aborts after 180 seconds if that condition is not reached.
This is passive benchmark preparation, **not a serving optimization**; no
fan, clock or power settings were changed. It deliberately excludes the
cooldown gaps from reported prefill. End-to-end workload throughput including
those gaps would be much lower and is not claimed here.

For the long measured pair, starting GPU temperature was 49.7/47.7 C, fans
1346/1463 versus1460/1584RPM. Approximate request-interior mean GPU frequency
was1482.0/1490.8MHz, and both reached 96.6 C. These are 1-second macmon windows
selected between HTTP start + 1 s and end − 1 s, not per-kernel timestamps. Earlier
short samples can include clock ramp/idle windows; do not normalize the
throughput by these MHz readings. The short control bracket changed -0.85%,
compared with roughly -10% in the previous uncooled sequence. Starting fan
states were not identical throughout the short series; they are retained.

The result supports a positive combined effect under these starting
conditions, while retaining the usual single-pair/order/machine-state limits.
Only about 2.3% further throughput is required to reach 2400 on this particular
HTTP workload. Subsequent candidates and the final decision to stop this research phase are
recorded in [the final report](20260913-final-prefill.md).

## Reproduce the screening

```sh
PLE_BENCH_TAG=cooled-combined python3 research/followups/bench-prefill-combined.py \
  --sequence 070770 --cooldown-c 50 --monitor /path/to/macmon \
  --out /tmp/cooled-short --lock /path/to/gpu_lock.sh
PLE_BENCH_LONG=1 PLE_BENCH_TAG=cooled-long \
  python3 research/followups/bench-prefill-combined.py --sequence 0707 \
  --warmup-short 2 --cooldown-c 50 --monitor /path/to/macmon \
  --out /tmp/cooled-long --lock /path/to/gpu_lock.sh
```

The script owns its server, monitor and GPU lock and
cleans up on failure. The bitmasks here remain 0=current, 7=previous combination
even after the driver adds support for the newer MPP option.

[Raw requests, GPU samples and summaries](cooled-prefill-20260913/).
