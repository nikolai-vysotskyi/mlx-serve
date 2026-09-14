# Current-head PR #408: M5 Max mixed-pack ladder

Measured production commit **d23d9df**, binary SHA256 `580f495dce825202ea2da7976e10f6b1c4db39158827e7d63735f60a79f0d476`. Base fa76a4b; not a benchmark of latest upstream main. Research PLE/grouping changes are disabled. Hardware M5 Max 128 GB, AC power, battery 100%, macOS 26.5, pinned MLX 0.32.2 (version label corrected September14 against the exported runtime version and archived dylib SHA). No other benchmark held the exclusive GPU lock. Power/frequency telemetry was unavailable, so no thermal cause is claimed.

## Full-model results

llmprobe 0.6.6, `--bench-only --rungs 4k,8k,16k,64k,128k --runs 3`. Rates below are llmprobe contextScaling prefill rates based on TTFT; raw server rates are retained separately. The standard short/speculative/cache/batch/drift checks also ran.

| Target tokens | Actual OFF / ON | OFF prefill tok/s | ON prefill tok/s | Observed change | OFF / ON decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 4096 | 4239 / 4235 | 1434 | 1489 | 3.84% | 55.1 / 50 |
| 8192 | 8235 / 8231 | 1599 | 1623 | 1.50% | 56.2 / 50.5 |
| 16384 | 16298 / 16311 | 1531 | 1735 | 13.32% | 57.2 / 50.8 |
| 65536 | 65841 / 65778 | 1517 | 1761 | 16.08% | 49.6 / 50.9 |
| 131072 | 131086 / 131133 | 1447 | 1771 | 22.39% | 48.5 / 49.1 |

**The requested 2400 tok/s at 65K has not been reached.** The measured 65K ON rate is 1761, about 26.6% more prefill time must be removed to reach 2400 on this workload. These observations are not a >1.5x result. They do not replace earlier, differently shaped HTTP screening with the PLE research branch.

## Comparison limits

- One complete OFF ladder followed by one complete ON ladder. Each rung has three novel requests and three counting-ceiling requests, as implemented by llmprobe. Not a repeated alternating ladder.
- OFF drift: 64.2 to 57.2 decode tok/s (-10.9%, degraded). ON drift: 54.2 to 56.8 (+4.8%, steady). Thus small-rung and decode differences are confounded by execution order/machine state; no statistically isolated 3.8% claim.
- Final standalone 64K OFF: **68,651 actual tokens, 1564 prefill, 50.4 decode**, drift +0.7%. This is a coarse drift control, NOT a matched third arm: llmprobe's filler calibration uses preceding rung observations, and each process creates a fresh cache-bust nonce. The standalone 64K prompt is longer than the ladder's 64K prompt. Do not average this arm into a precise speedup claim.
- ON logged paired QSA, native-inject HC and GDN engagement. At the 8K rung, actual prefill width 8230 exceeded the 8192 specialization cap and the new QSA fallback log correctly reported it. Longer requests split into eligible chunks. This is an opportunity to examine tail coalescing separately, not evidence that the 8K rung measured the specialized path.
- Prefix cache entries 0, KV quant off, MTP/PLD/drafter explicitly disabled, chunk 8192, context 262144. All three fusion flags forced 0 OFF and 1 ON. PLE PACKED/AHEAD and MoE GROUP forced 0 in both. See manifest for exact invocation and engagement.
- Peak MLX bytes: OFF 80,113,270,592; ON 80,131,854,112. Allocator peak is not total machine RSS.

## Validation and reproduction

`tests/bench_qwen4_prefill.py` at d23d9df reproduces this ladder on the mixed pack. A compatible GPU lock is optional via `--lock`. The default final single-rung drift check has the calibration limitation above.

Current source passed ReleaseFast build (7/7), full Zig tests (2330 passed, 154 skipped), compiled HC and compiled GDN reference fixtures, the mixed-pack HTTP recall/engagement wrapper, and the standalone float64 QSA oracle. The oracle checks every selected-key multiset and sampled output error at B2, ragged S/KB, 65K, 200K, 1M KV and S8192. QSA remains opt-in and approximate; these checks do not establish broad language-model quality or resolve the outstanding M4 deterministic-output report. HC/GDN default on, each with a `=0` kill switch.

Raw JSONs/logs and manifest in this directory retain measured values; local absolute workspace prefixes have been replaced by `<workspace>`. The startup inventory of unrelated local models was omitted. No measured result was edited.
