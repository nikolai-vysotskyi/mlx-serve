# Final Qwen3.8 Flash-Next prefill investigation — M5 Max 128 GB

The local research phase is complete. **PR [#408](https://github.com/ddalcu/mlx-serve/pull/408) is the single consolidated review target.** The measured result accepted for this phase is approximately **2300 tok/s at 64,947 input tokens**. It is an HTTP screening result, not a claim of sustained serving throughput or a >1.5× improvement over current upstream.

See the [September14 reviewer follow-up](20260914-review-and-compatibility.md) for independent measurements and the correction to the manual width pin. Historical measurements below retain their original provenance.

## Results and provenance

All new model measurements use `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`, M5 Max 128 GB, macOS 26.5, AC power, chunk8192, ctx131072, prefix-cache entries0, KV quantization off, MTP/PLD/drafter off. The request is public repository source plus a fixed passphrase, greedy seed1234, max_tokens32; the measured responses actually contain 10 output tokens. Every listed long request has cached_tokens=0 and recalls MAGNOLIA-7731. No model conversion, requantization, layer removal or changed routing selection was used.

| Evidence | Control / candidate | Prefill tok/s | Meaning |
|---|---|---:|---|
| Frozen 087c210, 13,515 tokens, warmed 0/7/7/0 bracket | Existing QSA+HC+GDN / add bounded PLE-ahead + MoE grouping + HC-upmix | 2182.1,2163.5 / 2335.6,2330.9 | +7.38% mean; control drift -0.85% |
| Same 087c210 binary/process, 64,947 tokens, one 0/7 pair | Same arms | **2173.8 / 2347.1** | **+7.97% observed**, decode50.0/51.1 |
| Later pre-wide32 snapshot, 64,947 tokens, reversed f/7 pair | Add optional MoE MPP / previous combination | 2284.7 / 2267.6 | Only +0.75%; starting conditions differ, no strong incremental MPP claim |
| Same pre-wide32 snapshot, fans Full blast, 64,947 tokens | Combination with optional MPP; no matched Automatic control | **2299.9** | Decode48.9; independent approximately2300 observation, not a fan-speed A/B |
| Rejected wide32 experiment, 64,947 tokens | Chunk32768, wider guards, per-layer eval, padded HC tails | **2016.9** | Regression; excluded from final runtime |

The 087c210 executable SHA256 is `cc7f4574620557b263212e009c7e933783a3e0fac4a00ff778d708467de59d48`. The later pre-wide32 snapshot executable is `283bc95a286117d72fd7f7fb5398acb058ad530fb092171545a3880e8b41bf4e`. Consolidation commit b7d9033 copies this already-built/tested runtime source without changing it; a source manifest identifies the restored snapshot and hashes. **2347.1 belongs to 087c210 without MPP; 2299.9 belongs to the later snapshot with MPP.** Packaging does not retroactively reassign measurements to a new commit or binary.

Long-prompt SHA256: `d6927da48d558a30339f665d0536288a7e07394c5f0c4daa23eb7e4246d0037e`; corpus SHA256: `1b0d4f22b5f08c471a93caac3d5f4729be34c64a04e4c0e303a3fc44bcde36a0`. Peak MLX allocations for the matched long pair were78,214,759,746/78,223,208,014 bytes. Full-blast MPP peak was78,216,733,974 bytes. These are MLX allocator observations, not total process/OS memory or a proof of admission safety under pressure.

The measured runtime uses the locally built MLX0.32.3 source `1f8e74e3f12f31365464a6867c6579f0e9b29d85`; the startup string saying0.32.2 is stale and is not the library fingerprint. Do not assume a checkout's submodule label alone identifies its installed dylib. Use isolated build caches per checkout.

### Earlier llmprobe acceptance-style evidence

On reviewed core-fusion head **d23d9df**, binary `580f495dce825202ea2da7976e10f6b1c4db39158827e7d63735f60a79f0d476`, llmprobe0.6.6 contextScaling, three runs per rung, fresh process per complete arm, ctx262144, same mixed pack:

| Target | Actual OFF / ON tokens | OFF / ON prefill tok/s | OFF / ON decode tok/s |
|---:|---:|---:|---:|
|4K|4239 /4235|1434 /1489|55.1 /50.0|
|8K|8235 /8231|1599 /1623|56.2 /50.5|
|16K|16298 /16311|1531 /1735|57.2 /50.8|
|64K|65841 /65778|1517 /1761|49.6 /50.9|
|128K|131086 /131133|1447 /1771|48.5 /49.1|

Observed improvements were+16.1% at64K and+22.4% at128K. OFF decode drift was-10.9%, ON+4.8%. A standalone final64K OFF gave1564 at68,651 actual tokens; llmprobe calibrates using preceding rungs, so this is a coarse drift check, not a matched A-B-A. The newer bf38063 tail/HC3 fixes and later combined additions have separate validation; this ladder is not relabelled as a measurement of them.

The new2347 HTTP cell and old1761 llmprobe cell use different prompts/protocols. Their ratio is **not an isolated code gain**. The user's earlier1868 upstreamfa76a4b cell likewise cannot serve as a causal denominator for this newer workload. Neither2400 nor>1.5× whole-model improvement has been established by the matched data. The earlier2509.4 short-context checkpoint was not a65K result; the old1879 long result preceded the HC inject correction.

## What the consolidated code contains

| Change | Mechanism and evidence | Default / scope |
|---|---|---|
| Paired QSA | Stage common selected K/V blocks for adjacent queries; component about29.5→14.19ms. Selected-key multiset preserved. FP32 probability high/low BF16 representation retains the tested accuracy bar, but reduction order changes. | Opt-in `MLX_SERVE_QSA_PAIR=1`, NAX; approximate, not bit-identical |
| HC prefill | Fuse pending write, normalization and stream mixing; retain native inject matmul after the original fused-inject rounding defect. Actual component4bit6.44→4.43ms;8bit6.58→4.73ms. | Default on, `MLX_SERVE_HC_PREFILL=0` rollback |
| GDN prefill | Fuse cold/warm prework and output normalization/gating, preserve BF16 rounding/table sigmoid. Compiled-production fixture parity. | Default on, `MLX_SERVE_GDN_PREFILL_FUSED=0` rollback |
| Planner/shape fixes | QSA planner71,909,376→19,611,648 bytes atS8192/B1; transient billed, KV support through1,048,576 with fallback log. HC geometry derived. Coalesced S<=8703 supported in core fusions. HC3 reciprocal rounding corrected. | Included in core path |
| Bounded PLE ahead | Request-owned CPU producer copies only selected100-byte quantized rows; two ready chunks plus one preparing,26.2MB raw-row cache plus metadata. MLX calls remain on inference thread. Compare all row IDs before consuming; join on cancellation. | Opt-in `MLX_SERVE_PLE_PACKED=1 MLX_SERVE_PLE_AHEAD=1`; fixed-boundary eligible text prefill; no whole-table GPU registration |
| MoE grouping/reduction | Counting/prefix/scatter replaces two general argsorts; fuse inverse-permutation, weighted reduction with native rounding. Native upstream already sorts experts: this does not fix an unsorted upstream path. | Opt-in `MLX_SERVE_MOE_PREFILL_GROUP=1`; B1,S2048..8192,E512,K10,H2560, supported affine4/BF16 geometry |
| HC upmix | Native MPP projection epilogue performs stream mixing without materializing the large up tensor. Compiled native-reference test and two live20,971,520-element comparisons exact. | Opt-in `MLX_SERVE_HC_UPMIX=1`; NAX, guarded HC4/H2560/K320/affine8, aligned rows; tail fallback |
| MoE MPP gate/up/SwiGLU | Expert-aligned64-row tiles, gate/up weight reuse and exact-table SwiGLU in epilogue; native down/shared-expert/reduction remain. Avoids419,430,400 logical intermediate bytes perS8192 layer. | Additional opt-in `MLX_SERVE_MOE_PREFILL_MPP=1`; implies grouping. Included for reproducibility, **not needed for2347 and not a demonstrated large model gain** |

MoE MPP component on real-routing fixtures with synthetic activations/weights: grouped-native23.34–23.41→21.284ms, another layer23.24–23.32→21.420ms. Live replay using actual model weights and inputs compared OFF/ON/ON/OFF: native30.288/30.236 vs group+MPP24.855/24.698ms at the first sampled layer; native29.609/30.247 vs24.943/24.886 at the fourth. Every20,971,520-element output comparison was bit-exact. Replay synchronizes the graph; its request tok/s is **diagnostic**, not a normal-serving gain.

New experimental paths remain opt-in because their general quality, architecture fallback, admission and pressure behavior are not established. This does not represent them as proven universal default-on wins. The core HC/GDN polarity requested in review remains implemented. Optional research replay/capture/sequence controls are disabled in ordinary serving; do not enable them for production measurements except the documented arm sequence used for this screening.

## Why a faster component did not always make the model faster

1. **A real integration regression:** registering the whole~32GB PLE table with Metal gave a~36× warm standalone gather ratio but actual15,715-token model prefill fell1987.7→1075.6. Page residency/registration and graph lifetime costs were outside the component timing. That integration was removed; bounded selected-row staging replaced it.
2. **Different numerical implementations:** the old HC fused inject was faster but did not match native dense BF16 reduction. Correcting it costs time. Old1879 cannot be used as evidence for the correction. HC3 reciprocal was another real rounding bug; HC4 is unchanged by that fix, so it does not explain the reported M4 HC4 divergence.
3. **Machine-state drift:** uncooled identical-binary sequences showed roughly10% control drift, large enough to bury modest component gains. Passive matched starting conditions reduced the short control bracket drift to0.85%. Temperature alone did not explain clock changes.
4. **A separate resource collision:** later, an unrelated local training process caused model-readiness HTTP503 (only16.11GB available against~77.1GB required). The user stopped training; the retry loaded and produced2299.9. The failed load is not a speed measurement and does not prove the earlier runs were all contaminated.
5. **Larger graphs changed the inclusive costs:**32K batches helped a fixed MoE component per token, yet the full model with wider guards, per-layer evaluation and HC padded tails only reached2016.9, with84.04GB MLX peak. The experiment was reverted. The per-component ratio omitted costs that moved elsewhere in the graph.

### Cooling result, without turning it into a code speedup

The2347 pair used Automatic fans. Before each request, outside its timer, the driver required three fresh samples with GPU<=50C, GPU power<=5W and both fans<=1600RPM. It aborts after180s. The long arms started49.7/47.7C, fan pairs1346/1463 and1460/1584RPM; request-interior approximate mean frequency1482.0/1490.8MHz, peak96.6C in both. Idle gaps are excluded; including them would lower end-to-end workload throughput substantially.

A later50C MPP comparison aborted after the old-combination long arm2335.6 because the next cooldown could not reach50C in180s. No missing candidate result is invented. The reversed55C pair2284.7/2267.6 started54.9/47.5C and had mean frequencies1442.6/1466.0MHz. Retain this mismatch when judging the small delta.

After training stopped, temporary Full blast fans around5354/5770RPM yielded2299.9 at65K, peak77.0C, approximate mean1467.9MHz and67.5W GPU power. No same-setting Automatic control followed, so there is no causal fan-speed ratio. A140W adapter was connected. All frequency/temperature observations use1-second telemetry windows inside HTTP request bounds, not kernel timestamps; do not normalize tok/s by MHz. **Fans were restored to Automatic.** No clock, power or permanent cooling changes are part of the PR, and Full blast is not required to reproduce the earlier2347 result.

## Validation and remaining review gates

The measured pre-wide32 runtime snapshot was built ReleaseFast7/7 and passed the full Zig suite **2333 passed /154 skipped** on this M5 before packaging. The source manifest confirms136 tracked runtime/build inputs at consolidation commit b7d9033 match the tested research snapshot after excluding the failedwide32 changes. The binary and prior test output are identified separately; no new GPU test was run during finalization because the owner reassigned GPU/RAM. No Swift/app code changed.

Coverage includes compiled HC/GDN references, generic HC2/3/4/8 and HC3 red/green, tail integration at8296/13515/15715 tokens, QSA selected-key/float64 oracle through1M KV, PLE row duplicates/lifetime/cancellation/history mismatch, grouping/inverse reduction, padded MoE input and exact tile coverage, HC-upmix compiled parity, live actual-weight/activation comparisons and uncached HTTP recall. Finite fixtures and a passphrase are not a broad model-quality evaluation.

Still open in **#408**, not silently marked completed:

- Corrected-head **M4 HC-only/GDN-only** deterministic-output isolation and requested fresh-process M4 ladder. Original M4 run on092ce2e had-3.9% at16K,+3.0% at64K and greedy divergence; QSA did not engage there because NAX was unavailable.
- QSA broad model-quality acceptance and cooperative-layout startup validation/fallback. New HC-upmix and MoE MPP also need a layout probe/fallback; a NAX capability guard alone is insufficient, and their current unsupported-layout diagnostic can yield NaN. They stay opt-in/draft.
- Fresh final-combination llmprobe ladder and rollout/admission review for new PLE/MoE buffers. The old ladder proves the old named core head; the new HTTP screen is supporting research evidence, not a replacement.
- Productionizing the bounded PLE/MoE paths, including pressure/cancellation/architecture review and removal or relocation of research diagnostics before a general rollout. No claim that pressure benefits or every non-M5 case are validated.
- Finalization found a merge conflict and then integrated upstream0814cf3ce2. The adapter retains upstream per-slot speculative PLE capture and deferred gather, and limits the experimental packed path to non-batched slots. See the final PR comment for the separate compile/test status of this integration. No model speed is assigned to it: the measured base remainsfa76a4b, and the new main was not silently substituted into the historical comparisons.

PR remains draft for these concrete reasons. Local research is finalized; merge/readiness and cross-machine validation are separate outstanding work, assigned explicitly in the final escalation.

### Upstream work already credited and retained

The baseline already contains upstream QSA NAX/block-select/score-sheet work, including our#385 contribution integrated by beamivalice through#388 (merge680e5a56d2ca6926785b404efed06d1d85b563f3). It also contains the owned conv-state tail and cadence/admission correction recorded from#366: upstream reported about2.5GB lower peak at4K chunks and neutral prefill, plus correction of the already-fixed-size QSA raw-key ring bill. These are valuable earlier upstream improvements, not new speedups attributable to this final PR. The final source integration also retains newer upstream MTP/batching and per-request Flash-Next state changes.

## Disposition of every related issue/PR from our side

| Thread | Final disposition |
|---|---|
|[#408](https://github.com/ddalcu/mlx-serve/pull/408)|**Only open consolidated PR**: code, reproduction and final report; draft pending gates above. Reviewers asked for stable-head M4/quality review, not another investigation thread.|
|[#375](https://github.com/ddalcu/mlx-serve/pull/375)|Close as superseded for our ongoing work. A gather path/test seam was already cherry-picked upstream according to13scoobie; it is not being reverted or claimed newly landed by#408. The remaining requested automatic-thread policy, accurate engagement label and purge pressure acceptance were **not completed**.|
|[#368](https://github.com/ddalcu/mlx-serve/issues/368)|Close the tracking issue: original prefetch gate fixed in862bddf; pressure follow-up from#375 remains unproven and is archived here. Closing means research consolidated, not pressure performance proven.|
|[#366](https://github.com/ddalcu/mlx-serve/issues/366)|Close the research ledger after linking this report/#408. Original apparent MoE throughput gap was a chunk-size comparison, not an unresolved kernel defect; upstream already sorts wide MoE. New findings and all rejected ideas remain linked.|
|[#385](https://github.com/ddalcu/mlx-serve/pull/385)|Already closed, integrated via[#388](https://github.com/ddalcu/mlx-serve/pull/388) by beamivalice. Keep closed; that upstream NAX baseline is distinct from paired QSA in#408.|
|[#365](https://github.com/ddalcu/mlx-serve/issues/365)|Already closed/fixed in26.9.2: Anthropic system-message preservation. No new action or prefill speed attribution.|

The#375 requested pressure protocol remains recorded precisely: mixed pack, `MLX_SERVE_NGRAM_WARM=0`, purge before each65K request, unique salt,max_tokens1, three alternating runs per arm/boot, plus no-purge control. Acceptance requested>=3% pressure gain with lower `ple gather S=4096` time and neutral warm control. It was not run; do not infer it from the bounded-ahead results.

## Rejected and unfinished hypotheses

These are archived to avoid repeating weeks of low-yield experiments. Component times are from each cited run's own controls, not composable model ratios.

| Hypothesis | Finding | Status / next condition |
|---|---|---|
| Whole-table GPU PLE / managed mapping | Standalone large win, actual model regression; managed/storage restrictions and page-ins matter | Rejected; use bounded selected rows |
| One resident MoE gate/up/down pass | About63ms vs25ms native; register/shared-memory schedule cost | Rejected |
| Larger register tile / BM192 gate-up | Earlier28vs25ms; newer MPP BM19221.945 vs BM64~21.452 | Rejected for current shape |
| Direct-input expert kernel | Inclusive improvement insufficient versus counted/sorted native; wider variant remained slower | Archived, no large model claim |
| Exact dense expert cache | At160rows/expert9.18ms cached vs~6.04native; inclusive13.66. At1280rows cached38.15 vs41.02–41.24, before wider model costs | Reject large-weight expansion as current lever |
| INT4 codebook / shuffle dequant |8.51vs7.29–7.38ms, exact | Rejected |
| Expert-aligned staging / virtual gate-up interleave | Roughly10–16% component signal, overlap with grouping benefit | Retained only where final MPP inclusive evidence supports it; no extra multiplied gain |
| New MPP down projection | BN128 whole chain22.001 vs gate-up-only21.694/22.677; BN25629.861 | Rejected; keep native down |
| Hoist MPP weight-bank address selection |21.772 vs22.237/22.021ms, exact | Small component gain only; not integrated |
| Bigger MoE batch |8K21.465/21.970ms;16K40.328;32K75.711, exact | Per-token component improves; whole-model32K experiment regressed |
| Wider whole-model prefill |32K/QSA precision/tail checks passed;65K prefill2016.9,peak84.04GB | Reverted; patch archived, not in final runtime |
| Increase command-buffer memory budget4096MB | No material improvement | Rejected |
| Cache exactly expanded dense/HC weights | Dense projections already use upstream DQGEMM at large M; HC4.159vs3.950/4.031ms even excluding expansion | Rejected |
| QSA Q-fragment reload / packed blocks |16.951/21.007ms vs retained~14.19 | Rejected |
| QSA separate QK/PV groups / pipeline |19.04/19.69ms (addressable-union variant72.85) vs~14.19 | Rejected; reduced spilling not a measured hardware counter |
| QSA full D256 in one SIMD group |62.060ms vs13.753paired | Rejected |
| Four-query QSA packing | K/V tile count-30%, active16-head tiles+3%; bucketing fragmented | No credible inclusive large gain; no full-model run |
| One-call FP32×BF16 PV |12.413vs13.70–13.85ms, but RMSE1.93e-5 vs1.16e-6 and larger max error | Rejected quality tradeoff |
| Strict FP32 PV cooperative layout | Incorrect max difference0.204834, slower | Rejected implementation; new layout derivation required |
| Coarsened online QSA max |16.131vs13.81ms | Rejected |
| Split QK→softmax→PV | About0.83GB FP32 score scratch on S4096 fixture, several GB extra traffic | Unimplemented; first demonstrate inclusive component win and bill scratch |
| Shape-specific tail JIT | Pinned MLX passes shapes/strides as runtime buffers; actual dimensions are not kernel-name keys for same type/template class | Proposed explanation disproved by source |
| Disabled MTP running during prefill | Generation/head history gated by mtp_active; some head/rerank preparation still at load | Not a timed-prefill lever; no change |
| CPU-only parallelism for1.5× | Host tens of ms vs GPU seconds on studied normal path | Insufficient budget; pressure PLE is separate unproven case |
| GDN core alone | Roughly0.288s/chunk; not the whole GDN block budget | Need full-stage cost reduction, not recurrence-only ratio |
| More cooling guarantees speed | Lower observed peak77C, clock still varied1095–1620MHz | Thermal state matters; no proven clock root cause or permanent fan requirement |

Other future work worth reconsidering only with a new inclusive cost model: shorten QSA register lifetimes without score-spill traffic; fuse larger exact MoE segments without register overcommit; reduce full GDN projection/materialization costs; bounded PLE pressure behavior. No evidence supports promising1.5× from any one of these remaining sketches. The current research phase does not schedule more tests.

## Reproduction and evidence map

Use the final PR runtime or explicitly checkout087c210 to reproduce the published2347 candidate. Build with the stated Zig/MLX/SDK provenance, ReleaseFast, private per-checkout caches. Do not share binaries across worktrees without hashing them. Use an exclusive GPU window; do not run while another training/UT session uses GPU/RAM. The driver owns and terminates only its own server/monitor.

```sh
PLE_BENCH_TAG=cooled-short python3 research/followups/bench-prefill-combined.py \
  --sequence 070770 --cooldown-c 50 --monitor /path/to/macmon \
  --out /tmp/qwen-cooled-short --lock /path/to/gpu_lock.sh
PLE_BENCH_LONG=1 PLE_BENCH_TAG=cooled-long \
  python3 research/followups/bench-prefill-combined.py --sequence 0707 \
  --warmup-short 2 --cooldown-c 50 --monitor /path/to/macmon \
  --out /tmp/qwen-cooled-long --lock /path/to/gpu_lock.sh
```

Run from repository root. Sequence0 keeps QSA+HC+GDN;7 adds group+PLEahead+HCupmix;f adds optional MoEMPP. The first two long-mode requests are13.5K warmups. Same-process arm selection is a research control, not a user-facing serving feature. The current driver supportsf; the original087c210 driver did not yet have all cooldown options, so use the published final driver with the explicitly selected binary/source corpus.

For manual serving of the2347 feature combination, set `MLX_SERVE_PREFILL_CHUNK=8192 MLX_SERVE_QSA_PAIR=1 MLX_SERVE_HC_PREFILL=1 MLX_SERVE_GDN_PREFILL_FUSED=1 MLX_SERVE_PLE_PACKED=1 MLX_SERVE_PLE_AHEAD=1 MLX_SERVE_MOE_PREFILL_GROUP=1 MLX_SERVE_HC_UPMIX=1 MLX_SERVE_MOE_PREFILL_MPP=0`, use `--prefill-chunk 8192 --ctx-size 131072 --prefix-cache-entries 0 --kv-quant off --no-mtp --no-pld --no-drafter`, and the exact mixed-pack model. Leave all replay/capture/sequence env unset for ordinary serving. Verify engagement, actual token count, cached_tokens, prompt hash and binary hash from logs rather than assuming launch flags prove a path ran.

For the maintainer's acceptance run use `tests/bench_qwen4_prefill.py`/`tests/bench.sh` and the full named OFF/ON llmprobe ladder; its existing core-arm driver must explicitly configure the new levers before treating it as a combined-bundle comparison. Save both prefill and decode, model/tokenizer and binary hashes, launch env/flags, engagement, real token counts, warmup policy, AC/thermal state and allocator peaks. A failure to meet cooldown or load readiness is an incomplete arm, not zero tok/s. Do not retry heavy tests for a weak component hypothesis.

- [Matched cooled requests and reduced telemetry](cooled-prefill-20260913/), [method report](20260913-cooled-prefill.md).
- [Final source manifest, full-suite output, live MPP and later/negative full-model artifacts](final-prefill-20260913/).
- [Earlier full llmprobe ladder and engagement](../reviewed_ladder_20260913/).
- [HC native inject red/green](20260912-hc-native-inject.md), [tail/HC3 validation](pr408-tail-validation/), [HC-upmix actual parity](20260913-hc-upmix-fusion.md).
- [PLE residency diagnosis](20260912-ple-residency-investigation.md), [bounded ahead and grouping](20260912-prefill-ahead-and-grouping.md), [live MoE/drift](20260913-live-moe-and-thermal-drift.md).
- [Earlier component hypothesis ledger](README.md), [larger lever analysis](20260912-large-levers.md), [next-lever budget](20260913-next-levers.md).

Only source, aggregate measurements and the public-source recall workload are published. Model-derived raw routing arrays/weights/activations remain local. A workspace GPU-lock helper was also corrected: status is read-only and EPERM must not be mistaken for a dead PID; that local orchestration fix is not an mlx-serve runtime speedup.
