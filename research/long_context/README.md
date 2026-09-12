# Long-context PR #408 validation

Measured on 2026-09-12 on Apple M5 Max 128 GB / macOS 26.5. These are same-binary, same-session comparisons of PR head `092ce2e`, based on upstream `fa76a4b`. The original 4-bit checkpoint has an off/on comparison; the recommended mixed-4-8bit checkpoint has an off/on/off confirmation.

The measured metric is `bench.contextScaling[0].prefillTokPerSec` from **llmprobe 0.6.6**, `--bench-only --rungs 64k --runs 3`. It is input tokens divided by client time to first text token, rounded after taking the median. The unrelated roughly 2K short-prefill headline is not used. Three coding requests contribute to each rung; three additional predictable-ceiling requests are interleaved by the unmodified harness. Earlier scenarios have discarded warmups; this version does not discard a separate context-rung warmup.

## Results

| Checkpoint / arm | Input tokens | Long prefill tok/s |
|---|---:|---:|
| Original 4-bit, off | 68,651 | **1644** |
| Original 4-bit, on | 68,447 | **1775** |
| Mixed 4/8-bit, off A | 68,447 | **1626** |
| Mixed 4/8-bit, on | 68,539 | **1879** |
| Mixed 4/8-bit, off B | 68,651 | **1643** |

Original 4-bit: **+7.97%**. Recommended mixed checkpoint: **+14.36% to +15.56%** against the two bracketing disabled controls. The two mixed controls differ by **+1.05%**. Each cell is a median of three long coding requests; the harness also sends three separate predictable-ceiling requests per boot.

**The earlier 2509.4 tok/s at 15,715 tokens was not reproduced at 64K+ context. Neither the 2500–3000 tok/s target nor >1.5× whole-prefill speedup is established.** The different checkpoint, prompt and length prevent treating the earlier short HTTP cell as the long-context result. The controls here are the frozen PR binary with the three opt-ins disabled, based on fa76a4b; they are not the user's historical 1868 tok/s measurement.

## Fixed setup and interpretation

- Original model `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`: affine group64, 793 quantized tensors at 4 bits and one 8-bit `language_model.lm_head.weight`, as inferred from checkpoint tensor geometry. This is a different checkpoint from `Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` used in the reviewer comment. Unquantized tensors retain their checkpoint types.
- Recommended model `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`: 646 quantized tensors at 8 bits and 148 at 4 bits, affine group64. Each model is compared only against itself.
- Explicit `--prefill-chunk 8192`, context131072, prefix-cache entries0, KV quant off. Long server traces must show nine chunks of width8192. The supported final tail also runs paired QSA.
- All three flags (`MLX_SERVE_QSA_PAIR`, `MLX_SERVE_HC_PREFILL`, `MLX_SERVE_GDN_PREFILL_FUSED`) are 0 for off and 1 for on. This compares the combined patch against its disabled upstream paths; it is not a separate clean-main binary measurement.
- MTP is default off. **PLD is requested on by the launch default and the HTTP n-gram gate, but the Qwen4 scheduler disables it before Generator initialization.** `moduleSpecWiring()` covers the shared Qwen4 module and `specInitWiring()` returns `use_pld=false`. All long traces lack the Generator’s `[pld]` suffix and PLD speculation statistics. The earlier short HTTP smoke explicitly requested PLD off; neither run actually executes PLD. Launch intent must not be used as engagement evidence.
- Standard llmprobe nonces differ between boots, and token calibration therefore produces slightly different prompt lengths. Exact counts are in every cell; this is the same harness/workload with a small reported length mismatch, not identical prompt bytes.
- AC power, 100% charge at start, exclusive site GPU lock. No foreign model/server was terminated. The post-load n-gram page warm completed before measurement in all arms.
- `QWEN4_PLE_PAR=16` was set by the existing driver, but PR #375 is absent from this base, so it has no effect.

`manifest.json` contains source, executable, MLX runtime and model-header fingerprints, settings and result cells. Model-header hashes identify metadata/geometry, not all weight payload bytes. The unchanged PR sources were verified against the earlier passing ReleaseFast and full-suite checkpoint in `../pr-validation.json`.

## Reproduction

Build PR head `092ce2e` in ReleaseFast using its pinned MLX/runtime. Reserve an otherwise idle M5 with this checkpoint. Run `reproduce.sh` from a shell with no unrelated performance overrides, passing absolute paths. It starts one owned server, performs the same short readiness request, then runs the public llmprobe command. It does not replace an occupied port or terminate another server.

```sh
bash research/long_context/reproduce.sh \
  /absolute/checkout/zig-out/bin/mlx-serve \
  /absolute/runtime/mlx/lib:/absolute/runtime/llama/lib \
  /absolute/site/gpu_lock.sh off /absolute/results/off-a
# Repeat with on /absolute/results/on, then off /absolute/results/off-b.
# The optional final argument selects the model; default is mixed-4-8bit.
```

The site lock implements `acquire NAME` and `release NAME`; all GPU workloads must honor it. The measured runs used the pre-existing local `bench_prefill_current.py` wrapper and `bench_prefill_mixed.py`, a copy changing only the model identifier. `reproduce.sh` is a portable version of the same launch/readiness/llmprobe sequence, syntax-checked without rerunning the model solely to validate this wrapper.

The PR remains opt-in. Long-context throughput is not broad model-quality validation; paired-kernel startup fallback, planner admission accounting and broader HC numerical coverage are separate readiness items. Do not infer a >1.5× model speedup from earlier component benchmarks.

## Short-query bridge and discrepancy checks

A fresh run of the original HTTP recall script, now explicitly selecting the **mixed-4-8bit** model, measured **1985.5 tok/s at 13,515 tokens and 2141.6 tok/s at 15,715 tokens**. It uses the same frozen PR binary as the long comparison. Both replies recalled the passphrase, with zero cached tokens and all three kernel engagement lines. `short-bridge.json` and its server excerpt hold the evidence. This is one ON smoke, not a new A/B or an improvement over the long result.

- The earlier **2509.4** was a 15,715-token HTTP cell on the original 4-bit pack. The new **2141.6** is a matching-length HTTP cell on the recommended mixed pack. The **1879** long cell uses that mixed pack but a different, roughly 68.5K llmprobe archive prompt. These are distinct workloads.
- Timing definitions do not explain the long discrepancy: the mixed ON coding-request median is **1882.5** in server logs and **1879** in llmprobe, only about 0.2% apart.
- PLD does not explain it either. The launch/HTTP gate prints it as enabled, but Qwen4 module wiring forces it off before Generator initialization. All relevant traces lack `[pld]` and PLD `spec-stats`. The metadata now distinguishes request intent from execution.
- The reviewer screenshot has disabled/enabled pairs **1877.4 → 2060.2** and **1684.5 → 2083.1**: arithmetic gains of 9.7% and 23.7%. Prompt lengths and complete commands have been requested in [the PR reply](https://github.com/ddalcu/mlx-serve/pull/408#issuecomment-5645273726); these cannot yet be treated as matched 64K cells.

First-chunk savings being spread across more chunks is a plausible contributor, not a measured explanation of the entire gap. The remaining prompt/length sensitivity still needs attribution before choosing the next substantial optimization. No additional long benchmark was run for this check.
