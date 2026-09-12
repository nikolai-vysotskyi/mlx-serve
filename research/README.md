# Qwen3.8-Flash-Next prefill: current research handoff

Updated 2026-09-12. Implementation checkpoint: `7797ccc` (same production sources as PR commit `44bd600`), branch `perf/qwen-prefill-paired-qsa` in `nikolai-vysotskyi/mlx-serve`. Upstream `ddalcu/mlx-serve/main` was refreshed and is still `fa76a4b50b3f54af7e9cd927279f5ba2870f02c6`.

## Objective and working style

The user's objective remains **>1.5× whole-model prefill on M5 Max 128 GB**, with the same model quality. Their stated clean-main baseline is 1868 tok/s; >1.5× means **>2802 tok/s**. The broader requested range is 2500–3000 tok/s. This target has **not** been achieved or validated by this branch.

Continue investigating and implementing autonomously. The user authorized code changes, reversible local experiments, GitHub issue updates, PRs when ready, and relevant maintainer pings when there are concrete results. Do not stop to ask whether to continue or whether to implement a promising idea. Follow the execution environment's actual permissions and the repository's contribution rules.

Prefer analysis and implementation over repeated long tests. Seek changes that remove substantial arithmetic, memory traffic, synchronization or an expensive dispatch path. Estimate the affected share of a normal forward before running a long model benchmark. A 2× component win on 10% of the time saves 5% of the total; component ratios do not multiply into a model ratio. Combine compatible savings on disjoint work. A single 5–10% component tweak usually does not justify another long run.

## What actually works

1. **Paired QSA** partitions adjacent queries' selected blocks into private-left, private-right and shared buckets. Common K/V is staged once. Exact selected-key multisets, causal tails and the attention budget are preserved; softmax traversal/reduction order changes.
   - S4096/KV36864/Q24/KV2/D256, real captured block map and synthetic BF16 Q/K/V: stock 29.5954 / 29.5001 ms -> 14.1949 ms, **2.08× component**, including the GPU planner.
   - `qsa_pair/README.md`, `qsa_pair/probe.cpp`, `qsa_pair/validate.cpp` and raw JSONL files contain reproduction and numerical evidence.
2. **HC write/read fusion** combines pending write, group RMS norm, norm weights and inject products; the last kernel mixes four streams without a full-width product buffer. Native projection GEMMs remain.
   - M8192/HC4/H2560/rank320: stock 6.14867 / 6.12183 -> 3.91179 ms, **1.57× component**.
   - BF16 write/norm/mix rounding is preserved. Inject reduction order changes. The float64 check covers 144 gates and is limited evidence, not a broad quality evaluation. See `hc_prefill/README.md`.
3. **First-chunk routing**: the old 8192-key gather crossover was inherited from an unshared kernel. It excluded the first chunk even with paired QSA enabled. The dispatcher now lowers that floor for supported paired-prefill geometry. Explicit `MLX_SERVE_QSA_GATHER_MIN_KV` / test overrides and unsupported shapes retain their previous policy.

Both optimizations remain **opt-in**, not production defaults:

```sh
MLX_SERVE_QSA_PAIR=1 MLX_SERVE_HC_PREFILL=1 \
  ./zig-out/bin/mlx-serve serve --host 127.0.0.1 --port 11234 \
  --ctx-size 131072 --prefix-cache-entries 0 --prefill-chunk 8192
```

Do not set `MLX_SERVE_QSA_GATHER_MIN_KV` for the new automatic dispatcher. Setting it to 8192 intentionally restores the legacy crossover. Set both optimization flags to 0 for the same-binary disabled control.

## Current live result: short HTTP comparison

M5 Max 128 GB, macOS 26.5, model `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`, base fa76a4b, server MLX source `1f8e74e3f12f31365464a6867c6579f0e9b29d85`. Same final binary, chunk8192, prefix cache entries0, KV quant off, MTP/PLD off in the request.

| Prompt | Disabled control | QSA + HC + first-chunk dispatch |
|---|---:|---:|
| 13,515 tokens, first long request | 1835.2 tok/s | 2251.1 tok/s |
| 15,715 tokens, second/warmed request | **1989.0 tok/s** | **2397.0 tok/s** |

The second-request ratio is **1.205× / +20.5%**. Each arm is one short HTTP cell; this is **not llmprobe, not a 65K comparison, and not evidence of >1.5× over 1868**. Earlier controls produced somewhat different numbers; do not cherry-pick the slowest one. A separate threshold-override experiment gave 2383.4 tok/s on the second request. The old first-chunk mask route with the two kernels enabled gave 2166.7 tok/s.

All on/off requests returned `MAGNOLIA-7731`, with zero cached prompt tokens. `hc_prefill/http-smoke.json` records every relevant arm, flags, token counts and engagement lines. This recall check establishes basic execution and routing; it is not a broad model-quality test.

## Validation already performed

- ReleaseFast server builds passed on this Mac, including the final first-chunk dispatcher: 7/7 build steps.
- Focused QSA and HC Zig regressions passed: each reported 9/9 build steps and 3/3 executed tests. These are **not** full-suite results.
- Paired-QSA original float64 validation: 43,008 sampled outputs over four shapes; every row's selected-key multiset matched. Includes B2, changing widths, strided inputs and tails.
- Newly enabled S8192/KV8192 first chunk: all 8192 rows' selected-key multisets matched, 8192 outputs checked against float64, including positions 2047/2048. Stock/new maximum errors match. See `qsa_pair/first-chunk-validation.jsonl`.
- Repeated two-shape HTTP runs on the built server passed. The final on arm logs `[qsa-pair] engaged: S=8192 kv=8192`; disabled control logs the existing NAX gather.
- **Full suite now passed on the clean PR worktree:** 9/9 steps, 2349 tests passed, 154 skipped, 0 failed. ReleaseFast 7/7 steps also passed. `pr-validation.json` identifies the exact production-source hashes.
- **Draft PR:** https://github.com/ddalcu/mlx-serve/pull/408 . It contains only the eight production/test files; component probes and research logs remain on this research branch.
- **Not done:** dedicated paired-kernel startup probe/fallback, planner-memory admission accounting, broad HC/model-quality validation, and final long-prompt same-session llmprobe A/B. The PR stays draft.

Before a PR, read `CONTRIBUTING.md`, `CLAUDE.md`, relevant engine gotchas and `.claude/skills/bench/SKILL.md`. The repository prohibits even a draft PR until that exact tree builds and its full Zig suite passes on a real Apple Silicon Mac in the session. A performance claim in a PR needs the prescribed llmprobe comparison. Keep research logs and discarded prototypes out of the eventual small production PR.

## How to reproduce without a long model run

Build the repository's pinned dependencies first; do not silently substitute another MLX release/metallib. Then, from repository root:

```sh
xz -dk research/qsa_pair/blocks-4096.bin.xz
clang++ -O3 -std=c++20 -mmacosx-version-min=26.3 research/qsa_pair/probe.cpp \
  -I lib/mlx-src -L lib/mlx/lib -lmlx -Wl,-rpath,"$PWD/lib/mlx/lib" -o /tmp/qsa-pair-probe
/tmp/qsa-pair-probe
clang++ -O3 -std=c++20 -mmacosx-version-min=26.3 research/qsa_pair/validate.cpp \
  -I lib/mlx-src -L lib/mlx/lib -lmlx -Wl,-rpath,"$PWD/lib/mlx/lib" -o /tmp/qsa-pair-validate
/tmp/qsa-pair-validate first-chunk
```

The last argument selects only the newly enabled first-chunk shape. Omit it for the earlier four-shape oracle. The compressed fixture contains selected block indices only, not prompt text, model weights or activations.

For HTTP, start a disposable server as above, make one short readiness request, then run:

```sh
python3 tests/test_qsa_pair_prefill.py --url http://127.0.0.1:11234 \
  --log /path/to/that-server.log --pair on --hc on --first-chunk
```

Start a fresh disabled server for `--pair off --hc off` without `--first-chunk`. The script uses fixed public-source snippets, disables MTP/PLD, checks uncached usage and the actual kernel engagement after its requests. Never terminate another person's server to obtain a port or GPU. Use the site's existing exclusive GPU lock and release it in cleanup.

For eventual long-prompt proof, freeze prompt bytes/token counts, model, context, KV dtype, cache policy, speculative settings, runtime, chunking and binary SHA. Use the same final binary's flags for isolation and identify the clean base separately. Measure prompt prefill, not decode speed or streaming delivery latency. Warm both arms equivalently; use the context-scaling rung in llmprobe, not its unrelated short-prefill headline. Run one justified A/B and the necessary confirmation, not a long sweep of speculative parameter combinations.

## Do not repeat these rejected directions blindly

`followups/README.md` and the component folders record the evidence. Larger generic NAX tiles, dense expert weight expansion, a physically transposed dense layout, direct native MPP GEMM, a BF16 INT4 codebook, Q reloads, packed K/V layouts, and several QSA role/pipeline arrangements did not beat the best relevant path by enough. An addressable union made one QSA role prototype extremely slow; explicit packed registers reduced 72.85 -> 19.04 ms, but the working paired kernel is still 14.19 ms. Pipelining those roles gave 19.69 ms. Do not ship a prototype just because it beats the old 29.5 ms baseline.

Native BF16 × packed UINT4 factorization is in `native_int4/`. It changes per-weight BF16 rounding and is slower than stock even after unrolling. Treat it as a rejected investigation, not lossless repacking or an accepted quality-preserving optimization.

The remaining large target still needs another substantive reduction of model work. Stronger candidates should explain how they avoid repeated expert projection work/data movement, improve the full GDN/projection chain, or change scheduling without keeping huge intermediate buffers alive. Do not present approximate expert pruning, smaller QSA budgets, lower precision, cached prefixes, or dropped context as the requested result.

## Continuing from the cloud

A cloud session with GitHub can fetch this branch, inspect upstream changes, derive and implement kernels, build CPU references, check geometry/lifetimes, and publish concrete hypotheses in issue #366 immediately. It need not ask the user to reconfirm the research scope. Lack of Metal blocks target-hardware validation only; continue independent analysis and implementation and prepare exact reproduction commands for an authorized M5 runner. Never substitute Linux/CUDA throughput for M5 results.

Do not assume ordinary hosted macOS CI is an M5 Max with 128 GB or that it holds this model. Use only provided runner access. If no such runner is available, clearly mark the hardware check pending and continue other useful work. Do not claim completion of the throughput objective.

Existing discussion: https://github.com/ddalcu/mlx-serve/issues/366 . The paired kernel and HC component results, code links and short combined results are recorded there. Prefer updating this issue to creating duplicates.
