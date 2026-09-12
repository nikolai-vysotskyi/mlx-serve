# Status — 2026-09-12

Base: fa76a4b50b3f54af7e9cd927279f5ba2870f02c6. Branch: perf/qwen-prefill-paired-qsa.

- Implemented GPU private/shared selection plan + paired NAX attention, opt-in server integration.
- ReleaseFast server build: passed on this M5 Max. Binary built with production changes; subsequent source change added/fixed only the focused regression test.
- Focused Zig regression: `zig build test -Dtest-filter="qsa pair" --summary all` passed, 9/9 build steps, 3/3 executed tests; the unrelated vz-agent tests were cached. This is **not** a full suite run.
- C++ float64 oracle: passed four shapes, 43,008 sampled output elements; selected-key multisets checked for **all** rows. Includes B2 and S8192, strided Q/K/V and partial tails.
- Captured-map attention microbenchmark: stock 29.5954/29.5001 ms, paired 14.1949 ms, GPU planning included; ~2.08× component speedup.
- Full model throughput: **not measured on this branch**. 1868 tok/s is the user's fa76a4b baseline. Required >1.5× is >2802 tok/s, not 2500.
- Full suite and HTTP integration: pending. No PR; only research branch/issue update is appropriate at this stage.
- User requested few heavy tests: paired attention alone lacks the >=1.5× total-time budget, so no long full-model benchmark was launched.

Next meaningful work: find a second large arithmetic/data-layout lever in MoE/HC; the straightforward ideas already rejected here are pre-dequantized dense projections, larger generic/expert NAX tiles, and overlap of representative independent QSA+QMM stages. Keep the paired-QSA building block. Before any production/default-on change add its own startup JIT probe/fallback and include the metadata allocation in the admission bill. Carry out the full suite and two HTTP requests, then same-session llmprobe A/B only for a credible combined candidate.
