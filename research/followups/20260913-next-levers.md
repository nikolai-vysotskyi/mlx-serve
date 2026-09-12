# Follow-up while PR #408's current-head ladder runs

Update: [live MoE parity and measured thermal/frequency drift](20260913-live-moe-and-thermal-drift.md) supersede the stationary-control assumption below. Actual-weight replay is faster; the full-model effect is still not established.

No new model speedup is asserted here. Target for this follow-up is 2400 tok/s at 65K, with the mixed pack and quality preserved.

1. **Measure normal forward scheduling.** `QWEN4_PREFILL_CADENCE_TIMING=1` in the research branch records host intervals and the existing `mlx_eval` intervals across the four-layer cadence. It introduces no extra GPU synchronization. These are host-side intervals, not GPU kernel timestamps; explicit evaluation inside an operation would be charged to the host interval. It excludes final mixer/head work. Only a substantial host interval would justify replacing the blocking cadence with a bounded, two-group producer/consumer scheme. Merely adding the old async ladder was already negative on another model.
2. **Weight reuse inside one SIMD group.** `moe-wide-register-gateup.metal` keeps the 192-row expert tile but reduces the six M SIMD groups to two (96 rows per group). It streams one A fragment at a time while reusing the B fragments across six M fragments, bounding the live input registers. The existing direct-input path used 32 rows per group; old large-tile experiments did not test this particular register-lifetime change. Native expert projections and the already measured counting-grouping chain remain controls. The model is not a valid next benchmark unless the inclusive chain improves substantially and matches BF16 outputs.
3. **Command-buffer batching.** This pinned MLX uses 50 operations OR 50 MiB as the M5 Max command-buffer commit threshold (`device.cpp`, `env::max_mb_per_buffer`). A single wide activation or expert weight array exceeds the byte threshold. A larger byte budget can reduce commits/fence bookkeeping without changing math, but no gain has been measured. Test the component with one larger budget before considering a model run; do not infer time savings from the count of commits alone. This can extend buffer lifetimes, so memory must be recorded.

The llmprobe calibration also depends on preceding rungs: a standalone 64K rung does not generate the same filler length as 64K inside a multi-rung ladder. A final single-rung OFF is therefore a coarse drift check, not a byte-matched A-B-A. Record actual token counts and do not use that extra arm to manufacture a precision speedup claim.


## Completed screens

- BM192/WM2 streamed-A gate/up: all synthetic BF16 outputs matched; inclusive S8192 chain 28.16 ms versus counting+native 25.85 ms and stock ~29.93 ms. The new tile lost; do not integrate or run it on the whole model.
- MLX_MAX_MB_PER_BUFFER=4096: counting+native ~26.38 ms, no benefit versus the default run. No model test justified by this result.
- Frozen binary `6302e6ba4be0307e30db60c32a0fcd36dfb918ed2a2bcf9b5abd2e0661f8451e`, d23d9df changes integrated on the PLE/MoE research branch, with a no-extra-sync cadence timer: matched 64,947-token HTTP current 1804.0 vs PLE+group 1485.6 tok/s. **Regression**, not an acceleration. Short cold current2082.1 vs combined1320.0. Same prompt hashes, zero cached tokens, exact passphrase recall. AC power; battery was70% charging for this screen. Not the same machine state or prompt as the older62,176-token HTTP pair.
- Follow-up short ablation, same binary and fixed prompts: current2102.6/2118.8, PLE-ahead1681.8/1973.6, group1985.0/2071.5 at13,515/15,715 tokens. Group's warmed change was -2.2%, PLE -6.9%. One screen, not a statistical claim. Cold PLE waited1764.65ms for packing; the warmed wait was7.80ms. This separates the first-request host penalty from the remaining graph execution regression, whose cause is not yet established.
- Cadence: the warmed current long forward spent tens of milliseconds building each chunk's graph (roughly55–80ms early) against several seconds of evaluation. The timer attributes explicit nested evaluations to its host interval, so it is not a CPU/GPU hardware profile. A CPU-only scheduling rewrite does not have a demonstrated large budget here.
- A Metal System Trace attached to the owned server, but the capture driver terminated the recorder after its finalization timeout; export failed `Document Missing Template Error`. No kernel-duration claim is supported by that incomplete artifact. Do not treat its request timings as uninstrumented throughput. Capture finalization needs a longer asynchronous wait before trying again.

Next: real-model routing fixtures (four first wide MoE calls) and their complete-chain component replay, then expert projections/layout. `QWEN4_MOE_CAPTURE_PATH=/output/routes` saves `.0.safetensors` through `.3.safetensors`, each with `indices`. It adds synchronization and must stay off for throughput. The probe accepts `large /output/routes.0.safetensors`; weights/activations remain synthetic, so this isolates routing distribution rather than establishing model numerical equivalence.


## Real routing result

The captured first/fourth wide MoE calls used496/486 distinct experts and a maximum1055/4296 rows for one expert, unlike the uniform fixture. Replaying those routes still improved the inclusive component from roughly29.57/30.20 to25.64/25.93ms, with exact BF16 outputs on the synthetic weights/activations. Therefore **routing skew alone does not explain the model regression**. Swapping in the exact production reduction source gave25.99ms versus25.97ms for the equivalent research source, also exact; a prototype-to-production reduction discrepancy was not reproduced. The remaining work is attribution on the complete live graph (actual expert weights/activation, compiled activation and neighboring work), not another tile-size sweep. No model throughput gain is inferred from these component checks.

The capture run uses an extra router synchronization and is explicitly diagnostic. It returned the expected passphrase, but its throughput number is excluded. Fixture provenance and component measurements are beside this report. The raw routing arrays are kept local; regenerate them with the command below.


## Regenerate routing fixtures locally

```sh
PREFILL_CAPTURE_ROUTES=1 PLE_BENCH_ARMS=current PLE_BENCH_TAG=route-capture \
  python3 research/followups/bench-prefill-combined.py \
  --binary zig-out/bin/mlx-serve --out /tmp/qwen-route-capture --lock /path/to/gpu_lock.sh
clang++ -O3 -std=c++20 -mmacosx-version-min=26.3 \
  research/followups/moe-wide-register-probe.cpp -I lib/mlx-src \
  -L lib/mlx/lib -lmlx -Wl,-rpath,"$PWD/lib/mlx/lib" -o /tmp/moe-route-probe
/path/to/gpu_lock.sh acquire route-replay
/tmp/moe-route-probe large /tmp/qwen-route-capture/routes.3.safetensors
/path/to/gpu_lock.sh release route-replay
```

Run acquire/probe/release in one shell parent, preferably with a cleanup trap. This uses the installed mixed pack and public-source benchmark prompt. Route capture synchronizes and is a diagnostic. Only code, aggregate timing/error measurements and ordinary benchmark logs are published here; the raw model-derived routing arrays are not distributed.
