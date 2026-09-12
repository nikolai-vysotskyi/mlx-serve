# HC native inject reduction after the M4 review

The [M4 reviewer report](https://github.com/ddalcu/mlx-serve/pull/408#issuecomment-5646303757) at production commit `092ce2e` observed deterministic OFF/ON generation divergence with HC and GDN enabled and QSA unavailable. This does not isolate the cause or measure quality degradation, but it makes the known HC inject reduction-order difference actionable.

The correction retains fused pending write, group normalization and stream mixing, while restoring native MLX `matmul(normalized, inject_weight)` for the four inject channels. It does not change model weights or quantization. A dense-weight regression fixture replaces the earlier one-hot mapping fixture: the old kernel fails exact parity (maximum raw-inject difference 0.0000076293945), while the corrected kernel passes across cold/pending shapes B2/S17, B1/S65 and B1/S513.

Synthetic S8192/HC4/H2560 component, three timed repetitions per mode, M5 Max 128 GB, same runtime as the long-context investigation:

| Projection quantization | Stock before | Previous fused HC | Native-inject HC | Stock after |
|---|---:|---:|---:|---:|
| 4-bit | 6.42583 ms | 4.10658 ms | 4.42950 ms | 6.46367 ms |
| 8-bit | 6.54479 ms | 4.20196 ms | 4.73458 ms | 6.62771 ms |

All five returned tensors (mixed stream, inject gate, normalized stream, raw inject, updated stream) are bit-identical to stock for the corrected variant in both fixtures. The previous fused variant differs by 0.0078125 in the raw/gated inject outputs on this fixture. Component acceleration remains about 1.45–1.46× for 4-bit and 1.38–1.40× for 8-bit; these are not whole-model gains or broad quality evidence. The correction is slower than the previous fused component, and the previous `092ce2e` long result of 1879 tok/s must not be attributed to the corrected head without a new measurement.

Reproduce from repository root:

```sh
clang++ -O3 -std=c++20 -mmacosx-version-min=26.3 \
  research/followups/hc-native-inject-probe.cpp -I lib/mlx-src -L lib/mlx/lib \
  -lmlx -Wl,-rpath,"$PWD/lib/mlx/lib" -o /tmp/hc-native-inject-probe
# Hold your site's exclusive GPU lock for each invocation.
/tmp/hc-native-inject-probe
/tmp/hc-native-inject-probe 8bit
```

The standalone probe keeps the historical unused partial-output argument to compare all three implementations. Production removes that argument and output entirely. Native matrix multiplication remains a separate graph node.

The M4 result still needs isolation or validation on the corrected head. This M5 component test does not establish that HC was the sole cause of M4 generation divergence. The PR remains draft.

Production correction `1b1a4df`: ReleaseFast 7/7 steps and full Zig suite 9/9 steps, 2329 passed, 154 skipped, zero failures (helper executable cached). The mixed-checkpoint HTTP smoke recalled the passphrase at 13,515 / 15,715 uncached tokens, with QSA, HC native inject and cold GDN engagement. Server rates 2038.8 / 2436.7 tok/s are single-arm observations, not a new speedup measurement. See `hc-native-http-smoke.json` and `../pr-validation.json`.
