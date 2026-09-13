# PR408 NAX availability regression

The failed CI run34762560533 reported `arch=air64_v27 macos=26.6.2 available=true`. Its trailing27 was parsed as a GPU generation. Strict Apple GPU family parsing fixes the shared runtime guard; no CI checks or numerical tolerances are weakened.

The pure regression failed before the fix with expected0/found27 and passes afterward. Positive cases retain desktop17+ and phone18+, case-insensitive family variants, verify-QMM and QSA eligibility. Negative cases cover the observed compiler target, other vendors, malformed suffixes, missing/oversized generation fields and embeddedNUL. The actual stock-gather fallback is bit-identical and builds no NAX kernel when availability is absent.

Local M5 Max128GB, macOS26.5, existing MLX0.32.3 runtime:

- Native architecture:2444 engine tests passed,155 skipped; `applegpu_g17s ... available=true`. The extra20 guest-helper tests were cached.
- `MLX_METAL_GPU_ARCH=air64_v27` rehearsal:2430 engine tests passed,169 skipped, plus20 guest-helper passes (2450 total passes); `air64_v27 ... available=false`. A separate Zig cache forced a fresh test run. This is a rehearsal onM5, not a claim to reproduce the CI hardware/SDK exactly.
- No full-model run and no new speed claim. See manifest.json for final build/commit and remote CI provenance.

Commands from PR checkout:

```sh
.zig-toolchain/zig build test -Dtest-filter=naxArchSupportedFrom --summary all
.zig-toolchain/zig build test --summary all
MLX_METAL_GPU_ARCH=air64_v27 .zig-toolchain/zig build test --cache-dir ../pr408-air-ci-cache --summary all
.zig-toolchain/zig build -Doptimize=ReleaseFast --summary all
```

Use the normal dylib setup and exclusive GPU lock. Logs are preserved as emitted, including the Zig runner's `failed command` diagnostic on successful runs; judge completion by exit0 and the final success/count summary.
