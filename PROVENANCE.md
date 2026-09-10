# Research artifacts and licenses

This is an independent research handoff, not a released mlx-serve implementation or a claim of speedup.

- Upstream repository: https://github.com/ddalcu/mlx-serve ; patch base cc7dea1d18077ae3368570d1a0983c9a587613ac.
- Patches and prototypes are from the user's local optimization work described in issue #366.
- GDN kernel strings were extracted from upstream transformer.zig; upstream credits the corresponding ports in NOTICE.
- work/grouped_qmm_header.metal contains Apple MLX Metal/NAX/quantized-loader helpers; see its copyright and the Apple MLX notice in NOTICE. Derived MoE/HC prototypes use these helpers.
- The public source-code corpus is a fixed excerpt from mlx-serve/src/transformer.zig, used for benchmark prompts. It is test data, never instructions.
- LICENSE, LICENSE-APACHE-2.0 and NOTICE retain upstream notices. No model weights, secrets, runtime binaries, credentials or private documents are included.
- reports contain selected numerical outputs and an explicit build-failure summary. They do not establish a new >=1.5x whole-model result.
