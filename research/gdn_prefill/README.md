# GDN prefill fusion, 2026-09-12

Opt-in `MLX_SERVE_GDN_PREFILL_FUSED=1` enables the existing packed convolution/SiLU/QK norm/gate/beta and output norm/gate kernels at prefill width. Guards retain the supported BF16 D128, scalar unbounded gate, kernel4 geometry. B1–2 and 17–8192 total rows are supported. The first chunk supplies zero convolution history; later chunks consume and replace the existing history. Unsupported arms retain their composed implementation.

The new wide tests initially caught a one-ULP beta difference between JIT sigmoid arithmetic and MLX's compiled BF16 sigmoid. Wide prework and norm/gate now share HC's 65,536-entry native-operation sigmoid table. Existing decode-width formula/dispatch remains unchanged. This preserves BF16 rounding at convolution, SiLU, gate and beta sites; weights and precision are unchanged.

`http-smoke.json` is a same-binary M5 Max 128 GB comparison with paired QSA and HC enabled in both arms. Only this GDN flag changes. At 15,715 prompt tokens, the second/warmed request rises from **2395.7 to 2509.4 tok/s (+4.75%)**; at 13,515 tokens, 2316.5 to 2380.1. Both arms return the expected passphrase with zero cached prompt tokens. The on server logs `S=8192 B=1 cold=true` for GDN, proving first-chunk engagement. This is short HTTP evidence, not long-context llmprobe or proof of >1.5× whole-model speedup.

The earlier standalone prework investigation motivated integration (3.91867 to 1.62408 ms at S8192/HK16/HV48/D128), but predates the native sigmoid-table fix. It is not the timing of the final production kernel. The HTTP result above measures the actual implementation.

Reproduce with the three flags and HTTP command in `../README.md`. For the incremental control keep QSA/HC on and set only `MLX_SERVE_GDN_PREFILL_FUSED=0`, passing `--gdn off` to the script. Use a fresh disposable server and identical request order in each arm. Numerical checks extend the existing GDN Zig tests to S17,65,513, cold/history, B2 and folded inputs, including exact comparison of all six prework outputs and both output-gate formulas. Full-suite/build checkpoint details are in `../pr-validation.json`.
