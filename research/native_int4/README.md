# Native packed INT4 research — rejected prototype

M5 Max 128 GB, same runtime as ../qsa_pair/provenance.json. This uses Metal Performance Primitives BF16 × packed UINT4 -> FP32 per group of 64, then applies scale and bias to the partial products. It removes software weight dequantization from that kernel. It does **not** preserve the stock per-weight BF16 rounding: algebraic factorization moves scale/bias outside the sum. It is not integrated or a quality-preserving model optimization.

Seeded affine4/gs64, E512, 160 tokens/expert, K2560, N640: stock 6.01958 / 5.96971 ms; unrolled prototype 10.457 ms, including input group sums. Relative RMSE 0.00247545. Dynamic cooperative-tensor indexing was still slower (20.3863 ms); explicit unrolling cut its cost but did not beat stock. Small shape has E1/M32/K256/N64. See JSONL files.

`factor_broadcast_failed.metal` is retained only as a rejected coordinate-mapping experiment: a source cooperative tensor's linear lane indexing did not match the accumulator's. The corrected kernel addresses per-output scale/bias with `get_multidimensional_index`. It is the only variant used in the reported corrected runs.

Compile `probe.cpp` from repository root like the HC probe. Pass any argument for the large shape; no argument selects the small arithmetic check. Do not ship either experiment or claim a full-model gain.
