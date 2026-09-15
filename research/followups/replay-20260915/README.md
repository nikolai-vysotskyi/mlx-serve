# M5 Max reproduction: 2323 tok/s at 64,947 tokens

Replayed the retained `087c210` executable and the frozen request on September 15, 2026. Full-model prefill reached **2323.423 tok/s**, **1.01% below the previous 2347.106**. The original 2300+ level reproduced.

| Same-process arm | Input tokens | Server prefill | Server time | Client wall |
|---|---:|---:|---:|---:|
| QSA-pair + HC + GDN |64,947|2163.887 tok/s|30.014049 s|30.260480 s|
| Add PLE-ahead/packed + MoE-group + HC-upmix |64,947|2323.423 tok/s|27.953158 s|28.165814 s|

The paired gain is **7.37%**. Both requests returned 10 output tokens, the expected passphrase, and explicitly zero cached input tokens. The faster arm still processes 2305.9 input tokens/s using the entire client wall time. This is one new A/B pair; the 13,515-token requests were warmups and excluded.

Same binary SHA, libmlx SHA and prompt/corpus SHA as the previous record, verified by assertions. Full configuration: `MLX_SERVE_PREFILL_CHUNK=8192`, context 131072, prefix entries 0, KV quantization off, MTP/PLD/drafter off, MoE MPP off; sequence 0707 with two short warmups. Before each request the original three-sample gate required GPU ≤ 50°C, power ≤ 5 W, fans ≤ 1600 RPM; no fan/power settings were changed. Preparation/cooldown time is outside request throughput. The first attempt before the user freed memory stopped at loading preflight and produced no speed result.

`result.json` contains the raw HTTP responses/timings, client wall times, matching hashes, model metadata fingerprints, engagement lines and pre-request telemetry. The original weights revision was not pinned, so the current metadata fingerprints are included without claiming a historical full-weight hash match.

The ~1800 results on another M5 remain a reproducibility gap to investigate. This replay establishes the 2300+ configuration again; the next task is isolating which enabled paths/settings deliver the gap in normal serving. PR #438 contains the HC/GDN subset; this is not a benchmark of that reduced tree. The combined-code reproduction stays in this research branch.
