# Reviewer confirmation and prefill configuration, September 14

[beamivalice's report](https://github.com/ddalcu/mlx-serve/pull/408#issuecomment-5658856761) provides an independent positive comparison of PR9644a6b against current main plus two local MTP commits. The exact control SHA and launch environment have been requested. These are reported results, not a new run on our local machine.

| Cell | PR | Control | Reported change |
|---|---:|---:|---:|
|2,042-token llmprobe prefill|1791|1694|+5.73%|
|~16.3K context prefill|1858|1761|+5.51%|
|Cold64K, mean of two runs|1849.5|1749.5|+5.72%|
|Cold16K|1850|1753|+5.53%|

The64K individual values were1830/1869 for the PR and1747/1752 for control. Both comparisons have a positive sign. PR first, then control, same reviewer machine/session, fans at maximum, llmprobe0.6.7, planner warmed with three requests and prefix caching off. Order/session effects remain possible; the report is useful evidence, not a confidence interval or a broad quality gate.

Our earlier+7.97% was a different comparison: coreQSA/HC/GDN versus core plus boundedPLE-ahead, grouping andHC-upmix in one087c210 executable. Matching percentages do not make these identical experiments. The reported lower absolute throughput does not erase the positive within-session comparison, and cannot yet be assigned entirely to temperature/session state.

## Two source-proven configuration gates

### The CLI maximum is not an8192 pin

On9644a6b, `effectivePrefillChunk` first honors `MLX_SERVE_PREFILL_CHUNK`. Otherwise it calls `boundedPrefillChunk`, whose default fused-causal non-sliding hd256 MoE path caps the result at4096, including when the CLI asks for8192. The local mixed pack has head_dim256,24 query heads,2 KV heads and512 experts; it uses that path.

Both published benchmark drivers already set `MLX_SERVE_PREFILL_CHUNK=8192`. The manual command in `docs/qwen4-prefill-final.md` omitted it; **92be945 fixes that documentation error**. Runtime sources remain identical to9644a6b. The existing pure policy test passed3/3 including its two harness tests; no model run was required.

Use the exact pin for the historical8K configuration:

```sh
MLX_SERVE_PREFILL_CHUNK=8192 MLX_SERVE_PREFILL_TRACE=1 \
  zig-out/bin/mlx-serve serve --prefill-chunk 8192 <remaining recorded flags>
```

This is a fragment, not a standalone launch command; use the full corrected command in the PR. Verify `[prefill-trace] chunk_size` and `chunk_widths`. The environment pin bypasses the legacy shape cap; do not describe a CLI maximum alone as proof of an8K forward. This is a configuration finding, not a measured4K→8K speedup in the reviewer's run.

### MTP declines the existing PLE producer

`generate.zig` at9644a6b has an explicit `mtp_active` refusal when creating `ctx.ple_ahead`. The PLE flag alone therefore cannot demonstrate ahead engagement with MTP on. MTP also chooses `forwardWithCaptureAll` and runs `appendHistory` for the relevant prefill chunks. Its default history window is0, meaning full history; windowing is a separate behavior and acceptance setting, not a free prefill optimization.

Our2347 screen disabled MTP, PLD and the drafter. The reviewer's86.6/95.8 decode numbers make speculative-mode metadata relevant but do not prove the mode. Likewise, two control commits restricted to verify/decode kernels do not establish equivalence between an MTP-on serving setup and our MTP-off screen.

Requested **existing** evidence: exact launch argv/env, control and local-patch SHAs, llmprobe JSON, actual chunk traces, speculative mode/history-window and the feature engagement lines. This request does not require another expensive benchmark before inspecting the recorded setup.

## Concrete next compatibility candidate

[native-mtp-ahead-proposal.patch](review-20260914/native-mtp-ahead-proposal.patch) is a narrow **unintegrated, untested proposal**, against9644a6b. It would retain the producer only for native Qwen4 MTP attached to the same Transformer. Other MTP heads, adaptive widths, vision, DFlash and compiled-forward exclusions remain.

Source facts supporting investigation:

- `preparePleAhead` clones immutable row IDs from the trunk's saved n-gram history and the fixed token boundaries.
- `forwardWithCaptureAll` temporarily changes only hidden-capture pointers on the trunk context.
- `MtpHeadRef.appendHistory` calls `qwen4MtpForward`; that function constructs a separate `ForwardCtx` and uses embedding/attention/MLP, without consuming the trunk's PLE queue.
- The existing producer checks every row ID before consumption and joins on destruction/cancellation.

These facts motivate a compatibility change; they do not prove it. Syntax/format checking passed for the proposal, but it has not been compiled or run. No quality, speed, memory-pressure or current-main compatibility result is assigned to it.

Minimum useful validation before integrating:

1. A small native-MTP fixture alternates trunk capture and head-history appends across at least two fixed chunks; PLE ahead on/off must preserve logits/hidden state and both trunk/head histories.
2. Positive producer engagement plus negative sidecar-head, batched/deferred input, history mismatch and cancellation paths. Verify the head cannot consume, stop or advance the trunk's queue.
3. Reconcile with the new per-request/grouped MTP implementation on main; preserve ownership and graph lifetimes.
4. Only after those checks and the reviewer's mode/width metadata, consider one matched full-model MTP-on pair with adequate headroom and engagement logs. No >1.5× prediction is supported yet.

A second possible implementation direction is a model-specific, memory-accounted8K default instead of the coarse generic MoE cap. The current cap is intentional protection for other expert-gather geometries; removing it globally is not justified by one model's result. Existing128GB measurements show the explicitly pinned8K configuration is viable in that tested setup, not under arbitrary context lengths or memory pressure.

## Integration state and evidence

Upstream moved from0814cf3 to1075630, including grouped MTP, draft/verify scheduling,3-bit decode experts and load-time verify compilation. A read-only merge preview found conflicts in transformer/documentation. No new upstream integration or runtime optimization is claimed by this follow-up; the reviewer's measured runtime remains9644a6b and our PR documentation update is92be945. The conflict requires an explicit integration/validation step, separate from attributing these performance numbers.

- [Our reply and metadata request](https://github.com/ddalcu/mlx-serve/pull/408#issuecomment-5661653812).
- [Machine-readable arithmetic and CPU policy check](review-20260914/).
- [Original final report](20260913-final-prefill.md), with its manual environment pin corrected.
- Source anchors: [width policy](https://github.com/nikolai-vysotskyi/mlx-serve/blob/9644a6b/src/generate.zig#L151-L207), [ahead refusal](https://github.com/nikolai-vysotskyi/mlx-serve/blob/9644a6b/src/generate.zig#L2364), [capture/history](https://github.com/nikolai-vysotskyi/mlx-serve/blob/9644a6b/src/generate.zig#L2455-L2515), [head's separate context](https://github.com/nikolai-vysotskyi/mlx-serve/blob/9644a6b/src/transformer.zig#L20980).

No GPU/model benchmark was started for this investigation. The conclusion to defend is a reproducible positive direction with precise conditions; the absolute gap and MTP-compatible PLE gain remain to be established.
