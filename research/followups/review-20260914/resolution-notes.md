# Main1075630 integration

No new performance measurement or MTP/PLE enablement.

- Preserve the strict, previously CI-tested NAX family parser; retain upstream's compiler-target regression cases as well.
- Factor PLE row-state advancement into pleRowsFromIds on evaluated IDs. Native/deferred grouped flush keeps upstream plePrepareIds/pleFillFromIds and the single grouped host sync; packed PLE reuses the same row/history transition.
- Extend the existing moeMLP2WithRouter core with the research group override. The normal/diagnostic wrapper supplies no router override and skip_shared=false; upstream joined verification supplies its router and skip_shared=true.
- Require !skip_shared for the wide prefill group. Preserve upstream route packing, indexed expert input, paired down and fused verification reduction. Select the existing prefill MPP activation/reduction only for its original guarded wide path.
- Keep all upstream additions and both documentation sections. Manual pin fix92be945 retained. The native-MTP-ahead patch stays unintegrated.
- ReleaseFast7/7 passed before the full suite. Full suite log:upstream-tests.log (currently running).
