# Build and correctness status, 2026-09-10

GDN patch on cc7dea1: targeted test before enabling wide dispatch failed with FusedDeclined (expected red). After dispatch changes, full suite completed 2307 passed / 153 skipped / 1 failed. Failing check: pre.beta versus MLX sigmoid; expected max difference 0, observed 0.0000076293945. Expanded inputs approximately [-8,8]. First short component experiment with random [-1,1] had all six outputs exact. Do not loosen the numeric bar to claim quality preservation.

ReleaseFast then failed because the local opencode2 submodule was empty. The local dependency link was repaired, but there was no rebuild. No working GDN server binary or full-model benchmark exists.

MoE gate/up patch: ReleaseFast build passed and full-model requests completed on M5. Full suite for that patch was not run. No confirmed whole-model speedup.

Conv compaction patch on cc7dea1: ReleaseFast and suite passed (2330 passed, 153 skipped). HTTP isolation was run on earlier v26.9.2 base, not repeated on cc7. Memory savings are measured; standalone speed gain was not found.

HC up/mix: C++ driver compiled on macOS; GPU correctness/speed not run.
