# PR #408 follow-up bf38063

Binary SHA256 d62beaf96ffb72c377e9c8cb845ae20022a4e0cfc00667a567252936f47fb203. HC/GDN/QSA remain engaged for coalesced tails throughS8703; QSA bills the corresponding planner. HC mean now rounds the reciprocal toBF16, matching compiled MLX forHC3 as well asHC2/4/8.

The new HC3 fixture failed before the fix (max absolute difference0.001953125) and passed afterward, including a fullHC4/H2560/S8703 fixture. Full Zig2330passed/154skipped, ReleaseFast7/7, float64 QSA oracle (includingS8703/KV65536) and the real-model wrapper passed. The wrapper checked8296,13515,15715 prompt tokens, exact passphrase, cache0; initialHC/GDN/QSAengagementS8295, GDNcold=true. The original24500-character tail fixture produced8153 tokens and failed the fixture boundary assertion; it was corrected to25000 characters before the passing run. This was a test-fixture failure, not a model correctness failure.

No full long-context speedup is claimed for this newer head. The published full ladder remains explicitly measured atd23d9df. The mathematical sequence in existingHC4 cases is unchanged by the reciprocal correction; expanded dispatch now runs the fusion on formerly declined coalesced tails.
