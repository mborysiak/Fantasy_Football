# Extra Trees / CatBoost ensemble result

## Primary RMSE comparisons

Negative deltas favor the challenger. Confidence intervals use a paired player-cluster bootstrap.

| Scoring | Challenger | Baseline | Challenger | Delta | 95% interval |
|---|---|---:|---:|---:|---:|
| beta | Both equal-fifths (secondary) | 2.892611 | 2.889969 | -0.002642 | [-0.009161, +0.003817] |
| dk | Both equal-fifths (secondary) | 3.090995 | 3.087811 | -0.003184 | [-0.010044, +0.003863] |
| beta | CatBoost equal-fourths | 2.892611 | 2.895369 | +0.002758 | [-0.001709, +0.007289] |
| dk | CatBoost equal-fourths | 3.090995 | 3.090309 | -0.000686 | [-0.005702, +0.004124] |
| beta | Extra Trees equal-fourths | 2.892611 | 2.886341 | -0.006270 | [-0.011945, -0.000534] |
| dk | Extra Trees equal-fourths | 3.090995 | 3.087534 | -0.003461 | [-0.009328, +0.002527] |

## Decision

Extra Trees advances as a research shadow candidate, not a production change. It improved pooled RMSE in DK and beta and won all six scoring-system/season cells. Its non-QB slices also improved in both systems, so the result is not solely a QB effect. All ten single-seed robustness cells remained favorable. The gain is small, DK uncertainty overlaps zero, and the confirmation seasons have been reused.

CatBoost is rejected as an equal-weight fourth member: its small DK gain reversed in beta. The equal-fifths blend is also rejected under the prespecified rule that both challengers must add value independently.

No production files were changed.
