# Boosting grid expansion result

Negative RMSE deltas favor the challenger.

| Scoring | Test | Baseline | Challenger | RMSE delta | 95% player-cluster interval |
|---|---|---:|---:|---:|---:|
| beta | Expanded CatBoost equal-four | 2.892611 | 2.895369 | +0.002758 | [-0.001733, +0.007299] |
| dk | Expanded CatBoost equal-four | 3.090995 | 3.090309 | -0.000686 | [-0.005689, +0.004123] |
| beta | Expanded LightGBM replacement | 2.892611 | 2.893194 | +0.000583 | [-0.001296, +0.002558] |
| dk | Expanded LightGBM replacement | 3.090995 | 3.090995 | +0.000000 | [-0.000000, +0.000000] |

## Decision

Retain the existing LightGBM grid and selected parameters. DK selected the exact incumbent 0.05 / 100-tree model. Beta selected the new 0.01 / 500-tree schedule by 0.001897 mean pre-2023 seasonal RMSE, but the resulting blend worsened pooled 2023-2025 RMSE by 0.000583. It slightly improved 2023 and 2024 and worsened 2025; its player-cluster interval crosses zero.

Retain CatBoost's rejection. Both scorings selected the original 0.03 / 300-iteration candidate despite the expanded schedules and boundary search, reproducing the earlier small DK gain and beta loss.

The expanded beta LightGBM also worsened the Extra Trees blend by 0.000414 RMSE versus the incumbent-LightGBM Extra Trees blend. Extra Trees therefore remains the only model-family shadow candidate from these tests. No production files were changed.
