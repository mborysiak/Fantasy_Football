# Ridge Swap Downstream Findings

## Decision

Do not replace Lasso with Ridge in the 2026 production point-center ensemble. Keep the active equal-third Lasso/RandomForest/LightGBM blend. The frozen Ridge swap fails the point-season replication and fixed-roster gates even though its pooled point and player-distribution metrics are slightly better.

## Gates

| gate | passed | detail |
| --- | --- | --- |
| point_ppg | False | pooled_both=True; recent_wins=4/6; position_guardrail=True |
| player_distribution | True | crps_both=True; coverage=True |
| weekly_template_transport | True | all core/depth PPG, contribution, and played CRPS relative deltas <= +0.25% |
| fixed_roster_snake | False | margin=False; nonworse_cells=1/4 |

## Point forecast

| league | production_rmse | ridge_swap_rmse | rmse_delta | bootstrap_p025 | bootstrap_p975 |
| --- | --- | --- | --- | --- | --- |
| dk | 3.105127 | 3.103726 | -0.001401 | -0.003952 | 0.001202 |
| beta | 2.878165 | 2.876902 | -0.001264 | -0.003512 | 0.001082 |

The swap lowers pooled RMSE by about 0.0013-0.0014 in both leagues, but every player-cluster interval crosses zero. It wins 2023 and 2024 and loses 2025 in both leagues, for only four of six recent season cells. Recent RB RMSE also worsens slightly in both leagues.

## Distribution and weekly templates

Strict-prior player CRPS improves slightly in both leagues and 50%/80% coverage remains calibrated. All eight core/depth league-period template cells stay inside the +0.25% PPG/contribution/played CRPS margins.

## Fixed-roster Snake replay

| league | period | rosters | score_crps_relative_delta | championship_brier_relative_delta |
| --- | --- | --- | --- | --- |
| dk | development_2018_2022 | 720.000000 | -0.007717 | -0.002374 |
| dk | temporal_2023_2025 | 432.000000 | 0.003109 | 0.001790 |
| beta | development_2018_2022 | 576.000000 | 0.003780 | -0.002928 |
| beta | temporal_2023_2025 | 432.000000 | 0.005270 | -0.000798 |

Only DK development improves roster-score CRPS. DK temporal and both beta periods worsen; beta temporal is +0.527%, just outside the +0.5% non-inferiority margin. Championship diagnostics do not rescue the point center because expected-score calibration is the primary gate.

Beta 2018 has no QB rows in the active locked whole-season forecast table, so a legal beta roster room cannot be formed for that origin. Beta player and template metrics retain 2018; beta roster metrics cover 2019-2025. DK roster metrics cover 2018-2025.

## 2026 shadow

The Ridge swap changes the 2026 center very little: mean PPG is lower by 0.026 in both leagues and rank correlation with production is about 0.997. Because the historical gates fail, the preregistered Auction shadow was not run. No production or app database was changed.
