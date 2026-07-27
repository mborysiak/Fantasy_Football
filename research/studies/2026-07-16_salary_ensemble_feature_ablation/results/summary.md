# Salary Ensemble-Feature Ablation

## Paired observed salary accuracy

| period | player_years | v1_mean_residual | v2_mean_residual | v1_mae | v2_mae | mae_delta_v2_minus_v1 | v1_rmse | v2_rmse |
|---|---|---|---|---|---|---|---|---|
| all_years | 644 | -0.706 | -0.477 | 4.307 | 4.488 | 0.181 | 6.276 | 6.262 |
| replay_development_2022_2024 | 389 | -0.480 | -0.222 | 4.388 | 4.441 | 0.053 | 6.346 | 6.310 |
| temporal_check_2025 | 137 | -0.474 | -0.413 | 3.726 | 4.203 | 0.477 | 5.850 | 5.925 |

Negative MAE/RMSE deltas favor v2. Residual is actual minus predicted.

## Coverage

| period | v1_rows | v2_rows | paired_rows | v1_only_rows | v2_only_rows |
|---|---|---|---|---|---|
| all_years | 644 | 645 | 644 | 0 | 1 |
| temporal_check_2025 | 137 | 138 | 137 | 0 | 1 |

## Frozen replay candidate universe

| period | method | player_origins | mean_residual | mae | rmse | selection_weighted_mean_residual |
|---|---|---|---|---|---|---|
| all_years | v1 | 518 | -0.388 | 4.310 | 6.307 | 1.427 |
| all_years | v2 | 518 | -0.195 | 4.469 | 6.315 | 1.581 |
| replay_development_2022_2024 | v1 | 384 | -0.456 | 4.479 | 6.450 | 1.333 |
| replay_development_2022_2024 | v2 | 384 | -0.211 | 4.513 | 6.402 | 1.636 |
| temporal_check_2025 | v1 | 134 | -0.194 | 3.826 | 5.876 | 1.696 |
| temporal_check_2025 | v2 | 134 | -0.148 | 4.342 | 6.059 | 1.422 |

## Strongest within-position value quintile

| method | player_origins | mean_residual | positive_residual_rate | selection_weighted_mean_residual |
|---|---|---|---|---|
| v1 | 81 | 1.636 | 0.481 | 4.815 |
| v2 | 70 | 1.222 | 0.386 | 3.906 |

## Fixed prior-v1 roster repricing

| period | rosters | mean_point_spend_shift_v2_minus_v1 | mean_actual_minus_point_v1 | mean_actual_minus_point_v2 | actual_minus_point_gap_change_v2_minus_v1 |
|---|---|---|---|---|---|
| all_years | 4000 | -1.253 | 17.762 | 19.015 | 1.253 |
| replay_development_2022_2024 | 3000 | -2.864 | 16.357 | 19.221 | 2.864 |
| temporal_check_2025 | 1000 | 3.579 | 21.974 | 18.395 | -3.579 |

These are the same rosters selected under v1. The repricing isolates the salary surface; it does not measure which rosters a v2 optimizer would select.

## Paired v1 versus v2 optimizer frontier

The full identical-seed v2 replay completed all 4,000 optimizer cells. See
`frontier_comparison_summary.md` for chance-level detail.

Across chance thresholds, v2 changed 79.1% of development rosters. It added 2.08
managed forecast season points on average, but reduced held-out modeled
affordability by 1.02 percentage points, left historical feasibility unchanged,
and increased historical overage by $1.09.

In 2025, v2 changed 80.1% of rosters, lost 1.79 managed forecast points, reduced
held-out modeled affordability by 0.43 points, improved historical feasibility
by 1.20 points, and reduced historical overage by $0.46. Directions varied
materially by season and chance threshold.

## Interpretation limits

- The data cutoff rolls by origin, but the model family and features are retrospectively locked as of 2026.
- Player-year rows within a season are not independent season-level outcome units.
- Fixed-roster repricing is deliberately conditional on old v1 selections and cannot replace a v2 optimizer replay.
- Historical actual-price fallbacks retain the prior replay's intentional `$1` treatment for unrecorded auction prices.
