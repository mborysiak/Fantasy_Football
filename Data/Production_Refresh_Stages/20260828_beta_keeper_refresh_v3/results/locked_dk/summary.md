# Locked V2 Whole-Season and Shadow Results

- Lock: `v2_conditional_ppg_2026_candidate_v1`
- Run: `v2_locked_final_dk_20260828T211340Z_2b29a4a7`
- Feature run: `milestone_3_20260828T210009Z_81f185d9`
- Whole-season forecast origins: 2017-2025
- Current shadow season: 2026
- Current candidate rows: 783
- Current conditional-PPG centers: 634

Every forecast, hyperparameter selection, router decision, and residual
calibration uses only seasons earlier than its forecast origin.

## Conditional PPG

| Method | Pooled RMSE |
|---|---:|
| `conditional_ppg_log_lasso_tree_blend` | 3.1033 |
| `conditional_ppg_history_gap_no_history_route` | 3.1045 |
| `conditional_ppg_qb_style_wrte_route` | 3.1057 |
| `conditional_ppg_primary_blend` | 3.1069 |
| `conditional_ppg_projection_history_router` | 3.1096 |
| `conditional_ppg_random_forest` | 3.1275 |
| `conditional_ppg_lightgbm` | 3.1293 |
| `conditional_ppg_lasso` | 3.1598 |
| `expert_recalibrated` | 3.1951 |
| `expert_team_game` | 4.2254 |

## Prespecified comparisons

Negative deltas favor the challenger.

| Comparison | Mean season delta | Wins | 2023-25 delta | Season bootstrap 95% |
|---|---:|---:|---:|---:|
| `primary_vs_expert` | -0.0874 | 9/9 | -0.0774 | [-0.1065, -0.0694] |
| `log_lasso_vs_primary` | -0.0036 | 6/9 | +0.0026 | [-0.0080, +0.0008] |
| `history_gap_route_vs_primary` | -0.0024 | 6/9 | -0.0008 | [-0.0052, -0.0001] |
| `projection_router_vs_primary` | +0.0025 | 1/9 | +0.0059 | [-0.0003, +0.0057] |
| `qb_style_route_vs_primary` | -0.0014 | 6/9 | -0.0047 | [-0.0037, +0.0011] |

## Participation

| Method | Pooled Brier |
|---|---:|
| `participation_lightgbm` | 0.1218 |
| `participation_logistic` | 0.1369 |
| `participation_prior_position_rate` | 0.2434 |

## Calibration and template handoff

- Strict-prior P25-P75 residual coverage: 0.496
- Strict-prior P10-P90 residual coverage: 0.793
- Residual reconstruction maximum absolute error: 1.78e-15
- The handoff uses the locked point prediction as the center and exactly one
  matched donor's centered residual plus weekly path. An additional independent
  model-residual draw is explicitly prohibited because it would approximately
  double residual variance.

The output remains shadow-only. Production projections, weekly-template
tables, and optimizer inputs are unchanged.
