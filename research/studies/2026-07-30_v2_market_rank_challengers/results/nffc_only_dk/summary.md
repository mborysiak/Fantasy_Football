# Locked V2 Whole-Season and Shadow Results

- Lock: `v2_conditional_ppg_2026_candidate_v1`
- Run: `v2_locked_final_dk_20260730T140449Z_8a9b4479`
- Feature run: `milestone_3_20260730T140041Z_e06ca8aa`
- Whole-season forecast origins: 2017-2025
- Current shadow season: 2026
- Current candidate rows: 745
- Current conditional-PPG centers: 715

Every forecast, hyperparameter selection, router decision, and residual
calibration uses only seasons earlier than its forecast origin.

## Conditional PPG

| Method | Pooled RMSE |
|---|---:|
| `conditional_ppg_history_gap_no_history_route` | 3.1048 |
| `conditional_ppg_log_lasso_tree_blend` | 3.1052 |
| `conditional_ppg_qb_style_wrte_route` | 3.1053 |
| `conditional_ppg_primary_blend` | 3.1076 |
| `conditional_ppg_projection_history_router` | 3.1107 |
| `conditional_ppg_random_forest` | 3.1316 |
| `conditional_ppg_lightgbm` | 3.1326 |
| `conditional_ppg_lasso` | 3.1603 |
| `expert_recalibrated` | 3.1951 |
| `expert_team_game` | 4.2254 |

## Prespecified comparisons

Negative deltas favor the challenger.

| Comparison | Mean season delta | Wins | 2023-25 delta | Season bootstrap 95% |
|---|---:|---:|---:|---:|
| `primary_vs_expert` | -0.0868 | 9/9 | -0.0716 | [-0.1053, -0.0692] |
| `log_lasso_vs_primary` | -0.0023 | 5/9 | +0.0013 | [-0.0068, +0.0023] |
| `history_gap_route_vs_primary` | -0.0026 | 7/9 | -0.0013 | [-0.0053, -0.0006] |
| `projection_router_vs_primary` | +0.0030 | 2/9 | +0.0066 | [+0.0001, +0.0071] |
| `qb_style_route_vs_primary` | -0.0024 | 6/9 | -0.0054 | [-0.0053, +0.0006] |

## Participation

| Method | Pooled Brier |
|---|---:|
| `participation_lightgbm` | 0.1222 |
| `participation_logistic` | 0.1371 |
| `participation_prior_position_rate` | 0.2434 |

## Calibration and template handoff

- Strict-prior P25-P75 residual coverage: 0.497
- Strict-prior P10-P90 residual coverage: 0.792
- Residual reconstruction maximum absolute error: 1.78e-15
- The handoff uses the locked point prediction as the center and exactly one
  matched donor's centered residual plus weekly path. An additional independent
  model-residual draw is explicitly prohibited because it would approximately
  double residual variance.

The output remains shadow-only. Production projections, weekly-template
tables, and optimizer inputs are unchanged.
