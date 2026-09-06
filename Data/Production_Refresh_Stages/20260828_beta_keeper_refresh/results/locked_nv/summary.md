# Locked V2 Whole-Season and Shadow Results

- Lock: `v2_conditional_ppg_2026_candidate_nv_v1`
- Run: `v2_locked_final_nv_20260828T193330Z_f710cfa4`
- Feature run: `milestone_3_20260828T191534Z_695cc236`
- Whole-season forecast origins: 2017-2025
- Current shadow season: 2026
- Current candidate rows: 783
- Current conditional-PPG centers: 615

Every forecast, hyperparameter selection, router decision, and residual
calibration uses only seasons earlier than its forecast origin.

## Conditional PPG

| Method | Pooled RMSE |
|---|---:|
| `conditional_ppg_qb_style_wrte_route` | 2.8368 |
| `conditional_ppg_log_lasso_tree_blend` | 2.8368 |
| `conditional_ppg_history_gap_no_history_route` | 2.8377 |
| `conditional_ppg_primary_blend` | 2.8384 |
| `conditional_ppg_projection_history_router` | 2.8394 |
| `conditional_ppg_lightgbm` | 2.8512 |
| `conditional_ppg_random_forest` | 2.8552 |
| `expert_recalibrated` | 2.9157 |
| `conditional_ppg_lasso` | 2.9193 |
| `expert_team_game` | 3.5402 |

## Prespecified comparisons

Negative deltas favor the challenger.

| Comparison | Mean season delta | Wins | 2023-25 delta | Season bootstrap 95% |
|---|---:|---:|---:|---:|
| `primary_vs_expert` | -0.0769 | 9/9 | -0.0713 | [-0.1004, -0.0541] |
| `log_lasso_vs_primary` | -0.0017 | 5/9 | +0.0024 | [-0.0054, +0.0012] |
| `history_gap_route_vs_primary` | -0.0008 | 5/9 | -0.0000 | [-0.0037, +0.0018] |
| `projection_router_vs_primary` | +0.0010 | 1/9 | +0.0029 | [-0.0008, +0.0034] |
| `qb_style_route_vs_primary` | -0.0017 | 7/9 | -0.0032 | [-0.0031, -0.0003] |

## Participation

| Method | Pooled Brier |
|---|---:|
| `participation_lightgbm` | 0.1251 |
| `participation_logistic` | 0.1398 |
| `participation_prior_position_rate` | 0.2434 |

## Calibration and template handoff

- Strict-prior P25-P75 residual coverage: 0.494
- Strict-prior P10-P90 residual coverage: 0.793
- Residual reconstruction maximum absolute error: 1.78e-15
- The handoff uses the locked point prediction as the center and exactly one
  matched donor's centered residual plus weekly path. An additional independent
  model-residual draw is explicitly prohibited because it would approximately
  double residual variance.

The output remains shadow-only. Production projections, weekly-template
tables, and optimizer inputs are unchanged.
