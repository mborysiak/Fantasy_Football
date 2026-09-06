# Locked V2 Whole-Season and Shadow Results

- Lock: `v2_conditional_ppg_2026_candidate_beta_v1`
- Run: `v2_locked_final_beta_20260828T201543Z_681a65fc`
- Feature run: `milestone_3_20260828T195933Z_e8a0ff4b`
- Whole-season forecast origins: 2017-2025
- Current shadow season: 2026
- Current candidate rows: 783
- Current conditional-PPG centers: 615

Every forecast, hyperparameter selection, router decision, and residual
calibration uses only seasons earlier than its forecast origin.

## Conditional PPG

| Method | Pooled RMSE |
|---|---:|
| `conditional_ppg_projection_history_router` | 2.8773 |
| `conditional_ppg_history_gap_no_history_route` | 2.8777 |
| `conditional_ppg_log_lasso_tree_blend` | 2.8780 |
| `conditional_ppg_qb_style_wrte_route` | 2.8787 |
| `conditional_ppg_primary_blend` | 2.8792 |
| `conditional_ppg_random_forest` | 2.8948 |
| `conditional_ppg_lightgbm` | 2.9135 |
| `conditional_ppg_lasso` | 2.9287 |
| `expert_recalibrated` | 2.9600 |
| `expert_team_game` | 3.6273 |

## Prespecified comparisons

Negative deltas favor the challenger.

| Comparison | Mean season delta | Wins | 2023-25 delta | Season bootstrap 95% |
|---|---:|---:|---:|---:|
| `primary_vs_expert` | -0.0803 | 9/9 | -0.0859 | [-0.1013, -0.0635] |
| `log_lasso_vs_primary` | -0.0012 | 4/9 | +0.0021 | [-0.0047, +0.0018] |
| `history_gap_route_vs_primary` | -0.0015 | 6/9 | -0.0006 | [-0.0034, +0.0002] |
| `projection_router_vs_primary` | -0.0020 | 1/9 | -0.0061 | [-0.0091, +0.0027] |
| `qb_style_route_vs_primary` | -0.0005 | 6/9 | -0.0023 | [-0.0022, +0.0014] |

## Participation

| Method | Pooled Brier |
|---|---:|
| `participation_lightgbm` | 0.1251 |
| `participation_logistic` | 0.1399 |
| `participation_prior_position_rate` | 0.2434 |

## Calibration and template handoff

- Strict-prior P25-P75 residual coverage: 0.497
- Strict-prior P10-P90 residual coverage: 0.793
- Residual reconstruction maximum absolute error: 1.78e-15
- The handoff uses the locked point prediction as the center and exactly one
  matched donor's centered residual plus weekly path. An additional independent
  model-residual draw is explicitly prohibited because it would approximately
  double residual variance.

The output remains shadow-only. Production projections, weekly-template
tables, and optimizer inputs are unchanged.
