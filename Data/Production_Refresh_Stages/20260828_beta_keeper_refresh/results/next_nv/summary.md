# V2 Next-Year Residual Validation (nv)

## Scope

- Run: `v2_next_year_nv_20260828T194449Z_ee62f18f`
- Feature run: `milestone_3_20260828T191534Z_695cc236`
- Target: following-season conditional PPG minus the origin-season expert
  team-game PPG consensus.
- Validation origins: 2017-2024.
- Each origin uses training labels only through origin minus two; the latest
  training target outcome season is origin minus one.
- Production projections, templates, and optimizers remain unchanged.

## Pooled scores

| target_name | method | metric | n_rows | value |
| --- | --- | --- | --- | --- |
| next_conditional_ppg | next_ppg_expert_carry_forward | rmse | 3368 | 4.2671 |
| next_conditional_ppg | next_ppg_expert_carry_forward | mae | 3368 | 3.0996 |
| next_conditional_ppg | next_ppg_expert_carry_forward | bias | 3368 | -1.3153 |
| next_conditional_ppg | next_ppg_expert_carry_forward | spearman | 3368 | 0.6126 |
| next_conditional_ppg | next_ppg_lasso | rmse | 3368 | 3.5604 |
| next_conditional_ppg | next_ppg_lasso | mae | 3368 | 2.7343 |
| next_conditional_ppg | next_ppg_lasso | bias | 3368 | 0.2748 |
| next_conditional_ppg | next_ppg_lasso | spearman | 3368 | 0.6875 |
| next_conditional_ppg | next_ppg_lightgbm | rmse | 3368 | 3.5133 |
| next_conditional_ppg | next_ppg_lightgbm | mae | 3368 | 2.6806 |
| next_conditional_ppg | next_ppg_lightgbm | bias | 3368 | 0.1661 |
| next_conditional_ppg | next_ppg_lightgbm | spearman | 3368 | 0.6893 |
| next_conditional_ppg | next_ppg_position_experience_aging | rmse | 3368 | 3.9117 |
| next_conditional_ppg | next_ppg_position_experience_aging | mae | 3368 | 2.8760 |
| next_conditional_ppg | next_ppg_position_experience_aging | bias | 3368 | -0.1165 |
| next_conditional_ppg | next_ppg_position_experience_aging | spearman | 3368 | 0.6481 |
| next_conditional_ppg | next_ppg_primary_blend | rmse | 3368 | 3.5057 |
| next_conditional_ppg | next_ppg_primary_blend | mae | 3368 | 2.6839 |
| next_conditional_ppg | next_ppg_primary_blend | bias | 3368 | 0.2194 |
| next_conditional_ppg | next_ppg_primary_blend | spearman | 3368 | 0.6935 |
| next_conditional_ppg | next_ppg_random_forest | rmse | 3368 | 3.5108 |
| next_conditional_ppg | next_ppg_random_forest | mae | 3368 | 2.6865 |
| next_conditional_ppg | next_ppg_random_forest | bias | 3368 | 0.2172 |
| next_conditional_ppg | next_ppg_random_forest | spearman | 3368 | 0.6910 |
| next_participation | next_participation_lightgbm | brier | 6988 | 0.1660 |
| next_participation | next_participation_lightgbm | log_loss | 6988 | 0.5023 |
| next_participation | next_participation_lightgbm | calibration_bias | 6988 | 0.0263 |
| next_participation | next_participation_lightgbm | roc_auc | 6988 | 0.8347 |
| next_participation | next_participation_logistic | brier | 6988 | 0.1746 |
| next_participation | next_participation_logistic | log_loss | 6988 | 0.5230 |
| next_participation | next_participation_logistic | calibration_bias | 6988 | 0.0444 |
| next_participation | next_participation_logistic | roc_auc | 6988 | 0.8180 |
| next_participation | next_participation_position_experience_prior | brier | 6988 | 0.2645 |
| next_participation | next_participation_position_experience_prior | log_loss | 6988 | 0.7248 |
| next_participation | next_participation_position_experience_prior | calibration_bias | 6988 | 0.1022 |
| next_participation | next_participation_position_experience_prior | roc_auc | 6988 | 0.4995 |

## Causal comparisons

| comparison | pooled_challenger | pooled_reference | mean_origin_delta | origin_wins | bootstrap_95_lower | bootstrap_95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| primary_vs_expert_carry | 3.5057 | 4.2671 | -0.7575 | 8 | -0.8496 | -0.6675 |
| aging_vs_expert_carry | 3.9117 | 4.2671 | -0.3516 | 8 | -0.4617 | -0.2324 |
| lasso_vs_expert_carry | 3.5604 | 4.2671 | -0.7034 | 8 | -0.8015 | -0.6064 |
| rf_vs_expert_carry | 3.5108 | 4.2671 | -0.7519 | 8 | -0.8446 | -0.6583 |
| lightgbm_vs_expert_carry | 3.5133 | 4.2671 | -0.7500 | 8 | -0.8296 | -0.6698 |
| participation_lgbm_vs_prior | 0.1660 | 0.2645 | -0.0969 | 8 | -0.1103 | -0.0838 |
| participation_lgbm_vs_logistic | 0.1660 | 0.1746 | -0.0081 | 8 | -0.0123 | -0.0045 |

## 2027 shadow

- Candidate origin rows: 783
- Conditional-PPG centers: 615
- Following-season participation probabilities:
  783
- Historical conditional training rows:
  6,062
- Historical participation labels:
  11,956

The prespecified conditional primary is the equal-third
Lasso/random-forest/LightGBM residual blend. The participation primary is
shallow LightGBM. Promotion depends on the comparison and subsequent
weekly-template feature replay; these outputs are shadow-only.

Runtime: 112.5 seconds.
