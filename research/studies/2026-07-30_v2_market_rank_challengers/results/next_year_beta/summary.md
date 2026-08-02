# V2 Next-Year Residual Validation (beta)

## Scope

- Run: `v2_next_year_beta_20260730T141950Z_88cf96e3`
- Feature run: `milestone_3_20260730T140041Z_8666f6b2`
- Target: following-season conditional PPG minus the origin-season expert
  team-game PPG consensus.
- Validation origins: 2017-2024.
- Each origin uses training labels only through origin minus two; the latest
  training target outcome season is origin minus one.
- Production projections, templates, and optimizers remain unchanged.

## Pooled scores

| target_name | method | metric | n_rows | value |
| --- | --- | --- | --- | --- |
| next_conditional_ppg | next_ppg_expert_carry_forward | rmse | 3368 | 4.3653 |
| next_conditional_ppg | next_ppg_expert_carry_forward | mae | 3368 | 3.1417 |
| next_conditional_ppg | next_ppg_expert_carry_forward | bias | 3368 | -1.3365 |
| next_conditional_ppg | next_ppg_expert_carry_forward | spearman | 3368 | 0.6138 |
| next_conditional_ppg | next_ppg_lasso | rmse | 3368 | 3.6096 |
| next_conditional_ppg | next_ppg_lasso | mae | 3368 | 2.7597 |
| next_conditional_ppg | next_ppg_lasso | bias | 3368 | 0.2833 |
| next_conditional_ppg | next_ppg_lasso | spearman | 3368 | 0.6953 |
| next_conditional_ppg | next_ppg_lightgbm | rmse | 3368 | 3.5688 |
| next_conditional_ppg | next_ppg_lightgbm | mae | 3368 | 2.7136 |
| next_conditional_ppg | next_ppg_lightgbm | bias | 3368 | 0.1656 |
| next_conditional_ppg | next_ppg_lightgbm | spearman | 3368 | 0.6923 |
| next_conditional_ppg | next_ppg_position_experience_aging | rmse | 3368 | 3.9997 |
| next_conditional_ppg | next_ppg_position_experience_aging | mae | 3368 | 2.9160 |
| next_conditional_ppg | next_ppg_position_experience_aging | bias | 3368 | -0.1157 |
| next_conditional_ppg | next_ppg_position_experience_aging | spearman | 3368 | 0.6525 |
| next_conditional_ppg | next_ppg_primary_blend | rmse | 3368 | 3.5538 |
| next_conditional_ppg | next_ppg_primary_blend | mae | 3368 | 2.7135 |
| next_conditional_ppg | next_ppg_primary_blend | bias | 3368 | 0.2309 |
| next_conditional_ppg | next_ppg_primary_blend | spearman | 3368 | 0.6987 |
| next_conditional_ppg | next_ppg_random_forest | rmse | 3368 | 3.5717 |
| next_conditional_ppg | next_ppg_random_forest | mae | 3368 | 2.7265 |
| next_conditional_ppg | next_ppg_random_forest | bias | 3368 | 0.2437 |
| next_conditional_ppg | next_ppg_random_forest | spearman | 3368 | 0.6952 |
| next_participation | next_participation_lightgbm | brier | 6988 | 0.1663 |
| next_participation | next_participation_lightgbm | log_loss | 6988 | 0.5034 |
| next_participation | next_participation_lightgbm | calibration_bias | 6988 | 0.0270 |
| next_participation | next_participation_lightgbm | roc_auc | 6988 | 0.8343 |
| next_participation | next_participation_logistic | brier | 6988 | 0.1763 |
| next_participation | next_participation_logistic | log_loss | 6988 | 0.5273 |
| next_participation | next_participation_logistic | calibration_bias | 6988 | 0.0466 |
| next_participation | next_participation_logistic | roc_auc | 6988 | 0.8152 |
| next_participation | next_participation_position_experience_prior | brier | 6988 | 0.2645 |
| next_participation | next_participation_position_experience_prior | log_loss | 6988 | 0.7248 |
| next_participation | next_participation_position_experience_prior | calibration_bias | 6988 | 0.1022 |
| next_participation | next_participation_position_experience_prior | roc_auc | 6988 | 0.4995 |

## Causal comparisons

| comparison | pooled_challenger | pooled_reference | mean_origin_delta | origin_wins | bootstrap_95_lower | bootstrap_95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| primary_vs_expert_carry | 3.5538 | 4.3653 | -0.8069 | 8 | -0.8963 | -0.7177 |
| aging_vs_expert_carry | 3.9997 | 4.3653 | -0.3620 | 8 | -0.4771 | -0.2360 |
| lasso_vs_expert_carry | 3.6096 | 4.3653 | -0.7517 | 8 | -0.8522 | -0.6512 |
| rf_vs_expert_carry | 3.5717 | 4.3653 | -0.7882 | 8 | -0.8886 | -0.6898 |
| lightgbm_vs_expert_carry | 3.5688 | 4.3653 | -0.7928 | 8 | -0.8632 | -0.7219 |
| participation_lgbm_vs_prior | 0.1663 | 0.2645 | -0.0966 | 8 | -0.1103 | -0.0833 |
| participation_lgbm_vs_logistic | 0.1663 | 0.1763 | -0.0094 | 8 | -0.0144 | -0.0052 |

## 2027 shadow

- Candidate origin rows: 745
- Conditional-PPG centers: 673
- Following-season participation probabilities:
  745
- Historical conditional training rows:
  6,062
- Historical participation labels:
  11,956

The prespecified conditional primary is the equal-third
Lasso/random-forest/LightGBM residual blend. The participation primary is
shallow LightGBM. Promotion depends on the comparison and subsequent
weekly-template feature replay; these outputs are shadow-only.

Runtime: 169.1 seconds.
