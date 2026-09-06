# V2 Next-Year Residual Validation (beta)

## Scope

- Run: `v2_next_year_beta_20260826T211927Z_b19d960b`
- Feature run: `milestone_3_20260826T210835Z_29025fcb`
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
| next_conditional_ppg | next_ppg_lasso | rmse | 3368 | 3.6193 |
| next_conditional_ppg | next_ppg_lasso | mae | 3368 | 2.7667 |
| next_conditional_ppg | next_ppg_lasso | bias | 3368 | 0.2889 |
| next_conditional_ppg | next_ppg_lasso | spearman | 3368 | 0.6930 |
| next_conditional_ppg | next_ppg_lightgbm | rmse | 3368 | 3.5621 |
| next_conditional_ppg | next_ppg_lightgbm | mae | 3368 | 2.7069 |
| next_conditional_ppg | next_ppg_lightgbm | bias | 3368 | 0.1624 |
| next_conditional_ppg | next_ppg_lightgbm | spearman | 3368 | 0.6939 |
| next_conditional_ppg | next_ppg_position_experience_aging | rmse | 3368 | 3.9996 |
| next_conditional_ppg | next_ppg_position_experience_aging | mae | 3368 | 2.9159 |
| next_conditional_ppg | next_ppg_position_experience_aging | bias | 3368 | -0.1158 |
| next_conditional_ppg | next_ppg_position_experience_aging | spearman | 3368 | 0.6525 |
| next_conditional_ppg | next_ppg_primary_blend | rmse | 3368 | 3.5571 |
| next_conditional_ppg | next_ppg_primary_blend | mae | 3368 | 2.7140 |
| next_conditional_ppg | next_ppg_primary_blend | bias | 3368 | 0.2291 |
| next_conditional_ppg | next_ppg_primary_blend | spearman | 3368 | 0.6983 |
| next_conditional_ppg | next_ppg_random_forest | rmse | 3368 | 3.5707 |
| next_conditional_ppg | next_ppg_random_forest | mae | 3368 | 2.7226 |
| next_conditional_ppg | next_ppg_random_forest | bias | 3368 | 0.2361 |
| next_conditional_ppg | next_ppg_random_forest | spearman | 3368 | 0.6945 |
| next_participation | next_participation_lightgbm | brier | 6988 | 0.1659 |
| next_participation | next_participation_lightgbm | log_loss | 6988 | 0.5022 |
| next_participation | next_participation_lightgbm | calibration_bias | 6988 | 0.0260 |
| next_participation | next_participation_lightgbm | roc_auc | 6988 | 0.8350 |
| next_participation | next_participation_logistic | brier | 6988 | 0.1750 |
| next_participation | next_participation_logistic | log_loss | 6988 | 0.5240 |
| next_participation | next_participation_logistic | calibration_bias | 6988 | 0.0451 |
| next_participation | next_participation_logistic | roc_auc | 6988 | 0.8176 |
| next_participation | next_participation_position_experience_prior | brier | 6988 | 0.2645 |
| next_participation | next_participation_position_experience_prior | log_loss | 6988 | 0.7248 |
| next_participation | next_participation_position_experience_prior | calibration_bias | 6988 | 0.1022 |
| next_participation | next_participation_position_experience_prior | roc_auc | 6988 | 0.4995 |

## Causal comparisons

| comparison | pooled_challenger | pooled_reference | mean_origin_delta | origin_wins | bootstrap_95_lower | bootstrap_95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| primary_vs_expert_carry | 3.5571 | 4.3653 | -0.8035 | 8 | -0.8951 | -0.7137 |
| aging_vs_expert_carry | 3.9996 | 4.3653 | -0.3620 | 8 | -0.4771 | -0.2360 |
| lasso_vs_expert_carry | 3.6193 | 4.3653 | -0.7423 | 8 | -0.8396 | -0.6458 |
| rf_vs_expert_carry | 3.5707 | 4.3653 | -0.7891 | 8 | -0.8879 | -0.6918 |
| lightgbm_vs_expert_carry | 3.5621 | 4.3653 | -0.7988 | 8 | -0.8767 | -0.7210 |
| participation_lgbm_vs_prior | 0.1659 | 0.2645 | -0.0970 | 8 | -0.1105 | -0.0839 |
| participation_lgbm_vs_logistic | 0.1659 | 0.1750 | -0.0086 | 8 | -0.0131 | -0.0048 |

## 2027 shadow

- Candidate origin rows: 785
- Conditional-PPG centers: 616
- Following-season participation probabilities:
  785
- Historical conditional training rows:
  6,062
- Historical participation labels:
  11,956

The prespecified conditional primary is the equal-third
Lasso/random-forest/LightGBM residual blend. The participation primary is
shallow LightGBM. Promotion depends on the comparison and subsequent
weekly-template feature replay; these outputs are shadow-only.

Runtime: 36.6 seconds.
