# V2 Next-Year Residual Validation (beta)

## Scope

- Run: `v2_next_year_beta_20260828T213408Z_741eb3cf`
- Feature run: `milestone_3_20260828T210736Z_903d8f44`
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
| next_conditional_ppg | next_ppg_lasso | rmse | 3368 | 3.6194 |
| next_conditional_ppg | next_ppg_lasso | mae | 3368 | 2.7668 |
| next_conditional_ppg | next_ppg_lasso | bias | 3368 | 0.2892 |
| next_conditional_ppg | next_ppg_lasso | spearman | 3368 | 0.6929 |
| next_conditional_ppg | next_ppg_lightgbm | rmse | 3368 | 3.5630 |
| next_conditional_ppg | next_ppg_lightgbm | mae | 3368 | 2.7096 |
| next_conditional_ppg | next_ppg_lightgbm | bias | 3368 | 0.1698 |
| next_conditional_ppg | next_ppg_lightgbm | spearman | 3368 | 0.6937 |
| next_conditional_ppg | next_ppg_position_experience_aging | rmse | 3368 | 3.9996 |
| next_conditional_ppg | next_ppg_position_experience_aging | mae | 3368 | 2.9159 |
| next_conditional_ppg | next_ppg_position_experience_aging | bias | 3368 | -0.1158 |
| next_conditional_ppg | next_ppg_position_experience_aging | spearman | 3368 | 0.6525 |
| next_conditional_ppg | next_ppg_primary_blend | rmse | 3368 | 3.5559 |
| next_conditional_ppg | next_ppg_primary_blend | mae | 3368 | 2.7138 |
| next_conditional_ppg | next_ppg_primary_blend | bias | 3368 | 0.2334 |
| next_conditional_ppg | next_ppg_primary_blend | spearman | 3368 | 0.6984 |
| next_conditional_ppg | next_ppg_random_forest | rmse | 3368 | 3.5714 |
| next_conditional_ppg | next_ppg_random_forest | mae | 3368 | 2.7216 |
| next_conditional_ppg | next_ppg_random_forest | bias | 3368 | 0.2412 |
| next_conditional_ppg | next_ppg_random_forest | spearman | 3368 | 0.6950 |
| next_participation | next_participation_lightgbm | brier | 6988 | 0.1657 |
| next_participation | next_participation_lightgbm | log_loss | 6988 | 0.5017 |
| next_participation | next_participation_lightgbm | calibration_bias | 6988 | 0.0263 |
| next_participation | next_participation_lightgbm | roc_auc | 6988 | 0.8354 |
| next_participation | next_participation_logistic | brier | 6988 | 0.1749 |
| next_participation | next_participation_logistic | log_loss | 6988 | 0.5238 |
| next_participation | next_participation_logistic | calibration_bias | 6988 | 0.0453 |
| next_participation | next_participation_logistic | roc_auc | 6988 | 0.8177 |
| next_participation | next_participation_position_experience_prior | brier | 6988 | 0.2645 |
| next_participation | next_participation_position_experience_prior | log_loss | 6988 | 0.7248 |
| next_participation | next_participation_position_experience_prior | calibration_bias | 6988 | 0.1022 |
| next_participation | next_participation_position_experience_prior | roc_auc | 6988 | 0.4995 |

## Causal comparisons

| comparison | pooled_challenger | pooled_reference | mean_origin_delta | origin_wins | bootstrap_95_lower | bootstrap_95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| primary_vs_expert_carry | 3.5559 | 4.3653 | -0.8047 | 8 | -0.8957 | -0.7143 |
| aging_vs_expert_carry | 3.9996 | 4.3653 | -0.3620 | 8 | -0.4771 | -0.2360 |
| lasso_vs_expert_carry | 3.6194 | 4.3653 | -0.7422 | 8 | -0.8396 | -0.6457 |
| rf_vs_expert_carry | 3.5714 | 4.3653 | -0.7884 | 8 | -0.8881 | -0.6906 |
| lightgbm_vs_expert_carry | 3.5630 | 4.3653 | -0.7978 | 8 | -0.8760 | -0.7190 |
| participation_lgbm_vs_prior | 0.1657 | 0.2645 | -0.0972 | 8 | -0.1107 | -0.0838 |
| participation_lgbm_vs_logistic | 0.1657 | 0.1749 | -0.0087 | 8 | -0.0131 | -0.0048 |

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

Runtime: 116.2 seconds.
