# V2 Next-Year Residual Validation (beta)

## Scope

- Run: `v2_next_year_beta_20260729T130342Z_55fecc3b`
- Feature run: `milestone_3_20260729T042626Z_54599d2e`
- Target: following-season conditional PPG minus the origin-season expert
  team-game PPG consensus.
- Validation origins: 2017-2024.
- Each origin uses training labels only through origin minus two; the latest
  training target outcome season is origin minus one.
- Production projections, templates, and optimizers remain unchanged.

## Pooled scores

| target_name | method | metric | n_rows | value |
| --- | --- | --- | --- | --- |
| next_conditional_ppg | next_ppg_expert_carry_forward | rmse | 3528 | 4.6685 |
| next_conditional_ppg | next_ppg_expert_carry_forward | mae | 3528 | 3.3153 |
| next_conditional_ppg | next_ppg_expert_carry_forward | bias | 3528 | -1.4227 |
| next_conditional_ppg | next_ppg_expert_carry_forward | spearman | 3528 | 0.5954 |
| next_conditional_ppg | next_ppg_lasso | rmse | 3528 | 3.7350 |
| next_conditional_ppg | next_ppg_lasso | mae | 3528 | 2.8376 |
| next_conditional_ppg | next_ppg_lasso | bias | 3528 | 0.2740 |
| next_conditional_ppg | next_ppg_lasso | spearman | 3528 | 0.6983 |
| next_conditional_ppg | next_ppg_lightgbm | rmse | 3528 | 3.6814 |
| next_conditional_ppg | next_ppg_lightgbm | mae | 3528 | 2.7782 |
| next_conditional_ppg | next_ppg_lightgbm | bias | 3528 | 0.1758 |
| next_conditional_ppg | next_ppg_lightgbm | spearman | 3528 | 0.6977 |
| next_conditional_ppg | next_ppg_position_experience_aging | rmse | 3528 | 4.2761 |
| next_conditional_ppg | next_ppg_position_experience_aging | mae | 3528 | 3.0924 |
| next_conditional_ppg | next_ppg_position_experience_aging | bias | 3528 | -0.1153 |
| next_conditional_ppg | next_ppg_position_experience_aging | spearman | 3528 | 0.6539 |
| next_conditional_ppg | next_ppg_primary_blend | rmse | 3528 | 3.6718 |
| next_conditional_ppg | next_ppg_primary_blend | mae | 3528 | 2.7780 |
| next_conditional_ppg | next_ppg_primary_blend | bias | 3528 | 0.2228 |
| next_conditional_ppg | next_ppg_primary_blend | spearman | 3528 | 0.7025 |
| next_conditional_ppg | next_ppg_random_forest | rmse | 3528 | 3.6783 |
| next_conditional_ppg | next_ppg_random_forest | mae | 3528 | 2.7785 |
| next_conditional_ppg | next_ppg_random_forest | bias | 3528 | 0.2184 |
| next_conditional_ppg | next_ppg_random_forest | spearman | 3528 | 0.6980 |
| next_participation | next_participation_lightgbm | brier | 7048 | 0.1623 |
| next_participation | next_participation_lightgbm | log_loss | 7048 | 0.4934 |
| next_participation | next_participation_lightgbm | calibration_bias | 7048 | 0.0274 |
| next_participation | next_participation_lightgbm | roc_auc | 7048 | 0.8425 |
| next_participation | next_participation_logistic | brier | 7048 | 0.1748 |
| next_participation | next_participation_logistic | log_loss | 7048 | 0.5241 |
| next_participation | next_participation_logistic | calibration_bias | 7048 | 0.0495 |
| next_participation | next_participation_logistic | roc_auc | 7048 | 0.8190 |
| next_participation | next_participation_position_experience_prior | brier | 7048 | 0.2648 |
| next_participation | next_participation_position_experience_prior | log_loss | 7048 | 0.7254 |
| next_participation | next_participation_position_experience_prior | calibration_bias | 7048 | 0.1022 |
| next_participation | next_participation_position_experience_prior | roc_auc | 7048 | 0.4974 |

## Causal comparisons

| comparison | pooled_challenger | pooled_reference | mean_origin_delta | origin_wins | bootstrap_95_lower | bootstrap_95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| primary_vs_expert_carry | 3.6718 | 4.6685 | -0.9917 | 8 | -1.1015 | -0.8917 |
| aging_vs_expert_carry | 4.2761 | 4.6685 | -0.3880 | 8 | -0.4946 | -0.2747 |
| lasso_vs_expert_carry | 3.7350 | 4.6685 | -0.9287 | 8 | -1.0507 | -0.8270 |
| rf_vs_expert_carry | 3.6783 | 4.6685 | -0.9852 | 8 | -1.0964 | -0.8845 |
| lightgbm_vs_expert_carry | 3.6814 | 4.6685 | -0.9824 | 8 | -1.0790 | -0.8913 |
| participation_lgbm_vs_prior | 0.1623 | 0.2648 | -0.1006 | 8 | -0.1172 | -0.0846 |
| participation_lgbm_vs_logistic | 0.1623 | 0.1748 | -0.0117 | 7 | -0.0179 | -0.0059 |

## 2027 shadow

- Candidate origin rows: 751
- Conditional-PPG centers: 720
- Following-season participation probabilities:
  751
- Historical conditional training rows:
  6,306
- Historical participation labels:
  12,106

The prespecified conditional primary is the equal-third
Lasso/random-forest/LightGBM residual blend. The participation primary is
shallow LightGBM. Promotion depends on the comparison and subsequent
weekly-template feature replay; these outputs are shadow-only.

Runtime: 160.8 seconds.
