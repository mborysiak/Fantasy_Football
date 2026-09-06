# V2 Next-Year Residual Validation (dk)

## Scope

- Run: `v2_next_year_dk_20260826T211457Z_64f6bb90`
- Feature run: `milestone_3_20260826T210455Z_b6326b6f`
- Target: following-season conditional PPG minus the origin-season expert
  team-game PPG consensus.
- Validation origins: 2017-2024.
- Each origin uses training labels only through origin minus two; the latest
  training target outcome season is origin minus one.
- Production projections, templates, and optimizers remain unchanged.

## Pooled scores

| target_name | method | metric | n_rows | value |
| --- | --- | --- | --- | --- |
| next_conditional_ppg | next_ppg_expert_carry_forward | rmse | 3505 | 5.2351 |
| next_conditional_ppg | next_ppg_expert_carry_forward | mae | 3505 | 3.7585 |
| next_conditional_ppg | next_ppg_expert_carry_forward | bias | 3505 | -2.0102 |
| next_conditional_ppg | next_ppg_expert_carry_forward | spearman | 3505 | 0.5906 |
| next_conditional_ppg | next_ppg_lasso | rmse | 3505 | 3.9706 |
| next_conditional_ppg | next_ppg_lasso | mae | 3505 | 3.0870 |
| next_conditional_ppg | next_ppg_lasso | bias | 3505 | 0.4004 |
| next_conditional_ppg | next_ppg_lasso | spearman | 3505 | 0.7170 |
| next_conditional_ppg | next_ppg_lightgbm | rmse | 3505 | 3.9268 |
| next_conditional_ppg | next_ppg_lightgbm | mae | 3505 | 3.0272 |
| next_conditional_ppg | next_ppg_lightgbm | bias | 3505 | 0.2412 |
| next_conditional_ppg | next_ppg_lightgbm | spearman | 3505 | 0.7189 |
| next_conditional_ppg | next_ppg_position_experience_aging | rmse | 3505 | 4.5046 |
| next_conditional_ppg | next_ppg_position_experience_aging | mae | 3505 | 3.3400 |
| next_conditional_ppg | next_ppg_position_experience_aging | bias | 3505 | -0.1915 |
| next_conditional_ppg | next_ppg_position_experience_aging | spearman | 3505 | 0.6736 |
| next_conditional_ppg | next_ppg_primary_blend | rmse | 3505 | 3.9105 |
| next_conditional_ppg | next_ppg_primary_blend | mae | 3505 | 3.0250 |
| next_conditional_ppg | next_ppg_primary_blend | bias | 3505 | 0.3060 |
| next_conditional_ppg | next_ppg_primary_blend | spearman | 3505 | 0.7228 |
| next_conditional_ppg | next_ppg_random_forest | rmse | 3505 | 3.9233 |
| next_conditional_ppg | next_ppg_random_forest | mae | 3505 | 3.0272 |
| next_conditional_ppg | next_ppg_random_forest | bias | 3505 | 0.2765 |
| next_conditional_ppg | next_ppg_random_forest | spearman | 3505 | 0.7198 |
| next_participation | next_participation_lightgbm | brier | 6988 | 0.1612 |
| next_participation | next_participation_lightgbm | log_loss | 6988 | 0.4906 |
| next_participation | next_participation_lightgbm | calibration_bias | 6988 | 0.0131 |
| next_participation | next_participation_lightgbm | roc_auc | 6988 | 0.8431 |
| next_participation | next_participation_logistic | brier | 6988 | 0.1723 |
| next_participation | next_participation_logistic | log_loss | 6988 | 0.5180 |
| next_participation | next_participation_logistic | calibration_bias | 6988 | 0.0418 |
| next_participation | next_participation_logistic | roc_auc | 6988 | 0.8224 |
| next_participation | next_participation_position_experience_prior | brier | 6988 | 0.2645 |
| next_participation | next_participation_position_experience_prior | log_loss | 6988 | 0.7248 |
| next_participation | next_participation_position_experience_prior | calibration_bias | 6988 | 0.1022 |
| next_participation | next_participation_position_experience_prior | roc_auc | 6988 | 0.4995 |

## Causal comparisons

| comparison | pooled_challenger | pooled_reference | mean_origin_delta | origin_wins | bootstrap_95_lower | bootstrap_95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| primary_vs_expert_carry | 3.9105 | 5.2351 | -1.3193 | 8 | -1.4533 | -1.1956 |
| aging_vs_expert_carry | 4.5046 | 5.2351 | -0.7260 | 8 | -0.8352 | -0.6144 |
| lasso_vs_expert_carry | 3.9706 | 5.2351 | -1.2589 | 8 | -1.4030 | -1.1295 |
| rf_vs_expert_carry | 3.9233 | 5.2351 | -1.3067 | 8 | -1.4447 | -1.1837 |
| lightgbm_vs_expert_carry | 3.9268 | 5.2351 | -1.3035 | 8 | -1.4270 | -1.1866 |
| participation_lgbm_vs_prior | 0.1612 | 0.2645 | -0.1012 | 8 | -0.1191 | -0.0845 |
| participation_lgbm_vs_logistic | 0.1612 | 0.1723 | -0.0103 | 8 | -0.0174 | -0.0045 |

## 2027 shadow

- Candidate origin rows: 785
- Conditional-PPG centers: 635
- Following-season participation probabilities:
  785
- Historical conditional training rows:
  6,233
- Historical participation labels:
  11,956

The prespecified conditional primary is the equal-third
Lasso/random-forest/LightGBM residual blend. The participation primary is
shallow LightGBM. Promotion depends on the comparison and subsequent
weekly-template feature replay; these outputs are shadow-only.

Runtime: 92.3 seconds.
