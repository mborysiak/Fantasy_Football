# V2 Next-Year Residual Validation (dk)

## Scope

- Run: `v2_next_year_dk_20260730T141950Z_c328fa47`
- Feature run: `milestone_3_20260730T140041Z_e06ca8aa`
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
| next_conditional_ppg | next_ppg_lasso | rmse | 3505 | 3.9845 |
| next_conditional_ppg | next_ppg_lasso | mae | 3505 | 3.0958 |
| next_conditional_ppg | next_ppg_lasso | bias | 3505 | 0.3785 |
| next_conditional_ppg | next_ppg_lasso | spearman | 3505 | 0.7161 |
| next_conditional_ppg | next_ppg_lightgbm | rmse | 3505 | 3.9256 |
| next_conditional_ppg | next_ppg_lightgbm | mae | 3505 | 3.0311 |
| next_conditional_ppg | next_ppg_lightgbm | bias | 3505 | 0.2031 |
| next_conditional_ppg | next_ppg_lightgbm | spearman | 3505 | 0.7182 |
| next_conditional_ppg | next_ppg_position_experience_aging | rmse | 3505 | 4.5046 |
| next_conditional_ppg | next_ppg_position_experience_aging | mae | 3505 | 3.3400 |
| next_conditional_ppg | next_ppg_position_experience_aging | bias | 3505 | -0.1915 |
| next_conditional_ppg | next_ppg_position_experience_aging | spearman | 3505 | 0.6736 |
| next_conditional_ppg | next_ppg_primary_blend | rmse | 3505 | 3.9137 |
| next_conditional_ppg | next_ppg_primary_blend | mae | 3505 | 3.0293 |
| next_conditional_ppg | next_ppg_primary_blend | bias | 3505 | 0.2860 |
| next_conditional_ppg | next_ppg_primary_blend | spearman | 3505 | 0.7222 |
| next_conditional_ppg | next_ppg_random_forest | rmse | 3505 | 3.9294 |
| next_conditional_ppg | next_ppg_random_forest | mae | 3505 | 3.0371 |
| next_conditional_ppg | next_ppg_random_forest | bias | 3505 | 0.2765 |
| next_conditional_ppg | next_ppg_random_forest | spearman | 3505 | 0.7190 |
| next_participation | next_participation_lightgbm | brier | 6988 | 0.1615 |
| next_participation | next_participation_lightgbm | log_loss | 6988 | 0.4913 |
| next_participation | next_participation_lightgbm | calibration_bias | 6988 | 0.0142 |
| next_participation | next_participation_lightgbm | roc_auc | 6988 | 0.8427 |
| next_participation | next_participation_logistic | brier | 6988 | 0.1734 |
| next_participation | next_participation_logistic | log_loss | 6988 | 0.5209 |
| next_participation | next_participation_logistic | calibration_bias | 6988 | 0.0433 |
| next_participation | next_participation_logistic | roc_auc | 6988 | 0.8205 |
| next_participation | next_participation_position_experience_prior | brier | 6988 | 0.2645 |
| next_participation | next_participation_position_experience_prior | log_loss | 6988 | 0.7248 |
| next_participation | next_participation_position_experience_prior | calibration_bias | 6988 | 0.1022 |
| next_participation | next_participation_position_experience_prior | roc_auc | 6988 | 0.4995 |

## Causal comparisons

| comparison | pooled_challenger | pooled_reference | mean_origin_delta | origin_wins | bootstrap_95_lower | bootstrap_95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| primary_vs_expert_carry | 3.9137 | 5.2351 | -1.3162 | 8 | -1.4506 | -1.1931 |
| aging_vs_expert_carry | 4.5046 | 5.2351 | -0.7259 | 8 | -0.8352 | -0.6144 |
| lasso_vs_expert_carry | 3.9845 | 5.2351 | -1.2451 | 8 | -1.3877 | -1.1184 |
| rf_vs_expert_carry | 3.9294 | 5.2351 | -1.3009 | 8 | -1.4400 | -1.1750 |
| lightgbm_vs_expert_carry | 3.9256 | 5.2351 | -1.3048 | 8 | -1.4223 | -1.1890 |
| participation_lgbm_vs_prior | 0.1615 | 0.2645 | -0.1009 | 8 | -0.1189 | -0.0842 |
| participation_lgbm_vs_logistic | 0.1615 | 0.1734 | -0.0111 | 8 | -0.0184 | -0.0051 |

## 2027 shadow

- Candidate origin rows: 745
- Conditional-PPG centers: 715
- Following-season participation probabilities:
  745
- Historical conditional training rows:
  6,233
- Historical participation labels:
  11,956

The prespecified conditional primary is the equal-third
Lasso/random-forest/LightGBM residual blend. The participation primary is
shallow LightGBM. Promotion depends on the comparison and subsequent
weekly-template feature replay; these outputs are shadow-only.

Runtime: 172.7 seconds.
