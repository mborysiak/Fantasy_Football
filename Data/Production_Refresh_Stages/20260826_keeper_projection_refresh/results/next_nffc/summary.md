# V2 Next-Year Residual Validation (nffc)

## Scope

- Run: `v2_next_year_nffc_20260826T203313Z_10c67e09`
- Feature run: `milestone_3_20260826T202043Z_1e4e26b9`
- Target: following-season conditional PPG minus the origin-season expert
  team-game PPG consensus.
- Validation origins: 2017-2024.
- Each origin uses training labels only through origin minus two; the latest
  training target outcome season is origin minus one.
- Production projections, templates, and optimizers remain unchanged.

## Pooled scores

| target_name | method | metric | n_rows | value |
| --- | --- | --- | --- | --- |
| next_conditional_ppg | next_ppg_expert_carry_forward | rmse | 3505 | 5.5962 |
| next_conditional_ppg | next_ppg_expert_carry_forward | mae | 3505 | 3.8566 |
| next_conditional_ppg | next_ppg_expert_carry_forward | bias | 3505 | -1.9814 |
| next_conditional_ppg | next_ppg_expert_carry_forward | spearman | 3505 | 0.5888 |
| next_conditional_ppg | next_ppg_lasso | rmse | 3505 | 4.0248 |
| next_conditional_ppg | next_ppg_lasso | mae | 3505 | 3.1022 |
| next_conditional_ppg | next_ppg_lasso | bias | 3505 | 0.4553 |
| next_conditional_ppg | next_ppg_lasso | spearman | 3505 | 0.7265 |
| next_conditional_ppg | next_ppg_lightgbm | rmse | 3505 | 3.9899 |
| next_conditional_ppg | next_ppg_lightgbm | mae | 3505 | 3.0459 |
| next_conditional_ppg | next_ppg_lightgbm | bias | 3505 | 0.2564 |
| next_conditional_ppg | next_ppg_lightgbm | spearman | 3505 | 0.7253 |
| next_conditional_ppg | next_ppg_position_experience_aging | rmse | 3505 | 4.8282 |
| next_conditional_ppg | next_ppg_position_experience_aging | mae | 3505 | 3.4641 |
| next_conditional_ppg | next_ppg_position_experience_aging | bias | 3505 | -0.2592 |
| next_conditional_ppg | next_ppg_position_experience_aging | spearman | 3505 | 0.6810 |
| next_conditional_ppg | next_ppg_primary_blend | rmse | 3505 | 3.9706 |
| next_conditional_ppg | next_ppg_primary_blend | mae | 3505 | 3.0423 |
| next_conditional_ppg | next_ppg_primary_blend | bias | 3505 | 0.3320 |
| next_conditional_ppg | next_ppg_primary_blend | spearman | 3505 | 0.7304 |
| next_conditional_ppg | next_ppg_random_forest | rmse | 3505 | 3.9898 |
| next_conditional_ppg | next_ppg_random_forest | mae | 3505 | 3.0402 |
| next_conditional_ppg | next_ppg_random_forest | bias | 3505 | 0.2844 |
| next_conditional_ppg | next_ppg_random_forest | spearman | 3505 | 0.7282 |
| next_participation | next_participation_lightgbm | brier | 6988 | 0.1606 |
| next_participation | next_participation_lightgbm | log_loss | 6988 | 0.4886 |
| next_participation | next_participation_lightgbm | calibration_bias | 6988 | 0.0122 |
| next_participation | next_participation_lightgbm | roc_auc | 6988 | 0.8445 |
| next_participation | next_participation_logistic | brier | 6988 | 0.1734 |
| next_participation | next_participation_logistic | log_loss | 6988 | 0.5206 |
| next_participation | next_participation_logistic | calibration_bias | 6988 | 0.0451 |
| next_participation | next_participation_logistic | roc_auc | 6988 | 0.8210 |
| next_participation | next_participation_position_experience_prior | brier | 6988 | 0.2645 |
| next_participation | next_participation_position_experience_prior | log_loss | 6988 | 0.7248 |
| next_participation | next_participation_position_experience_prior | calibration_bias | 6988 | 0.1022 |
| next_participation | next_participation_position_experience_prior | roc_auc | 6988 | 0.4995 |

## Causal comparisons

| comparison | pooled_challenger | pooled_reference | mean_origin_delta | origin_wins | bootstrap_95_lower | bootstrap_95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| primary_vs_expert_carry | 3.9706 | 5.5962 | -1.6209 | 8 | -1.7448 | -1.5030 |
| aging_vs_expert_carry | 4.8282 | 5.5962 | -0.7648 | 8 | -0.8716 | -0.6508 |
| lasso_vs_expert_carry | 4.0248 | 5.5962 | -1.5666 | 8 | -1.7007 | -1.4407 |
| rf_vs_expert_carry | 3.9898 | 5.5962 | -1.6020 | 8 | -1.7287 | -1.4894 |
| lightgbm_vs_expert_carry | 3.9899 | 5.5962 | -1.6019 | 8 | -1.7144 | -1.4883 |
| participation_lgbm_vs_prior | 0.1606 | 0.2645 | -0.1018 | 8 | -0.1200 | -0.0847 |
| participation_lgbm_vs_logistic | 0.1606 | 0.1734 | -0.0120 | 8 | -0.0200 | -0.0053 |

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

Runtime: 93.6 seconds.
