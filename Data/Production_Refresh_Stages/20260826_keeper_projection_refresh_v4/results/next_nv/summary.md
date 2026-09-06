# V2 Next-Year Residual Validation (nv)

## Scope

- Run: `v2_next_year_nv_20260826T214443Z_10f3fe58`
- Feature run: `milestone_3_20260826T213154Z_362a6695`
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
| next_conditional_ppg | next_ppg_lasso | rmse | 3368 | 3.5603 |
| next_conditional_ppg | next_ppg_lasso | mae | 3368 | 2.7343 |
| next_conditional_ppg | next_ppg_lasso | bias | 3368 | 0.2746 |
| next_conditional_ppg | next_ppg_lasso | spearman | 3368 | 0.6876 |
| next_conditional_ppg | next_ppg_lightgbm | rmse | 3368 | 3.5123 |
| next_conditional_ppg | next_ppg_lightgbm | mae | 3368 | 2.6798 |
| next_conditional_ppg | next_ppg_lightgbm | bias | 3368 | 0.1677 |
| next_conditional_ppg | next_ppg_lightgbm | spearman | 3368 | 0.6897 |
| next_conditional_ppg | next_ppg_position_experience_aging | rmse | 3368 | 3.9117 |
| next_conditional_ppg | next_ppg_position_experience_aging | mae | 3368 | 2.8760 |
| next_conditional_ppg | next_ppg_position_experience_aging | bias | 3368 | -0.1165 |
| next_conditional_ppg | next_ppg_position_experience_aging | spearman | 3368 | 0.6481 |
| next_conditional_ppg | next_ppg_primary_blend | rmse | 3368 | 3.5056 |
| next_conditional_ppg | next_ppg_primary_blend | mae | 3368 | 2.6839 |
| next_conditional_ppg | next_ppg_primary_blend | bias | 3368 | 0.2193 |
| next_conditional_ppg | next_ppg_primary_blend | spearman | 3368 | 0.6937 |
| next_conditional_ppg | next_ppg_random_forest | rmse | 3368 | 3.5118 |
| next_conditional_ppg | next_ppg_random_forest | mae | 3368 | 2.6864 |
| next_conditional_ppg | next_ppg_random_forest | bias | 3368 | 0.2158 |
| next_conditional_ppg | next_ppg_random_forest | spearman | 3368 | 0.6910 |
| next_participation | next_participation_lightgbm | brier | 6988 | 0.1657 |
| next_participation | next_participation_lightgbm | log_loss | 6988 | 0.5015 |
| next_participation | next_participation_lightgbm | calibration_bias | 6988 | 0.0259 |
| next_participation | next_participation_lightgbm | roc_auc | 6988 | 0.8354 |
| next_participation | next_participation_logistic | brier | 6988 | 0.1747 |
| next_participation | next_participation_logistic | log_loss | 6988 | 0.5232 |
| next_participation | next_participation_logistic | calibration_bias | 6988 | 0.0442 |
| next_participation | next_participation_logistic | roc_auc | 6988 | 0.8179 |
| next_participation | next_participation_position_experience_prior | brier | 6988 | 0.2645 |
| next_participation | next_participation_position_experience_prior | log_loss | 6988 | 0.7248 |
| next_participation | next_participation_position_experience_prior | calibration_bias | 6988 | 0.1022 |
| next_participation | next_participation_position_experience_prior | roc_auc | 6988 | 0.4995 |

## Causal comparisons

| comparison | pooled_challenger | pooled_reference | mean_origin_delta | origin_wins | bootstrap_95_lower | bootstrap_95_upper |
| --- | --- | --- | --- | --- | --- | --- |
| primary_vs_expert_carry | 3.5056 | 4.2671 | -0.7576 | 8 | -0.8495 | -0.6678 |
| aging_vs_expert_carry | 3.9117 | 4.2671 | -0.3516 | 8 | -0.4617 | -0.2324 |
| lasso_vs_expert_carry | 3.5603 | 4.2671 | -0.7035 | 8 | -0.8020 | -0.6067 |
| rf_vs_expert_carry | 3.5118 | 4.2671 | -0.7510 | 8 | -0.8431 | -0.6582 |
| lightgbm_vs_expert_carry | 3.5123 | 4.2671 | -0.7511 | 8 | -0.8308 | -0.6710 |
| participation_lgbm_vs_prior | 0.1657 | 0.2645 | -0.0973 | 8 | -0.1108 | -0.0840 |
| participation_lgbm_vs_logistic | 0.1657 | 0.1747 | -0.0085 | 8 | -0.0130 | -0.0048 |

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

Runtime: 110.3 seconds.
