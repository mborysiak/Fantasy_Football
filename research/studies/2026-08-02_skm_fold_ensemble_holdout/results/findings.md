# SKM fold-ensemble sealed-holdout findings

## Decision

Retain the current single-fit methodology. Applying the legacy-style five-member fold-parameter and seed bag to every component does not improve the sealed 2023-2025 holdout: it is effectively tied in DK and materially worse in beta. Merely averaging five estimator seeds around the current parameters is also neutral across leagues.

The beta loss is parameter-bagging instability rather than estimator seed noise. One Lasso fold selected alpha 0.1; the bag worsened beta Lasso RMSE by 0.015941, while fold-bagged RF improved by 0.002783 and LightGBM was flat. DK shows the same offset at smaller magnitude: RF improves but Lasso worsens, leaving the blend tied.

## Primary sealed-holdout scores

| league | method | rows | rmse | rmse_delta | mae_delta | absolute_bias_delta |
|---|---|---|---|---|---|---|
| dk | current_seed_bag | 1237.000000 | 3.090479 | -0.000515 | -0.000386 | 0.001446 |
| dk | current_single | 1237.000000 | 3.090995 | 0.000000 | 0.000000 | 0.000000 |
| dk | skm_fold_seed_bag | 1237.000000 | 3.091207 | 0.000213 | -0.000147 | 0.002051 |
| beta | current_seed_bag | 1226.000000 | 2.892639 | 0.000028 | 0.000993 | 0.003831 |
| beta | current_single | 1226.000000 | 2.892611 | 0.000000 | 0.000000 | 0.000000 |
| beta | skm_fold_seed_bag | 1226.000000 | 2.897024 | 0.004413 | 0.007148 | 0.011041 |

Negative deltas favor the challenger.

## Paired player-cluster uncertainty for the full SKM bag

| league | rmse_delta | bootstrap_low | bootstrap_high |
|---|---|---|---|
| dk | 0.000213 | -0.002066 | 0.002491 |
| beta | 0.004413 | 0.000917 | 0.007850 |

## Exploratory RF-only hybrid

| league | rf_method | hybrid_method | rmse_delta_vs_current_blend | bootstrap_low | bootstrap_high | prespecified |
|---|---|---|---|---|---|---|
| dk | skm_fold_param_bag | current_plus_skm_fold_param_bag_rf | -0.001193 | -0.002324 | -0.000057 | False |
| beta | skm_fold_param_bag | current_plus_skm_fold_param_bag_rf | -0.000516 | -0.001313 | 0.000266 | False |

Keeping current Lasso and LightGBM while replacing only RF with its fold-parameter bag improves the point estimate in both leagues, but the arm was identified after inspecting component results. The DK interval ends narrowly below zero while the beta interval crosses zero; without prespecification or multiplicity protection, this is a follow-up hypothesis rather than promotion evidence.

## Governance

The 2023-2025 outcomes were never used in fitting, fold assignment, or hyperparameter selection. No production database, feature set, model, or projection changed.
