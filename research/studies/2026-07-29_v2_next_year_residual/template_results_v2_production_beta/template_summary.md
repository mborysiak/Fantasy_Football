# Next-Year Weekly-Template Feature Replay (beta)

## Scope

- Strict rolling target seasons: 2017-2025.
- Held-out player-seasons: 1,620.
- Every weekly donor season precedes its target season.
- The next-year fields are themselves causal forecasts with a one-origin
  outcome embargo.
- Baseline is the current production matcher with the 12-season recency prior.
- Production templates and optimizer inputs remain unchanged.

## Context coverage

| universe | rows | next_context_coverage |
| --- | --- | --- |
| all_templates | 5298 | 0.84598 |
| held_out_targets | 1620 | 1.00000 |

## Period results

| period | method | ppg_crps | contribution_crps | played_crps | ppg_bias | played_bias |
| --- | --- | --- | --- | --- | --- | --- |
| all_2017_2025 | next_both_w050 | 2.34762 | 26.84610 | 1.51949 | -1.65145 | 0.13461 |
| all_2017_2025 | next_participation_w025 | 2.34924 | 26.82760 | 1.51885 | -1.65170 | 0.13660 |
| all_2017_2025 | next_participation_w050 | 2.34881 | 26.84546 | 1.51903 | -1.65171 | 0.13249 |
| all_2017_2025 | next_residual_w025 | 2.34866 | 26.82674 | 1.51857 | -1.65159 | 0.13814 |
| all_2017_2025 | next_residual_w050 | 2.34813 | 26.83016 | 1.51852 | -1.65140 | 0.13722 |
| all_2017_2025 | next_residual_w100 | 2.34721 | 26.84168 | 1.51949 | -1.65108 | 0.13560 |
| all_2017_2025 | production_no_next | 2.34897 | 26.81375 | 1.51908 | -1.65176 | 0.13978 |
| temporal_2023_2025 | next_both_w050 | 2.33941 | 26.41909 | 1.53398 | -1.43337 | 0.01982 |
| temporal_2023_2025 | next_participation_w025 | 2.33986 | 26.38658 | 1.53429 | -1.43345 | 0.01635 |
| temporal_2023_2025 | next_participation_w050 | 2.33931 | 26.41152 | 1.53486 | -1.43350 | 0.01534 |
| temporal_2023_2025 | next_residual_w025 | 2.33905 | 26.37930 | 1.53257 | -1.43317 | 0.01912 |
| temporal_2023_2025 | next_residual_w050 | 2.33806 | 26.38064 | 1.53173 | -1.43307 | 0.01690 |
| temporal_2023_2025 | next_residual_w100 | 2.33865 | 26.39317 | 1.53330 | -1.43291 | 0.01969 |
| temporal_2023_2025 | production_no_next | 2.33903 | 26.37226 | 1.53387 | -1.43337 | 0.01769 |

## Paired candidate-minus-baseline results

| candidate_method | period | metric | candidate_minus_baseline | bootstrap_p025 | bootstrap_p975 | probability_candidate_better |
| --- | --- | --- | --- | --- | --- | --- |
| next_residual_w025 | all_2017_2025 | ppg_crps | -0.00031 | -0.00105 | 0.00055 | 0.79250 |
| next_residual_w025 | all_2017_2025 | contribution_crps | 0.01298 | 0.00573 | 0.01994 | 0.00000 |
| next_residual_w025 | all_2017_2025 | played_crps | -0.00051 | -0.00185 | 0.00083 | 0.75700 |
| next_residual_w025 | temporal_2023_2025 | ppg_crps | 0.00002 | -0.00058 | 0.00067 | 0.40450 |
| next_residual_w025 | temporal_2023_2025 | contribution_crps | 0.00704 | -0.00385 | 0.01754 | 0.14950 |
| next_residual_w025 | temporal_2023_2025 | played_crps | -0.00129 | -0.00431 | 0.00073 | 0.75600 |
| next_residual_w050 | all_2017_2025 | ppg_crps | -0.00085 | -0.00235 | 0.00081 | 0.86200 |
| next_residual_w050 | all_2017_2025 | contribution_crps | 0.01641 | -0.00118 | 0.03471 | 0.03500 |
| next_residual_w050 | all_2017_2025 | played_crps | -0.00056 | -0.00239 | 0.00130 | 0.70750 |
| next_residual_w050 | temporal_2023_2025 | ppg_crps | -0.00098 | -0.00302 | 0.00109 | 0.86650 |
| next_residual_w050 | temporal_2023_2025 | contribution_crps | 0.00838 | -0.01931 | 0.04199 | 0.26550 |
| next_residual_w050 | temporal_2023_2025 | played_crps | -0.00213 | -0.00596 | 0.00012 | 0.96050 |
| next_residual_w100 | all_2017_2025 | ppg_crps | -0.00177 | -0.00424 | 0.00050 | 0.93050 |
| next_residual_w100 | all_2017_2025 | contribution_crps | 0.02793 | 0.00829 | 0.04714 | 0.00300 |
| next_residual_w100 | all_2017_2025 | played_crps | 0.00041 | -0.00178 | 0.00248 | 0.33550 |
| next_residual_w100 | temporal_2023_2025 | ppg_crps | -0.00038 | -0.00318 | 0.00234 | 0.64450 |
| next_residual_w100 | temporal_2023_2025 | contribution_crps | 0.02091 | -0.01556 | 0.05675 | 0.14950 |
| next_residual_w100 | temporal_2023_2025 | played_crps | -0.00056 | -0.00395 | 0.00277 | 0.61950 |
| next_participation_w025 | all_2017_2025 | ppg_crps | 0.00026 | -0.00065 | 0.00120 | 0.30300 |
| next_participation_w025 | all_2017_2025 | contribution_crps | 0.01385 | 0.00484 | 0.02237 | 0.00050 |
| next_participation_w025 | all_2017_2025 | played_crps | -0.00023 | -0.00131 | 0.00090 | 0.67150 |
| next_participation_w025 | temporal_2023_2025 | ppg_crps | 0.00082 | -0.00047 | 0.00223 | 0.14800 |
| next_participation_w025 | temporal_2023_2025 | contribution_crps | 0.01432 | -0.00898 | 0.02956 | 0.03500 |
| next_participation_w025 | temporal_2023_2025 | played_crps | 0.00042 | -0.00260 | 0.00318 | 0.37850 |
| next_participation_w050 | all_2017_2025 | ppg_crps | -0.00016 | -0.00143 | 0.00108 | 0.59900 |
| next_participation_w050 | all_2017_2025 | contribution_crps | 0.03171 | 0.01152 | 0.05065 | 0.00000 |
| next_participation_w050 | all_2017_2025 | played_crps | -0.00005 | -0.00209 | 0.00196 | 0.52350 |
| next_participation_w050 | temporal_2023_2025 | ppg_crps | 0.00028 | -0.00123 | 0.00108 | 0.26250 |
| next_participation_w050 | temporal_2023_2025 | contribution_crps | 0.03926 | -0.00379 | 0.08312 | 0.03500 |
| next_participation_w050 | temporal_2023_2025 | played_crps | 0.00100 | -0.00337 | 0.00546 | 0.37850 |
| next_both_w050 | all_2017_2025 | ppg_crps | -0.00135 | -0.00351 | 0.00087 | 0.88950 |
| next_both_w050 | all_2017_2025 | contribution_crps | 0.03235 | 0.01024 | 0.05443 | 0.00000 |
| next_both_w050 | all_2017_2025 | played_crps | 0.00041 | -0.00172 | 0.00254 | 0.34550 |
| next_both_w050 | temporal_2023_2025 | ppg_crps | 0.00037 | -0.00147 | 0.00167 | 0.26250 |
| next_both_w050 | temporal_2023_2025 | contribution_crps | 0.04683 | 0.01579 | 0.06801 | 0.00000 |
| next_both_w050 | temporal_2023_2025 | played_crps | 0.00011 | -0.00268 | 0.00322 | 0.41800 |

The residual feature is the within-position percentile of predicted
following-season PPG change versus the origin expert projection. The
participation feature is the predicted probability of any following-season
appearance. Both are matching context only; neither creates another residual
draw or directly changes current-season games played.

Runtime: 52.9 seconds.
