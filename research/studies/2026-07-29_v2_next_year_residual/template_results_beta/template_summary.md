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
| all_2017_2025 | next_both_w050 | 2.34218 | 26.80627 | 1.51493 | -1.65043 | 0.13234 |
| all_2017_2025 | next_participation_w025 | 2.34365 | 26.79307 | 1.51693 | -1.65094 | 0.13227 |
| all_2017_2025 | next_participation_w050 | 2.34388 | 26.80692 | 1.51521 | -1.65099 | 0.12975 |
| all_2017_2025 | next_residual_w025 | 2.34320 | 26.78523 | 1.51591 | -1.65085 | 0.13444 |
| all_2017_2025 | next_residual_w050 | 2.34217 | 26.79219 | 1.51648 | -1.65051 | 0.13415 |
| all_2017_2025 | next_residual_w100 | 2.34154 | 26.79382 | 1.51566 | -1.65011 | 0.13367 |
| all_2017_2025 | production_no_next | 2.34383 | 26.77231 | 1.51582 | -1.65107 | 0.13445 |
| temporal_2023_2025 | next_both_w050 | 2.33098 | 26.32974 | 1.52375 | -1.43314 | 0.01970 |
| temporal_2023_2025 | next_participation_w025 | 2.33142 | 26.32195 | 1.52662 | -1.43341 | 0.01143 |
| temporal_2023_2025 | next_participation_w050 | 2.33136 | 26.32957 | 1.52477 | -1.43359 | 0.01217 |
| temporal_2023_2025 | next_residual_w025 | 2.33093 | 26.30970 | 1.52462 | -1.43323 | 0.01467 |
| temporal_2023_2025 | next_residual_w050 | 2.32944 | 26.30464 | 1.52547 | -1.43306 | 0.01697 |
| temporal_2023_2025 | next_residual_w100 | 2.32957 | 26.30489 | 1.52449 | -1.43289 | 0.02050 |
| temporal_2023_2025 | production_no_next | 2.33220 | 26.30861 | 1.52646 | -1.43356 | 0.01145 |

## Paired candidate-minus-baseline results

| candidate_method | period | metric | candidate_minus_baseline | bootstrap_p025 | bootstrap_p975 | probability_candidate_better |
| --- | --- | --- | --- | --- | --- | --- |
| next_residual_w025 | all_2017_2025 | ppg_crps | -0.00063 | -0.00175 | 0.00058 | 0.85900 |
| next_residual_w025 | all_2017_2025 | contribution_crps | 0.01292 | -0.00126 | 0.02851 | 0.03450 |
| next_residual_w025 | all_2017_2025 | played_crps | 0.00009 | -0.00144 | 0.00188 | 0.45150 |
| next_residual_w025 | temporal_2023_2025 | ppg_crps | -0.00127 | -0.00170 | -0.00067 | 1.00000 |
| next_residual_w025 | temporal_2023_2025 | contribution_crps | 0.00109 | -0.02218 | 0.02951 | 0.41800 |
| next_residual_w025 | temporal_2023_2025 | played_crps | -0.00184 | -0.00244 | -0.00097 | 1.00000 |
| next_residual_w050 | all_2017_2025 | ppg_crps | -0.00166 | -0.00364 | 0.00048 | 0.94200 |
| next_residual_w050 | all_2017_2025 | contribution_crps | 0.01988 | -0.00812 | 0.04429 | 0.07800 |
| next_residual_w050 | all_2017_2025 | played_crps | 0.00066 | -0.00110 | 0.00255 | 0.22000 |
| next_residual_w050 | temporal_2023_2025 | ppg_crps | -0.00276 | -0.00524 | -0.00143 | 1.00000 |
| next_residual_w050 | temporal_2023_2025 | contribution_crps | -0.00397 | -0.06969 | 0.04537 | 0.60500 |
| next_residual_w050 | temporal_2023_2025 | played_crps | -0.00099 | -0.00372 | 0.00134 | 0.75600 |
| next_residual_w100 | all_2017_2025 | ppg_crps | -0.00229 | -0.00438 | -0.00005 | 0.97800 |
| next_residual_w100 | all_2017_2025 | contribution_crps | 0.02151 | -0.00978 | 0.04517 | 0.08650 |
| next_residual_w100 | all_2017_2025 | played_crps | -0.00016 | -0.00268 | 0.00258 | 0.55100 |
| next_residual_w100 | temporal_2023_2025 | ppg_crps | -0.00263 | -0.00371 | -0.00165 | 1.00000 |
| next_residual_w100 | temporal_2023_2025 | contribution_crps | -0.00371 | -0.09181 | 0.04232 | 0.71650 |
| next_residual_w100 | temporal_2023_2025 | played_crps | -0.00197 | -0.00356 | -0.00058 | 1.00000 |
| next_participation_w025 | all_2017_2025 | ppg_crps | -0.00017 | -0.00085 | 0.00064 | 0.69600 |
| next_participation_w025 | all_2017_2025 | contribution_crps | 0.02076 | 0.01002 | 0.03181 | 0.00000 |
| next_participation_w025 | all_2017_2025 | played_crps | 0.00112 | 0.00009 | 0.00213 | 0.01600 |
| next_participation_w025 | temporal_2023_2025 | ppg_crps | -0.00078 | -0.00143 | -0.00025 | 1.00000 |
| next_participation_w025 | temporal_2023_2025 | contribution_crps | 0.01335 | -0.00500 | 0.03031 | 0.03500 |
| next_participation_w025 | temporal_2023_2025 | played_crps | 0.00016 | -0.00157 | 0.00163 | 0.35800 |
| next_participation_w050 | all_2017_2025 | ppg_crps | 0.00005 | -0.00101 | 0.00113 | 0.44600 |
| next_participation_w050 | all_2017_2025 | contribution_crps | 0.03461 | 0.01414 | 0.05334 | 0.00150 |
| next_participation_w050 | all_2017_2025 | played_crps | -0.00061 | -0.00163 | 0.00043 | 0.86550 |
| next_participation_w050 | temporal_2023_2025 | ppg_crps | -0.00084 | -0.00272 | 0.00055 | 0.86650 |
| next_participation_w050 | temporal_2023_2025 | contribution_crps | 0.02096 | -0.03432 | 0.05221 | 0.26250 |
| next_participation_w050 | temporal_2023_2025 | played_crps | -0.00169 | -0.00311 | 0.00012 | 0.96500 |
| next_both_w050 | all_2017_2025 | ppg_crps | -0.00164 | -0.00378 | 0.00066 | 0.92050 |
| next_both_w050 | all_2017_2025 | contribution_crps | 0.03396 | -0.00059 | 0.06454 | 0.02700 |
| next_both_w050 | all_2017_2025 | played_crps | -0.00088 | -0.00257 | 0.00098 | 0.83000 |
| next_both_w050 | temporal_2023_2025 | ppg_crps | -0.00122 | -0.00245 | -0.00020 | 1.00000 |
| next_both_w050 | temporal_2023_2025 | contribution_crps | 0.02113 | -0.07263 | 0.09383 | 0.26250 |
| next_both_w050 | temporal_2023_2025 | played_crps | -0.00271 | -0.00449 | -0.00153 | 1.00000 |

The residual feature is the within-position percentile of predicted
following-season PPG change versus the origin expert projection. The
participation feature is the predicted probability of any following-season
appearance. Both are matching context only; neither creates another residual
draw or directly changes current-season games played.

Runtime: 48.1 seconds.
