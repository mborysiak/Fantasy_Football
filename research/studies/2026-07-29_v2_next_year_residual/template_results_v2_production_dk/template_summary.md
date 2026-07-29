# Next-Year Weekly-Template Feature Replay (dk)

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
| all_2017_2025 | next_both_w050 | 2.34772 | 26.82603 | 1.51609 | -1.65502 | 0.12345 |
| all_2017_2025 | next_participation_w025 | 2.34951 | 26.79413 | 1.51746 | -1.65486 | 0.13014 |
| all_2017_2025 | next_participation_w050 | 2.34967 | 26.81536 | 1.51529 | -1.65500 | 0.12913 |
| all_2017_2025 | next_residual_w025 | 2.34795 | 26.78154 | 1.51627 | -1.65483 | 0.13025 |
| all_2017_2025 | next_residual_w050 | 2.34754 | 26.79789 | 1.51668 | -1.65484 | 0.12681 |
| all_2017_2025 | next_residual_w100 | 2.34698 | 26.81472 | 1.51474 | -1.65473 | 0.11906 |
| all_2017_2025 | production_no_next | 2.34899 | 26.76900 | 1.51592 | -1.65489 | 0.13321 |
| temporal_2023_2025 | next_both_w050 | 2.35429 | 26.28185 | 1.53239 | -1.40695 | 0.02902 |
| temporal_2023_2025 | next_participation_w025 | 2.35510 | 26.23528 | 1.53187 | -1.40642 | 0.03217 |
| temporal_2023_2025 | next_participation_w050 | 2.35477 | 26.26826 | 1.53051 | -1.40667 | 0.03092 |
| temporal_2023_2025 | next_residual_w025 | 2.35295 | 26.21216 | 1.53012 | -1.40619 | 0.03116 |
| temporal_2023_2025 | next_residual_w050 | 2.35305 | 26.22995 | 1.53115 | -1.40629 | 0.02512 |
| temporal_2023_2025 | next_residual_w100 | 2.35392 | 26.25451 | 1.52876 | -1.40634 | 0.02194 |
| temporal_2023_2025 | production_no_next | 2.35449 | 26.21683 | 1.53036 | -1.40621 | 0.03530 |

## Paired candidate-minus-baseline results

| candidate_method | period | metric | candidate_minus_baseline | bootstrap_p025 | bootstrap_p975 | probability_candidate_better |
| --- | --- | --- | --- | --- | --- | --- |
| next_residual_w025 | all_2017_2025 | ppg_crps | -0.00104 | -0.00240 | 0.00022 | 0.94700 |
| next_residual_w025 | all_2017_2025 | contribution_crps | 0.01254 | 0.00206 | 0.02286 | 0.01350 |
| next_residual_w025 | all_2017_2025 | played_crps | 0.00035 | -0.00200 | 0.00265 | 0.39150 |
| next_residual_w025 | temporal_2023_2025 | ppg_crps | -0.00154 | -0.00321 | 0.00008 | 0.96500 |
| next_residual_w025 | temporal_2023_2025 | contribution_crps | -0.00468 | -0.01419 | 0.01430 | 0.73400 |
| next_residual_w025 | temporal_2023_2025 | played_crps | -0.00024 | -0.00466 | 0.00403 | 0.64450 |
| next_residual_w050 | all_2017_2025 | ppg_crps | -0.00145 | -0.00322 | 0.00025 | 0.95250 |
| next_residual_w050 | all_2017_2025 | contribution_crps | 0.02889 | 0.01415 | 0.04555 | 0.00000 |
| next_residual_w050 | all_2017_2025 | played_crps | 0.00076 | -0.00156 | 0.00289 | 0.24750 |
| next_residual_w050 | temporal_2023_2025 | ppg_crps | -0.00144 | -0.00388 | 0.00026 | 0.96500 |
| next_residual_w050 | temporal_2023_2025 | contribution_crps | 0.01312 | 0.00119 | 0.02478 | 0.00000 |
| next_residual_w050 | temporal_2023_2025 | played_crps | 0.00079 | -0.00463 | 0.00373 | 0.26250 |
| next_residual_w100 | all_2017_2025 | ppg_crps | -0.00200 | -0.00498 | 0.00082 | 0.92150 |
| next_residual_w100 | all_2017_2025 | contribution_crps | 0.04572 | 0.03117 | 0.05943 | 0.00000 |
| next_residual_w100 | all_2017_2025 | played_crps | -0.00119 | -0.00619 | 0.00369 | 0.67100 |
| next_residual_w100 | temporal_2023_2025 | ppg_crps | -0.00057 | -0.00510 | 0.00303 | 0.59550 |
| next_residual_w100 | temporal_2023_2025 | contribution_crps | 0.03768 | 0.02486 | 0.05155 | 0.00000 |
| next_residual_w100 | temporal_2023_2025 | played_crps | -0.00160 | -0.00983 | 0.00384 | 0.71650 |
| next_participation_w025 | all_2017_2025 | ppg_crps | 0.00052 | -0.00019 | 0.00128 | 0.09400 |
| next_participation_w025 | all_2017_2025 | contribution_crps | 0.02514 | 0.01518 | 0.03525 | 0.00000 |
| next_participation_w025 | all_2017_2025 | played_crps | 0.00154 | 0.00020 | 0.00288 | 0.01200 |
| next_participation_w025 | temporal_2023_2025 | ppg_crps | 0.00061 | -0.00054 | 0.00127 | 0.03950 |
| next_participation_w025 | temporal_2023_2025 | contribution_crps | 0.01845 | -0.00429 | 0.03454 | 0.03500 |
| next_participation_w025 | temporal_2023_2025 | played_crps | 0.00151 | 0.00030 | 0.00270 | 0.00000 |
| next_participation_w050 | all_2017_2025 | ppg_crps | 0.00068 | -0.00079 | 0.00226 | 0.19350 |
| next_participation_w050 | all_2017_2025 | contribution_crps | 0.04636 | 0.02703 | 0.06520 | 0.00000 |
| next_participation_w050 | all_2017_2025 | played_crps | -0.00063 | -0.00204 | 0.00062 | 0.81350 |
| next_participation_w050 | temporal_2023_2025 | ppg_crps | 0.00028 | -0.00105 | 0.00172 | 0.37850 |
| next_participation_w050 | temporal_2023_2025 | contribution_crps | 0.05142 | 0.01938 | 0.08011 | 0.00000 |
| next_participation_w050 | temporal_2023_2025 | played_crps | 0.00015 | -0.00058 | 0.00085 | 0.35800 |
| next_both_w050 | all_2017_2025 | ppg_crps | -0.00126 | -0.00360 | 0.00103 | 0.85100 |
| next_both_w050 | all_2017_2025 | contribution_crps | 0.05703 | 0.04378 | 0.06871 | 0.00000 |
| next_both_w050 | all_2017_2025 | played_crps | 0.00017 | -0.00313 | 0.00335 | 0.44500 |
| next_both_w050 | temporal_2023_2025 | ppg_crps | -0.00020 | -0.00523 | 0.00251 | 0.71000 |
| next_both_w050 | temporal_2023_2025 | contribution_crps | 0.06502 | 0.05267 | 0.07238 | 0.00000 |
| next_both_w050 | temporal_2023_2025 | played_crps | 0.00203 | -0.00080 | 0.00367 | 0.03500 |

The residual feature is the within-position percentile of predicted
following-season PPG change versus the origin expert projection. The
participation feature is the predicted probability of any following-season
appearance. Both are matching context only; neither creates another residual
draw or directly changes current-season games played.

Runtime: 52.7 seconds.
