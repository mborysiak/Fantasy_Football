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
| all_2017_2025 | next_both_w050 | 2.34102 | 26.79269 | 1.51252 | -1.65388 | 0.12502 |
| all_2017_2025 | next_participation_w025 | 2.34327 | 26.77664 | 1.51430 | -1.65422 | 0.13231 |
| all_2017_2025 | next_participation_w050 | 2.34401 | 26.79705 | 1.51324 | -1.65419 | 0.12981 |
| all_2017_2025 | next_residual_w025 | 2.34241 | 26.76517 | 1.51368 | -1.65415 | 0.13333 |
| all_2017_2025 | next_residual_w050 | 2.34184 | 26.77201 | 1.51388 | -1.65382 | 0.13055 |
| all_2017_2025 | next_residual_w100 | 2.34129 | 26.79431 | 1.51381 | -1.65349 | 0.12255 |
| all_2017_2025 | production_no_next | 2.34329 | 26.75515 | 1.51372 | -1.65419 | 0.13657 |
| temporal_2023_2025 | next_both_w050 | 2.34434 | 26.26973 | 1.52569 | -1.40580 | 0.03191 |
| temporal_2023_2025 | next_participation_w025 | 2.34435 | 26.24146 | 1.52669 | -1.40607 | 0.02884 |
| temporal_2023_2025 | next_participation_w050 | 2.34501 | 26.25753 | 1.52708 | -1.40603 | 0.03000 |
| temporal_2023_2025 | next_residual_w025 | 2.34355 | 26.21921 | 1.52521 | -1.40574 | 0.03232 |
| temporal_2023_2025 | next_residual_w050 | 2.34382 | 26.22305 | 1.52748 | -1.40531 | 0.03052 |
| temporal_2023_2025 | next_residual_w100 | 2.34369 | 26.25548 | 1.52680 | -1.40534 | 0.02559 |
| temporal_2023_2025 | production_no_next | 2.34574 | 26.23136 | 1.52544 | -1.40589 | 0.03347 |

## Paired candidate-minus-baseline results

| candidate_method | period | metric | candidate_minus_baseline | bootstrap_p025 | bootstrap_p975 | probability_candidate_better |
| --- | --- | --- | --- | --- | --- | --- |
| next_residual_w025 | all_2017_2025 | ppg_crps | -0.00088 | -0.00187 | 0.00007 | 0.96600 |
| next_residual_w025 | all_2017_2025 | contribution_crps | 0.01002 | -0.00414 | 0.02394 | 0.08900 |
| next_residual_w025 | all_2017_2025 | played_crps | -0.00005 | -0.00205 | 0.00193 | 0.53350 |
| next_residual_w025 | temporal_2023_2025 | ppg_crps | -0.00219 | -0.00356 | -0.00100 | 1.00000 |
| next_residual_w025 | temporal_2023_2025 | contribution_crps | -0.01214 | -0.02830 | 0.01430 | 0.73400 |
| next_residual_w025 | temporal_2023_2025 | played_crps | -0.00022 | -0.00380 | 0.00232 | 0.58200 |
| next_residual_w050 | all_2017_2025 | ppg_crps | -0.00145 | -0.00279 | -0.00001 | 0.97500 |
| next_residual_w050 | all_2017_2025 | contribution_crps | 0.01686 | 0.00035 | 0.03439 | 0.02050 |
| next_residual_w050 | all_2017_2025 | played_crps | 0.00015 | -0.00190 | 0.00200 | 0.43250 |
| next_residual_w050 | temporal_2023_2025 | ppg_crps | -0.00191 | -0.00273 | -0.00135 | 1.00000 |
| next_residual_w050 | temporal_2023_2025 | contribution_crps | -0.00830 | -0.02209 | 0.01198 | 0.73400 |
| next_residual_w050 | temporal_2023_2025 | played_crps | 0.00204 | 0.00045 | 0.00370 | 0.00000 |
| next_residual_w100 | all_2017_2025 | ppg_crps | -0.00200 | -0.00418 | 0.00015 | 0.96350 |
| next_residual_w100 | all_2017_2025 | contribution_crps | 0.03916 | 0.02614 | 0.05523 | 0.00000 |
| next_residual_w100 | all_2017_2025 | played_crps | 0.00008 | -0.00316 | 0.00346 | 0.47700 |
| next_residual_w100 | temporal_2023_2025 | ppg_crps | -0.00204 | -0.00279 | -0.00164 | 1.00000 |
| next_residual_w100 | temporal_2023_2025 | contribution_crps | 0.02413 | 0.01078 | 0.03563 | 0.00000 |
| next_residual_w100 | temporal_2023_2025 | played_crps | 0.00136 | -0.00185 | 0.00442 | 0.14800 |
| next_participation_w025 | all_2017_2025 | ppg_crps | -0.00002 | -0.00094 | 0.00103 | 0.50400 |
| next_participation_w025 | all_2017_2025 | contribution_crps | 0.02150 | 0.00442 | 0.03791 | 0.00600 |
| next_participation_w025 | all_2017_2025 | played_crps | 0.00058 | -0.00072 | 0.00168 | 0.17050 |
| next_participation_w025 | temporal_2023_2025 | ppg_crps | -0.00138 | -0.00172 | -0.00073 | 1.00000 |
| next_participation_w025 | temporal_2023_2025 | contribution_crps | 0.01010 | -0.01631 | 0.04578 | 0.25500 |
| next_participation_w025 | temporal_2023_2025 | played_crps | 0.00125 | 0.00009 | 0.00268 | 0.00000 |
| next_participation_w050 | all_2017_2025 | ppg_crps | 0.00072 | -0.00096 | 0.00262 | 0.22000 |
| next_participation_w050 | all_2017_2025 | contribution_crps | 0.04190 | 0.02350 | 0.06041 | 0.00000 |
| next_participation_w050 | all_2017_2025 | played_crps | -0.00048 | -0.00224 | 0.00114 | 0.71450 |
| next_participation_w050 | temporal_2023_2025 | ppg_crps | -0.00073 | -0.00295 | 0.00102 | 0.73000 |
| next_participation_w050 | temporal_2023_2025 | contribution_crps | 0.02617 | -0.00059 | 0.05572 | 0.03050 |
| next_participation_w050 | temporal_2023_2025 | played_crps | 0.00164 | -0.00024 | 0.00309 | 0.03050 |
| next_both_w050 | all_2017_2025 | ppg_crps | -0.00227 | -0.00439 | -0.00020 | 0.98200 |
| next_both_w050 | all_2017_2025 | contribution_crps | 0.03755 | 0.02203 | 0.05350 | 0.00000 |
| next_both_w050 | all_2017_2025 | played_crps | -0.00120 | -0.00305 | 0.00043 | 0.91850 |
| next_both_w050 | temporal_2023_2025 | ppg_crps | -0.00140 | -0.00335 | -0.00031 | 1.00000 |
| next_both_w050 | temporal_2023_2025 | contribution_crps | 0.03837 | 0.02724 | 0.05294 | 0.00000 |
| next_both_w050 | temporal_2023_2025 | played_crps | 0.00025 | -0.00200 | 0.00151 | 0.24500 |

The residual feature is the within-position percentile of predicted
following-season PPG change versus the origin expert projection. The
participation feature is the predicted probability of any following-season
appearance. Both are matching context only; neither creates another residual
draw or directly changes current-season games played.

Runtime: 52.8 seconds.
