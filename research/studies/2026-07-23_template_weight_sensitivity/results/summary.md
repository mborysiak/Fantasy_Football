# Weekly Template Weight Sensitivity

## Design

- Held out 1,620 player-seasons at strict rolling origins.
- Evaluated 36 paired local weight specifications.
- The reference removes `projection_x_exp` and fixes 12-season recency; all other matcher mechanics remain unchanged.
- Selection uses 2017-2022 only, with aggregate and position guardrails. The promotion threshold also requires at least 0.1% development composite improvement, non-worse temporal composite, temporal position safety, and two of three recent nested selections.

## Development leaderboard

| method | family | multiplier | selection_loss | ppg_crps | contribution_crps | played_crps | guardrail_pass | max_position_composite_delta | max_position_metric_delta | within_one_se | development_winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_weights__w005 | all_weights | 0.05 | 0.997878 | 1.871158 | 20.411251 | 1.507116 | True | 0.001942 | 0.003216 | 1 | 1 |
| all_weights__w001 | all_weights | 0.01 | 0.997885 | 1.871114 | 20.411368 | 1.507173 | True | 0.002322 | 0.004009 | 1 | 0 |
| all_weights__w012 | all_weights | 0.125 | 0.997894 | 1.871254 | 20.411353 | 1.507103 | True | 0.001301 | 0.001977 | 1 | 0 |
| all_weights__w025 | all_weights | 0.25 | 0.997999 | 1.871454 | 20.412642 | 1.507321 | True | 0.000399 | 0.000624 | 1 | 0 |
| all_weights__w038 | all_weights | 0.375 | 0.998212 | 1.871837 | 20.415381 | 1.507773 | True | -0.000201 | 0.000275 | 1 | 0 |
| all_weights__w050 | all_weights | 0.5 | 0.998525 | 1.872425 | 20.419694 | 1.508403 | True | -0.000433 | 0.000118 | 0 | 0 |
| all_weights__w062 | all_weights | 0.625 | 0.998914 | 1.873117 | 20.424811 | 1.509229 | True | -0.000416 | 3.8e-05 | 0 | 0 |
| all_weights__w075 | all_weights | 0.75 | 0.999345 | 1.873866 | 20.430429 | 1.510163 | True | -0.000171 | 7.6e-05 | 0 | 0 |
| all_weights__w088 | all_weights | 0.875 | 0.999705 | 1.874565 | 20.435431 | 1.510863 | True | -7.4e-05 | -1.2e-05 | 0 | 0 |
| projection_rank__w075 | projection_rank | 0.75 | 0.999879 | 1.874697 | 20.436244 | 1.511485 | True | 0.001139 | 0.002494 | 0 | 0 |
| adp_rank__w125 | adp_rank | 1.25 | 0.999896 | 1.875187 | 20.436108 | 1.511177 | True | 8.3e-05 | 0.000723 | 0 | 0 |
| room_hierarchy__w125 | room_hierarchy | 1.25 | 0.999932 | 1.876112 | 20.434529 | 1.510714 | True | 0.000306 | 0.001744 | 0 | 0 |

## Family sensitivity

Negative composite deltas favor the perturbation.

| family | dev_w75 | dev_w125 | temporal_w75 | temporal_w125 |
| --- | --- | --- | --- | --- |
| absolute_ppg | 0.000492 | 0.00078 | -0.000322 | -0.001411 |
| adp_rank | -0.0 | -0.000104 | -0.000518 | -0.000214 |
| component_ranks | 0.000291 | 0.000257 | 7.7e-05 | -0.000682 |
| concentration | 0.000176 | 0.000172 | -0.000286 | -0.000571 |
| disagreement | 2.1e-05 | 0.000187 | -0.000341 | -0.000459 |
| experience | 0.000109 | 0.000388 | -0.001656 | -0.000336 |
| market_gap | 0.000106 | -3.9e-05 | -0.000325 | -0.00039 |
| projection_rank | -0.000121 | 0.000897 | -0.001232 | 0.00064 |
| room_hierarchy | 0.000382 | -6.8e-05 | -0.000156 | -0.000772 |
| room_share | 0.000442 | 4e-05 | 5.7e-05 | -0.000498 |
| scoring_mix | 6.6e-05 | 0.000137 | 0.000111 | -0.000582 |
| team_pass_environment | 0.000104 | 0.000246 | -0.000126 | -0.00078 |

## Overall distance-sharpness curve

| multiplier | loss_dev | loss_recent | loss_temporal | ess_dev | ess_recent | ess_temporal |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | -0.002115 | -0.000248 | 0.000407 | 77.248178 | 75.855912 | 75.087995 |
| 0.05 | -0.002122 | -0.000352 | 0.000273 | 77.138571 | 75.758675 | 75.004415 |
| 0.125 | -0.002106 | -0.00051 | 6.9e-05 | 76.694578 | 75.388866 | 74.68213 |
| 0.25 | -0.002001 | -0.000692 | -0.000172 | 75.355194 | 74.293328 | 73.714855 |
| 0.375 | -0.001788 | -0.000762 | -0.000305 | 73.427572 | 72.707162 | 72.294996 |
| 0.5 | -0.001475 | -0.000727 | -0.000361 | 71.108723 | 70.766159 | 70.533275 |
| 0.625 | -0.001086 | -0.000608 | -0.000351 | 68.630886 | 68.631091 | 68.565329 |
| 0.75 | -0.000655 | -0.000442 | -0.000288 | 66.171845 | 66.441397 | 66.52498 |
| 0.875 | -0.000295 | -0.000228 | -0.000139 | 63.889434 | 64.335947 | 64.530588 |
| 1.25 | 0.000551 | 0.000588 | 0.000421 | 58.75731 | 59.27066 | 59.612007 |
| 1.5 | 0.000911 | 0.001059 | 0.000862 | 56.719219 | 57.097695 | 57.436257 |

## Nested rolling selection

| target_season | training_start | training_end | training_target_rows | selected_method | selected_family | selected_multiplier | selected_development_loss | baseline_development_loss | selected_minus_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021 | 2017 | 2020 | 720 | all_weights__w001 | all_weights | 0.01 | 0.9966044599387366 | 1.0 | -0.0033955400612634135 |
| 2022 | 2017 | 2021 | 900 | all_weights__w001 | all_weights | 0.01 | 0.9974936584043761 | 0.9999999999999999 | -0.0025063415956237955 |
| 2023 | 2017 | 2022 | 1080 | all_weights__w005 | all_weights | 0.05 | 0.9978784728369844 | 1.0 | -0.002121527163015613 |
| 2024 | 2017 | 2023 | 1260 | all_weights__w012 | all_weights | 0.125 | 0.9980972452940775 | 1.0 | -0.0019027547059224936 |
| 2025 | 2017 | 2024 | 1440 | all_weights__w012 | all_weights | 0.125 | 0.9984039560285423 | 1.0 | -0.001596043971457739 |

| method | n | ppg_crps | contribution_crps | played_crps | plus5_brier | impact_brier | impact_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nested_rolling_selected | 900.0 | 1.935893 | 20.558639 | 1.541415 | 0.060011 | 0.09438 | 0.659347 |
| recommended | 900.0 | 1.936863 | 20.577614 | 1.538199 | 0.060159 | 0.094957 | 0.650313 |

## Development-winner clustered uncertainty

| candidate_method | period | metric | candidate_minus_baseline | bootstrap_p025 | bootstrap_p975 | probability_candidate_better |
| --- | --- | --- | --- | --- | --- | --- |
| all_weights__w005 | development_2017_2022 | ppg_crps | -0.003969 | -0.00604 | -0.001528 | 0.9995 |
| all_weights__w005 | development_2017_2022 | contribution_crps | -0.029401 | -0.044776 | -0.013763 | 1.0 |
| all_weights__w005 | development_2017_2022 | played_crps | -0.004246 | -0.010331 | 0.001714 | 0.908 |
| all_weights__w005 | recent_2020_2025 | ppg_crps | -0.001119 | -0.003416 | 0.000879 | 0.84 |
| all_weights__w005 | recent_2020_2025 | contribution_crps | -0.024281 | -0.042769 | -0.003998 | 0.99 |
| all_weights__w005 | recent_2020_2025 | played_crps | 0.00109 | -0.003815 | 0.004422 | 0.267 |
| all_weights__w005 | temporal_2023_2025 | ppg_crps | -0.000197 | -0.002966 | 0.00223 | 0.5965 |
| all_weights__w005 | temporal_2023_2025 | contribution_crps | -0.019875 | -0.039977 | 0.019346 | 0.955 |
| all_weights__w005 | temporal_2023_2025 | played_crps | 0.002837 | 0.000644 | 0.004447 | 0.0 |
| all_weights__w001 | development_2017_2022 | ppg_crps | -0.004013 | -0.00626 | -0.00135 | 0.999 |
| all_weights__w001 | development_2017_2022 | contribution_crps | -0.029285 | -0.045079 | -0.013198 | 1.0 |
| all_weights__w001 | development_2017_2022 | played_crps | -0.004189 | -0.010543 | 0.001984 | 0.897 |
| all_weights__w001 | recent_2020_2025 | ppg_crps | -0.001007 | -0.003459 | 0.001093 | 0.793 |
| all_weights__w001 | recent_2020_2025 | contribution_crps | -0.023826 | -0.043175 | -0.002145 | 0.984 |
| all_weights__w001 | recent_2020_2025 | played_crps | 0.001442 | -0.003611 | 0.004847 | 0.248 |
| all_weights__w001 | temporal_2023_2025 | ppg_crps | -6.5e-05 | -0.003011 | 0.002473 | 0.5965 |
| all_weights__w001 | temporal_2023_2025 | contribution_crps | -0.019153 | -0.04145 | 0.02258 | 0.7365 |
| all_weights__w001 | temporal_2023_2025 | played_crps | 0.003294 | 0.001174 | 0.004962 | 0.0 |
| all_weights__w012 | development_2017_2022 | ppg_crps | -0.003873 | -0.005618 | -0.001851 | 1.0 |
| all_weights__w012 | development_2017_2022 | contribution_crps | -0.029299 | -0.04378 | -0.01453 | 1.0 |
| all_weights__w012 | development_2017_2022 | played_crps | -0.004259 | -0.009831 | 0.00128 | 0.9245 |
| all_weights__w012 | recent_2020_2025 | ppg_crps | -0.001285 | -0.003324 | 0.000467 | 0.918 |
| all_weights__w012 | recent_2020_2025 | contribution_crps | -0.024798 | -0.041884 | -0.005896 | 0.993 |
| all_weights__w012 | recent_2020_2025 | played_crps | 0.000534 | -0.00398 | 0.003718 | 0.331 |
| all_weights__w012 | temporal_2023_2025 | ppg_crps | -0.000354 | -0.002855 | 0.00184 | 0.636 |
| all_weights__w012 | temporal_2023_2025 | contribution_crps | -0.020675 | -0.038811 | 0.013919 | 0.955 |
| all_weights__w012 | temporal_2023_2025 | played_crps | 0.002084 | -0.00019 | 0.003552 | 0.0395 |
| all_weights__w025 | development_2017_2022 | ppg_crps | -0.003673 | -0.00487 | -0.002313 | 1.0 |
| all_weights__w025 | development_2017_2022 | contribution_crps | -0.028011 | -0.040769 | -0.015131 | 1.0 |
| all_weights__w025 | development_2017_2022 | played_crps | -0.004041 | -0.008743 | 0.000742 | 0.9435 |
| all_weights__w025 | recent_2020_2025 | ppg_crps | -0.00151 | -0.003134 | -6e-05 | 0.9795 |
| all_weights__w025 | recent_2020_2025 | contribution_crps | -0.024622 | -0.039091 | -0.008657 | 0.999 |
| all_weights__w025 | recent_2020_2025 | played_crps | -0.000135 | -0.003914 | 0.002647 | 0.4865 |
| all_weights__w025 | temporal_2023_2025 | ppg_crps | -0.000513 | -0.002651 | 0.001338 | 0.636 |
| all_weights__w025 | temporal_2023_2025 | contribution_crps | -0.020659 | -0.03654 | 0.006639 | 0.955 |
| all_weights__w025 | temporal_2023_2025 | played_crps | 0.001106 | -0.001183 | 0.002257 | 0.2465 |
| all_weights__w038 | development_2017_2022 | ppg_crps | -0.00329 | -0.004092 | -0.002474 | 1.0 |
| all_weights__w038 | development_2017_2022 | contribution_crps | -0.025271 | -0.03622 | -0.014132 | 1.0 |
| all_weights__w038 | development_2017_2022 | played_crps | -0.003588 | -0.007482 | 0.000395 | 0.965 |
| all_weights__w038 | recent_2020_2025 | ppg_crps | -0.001548 | -0.002837 | -0.000225 | 0.992 |
| all_weights__w038 | recent_2020_2025 | contribution_crps | -0.023237 | -0.035345 | -0.010494 | 1.0 |
| all_weights__w038 | recent_2020_2025 | played_crps | -0.000533 | -0.003774 | 0.00196 | 0.6095 |
| all_weights__w038 | temporal_2023_2025 | ppg_crps | -0.000544 | -0.002384 | 0.001061 | 0.755 |
| all_weights__w038 | temporal_2023_2025 | contribution_crps | -0.019351 | -0.032098 | 0.001324 | 0.955 |
| all_weights__w038 | temporal_2023_2025 | played_crps | 0.000432 | -0.001754 | 0.001849 | 0.2465 |

## Promotion screen

| method | development_improvement | temporal_composite_delta | max_temporal_position_composite_delta | max_temporal_position_metric_delta | recent_nested_wins | qualifies_for_promotion |
| --- | --- | --- | --- | --- | --- | --- |
| all_weights__w005 | 0.002122 | 0.000273 | 0.004629 | 0.011017 | 1 | False |
| all_weights__w001 | 0.002115 | 0.000407 | 0.00508 | 0.011995 | 0 | False |
| all_weights__w012 | 0.002106 | 6.9e-05 | 0.00395 | 0.009409 | 2 | False |
| all_weights__w025 | 0.002001 | -0.000172 | 0.003105 | 0.00727 | 0 | False |
| all_weights__w038 | 0.001788 | -0.000305 | 0.002515 | 0.005661 | 0 | False |
| all_weights__w050 | 0.001475 | -0.000361 | 0.002047 | 0.004495 | 0 | False |
| all_weights__w062 | 0.001086 | -0.000351 | 0.001669 | 0.003637 | 0 | False |
| all_weights__w075 | 0.000655 | -0.000288 | 0.001162 | 0.002499 | 0 | False |
| all_weights__w088 | 0.000295 | -0.000139 | 0.000635 | 0.001387 | 0 | False |
| projection_rank__w075 | 0.000121 | -0.001232 | 0.0001 | 0.000936 | 0 | False |
| adp_rank__w125 | 0.000104 | -0.000214 | 0.001484 | 0.002166 | 0 | False |
| room_hierarchy__w125 | 6.8e-05 | -0.000772 | -0.000381 | 0.000534 | 0 | False |

## Decision

Retain the reference weights; no tested perturbation clears every promotion threshold.

- Development winner: `all_weights__w005`.
- Development improvement: 0.2122%.
- Temporal composite delta: +0.0273%.
- Worst temporal position composite / metric deltas: +0.4629% / +1.1017%.
- Same winner selected in 1/3 recent nested origins.
- No individual feature-family change improved development composite by more than 0.0121%.
- The near-uniform development winner worsened temporal played-games CRPS by +0.00284 (cluster interval +0.00064 to +0.00445).
- Lower overall sharpness remains a useful future sampling-kernel hypothesis, but its exact scale drifted across rolling origins and should not be bundled into the feature/recency update.
- Production remains unchanged.

Runtime: 125.4 seconds.
