# Weekly Template Feature Pruning

## Design

- Held out 1,620 player-seasons at strict rolling origins.
- Evaluated 30 paired feature/recency specifications.
- Every donor season is strictly earlier than its target season.
- Development selection uses 2017-2022 only, predeclared guardrails, position-level safety checks, and a paired season-level one-standard-error rule.

## Development selection

Selected specification: `no_exp_interaction__r12`.

| method | feature_count_total | complexity_score | selection_loss | ppg_crps | contribution_crps | played_crps | guardrail_pass | position_guardrail_pass | max_position_composite_delta | max_position_metric_delta | within_one_se | selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_exp_interaction__r12 | 55 | 55 | 0.99875 | 1.87513 | 20.44065 | 1.51136 | True | True | -0.00044 | 0.00075 | 1 | 1 |
| no_receiver_team_pass__r12 | 57 | 57 | 0.99889 | 1.87648 | 20.4439 | 1.51065 | True | True | 0.00028 | 0.00145 | 1 | 0 |
| no_exp_interaction | 55 | 55 | 0.9992 | 1.87639 | 20.44466 | 1.51207 | True | True | -0.00068 | 0.00028 | 1 | 0 |
| no_disagreement__r12 | 51 | 51 | 0.9993 | 1.87577 | 20.44631 | 1.51292 | True | True | 0.00027 | 0.00072 | 0 | 0 |
| no_receiver_team_pass | 57 | 57 | 0.99946 | 1.8776 | 20.4509 | 1.51181 | True | True | 0.0 | 0.00115 | 0 | 0 |
| no_concentration__r12 | 56 | 56 | 0.9998 | 1.87511 | 20.45438 | 1.51512 | True | True | 0.00091 | 0.0025 | 0 | 0 |
| no_adp_rank__r12 | 55 | 55 | 0.99981 | 1.87578 | 20.44806 | 1.5151 | True | True | 0.00279 | 0.00671 | 0 | 0 |
| no_disagreement | 51 | 51 | 0.99996 | 1.87697 | 20.45159 | 1.51457 | True | True | 0.00021 | 0.00066 | 0 | 0 |
| full | 59 | 59 | 1.0 | 1.87665 | 20.45543 | 1.51472 | True | True | 0.0 | 0.0 | 0 | 0 |
| no_market_gap__r12 | 55 | 55 | 1.00019 | 1.8762 | 20.4585 | 1.5157 | True | True | 0.00126 | 0.00299 | 0 | 0 |
| no_market__r12 | 51 | 51 | 1.0002 | 1.87727 | 20.45418 | 1.5152 | True | True | 0.0025 | 0.00566 | 0 | 0 |
| no_concentration | 56 | 56 | 1.00046 | 1.87642 | 20.46125 | 1.51657 | True | True | 0.00105 | 0.00282 | 0 | 0 |

## Fixed-method period checks

| period | method | ppg_crps | contribution_crps | played_crps | plus5_brier | impact_brier | impact_auc | weighted_season_gap | weight_10plus_seasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development_2017_2022 | full | 1.87665 | 20.45543 | 1.51472 | 0.06987 | 0.09815 | 0.68542 | 5.86725 | 0.17581 |
| development_2017_2022 | full__r12 | 1.87527 | 20.45039 | 1.51347 | 0.06965 | 0.09793 | 0.68971 | 5.2606 | 0.13205 |
| development_2017_2022 | no_exp_interaction__r12 | 1.87513 | 20.44065 | 1.51136 | 0.0698 | 0.09766 | 0.69463 | 5.24328 | 0.13211 |
| recent_2020_2025 | full | 1.91827 | 20.57625 | 1.53839 | 0.0639 | 0.09632 | 0.65477 | 7.21399 | 0.32338 |
| recent_2020_2025 | full__r12 | 1.91713 | 20.55989 | 1.5342 | 0.06376 | 0.09583 | 0.66434 | 6.27503 | 0.24135 |
| recent_2020_2025 | no_exp_interaction__r12 | 1.91483 | 20.54817 | 1.53327 | 0.06364 | 0.09564 | 0.66787 | 6.24627 | 0.23917 |
| temporal_2023_2025 | full | 1.99298 | 20.95177 | 1.52278 | 0.06869 | 0.09786 | 0.61444 | 7.91886 | 0.38092 |
| temporal_2023_2025 | full__r12 | 1.99335 | 20.94136 | 1.51871 | 0.06878 | 0.09736 | 0.6267 | 6.79172 | 0.28376 |
| temporal_2023_2025 | no_exp_interaction__r12 | 1.98893 | 20.91932 | 1.51836 | 0.06857 | 0.0972 | 0.62969 | 6.75185 | 0.27944 |

## Nested rolling selection

| target_season | training_start | training_end | training_target_rows | selected_method | selected_variant | selected_uses_recency | selected_feature_count_total | selected_complexity_score | selected_development_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021 | 2017 | 2020 | 720 | no_disagreement__r12 | no_disagreement | 1 | 51 | 51 | 0.9999918880801112 |
| 2022 | 2017 | 2021 | 900 | no_disagreement__r12 | no_disagreement | 1 | 51 | 51 | 0.9996649025187552 |
| 2023 | 2017 | 2022 | 1080 | no_exp_interaction__r12 | no_exp_interaction | 1 | 55 | 55 | 0.9987508333878433 |
| 2024 | 2017 | 2023 | 1260 | no_exp_interaction__r12 | no_exp_interaction | 1 | 55 | 55 | 0.9985026826916195 |
| 2025 | 2017 | 2024 | 1440 | no_exp_interaction__r12 | no_exp_interaction | 1 | 55 | 55 | 0.9986795905167638 |

| method | n | ppg_crps | contribution_crps | played_crps | plus5_brier | impact_brier | impact_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | 900.0 | 1.94064 | 20.60994 | 1.54371 | 0.06035 | 0.0955 | 0.63689 |
| full__r12 | 900.0 | 1.93947 | 20.59033 | 1.53979 | 0.0602 | 0.09515 | 0.64512 |
| nested_rolling_selected | 900.0 | 1.93691 | 20.57641 | 1.53935 | 0.06001 | 0.09505 | 0.64657 |

## Selected-vs-production clustered uncertainty

Negative score deltas favor the selected specification.

| period | baseline_method | metric | candidate_minus_baseline | bootstrap_p025 | bootstrap_p975 | probability_candidate_better |
| --- | --- | --- | --- | --- | --- | --- |
| development_2017_2022 | full | ppg_crps | -0.00152 | -0.003 | 0.00036 | 0.9505 |
| development_2017_2022 | full | contribution_crps | -0.01478 | -0.04637 | 0.00869 | 0.8385 |
| development_2017_2022 | full | played_crps | -0.00336 | -0.00673 | -0.0011 | 1.0 |
| recent_2020_2025 | full | ppg_crps | -0.00344 | -0.00643 | -0.00125 | 1.0 |
| recent_2020_2025 | full | contribution_crps | -0.02808 | -0.0654 | 0.00919 | 0.9355 |
| recent_2020_2025 | full | played_crps | -0.00512 | -0.00845 | -0.00197 | 1.0 |
| temporal_2023_2025 | full | ppg_crps | -0.00405 | -0.01087 | -0.00024 | 1.0 |
| temporal_2023_2025 | full | contribution_crps | -0.03245 | -0.06928 | 0.03643 | 0.7365 |
| temporal_2023_2025 | full | played_crps | -0.00441 | -0.01103 | 6e-05 | 0.9605 |

## Interpretation

- Recommend `no_exp_interaction__r12` for the next production update: retain direct projected PPG and uncapped experience, but remove their redundant projection-by-experience interaction.
- Versus production in untouched 2023-2025, PPG CRPS changed by -0.00405, contribution CRPS by -0.03245, and played-games CRPS by -0.00441; negative changes are improvements.
- Dropping component ranks was rejected despite its aggregate development score: its worst position composite moved by +0.592% and its worst individual position metric by +2.048%.
- Keep ADP/market context, disagreement, component ranks, room hierarchy, concentration, and pass-catcher team environment. The aggressive compact variants lost too much downside and availability precision.
- The gains are small, as expected for pruning a redundant feature. Production remains unchanged until this recommendation is explicitly promoted.

Runtime: 106.2 seconds.
