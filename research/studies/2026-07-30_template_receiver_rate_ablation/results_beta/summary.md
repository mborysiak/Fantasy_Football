# Receiver-Rate Weekly-Template Replay (beta)

## Scope

- Strict rolling target seasons: 2017-2025.
- Held-out player-seasons: 1,620.
- Primary comparison: `both_w050_wrte` versus `production` for WR/TE.
- Rates are preseason V2 projections, not realized outcomes.
- Every donor precedes its target season.
- The production pool size, kernel, recency prior, donor cap, and joint outcome
  transport are unchanged.
- Production code and databases are unchanged.

## Coverage

| population | pos | rows | rate_available | rate_missing | coverage |
| --- | --- | --- | --- | --- | --- |
| historical_templates | QB | 706 | 0 | 706 | 0.000000 |
| historical_templates | RB | 1549 | 1548 | 1 | 0.999354 |
| historical_templates | TE | 833 | 833 | 0 | 1.000000 |
| historical_templates | WR | 2210 | 2210 | 0 | 1.000000 |
| rolling_targets | QB | 216 | 0 | 216 | 0.000000 |
| rolling_targets | RB | 540 | 540 | 0 | 1.000000 |
| rolling_targets | TE | 216 | 216 | 0 | 1.000000 |
| rolling_targets | WR | 648 | 648 | 0 | 1.000000 |

## Outcome summary

| scope | period | method | n | ppg_crps | contribution_crps | played_crps | plus3_brier | impact_brier | impact_auc | effective_sample_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | all_2017_2025 | both_w050_rbwrte | 1620.000000 | 1.912458 | 20.610721 | 1.517271 | 0.153160 | 0.097647 | 0.673246 | 61.610668 |
| all | all_2017_2025 | both_w050_wrte | 1620.000000 | 1.912471 | 20.606717 | 1.516338 | 0.153282 | 0.097777 | 0.671031 | 61.327648 |
| all | all_2017_2025 | production | 1620.000000 | 1.913075 | 20.614248 | 1.515816 | 0.153260 | 0.097522 | 0.676271 | 61.087842 |
| all | all_2017_2025 | tdrate_w050_wrte | 1620.000000 | 1.912901 | 20.614373 | 1.515905 | 0.153120 | 0.097577 | 0.674994 | 61.254440 |
| all | all_2017_2025 | ypr_w050_wrte | 1620.000000 | 1.913244 | 20.611899 | 1.516031 | 0.153319 | 0.097613 | 0.673966 | 61.213097 |
| all | temporal_2023_2025 | both_w050_rbwrte | 540.000000 | 1.987719 | 20.936939 | 1.525234 | 0.162358 | 0.098445 | 0.613368 | 62.289917 |
| all | temporal_2023_2025 | both_w050_wrte | 540.000000 | 1.987303 | 20.932586 | 1.525537 | 0.162502 | 0.098393 | 0.612465 | 62.031655 |
| all | temporal_2023_2025 | production | 540.000000 | 1.989000 | 20.946473 | 1.526458 | 0.162549 | 0.097952 | 0.624062 | 61.825461 |
| all | temporal_2023_2025 | tdrate_w050_wrte | 540.000000 | 1.986992 | 20.936657 | 1.524676 | 0.161955 | 0.098090 | 0.621076 | 62.000184 |
| all | temporal_2023_2025 | ypr_w050_wrte | 540.000000 | 1.988275 | 20.934506 | 1.524559 | 0.162405 | 0.097963 | 0.621111 | 61.916936 |
| wr_te | all_2017_2025 | both_w050_rbwrte | 864.000000 | 1.674398 | 18.524613 | 1.479773 | 0.138253 | 0.086321 | 0.664260 | 61.981558 |
| wr_te | all_2017_2025 | both_w050_wrte | 864.000000 | 1.674398 | 18.524613 | 1.479773 | 0.138253 | 0.086321 | 0.664260 | 61.981558 |
| wr_te | all_2017_2025 | production | 864.000000 | 1.675530 | 18.538733 | 1.478795 | 0.138212 | 0.085844 | 0.677218 | 61.531921 |
| wr_te | all_2017_2025 | tdrate_w050_wrte | 864.000000 | 1.675204 | 18.538969 | 1.478962 | 0.137950 | 0.085946 | 0.673669 | 61.844293 |
| wr_te | all_2017_2025 | ypr_w050_wrte | 864.000000 | 1.675846 | 18.534330 | 1.479198 | 0.138322 | 0.086014 | 0.671419 | 61.766774 |
| wr_te | temporal_2023_2025 | both_w050_rbwrte | 288.000000 | 1.679834 | 18.435926 | 1.428484 | 0.147801 | 0.091071 | 0.649871 | 62.253584 |
| wr_te | temporal_2023_2025 | both_w050_wrte | 288.000000 | 1.679834 | 18.435926 | 1.428484 | 0.147801 | 0.091071 | 0.649871 | 62.253584 |
| wr_te | temporal_2023_2025 | production | 288.000000 | 1.683015 | 18.461964 | 1.430211 | 0.147889 | 0.090245 | 0.673773 | 61.866969 |
| wr_te | temporal_2023_2025 | tdrate_w050_wrte | 288.000000 | 1.679251 | 18.443558 | 1.426870 | 0.146775 | 0.090503 | 0.667829 | 62.194576 |
| wr_te | temporal_2023_2025 | ypr_w050_wrte | 288.000000 | 1.681657 | 18.439526 | 1.426651 | 0.147618 | 0.090266 | 0.667183 | 62.038487 |

## Primary WR/TE clustered comparisons

| scope | candidate_method | baseline_method | period | metric | cluster_type | n | clusters | candidate_minus_baseline | bootstrap_p025 | bootstrap_p975 | probability_candidate_better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wr_te | both_w050_wrte | production | all_2017_2025 | ppg_crps | season | 864 | 9 | -0.001131 | -0.003631 | 0.001518 | 0.815000 |
| wr_te | both_w050_wrte | production | all_2017_2025 | contribution_crps | season | 864 | 9 | -0.014121 | -0.037901 | 0.011752 | 0.867500 |
| wr_te | both_w050_wrte | production | all_2017_2025 | played_crps | season | 864 | 9 | 0.000979 | -0.004603 | 0.006022 | 0.371500 |
| wr_te | both_w050_wrte | production | all_2017_2025 | plus3_brier_row | season | 864 | 9 | 0.000041 | -0.000324 | 0.000430 | 0.422500 |
| wr_te | both_w050_wrte | production | all_2017_2025 | impact_brier_row | season | 864 | 9 | 0.000477 | -0.000039 | 0.000960 | 0.034000 |
| wr_te | both_w050_wrte | production | all_2017_2025 | ppg_crps | player | 864 | 272 | -0.001131 | -0.003644 | 0.001409 | 0.806000 |
| wr_te | both_w050_wrte | production | all_2017_2025 | contribution_crps | player | 864 | 272 | -0.014121 | -0.048459 | 0.020207 | 0.787500 |
| wr_te | both_w050_wrte | production | all_2017_2025 | played_crps | player | 864 | 272 | 0.000979 | -0.003646 | 0.005675 | 0.356000 |
| wr_te | both_w050_wrte | production | all_2017_2025 | plus3_brier_row | player | 864 | 272 | 0.000041 | -0.000666 | 0.000722 | 0.461500 |
| wr_te | both_w050_wrte | production | all_2017_2025 | impact_brier_row | player | 864 | 272 | 0.000477 | -0.000111 | 0.001057 | 0.066000 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | ppg_crps | season | 288 | 3 | -0.003181 | -0.004956 | -0.001124 | 1.000000 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | contribution_crps | season | 288 | 3 | -0.026038 | -0.067575 | 0.000248 | 0.955000 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | played_crps | season | 288 | 3 | -0.001727 | -0.014056 | 0.011919 | 0.625000 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | plus3_brier_row | season | 288 | 3 | -0.000089 | -0.000752 | 0.000653 | 0.643500 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | impact_brier_row | season | 288 | 3 | 0.000826 | -0.000098 | 0.001682 | 0.035000 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | ppg_crps | player | 288 | 146 | -0.003181 | -0.008174 | 0.002134 | 0.878500 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | contribution_crps | player | 288 | 146 | -0.026038 | -0.092244 | 0.045393 | 0.769500 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | played_crps | player | 288 | 146 | -0.001727 | -0.010713 | 0.006935 | 0.667500 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | plus3_brier_row | player | 288 | 146 | -0.000089 | -0.001449 | 0.001279 | 0.567500 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | impact_brier_row | player | 288 | 146 | 0.000826 | -0.000405 | 0.002102 | 0.100500 |

Lower CRPS and Brier scores are better. `candidate_minus_baseline < 0` favors
the receiver-rate matcher.

## Pool-composition audit

| scope | n | mean_pool_overlap_share | median_pool_overlap_share | mean_ypr_profile_distance_delta | mean_td_rate_profile_distance_delta | mean_effective_sample_size_delta |
| --- | --- | --- | --- | --- | --- | --- |
| all | 1620 | 0.915509 | 0.925000 | -0.027727 | -0.021328 | 0.522825 |
| wr_te | 864 | 0.885503 | 0.887500 | -0.039575 | -0.030914 | 0.449637 |
| wr | 648 | 0.874209 | 0.875000 | -0.044562 | -0.032850 | 0.334390 |
| te | 216 | 0.919387 | 0.925000 | -0.024615 | -0.025105 | 0.795377 |
| rb | 540 | 0.929722 | 0.937500 | -0.019861 | -0.014521 | 0.849058 |

Negative profile-distance deltas mean the position-relevant candidate selected
donors closer to the target on that projected rate. The candidate is the primary
WR/TE arm for WR/TE and the RB-extension arm for RB. Pool overlap is the share
of baseline top-80 donors retained.

Runtime: 33.0 seconds.
