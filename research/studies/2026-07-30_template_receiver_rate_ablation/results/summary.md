# Receiver-Rate Weekly-Template Replay (dk)

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
| all | all_2017_2025 | both_w050_rbwrte | 1620.000000 | 2.343037 | 26.745624 | 1.514883 | 0.258431 | 0.135025 | 0.742177 | 62.466720 |
| all | all_2017_2025 | both_w050_wrte | 1620.000000 | 2.343139 | 26.744108 | 1.514886 | 0.258531 | 0.134975 | 0.743809 | 62.179313 |
| all | all_2017_2025 | production | 1620.000000 | 2.343290 | 26.755146 | 1.513724 | 0.258514 | 0.135024 | 0.745393 | 62.067682 |
| all | all_2017_2025 | tdrate_w050_wrte | 1620.000000 | 2.342953 | 26.749512 | 1.515272 | 0.258357 | 0.134942 | 0.746234 | 62.190927 |
| all | all_2017_2025 | ypr_w050_wrte | 1620.000000 | 2.343109 | 26.741411 | 1.514170 | 0.258737 | 0.134948 | 0.745346 | 62.101352 |
| all | temporal_2023_2025 | both_w050_rbwrte | 540.000000 | 2.344301 | 26.209426 | 1.530372 | 0.252494 | 0.134062 | 0.712898 | 63.344845 |
| all | temporal_2023_2025 | both_w050_wrte | 540.000000 | 2.344222 | 26.210785 | 1.530707 | 0.252490 | 0.134041 | 0.712923 | 63.105609 |
| all | temporal_2023_2025 | production | 540.000000 | 2.345735 | 26.231355 | 1.525437 | 0.252948 | 0.133749 | 0.721406 | 63.057980 |
| all | temporal_2023_2025 | tdrate_w050_wrte | 540.000000 | 2.344900 | 26.224111 | 1.529053 | 0.252353 | 0.133792 | 0.722290 | 63.147377 |
| all | temporal_2023_2025 | ypr_w050_wrte | 540.000000 | 2.344496 | 26.201829 | 1.526867 | 0.252402 | 0.133764 | 0.720446 | 63.052403 |
| wr_te | all_2017_2025 | both_w050_rbwrte | 864.000000 | 2.352421 | 27.419490 | 1.480770 | 0.289757 | 0.143481 | 0.786011 | 63.600473 |
| wr_te | all_2017_2025 | both_w050_wrte | 864.000000 | 2.352421 | 27.419490 | 1.480770 | 0.289757 | 0.143481 | 0.786011 | 63.600473 |
| wr_te | all_2017_2025 | production | 864.000000 | 2.352704 | 27.440187 | 1.478591 | 0.289724 | 0.143573 | 0.790888 | 63.391164 |
| wr_te | all_2017_2025 | tdrate_w050_wrte | 864.000000 | 2.352073 | 27.429623 | 1.481494 | 0.289430 | 0.143420 | 0.792905 | 63.622248 |
| wr_te | all_2017_2025 | ypr_w050_wrte | 864.000000 | 2.352364 | 27.414433 | 1.479428 | 0.290143 | 0.143431 | 0.789499 | 63.454296 |
| wr_te | temporal_2023_2025 | both_w050_rbwrte | 288.000000 | 2.289870 | 26.312681 | 1.439088 | 0.276688 | 0.141531 | 0.745181 | 64.496884 |
| wr_te | temporal_2023_2025 | both_w050_wrte | 288.000000 | 2.289870 | 26.312681 | 1.439088 | 0.276688 | 0.141531 | 0.745181 | 64.496884 |
| wr_te | temporal_2023_2025 | production | 288.000000 | 2.292708 | 26.351251 | 1.429208 | 0.277549 | 0.140985 | 0.761976 | 64.407580 |
| wr_te | temporal_2023_2025 | tdrate_w050_wrte | 288.000000 | 2.291142 | 26.337668 | 1.435988 | 0.276432 | 0.141065 | 0.765120 | 64.575198 |
| wr_te | temporal_2023_2025 | ypr_w050_wrte | 288.000000 | 2.290384 | 26.295889 | 1.431889 | 0.276524 | 0.141013 | 0.758170 | 64.397123 |

## Primary WR/TE clustered comparisons

| scope | candidate_method | baseline_method | period | metric | cluster_type | n | clusters | candidate_minus_baseline | bootstrap_p025 | bootstrap_p975 | probability_candidate_better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wr_te | both_w050_wrte | production | all_2017_2025 | ppg_crps | season | 864 | 9 | -0.000283 | -0.002731 | 0.001926 | 0.580500 |
| wr_te | both_w050_wrte | production | all_2017_2025 | contribution_crps | season | 864 | 9 | -0.020697 | -0.055650 | 0.014834 | 0.873500 |
| wr_te | both_w050_wrte | production | all_2017_2025 | played_crps | season | 864 | 9 | 0.002179 | -0.004519 | 0.009705 | 0.286000 |
| wr_te | both_w050_wrte | production | all_2017_2025 | plus3_brier_row | season | 864 | 9 | 0.000033 | -0.001286 | 0.001232 | 0.471000 |
| wr_te | both_w050_wrte | production | all_2017_2025 | impact_brier_row | season | 864 | 9 | -0.000092 | -0.000912 | 0.000639 | 0.582500 |
| wr_te | both_w050_wrte | production | all_2017_2025 | ppg_crps | player | 864 | 272 | -0.000283 | -0.003329 | 0.002612 | 0.567000 |
| wr_te | both_w050_wrte | production | all_2017_2025 | contribution_crps | player | 864 | 272 | -0.020697 | -0.062408 | 0.023169 | 0.829500 |
| wr_te | both_w050_wrte | production | all_2017_2025 | played_crps | player | 864 | 272 | 0.002179 | -0.002744 | 0.007116 | 0.207500 |
| wr_te | both_w050_wrte | production | all_2017_2025 | plus3_brier_row | player | 864 | 272 | 0.000033 | -0.001147 | 0.001152 | 0.477500 |
| wr_te | both_w050_wrte | production | all_2017_2025 | impact_brier_row | player | 864 | 272 | -0.000092 | -0.000755 | 0.000564 | 0.611000 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | ppg_crps | season | 288 | 3 | -0.002839 | -0.006991 | 0.001341 | 0.963500 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | contribution_crps | season | 288 | 3 | -0.038569 | -0.063043 | -0.017612 | 1.000000 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | played_crps | season | 288 | 3 | 0.009880 | -0.008012 | 0.024159 | 0.139000 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | plus3_brier_row | season | 288 | 3 | -0.000860 | -0.002012 | 0.001249 | 0.747500 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | impact_brier_row | season | 288 | 3 | 0.000547 | -0.000117 | 0.001098 | 0.035000 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | ppg_crps | player | 288 | 145 | -0.002839 | -0.008157 | 0.002479 | 0.844000 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | contribution_crps | player | 288 | 145 | -0.038569 | -0.113235 | 0.039181 | 0.845000 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | played_crps | player | 288 | 145 | 0.009880 | 0.001879 | 0.017941 | 0.006500 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | plus3_brier_row | player | 288 | 145 | -0.000860 | -0.002708 | 0.000901 | 0.828500 |
| wr_te | both_w050_wrte | production | temporal_2023_2025 | impact_brier_row | player | 288 | 145 | 0.000547 | -0.000666 | 0.001842 | 0.199500 |

Lower CRPS and Brier scores are better. `candidate_minus_baseline < 0` favors
the receiver-rate matcher.

## Pool-composition audit

| scope | n | mean_pool_overlap_share | median_pool_overlap_share | mean_ypr_profile_distance_delta | mean_td_rate_profile_distance_delta | mean_effective_sample_size_delta |
| --- | --- | --- | --- | --- | --- | --- |
| all | 1620 | 0.914321 | 0.925000 | -0.028112 | -0.021509 | 0.399037 |
| wr_te | 864 | 0.882827 | 0.887500 | -0.040426 | -0.031292 | 0.209309 |
| wr | 648 | 0.870255 | 0.875000 | -0.045762 | -0.033475 | -0.010314 |
| te | 216 | 0.920544 | 0.925000 | -0.024416 | -0.024741 | 0.868175 |
| rb | 540 | 0.930440 | 0.937500 | -0.019654 | -0.014462 | 0.862219 |

Negative profile-distance deltas mean the position-relevant candidate selected
donors closer to the target on that projected rate. The candidate is the primary
WR/TE arm for WR/TE and the RB-extension arm for RB. Pool overlap is the share
of baseline top-80 donors retained.

Runtime: 32.7 seconds.
