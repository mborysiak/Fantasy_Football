# Height/Weight Weekly-Template Replay (beta)

## Scope

- Strict rolling target seasons: 2017-2025.
- Held-out player-seasons: 1,620.
- Primary comparison: `height_weight_w025_all` versus `production`.
- Measurements come from the existing nflverse player master.
- Every donor precedes its target season.
- The production pool size, kernel, recency prior, donor cap, and joint outcome
  transport are unchanged.
- Production code and databases are unchanged.

## Coverage

| population | pos | rows | height_available | weight_available | both_available | both_coverage | position_mismatches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| historical_templates | QB | 706 | 706 | 706 | 706 | 1.000000 | 3 |
| historical_templates | RB | 1549 | 1549 | 1549 | 1549 | 1.000000 | 12 |
| historical_templates | TE | 833 | 831 | 831 | 831 | 0.997599 | 1 |
| historical_templates | WR | 2210 | 2205 | 2205 | 2205 | 0.997738 | 17 |
| rolling_targets | QB | 216 | 216 | 216 | 216 | 1.000000 | 2 |
| rolling_targets | RB | 540 | 540 | 540 | 540 | 1.000000 | 2 |
| rolling_targets | TE | 216 | 216 | 216 | 216 | 1.000000 | 0 |
| rolling_targets | WR | 648 | 648 | 648 | 648 | 1.000000 | 3 |

## Outcome summary

| scope | period | method | n | ppg_crps | contribution_crps | played_crps | plus3_brier | impact_brier | impact_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | all_2017_2025 | height_weight_w025_all | 1620.000000 | 1.911973 | 20.595586 | 1.512978 | 0.153211 | 0.097223 | 0.682362 |
| all | all_2017_2025 | production | 1620.000000 | 1.913075 | 20.614248 | 1.515816 | 0.153260 | 0.097522 | 0.676271 |
| all | temporal_2023_2025 | height_weight_w025_all | 540.000000 | 1.986455 | 20.920497 | 1.522336 | 0.162311 | 0.097039 | 0.644028 |
| all | temporal_2023_2025 | production | 540.000000 | 1.989000 | 20.946473 | 1.526458 | 0.162549 | 0.097952 | 0.624062 |
| qb | all_2017_2025 | height_weight_w025_all | 216.000000 | 2.407279 | 21.993285 | 1.341108 | 0.172756 | 0.118731 | 0.603047 |
| qb | all_2017_2025 | production | 216.000000 | 2.409254 | 22.019568 | 1.340486 | 0.173306 | 0.119284 | 0.593369 |
| qb | temporal_2023_2025 | height_weight_w025_all | 72.000000 | 2.421631 | 20.883095 | 1.754601 | 0.148924 | 0.090980 | 0.467172 |
| qb | temporal_2023_2025 | production | 72.000000 | 2.429462 | 20.994026 | 1.742280 | 0.150787 | 0.093278 | 0.381313 |
| rb | all_2017_2025 | height_weight_w025_all | 540.000000 | 2.094584 | 23.354515 | 1.644470 | 0.169081 | 0.107220 | 0.695593 |
| rb | all_2017_2025 | production | 540.000000 | 2.094675 | 23.372942 | 1.645181 | 0.169318 | 0.107503 | 0.690821 |
| rb | temporal_2023_2025 | height_weight_w025_all | 180.000000 | 2.301510 | 24.868997 | 1.593603 | 0.190112 | 0.111469 | 0.692308 |
| rb | temporal_2023_2025 | production | 180.000000 | 2.302390 | 24.902667 | 1.594123 | 0.190711 | 0.112153 | 0.686432 |
| wr | all_2017_2025 | height_weight_w025_all | 648.000000 | 1.745484 | 19.005927 | 1.453461 | 0.147041 | 0.083704 | 0.710487 |
| wr | all_2017_2025 | production | 648.000000 | 1.746957 | 19.018850 | 1.457521 | 0.146788 | 0.083929 | 0.704029 |
| wr | temporal_2023_2025 | height_weight_w025_all | 216.000000 | 1.753200 | 19.256702 | 1.419995 | 0.154526 | 0.086787 | 0.717432 |
| wr | temporal_2023_2025 | production | 216.000000 | 1.755591 | 19.240151 | 1.429668 | 0.153959 | 0.087580 | 0.696111 |
| te | all_2017_2025 | height_weight_w025_all | 216.000000 | 1.459607 | 17.069541 | 1.534668 | 0.112502 | 0.091281 | 0.575211 |
| te | all_2017_2025 | production | 216.000000 | 1.461246 | 17.098385 | 1.542617 | 0.112483 | 0.091590 | 0.566307 |
| te | temporal_2023_2025 | height_weight_w025_all | 72.000000 | 1.463406 | 16.078037 | 1.418928 | 0.129553 | 0.097778 | 0.615234 |
| te | temporal_2023_2025 | production | 72.000000 | 1.465289 | 16.127402 | 1.431841 | 0.129681 | 0.098243 | 0.585938 |
| wr_te | all_2017_2025 | height_weight_w025_all | 864.000000 | 1.674014 | 18.521830 | 1.473763 | 0.138406 | 0.085598 | 0.684286 |
| wr_te | all_2017_2025 | production | 864.000000 | 1.675530 | 18.538733 | 1.478795 | 0.138212 | 0.085844 | 0.677218 |
| wr_te | temporal_2023_2025 | height_weight_w025_all | 288.000000 | 1.680752 | 18.462035 | 1.419728 | 0.148283 | 0.089534 | 0.694574 |
| wr_te | temporal_2023_2025 | production | 288.000000 | 1.683015 | 18.461964 | 1.430211 | 0.147889 | 0.090245 | 0.673773 |

## Primary clustered comparisons

| scope | candidate_method | baseline_method | period | metric | cluster_type | n | clusters | candidate_minus_baseline | bootstrap_p025 | bootstrap_p975 | probability_candidate_better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | height_weight_w025_all | production | all_2017_2025 | ppg_crps | season | 1620 | 9 | -0.001102 | -0.002202 | 0.000024 | 0.971500 |
| all | height_weight_w025_all | production | all_2017_2025 | contribution_crps | season | 1620 | 9 | -0.018661 | -0.045874 | 0.007208 | 0.918500 |
| all | height_weight_w025_all | production | all_2017_2025 | played_crps | season | 1620 | 9 | -0.002838 | -0.006653 | 0.000583 | 0.942000 |
| all | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | season | 1620 | 9 | -0.000049 | -0.000655 | 0.000431 | 0.542000 |
| all | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | season | 1620 | 9 | -0.000299 | -0.000836 | 0.000144 | 0.893000 |
| all | height_weight_w025_all | production | all_2017_2025 | ppg_crps | player | 1620 | 519 | -0.001102 | -0.002916 | 0.000676 | 0.879000 |
| all | height_weight_w025_all | production | all_2017_2025 | contribution_crps | player | 1620 | 519 | -0.018661 | -0.043292 | 0.004628 | 0.929000 |
| all | height_weight_w025_all | production | all_2017_2025 | played_crps | player | 1620 | 519 | -0.002838 | -0.006243 | 0.000737 | 0.933500 |
| all | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | player | 1620 | 519 | -0.000049 | -0.000647 | 0.000552 | 0.549000 |
| all | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | player | 1620 | 519 | -0.000299 | -0.000775 | 0.000158 | 0.898500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | season | 540 | 3 | -0.002545 | -0.003643 | -0.001974 | 1.000000 |
| all | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | season | 540 | 3 | -0.025976 | -0.081770 | 0.009196 | 0.853500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | played_crps | season | 540 | 3 | -0.004121 | -0.011864 | 0.002278 | 0.853500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | season | 540 | 3 | -0.000238 | -0.001770 | 0.000650 | 0.684500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | season | 540 | 3 | -0.000914 | -0.001706 | 0.000114 | 0.959500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | player | 540 | 269 | -0.002545 | -0.005730 | 0.000665 | 0.939000 |
| all | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | player | 540 | 269 | -0.025976 | -0.072693 | 0.019773 | 0.867500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | played_crps | player | 540 | 269 | -0.004121 | -0.010520 | 0.002530 | 0.889500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | player | 540 | 269 | -0.000238 | -0.001405 | 0.000870 | 0.654500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | player | 540 | 269 | -0.000914 | -0.001888 | -0.000034 | 0.981500 |
| qb | height_weight_w025_all | production | all_2017_2025 | ppg_crps | season | 216 | 9 | -0.001975 | -0.008588 | 0.004458 | 0.713000 |
| qb | height_weight_w025_all | production | all_2017_2025 | contribution_crps | season | 216 | 9 | -0.026283 | -0.102540 | 0.023257 | 0.704500 |
| qb | height_weight_w025_all | production | all_2017_2025 | played_crps | season | 216 | 9 | 0.000622 | -0.008791 | 0.011320 | 0.471500 |
| qb | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | season | 216 | 9 | -0.000549 | -0.002615 | 0.001088 | 0.692500 |
| qb | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | season | 216 | 9 | -0.000553 | -0.001903 | 0.000430 | 0.812500 |
| qb | height_weight_w025_all | production | all_2017_2025 | ppg_crps | player | 216 | 59 | -0.001975 | -0.007137 | 0.003747 | 0.751500 |
| qb | height_weight_w025_all | production | all_2017_2025 | contribution_crps | player | 216 | 59 | -0.026283 | -0.092470 | 0.039678 | 0.784500 |
| qb | height_weight_w025_all | production | all_2017_2025 | played_crps | player | 216 | 59 | 0.000622 | -0.009141 | 0.011104 | 0.490000 |
| qb | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | player | 216 | 59 | -0.000549 | -0.002250 | 0.001210 | 0.744500 |
| qb | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | player | 216 | 59 | -0.000553 | -0.001979 | 0.000772 | 0.777500 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | season | 72 | 3 | -0.007831 | -0.020060 | 0.007906 | 0.851000 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | season | 72 | 3 | -0.110931 | -0.307077 | 0.007874 | 0.970000 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | played_crps | season | 72 | 3 | 0.012321 | -0.004932 | 0.038284 | 0.137500 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | season | 72 | 3 | -0.001863 | -0.007460 | 0.001578 | 0.693000 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | season | 72 | 3 | -0.002298 | -0.004971 | -0.000600 | 1.000000 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | player | 72 | 32 | -0.007831 | -0.018988 | 0.003271 | 0.910000 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | player | 72 | 32 | -0.110931 | -0.240836 | 0.021640 | 0.951500 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | played_crps | player | 72 | 32 | 0.012321 | -0.009968 | 0.038477 | 0.161000 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | player | 72 | 32 | -0.001863 | -0.005285 | 0.000980 | 0.881500 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | player | 72 | 32 | -0.002298 | -0.005310 | 0.000393 | 0.941500 |
| rb | height_weight_w025_all | production | all_2017_2025 | ppg_crps | season | 540 | 9 | -0.000092 | -0.002026 | 0.001970 | 0.549500 |
| rb | height_weight_w025_all | production | all_2017_2025 | contribution_crps | season | 540 | 9 | -0.018426 | -0.037906 | 0.001926 | 0.964000 |
| rb | height_weight_w025_all | production | all_2017_2025 | played_crps | season | 540 | 9 | -0.000710 | -0.003808 | 0.002695 | 0.678500 |
| rb | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | season | 540 | 9 | -0.000238 | -0.001156 | 0.000895 | 0.686500 |
| rb | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | season | 540 | 9 | -0.000283 | -0.000901 | 0.000535 | 0.775500 |
| rb | height_weight_w025_all | production | all_2017_2025 | ppg_crps | player | 540 | 189 | -0.000092 | -0.002637 | 0.002570 | 0.527500 |
| rb | height_weight_w025_all | production | all_2017_2025 | contribution_crps | player | 540 | 189 | -0.018426 | -0.060118 | 0.025620 | 0.800000 |
| rb | height_weight_w025_all | production | all_2017_2025 | played_crps | player | 540 | 189 | -0.000710 | -0.007288 | 0.005879 | 0.572500 |
| rb | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | player | 540 | 189 | -0.000238 | -0.001344 | 0.000864 | 0.658500 |
| rb | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | player | 540 | 189 | -0.000283 | -0.001032 | 0.000417 | 0.774000 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | season | 180 | 3 | -0.000881 | -0.003413 | 0.001319 | 0.734500 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | season | 180 | 3 | -0.033670 | -0.059617 | -0.019855 | 1.000000 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | played_crps | season | 180 | 3 | -0.000520 | -0.004637 | 0.004424 | 0.626000 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | season | 180 | 3 | -0.000599 | -0.001449 | 0.000143 | 0.963500 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | season | 180 | 3 | -0.000684 | -0.001477 | 0.000546 | 0.959000 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | player | 180 | 91 | -0.000881 | -0.006409 | 0.004347 | 0.648500 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | player | 180 | 91 | -0.033670 | -0.125144 | 0.053045 | 0.783000 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | played_crps | player | 180 | 91 | -0.000520 | -0.010677 | 0.009311 | 0.543000 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | player | 180 | 91 | -0.000599 | -0.002777 | 0.001409 | 0.709500 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | player | 180 | 91 | -0.000684 | -0.002195 | 0.000645 | 0.829000 |
| wr | height_weight_w025_all | production | all_2017_2025 | ppg_crps | season | 648 | 9 | -0.001474 | -0.004330 | 0.000982 | 0.858000 |
| wr | height_weight_w025_all | production | all_2017_2025 | contribution_crps | season | 648 | 9 | -0.012923 | -0.072212 | 0.048776 | 0.664000 |
| wr | height_weight_w025_all | production | all_2017_2025 | played_crps | season | 648 | 9 | -0.004060 | -0.012207 | 0.004317 | 0.836500 |
| wr | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | season | 648 | 9 | 0.000253 | -0.000639 | 0.001246 | 0.320500 |
| wr | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | season | 648 | 9 | -0.000225 | -0.000814 | 0.000333 | 0.780500 |
| wr | height_weight_w025_all | production | all_2017_2025 | ppg_crps | player | 648 | 202 | -0.001474 | -0.004814 | 0.001859 | 0.804500 |
| wr | height_weight_w025_all | production | all_2017_2025 | contribution_crps | player | 648 | 202 | -0.012923 | -0.057950 | 0.031275 | 0.705000 |
| wr | height_weight_w025_all | production | all_2017_2025 | played_crps | player | 648 | 202 | -0.004060 | -0.010116 | 0.001631 | 0.919500 |
| wr | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | player | 648 | 202 | 0.000253 | -0.000781 | 0.001254 | 0.308500 |
| wr | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | player | 648 | 202 | -0.000225 | -0.001077 | 0.000589 | 0.690000 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | season | 216 | 3 | -0.002391 | -0.002976 | -0.001729 | 1.000000 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | season | 216 | 3 | 0.016550 | -0.125654 | 0.153373 | 0.371500 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | played_crps | season | 216 | 3 | -0.009673 | -0.026133 | 0.017427 | 0.730000 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | season | 216 | 3 | 0.000567 | -0.000262 | 0.001879 | 0.264500 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | season | 216 | 3 | -0.000793 | -0.001770 | 0.000237 | 0.963000 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | player | 216 | 108 | -0.002391 | -0.007560 | 0.002698 | 0.811000 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | player | 216 | 108 | 0.016550 | -0.059826 | 0.087583 | 0.327500 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | played_crps | player | 216 | 108 | -0.009673 | -0.020179 | 0.001035 | 0.959500 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | player | 216 | 108 | 0.000567 | -0.001219 | 0.002359 | 0.256000 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | player | 216 | 108 | -0.000793 | -0.002471 | 0.000767 | 0.833000 |
| te | height_weight_w025_all | production | all_2017_2025 | ppg_crps | season | 216 | 9 | -0.001640 | -0.003545 | 0.000210 | 0.958500 |
| te | height_weight_w025_all | production | all_2017_2025 | contribution_crps | season | 216 | 9 | -0.028844 | -0.055226 | 0.002455 | 0.965500 |
| te | height_weight_w025_all | production | all_2017_2025 | played_crps | season | 216 | 9 | -0.007949 | -0.017524 | 0.000913 | 0.951500 |
| te | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | season | 216 | 9 | 0.000018 | -0.000662 | 0.000782 | 0.493000 |
| te | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | season | 216 | 9 | -0.000308 | -0.000735 | 0.000113 | 0.916500 |
| te | height_weight_w025_all | production | all_2017_2025 | ppg_crps | player | 216 | 70 | -0.001640 | -0.004383 | 0.001266 | 0.868000 |
| te | height_weight_w025_all | production | all_2017_2025 | contribution_crps | player | 216 | 70 | -0.028844 | -0.077653 | 0.024530 | 0.862500 |
| te | height_weight_w025_all | production | all_2017_2025 | played_crps | player | 216 | 70 | -0.007949 | -0.016779 | 0.002254 | 0.942500 |
| te | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | player | 216 | 70 | 0.000018 | -0.000739 | 0.000813 | 0.480000 |
| te | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | player | 216 | 70 | -0.000308 | -0.000840 | 0.000227 | 0.882500 |
| te | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | season | 72 | 3 | -0.001882 | -0.007103 | 0.002163 | 0.763500 |
| te | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | season | 72 | 3 | -0.049365 | -0.095147 | -0.018512 | 1.000000 |
| te | height_weight_w025_all | production | temporal_2023_2025 | played_crps | season | 72 | 3 | -0.012913 | -0.035446 | 0.005947 | 0.841000 |
| te | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | season | 72 | 3 | -0.000128 | -0.001406 | 0.002128 | 0.725000 |
| te | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | season | 72 | 3 | -0.000466 | -0.001597 | 0.000822 | 0.725000 |
| te | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | player | 72 | 38 | -0.001882 | -0.007671 | 0.003294 | 0.749000 |
| te | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | player | 72 | 38 | -0.049365 | -0.131437 | 0.028772 | 0.885000 |
| te | height_weight_w025_all | production | temporal_2023_2025 | played_crps | player | 72 | 38 | -0.012913 | -0.024866 | -0.001387 | 0.984500 |
| te | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | player | 72 | 38 | -0.000128 | -0.001703 | 0.001601 | 0.558500 |
| te | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | player | 72 | 38 | -0.000466 | -0.001738 | 0.000767 | 0.763000 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | ppg_crps | season | 864 | 9 | -0.001515 | -0.003833 | 0.000394 | 0.925000 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | contribution_crps | season | 864 | 9 | -0.016903 | -0.065505 | 0.033753 | 0.722500 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | played_crps | season | 864 | 9 | -0.005032 | -0.011098 | 0.000689 | 0.954500 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | season | 864 | 9 | 0.000194 | -0.000438 | 0.000829 | 0.280000 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | season | 864 | 9 | -0.000246 | -0.000636 | 0.000157 | 0.881500 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | ppg_crps | player | 864 | 272 | -0.001515 | -0.004048 | 0.000873 | 0.873000 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | contribution_crps | player | 864 | 272 | -0.016903 | -0.050476 | 0.017371 | 0.833500 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | played_crps | player | 864 | 272 | -0.005032 | -0.009724 | -0.000241 | 0.977500 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | player | 864 | 272 | 0.000194 | -0.000593 | 0.000968 | 0.307000 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | player | 864 | 272 | -0.000246 | -0.000876 | 0.000388 | 0.774000 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | season | 288 | 3 | -0.002264 | -0.003626 | -0.001473 | 1.000000 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | season | 288 | 3 | 0.000071 | -0.118027 | 0.106421 | 0.375000 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | played_crps | season | 288 | 3 | -0.010483 | -0.018113 | 0.004209 | 0.959000 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | season | 288 | 3 | 0.000393 | -0.000548 | 0.001133 | 0.150000 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | season | 288 | 3 | -0.000711 | -0.001122 | 0.000022 | 0.970500 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | player | 288 | 146 | -0.002264 | -0.006241 | 0.001722 | 0.869500 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | player | 288 | 146 | 0.000071 | -0.058394 | 0.058761 | 0.475500 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | played_crps | player | 288 | 146 | -0.010483 | -0.019252 | -0.002302 | 0.992000 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | player | 288 | 146 | 0.000393 | -0.000906 | 0.001770 | 0.289000 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | player | 288 | 146 | -0.000711 | -0.001964 | 0.000423 | 0.872500 |

Lower CRPS and Brier scores are better. `candidate_minus_baseline < 0` favors
the height/weight matcher.

## Pool-composition audit

| scope | n | mean_pool_overlap_share | median_pool_overlap_share | mean_height_distance_delta | mean_weight_distance_delta | mean_effective_sample_size_delta |
| --- | --- | --- | --- | --- | --- | --- |
| all | 1620 | 0.912955 | 0.925000 | -0.046664 | -0.047535 | 0.816618 |
| qb | 216 | 0.953125 | 0.956250 | -0.026830 | -0.031448 | 1.245319 |
| rb | 540 | 0.929722 | 0.937500 | -0.036030 | -0.033181 | 0.963637 |
| wr | 648 | 0.878704 | 0.887500 | -0.066208 | -0.069310 | 0.544776 |
| te | 216 | 0.933623 | 0.937500 | -0.034447 | -0.034185 | 0.835895 |
| wr_te | 864 | 0.892433 | 0.900000 | -0.058268 | -0.060528 | 0.617555 |

Negative size-distance deltas mean the primary candidate selected donors closer
to the target on that measurement. Pool overlap is the share of baseline top-80
donors retained.

Runtime: 31.6 seconds.
