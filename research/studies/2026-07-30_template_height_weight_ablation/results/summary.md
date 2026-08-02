# Height/Weight Weekly-Template Replay (dk)

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
| all | all_2017_2025 | height_weight_w025_all | 1620.000000 | 2.343960 | 26.762260 | 1.511470 | 0.258896 | 0.135242 | 0.742094 |
| all | all_2017_2025 | production | 1620.000000 | 2.343290 | 26.755146 | 1.513724 | 0.258514 | 0.135024 | 0.745393 |
| all | temporal_2023_2025 | height_weight_w025_all | 540.000000 | 2.344674 | 26.209602 | 1.525886 | 0.252342 | 0.133878 | 0.716635 |
| all | temporal_2023_2025 | production | 540.000000 | 2.345735 | 26.231355 | 1.525437 | 0.252948 | 0.133749 | 0.721406 |
| qb | all_2017_2025 | height_weight_w025_all | 216.000000 | 2.392453 | 23.781286 | 1.331314 | 0.266994 | 0.142360 | 0.601698 |
| qb | all_2017_2025 | production | 216.000000 | 2.394929 | 23.778347 | 1.329287 | 0.268155 | 0.142749 | 0.599923 |
| qb | temporal_2023_2025 | height_weight_w025_all | 72.000000 | 2.223336 | 20.095796 | 1.774501 | 0.211250 | 0.114869 | 0.463845 |
| qb | temporal_2023_2025 | production | 72.000000 | 2.229711 | 20.181306 | 1.752653 | 0.214352 | 0.115233 | 0.470899 |
| rb | all_2017_2025 | height_weight_w025_all | 540.000000 | 2.309131 | 26.844534 | 1.640002 | 0.204065 | 0.118556 | 0.747337 |
| rb | all_2017_2025 | production | 540.000000 | 2.307571 | 26.849801 | 1.643712 | 0.204720 | 0.118256 | 0.752396 |
| rb | temporal_2023_2025 | height_weight_w025_all | 180.000000 | 2.477873 | 28.433639 | 1.583539 | 0.227486 | 0.129252 | 0.750171 |
| rb | temporal_2023_2025 | production | 180.000000 | 2.476989 | 28.459543 | 1.588518 | 0.229027 | 0.129579 | 0.749715 |
| wr | all_2017_2025 | height_weight_w025_all | 648.000000 | 2.457975 | 28.216656 | 1.454106 | 0.303380 | 0.145318 | 0.806530 |
| wr | all_2017_2025 | production | 648.000000 | 2.456207 | 28.184382 | 1.457096 | 0.301773 | 0.144990 | 0.811141 |
| wr | temporal_2023_2025 | height_weight_w025_all | 216.000000 | 2.366963 | 27.121066 | 1.411652 | 0.273159 | 0.142766 | 0.767637 |
| wr | temporal_2023_2025 | production | 216.000000 | 2.369234 | 27.134739 | 1.412732 | 0.273221 | 0.142208 | 0.770824 |
| te | all_2017_2025 | height_weight_w025_all | 216.000000 | 2.040496 | 25.174360 | 1.542384 | 0.254427 | 0.139611 | 0.712819 |
| te | all_2017_2025 | production | 216.000000 | 2.042195 | 25.207602 | 1.543073 | 0.253577 | 0.139325 | 0.721274 |
| te | temporal_2023_2025 | height_weight_w025_all | 72.000000 | 2.066147 | 24.028919 | 1.475838 | 0.293123 | 0.137790 | 0.725000 |
| te | temporal_2023_2025 | production | 72.000000 | 2.063132 | 24.000786 | 1.478636 | 0.290531 | 0.137315 | 0.741667 |
| wr_te | all_2017_2025 | height_weight_w025_all | 864.000000 | 2.353606 | 27.456082 | 1.476176 | 0.291142 | 0.143891 | 0.787159 |
| wr_te | all_2017_2025 | production | 864.000000 | 2.352704 | 27.440187 | 1.478591 | 0.289724 | 0.143573 | 0.790888 |
| wr_te | temporal_2023_2025 | height_weight_w025_all | 288.000000 | 2.291759 | 26.348029 | 1.427698 | 0.278150 | 0.141522 | 0.756846 |
| wr_te | temporal_2023_2025 | production | 288.000000 | 2.292708 | 26.351251 | 1.429208 | 0.277549 | 0.140985 | 0.761976 |

## Primary clustered comparisons

| scope | candidate_method | baseline_method | period | metric | cluster_type | n | clusters | candidate_minus_baseline | bootstrap_p025 | bootstrap_p975 | probability_candidate_better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | height_weight_w025_all | production | all_2017_2025 | ppg_crps | season | 1620 | 9 | 0.000670 | -0.001170 | 0.002692 | 0.283000 |
| all | height_weight_w025_all | production | all_2017_2025 | contribution_crps | season | 1620 | 9 | 0.007114 | -0.032973 | 0.050160 | 0.406000 |
| all | height_weight_w025_all | production | all_2017_2025 | played_crps | season | 1620 | 9 | -0.002254 | -0.005530 | 0.001903 | 0.888500 |
| all | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | season | 1620 | 9 | 0.000383 | -0.000306 | 0.000990 | 0.124500 |
| all | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | season | 1620 | 9 | 0.000218 | -0.000143 | 0.000555 | 0.109500 |
| all | height_weight_w025_all | production | all_2017_2025 | ppg_crps | player | 1620 | 518 | 0.000670 | -0.001468 | 0.002838 | 0.265000 |
| all | height_weight_w025_all | production | all_2017_2025 | contribution_crps | player | 1620 | 518 | 0.007114 | -0.021799 | 0.037311 | 0.320000 |
| all | height_weight_w025_all | production | all_2017_2025 | played_crps | player | 1620 | 518 | -0.002254 | -0.005926 | 0.001423 | 0.887500 |
| all | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | player | 1620 | 518 | 0.000383 | -0.000323 | 0.001101 | 0.151000 |
| all | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | player | 1620 | 518 | 0.000218 | -0.000221 | 0.000659 | 0.169500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | season | 540 | 3 | -0.001061 | -0.002807 | 0.000190 | 0.959500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | season | 540 | 3 | -0.021754 | -0.095915 | 0.029288 | 0.705000 |
| all | height_weight_w025_all | production | temporal_2023_2025 | played_crps | season | 540 | 3 | 0.000448 | -0.007411 | 0.009488 | 0.423500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | season | 540 | 3 | -0.000607 | -0.001384 | 0.000480 | 0.851500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | season | 540 | 3 | 0.000129 | -0.000815 | 0.001246 | 0.402500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | player | 540 | 268 | -0.001061 | -0.005232 | 0.003144 | 0.696500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | player | 540 | 268 | -0.021754 | -0.088587 | 0.038781 | 0.776500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | played_crps | player | 540 | 268 | 0.000448 | -0.006166 | 0.007274 | 0.476000 |
| all | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | player | 540 | 268 | -0.000607 | -0.001938 | 0.000696 | 0.826500 |
| all | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | player | 540 | 268 | 0.000129 | -0.000706 | 0.000995 | 0.393000 |
| qb | height_weight_w025_all | production | all_2017_2025 | ppg_crps | season | 216 | 9 | -0.002476 | -0.008755 | 0.005177 | 0.744000 |
| qb | height_weight_w025_all | production | all_2017_2025 | contribution_crps | season | 216 | 9 | 0.002940 | -0.093349 | 0.096132 | 0.469500 |
| qb | height_weight_w025_all | production | all_2017_2025 | played_crps | season | 216 | 9 | 0.002027 | -0.008890 | 0.014443 | 0.383500 |
| qb | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | season | 216 | 9 | -0.001161 | -0.002677 | 0.000270 | 0.940000 |
| qb | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | season | 216 | 9 | -0.000389 | -0.001484 | 0.000721 | 0.752500 |
| qb | height_weight_w025_all | production | all_2017_2025 | ppg_crps | player | 216 | 58 | -0.002476 | -0.007015 | 0.002110 | 0.866000 |
| qb | height_weight_w025_all | production | all_2017_2025 | contribution_crps | player | 216 | 58 | 0.002940 | -0.057967 | 0.066428 | 0.462000 |
| qb | height_weight_w025_all | production | all_2017_2025 | played_crps | player | 216 | 58 | 0.002027 | -0.006873 | 0.012394 | 0.348000 |
| qb | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | player | 216 | 58 | -0.001161 | -0.002865 | 0.000662 | 0.895500 |
| qb | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | player | 216 | 58 | -0.000389 | -0.001232 | 0.000405 | 0.827500 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | season | 72 | 3 | -0.006375 | -0.016499 | 0.006857 | 0.851000 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | season | 72 | 3 | -0.085510 | -0.279680 | 0.082843 | 0.851000 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | played_crps | season | 72 | 3 | 0.021848 | -0.000410 | 0.044527 | 0.034000 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | season | 72 | 3 | -0.003102 | -0.005153 | -0.001159 | 1.000000 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | season | 72 | 3 | -0.000364 | -0.002468 | 0.002025 | 0.606000 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | player | 72 | 33 | -0.006375 | -0.018673 | 0.006335 | 0.825500 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | player | 72 | 33 | -0.085510 | -0.220058 | 0.052717 | 0.877000 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | played_crps | player | 72 | 33 | 0.021848 | 0.000997 | 0.046099 | 0.018500 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | player | 72 | 33 | -0.003102 | -0.007176 | 0.000561 | 0.944000 |
| qb | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | player | 72 | 33 | -0.000364 | -0.001880 | 0.001208 | 0.673000 |
| rb | height_weight_w025_all | production | all_2017_2025 | ppg_crps | season | 540 | 9 | 0.001559 | 0.000116 | 0.003128 | 0.013500 |
| rb | height_weight_w025_all | production | all_2017_2025 | contribution_crps | season | 540 | 9 | -0.005267 | -0.037946 | 0.021467 | 0.613000 |
| rb | height_weight_w025_all | production | all_2017_2025 | played_crps | season | 540 | 9 | -0.003710 | -0.007463 | -0.000295 | 0.983000 |
| rb | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | season | 540 | 9 | -0.000655 | -0.001633 | 0.000445 | 0.881000 |
| rb | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | season | 540 | 9 | 0.000301 | -0.000146 | 0.000775 | 0.105500 |
| rb | height_weight_w025_all | production | all_2017_2025 | ppg_crps | player | 540 | 189 | 0.001559 | -0.001840 | 0.004873 | 0.177500 |
| rb | height_weight_w025_all | production | all_2017_2025 | contribution_crps | player | 540 | 189 | -0.005267 | -0.047374 | 0.035910 | 0.608000 |
| rb | height_weight_w025_all | production | all_2017_2025 | played_crps | player | 540 | 189 | -0.003710 | -0.010275 | 0.003163 | 0.854500 |
| rb | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | player | 540 | 189 | -0.000655 | -0.001675 | 0.000433 | 0.891000 |
| rb | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | player | 540 | 189 | 0.000301 | -0.000445 | 0.001055 | 0.220000 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | season | 180 | 3 | 0.000884 | -0.002235 | 0.004551 | 0.261500 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | season | 180 | 3 | -0.025903 | -0.080426 | 0.007530 | 0.855500 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | played_crps | season | 180 | 3 | -0.004978 | -0.011038 | 0.000246 | 0.969500 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | season | 180 | 3 | -0.001541 | -0.002600 | -0.000261 | 1.000000 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | season | 180 | 3 | -0.000327 | -0.000756 | 0.000023 | 0.963500 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | player | 180 | 90 | 0.000884 | -0.006285 | 0.008151 | 0.396000 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | player | 180 | 90 | -0.025903 | -0.117011 | 0.061819 | 0.712000 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | played_crps | player | 180 | 90 | -0.004978 | -0.014649 | 0.004976 | 0.839500 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | player | 180 | 90 | -0.001541 | -0.003378 | 0.000241 | 0.950000 |
| rb | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | player | 180 | 90 | -0.000327 | -0.001425 | 0.000635 | 0.711000 |
| wr | height_weight_w025_all | production | all_2017_2025 | ppg_crps | season | 648 | 9 | 0.001768 | -0.002021 | 0.005547 | 0.193500 |
| wr | height_weight_w025_all | production | all_2017_2025 | contribution_crps | season | 648 | 9 | 0.032274 | -0.039951 | 0.102785 | 0.179000 |
| wr | height_weight_w025_all | production | all_2017_2025 | played_crps | season | 648 | 9 | -0.002990 | -0.011254 | 0.006783 | 0.742000 |
| wr | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | season | 648 | 9 | 0.001607 | 0.000538 | 0.002625 | 0.002000 |
| wr | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | season | 648 | 9 | 0.000328 | -0.000383 | 0.001045 | 0.213500 |
| wr | height_weight_w025_all | production | all_2017_2025 | ppg_crps | player | 648 | 201 | 0.001768 | -0.001686 | 0.005159 | 0.173000 |
| wr | height_weight_w025_all | production | all_2017_2025 | contribution_crps | player | 648 | 201 | 0.032274 | -0.023204 | 0.091709 | 0.130500 |
| wr | height_weight_w025_all | production | all_2017_2025 | played_crps | player | 648 | 201 | -0.002990 | -0.008805 | 0.002775 | 0.845500 |
| wr | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | player | 648 | 201 | 0.001607 | 0.000183 | 0.002889 | 0.016500 |
| wr | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | player | 648 | 201 | 0.000328 | -0.000549 | 0.001154 | 0.224500 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | season | 216 | 3 | -0.002270 | -0.003476 | -0.000585 | 1.000000 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | season | 216 | 3 | -0.013673 | -0.168336 | 0.086693 | 0.588000 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | played_crps | season | 216 | 3 | -0.001080 | -0.025735 | 0.028565 | 0.620500 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | season | 216 | 3 | -0.000062 | -0.001512 | 0.001152 | 0.588000 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | season | 216 | 3 | 0.000558 | -0.001212 | 0.002345 | 0.379500 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | player | 216 | 108 | -0.002270 | -0.009204 | 0.005070 | 0.725500 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | player | 216 | 108 | -0.013673 | -0.133652 | 0.103737 | 0.590000 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | played_crps | player | 216 | 108 | -0.001080 | -0.012060 | 0.010758 | 0.566000 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | player | 216 | 108 | -0.000062 | -0.002521 | 0.002319 | 0.518000 |
| wr | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | player | 216 | 108 | 0.000558 | -0.001239 | 0.002341 | 0.269000 |
| te | height_weight_w025_all | production | all_2017_2025 | ppg_crps | season | 216 | 9 | -0.001699 | -0.005909 | 0.002856 | 0.768500 |
| te | height_weight_w025_all | production | all_2017_2025 | contribution_crps | season | 216 | 9 | -0.033243 | -0.078561 | 0.028225 | 0.874500 |
| te | height_weight_w025_all | production | all_2017_2025 | played_crps | season | 216 | 9 | -0.000690 | -0.007223 | 0.005511 | 0.570500 |
| te | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | season | 216 | 9 | 0.000850 | -0.000511 | 0.002194 | 0.112000 |
| te | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | season | 216 | 9 | 0.000286 | -0.000532 | 0.001096 | 0.269500 |
| te | height_weight_w025_all | production | all_2017_2025 | ppg_crps | player | 216 | 71 | -0.001699 | -0.006369 | 0.002863 | 0.766000 |
| te | height_weight_w025_all | production | all_2017_2025 | contribution_crps | player | 216 | 71 | -0.033243 | -0.104824 | 0.045084 | 0.796500 |
| te | height_weight_w025_all | production | all_2017_2025 | played_crps | player | 216 | 71 | -0.000690 | -0.008075 | 0.006938 | 0.572000 |
| te | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | player | 216 | 71 | 0.000850 | -0.000324 | 0.002074 | 0.082500 |
| te | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | player | 216 | 71 | 0.000286 | -0.000362 | 0.001041 | 0.191500 |
| te | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | season | 72 | 3 | 0.003015 | -0.010199 | 0.009958 | 0.275000 |
| te | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | season | 72 | 3 | 0.028133 | -0.096128 | 0.169511 | 0.383000 |
| te | height_weight_w025_all | production | temporal_2023_2025 | played_crps | season | 72 | 3 | -0.002799 | -0.014743 | 0.004690 | 0.692500 |
| te | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | season | 72 | 3 | 0.002592 | 0.001327 | 0.004894 | 0.000000 |
| te | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | season | 72 | 3 | 0.000475 | -0.001362 | 0.002179 | 0.383000 |
| te | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | player | 72 | 37 | 0.003015 | -0.006237 | 0.011809 | 0.263000 |
| te | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | player | 72 | 37 | 0.028133 | -0.107101 | 0.165898 | 0.355500 |
| te | height_weight_w025_all | production | temporal_2023_2025 | played_crps | player | 72 | 37 | -0.002799 | -0.012658 | 0.006988 | 0.696500 |
| te | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | player | 72 | 37 | 0.002592 | 0.000256 | 0.005180 | 0.012500 |
| te | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | player | 72 | 37 | 0.000475 | -0.001040 | 0.002109 | 0.292500 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | ppg_crps | season | 864 | 9 | 0.000901 | -0.001801 | 0.003684 | 0.256500 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | contribution_crps | season | 864 | 9 | 0.015895 | -0.042369 | 0.073733 | 0.274500 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | played_crps | season | 864 | 9 | -0.002415 | -0.008643 | 0.004389 | 0.775500 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | season | 864 | 9 | 0.001418 | 0.000613 | 0.002231 | 0.000500 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | season | 864 | 9 | 0.000318 | -0.000348 | 0.000959 | 0.171500 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | ppg_crps | player | 864 | 272 | 0.000901 | -0.002116 | 0.003846 | 0.268000 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | contribution_crps | player | 864 | 272 | 0.015895 | -0.032230 | 0.062413 | 0.256000 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | played_crps | player | 864 | 272 | -0.002415 | -0.006825 | 0.002134 | 0.845500 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | plus3_brier_row | player | 864 | 272 | 0.001418 | 0.000350 | 0.002449 | 0.005000 |
| wr_te | height_weight_w025_all | production | all_2017_2025 | impact_brier_row | player | 864 | 272 | 0.000318 | -0.000345 | 0.000950 | 0.170500 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | season | 288 | 3 | -0.000949 | -0.002988 | 0.000258 | 0.727000 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | season | 288 | 3 | -0.003221 | -0.150284 | 0.072846 | 0.697500 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | played_crps | season | 288 | 3 | -0.001510 | -0.018129 | 0.017738 | 0.622500 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | season | 288 | 3 | 0.000601 | -0.000745 | 0.001354 | 0.261500 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | season | 288 | 3 | 0.000537 | -0.000757 | 0.002303 | 0.262000 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | ppg_crps | player | 288 | 145 | -0.000949 | -0.006751 | 0.004984 | 0.624500 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | contribution_crps | player | 288 | 145 | -0.003221 | -0.097535 | 0.094825 | 0.519500 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | played_crps | player | 288 | 145 | -0.001510 | -0.010446 | 0.007473 | 0.622000 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | plus3_brier_row | player | 288 | 145 | 0.000601 | -0.001302 | 0.002489 | 0.259000 |
| wr_te | height_weight_w025_all | production | temporal_2023_2025 | impact_brier_row | player | 288 | 145 | 0.000537 | -0.000866 | 0.001935 | 0.229500 |

Lower CRPS and Brier scores are better. `candidate_minus_baseline < 0` favors
the height/weight matcher.

## Pool-composition audit

| scope | n | mean_pool_overlap_share | median_pool_overlap_share | mean_height_distance_delta | mean_weight_distance_delta | mean_effective_sample_size_delta |
| --- | --- | --- | --- | --- | --- | --- |
| all | 1620 | 0.912022 | 0.925000 | -0.047030 | -0.048139 | 0.716048 |
| qb | 216 | 0.951736 | 0.950000 | -0.027367 | -0.031642 | 1.314445 |
| rb | 540 | 0.929838 | 0.937500 | -0.035568 | -0.033010 | 0.969512 |
| wr | 648 | 0.876215 | 0.887500 | -0.067466 | -0.070972 | 0.251075 |
| te | 216 | 0.935185 | 0.937500 | -0.034039 | -0.033960 | 0.878912 |
| wr_te | 864 | 0.890958 | 0.900000 | -0.059109 | -0.061719 | 0.408034 |

Negative size-distance deltas mean the primary candidate selected donors closer
to the target on that measurement. Pool overlap is the share of baseline top-80
donors retained.

Runtime: 31.6 seconds.
