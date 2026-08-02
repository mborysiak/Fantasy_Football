# RB role-tier findings

| method | mean_development_ppg_relative_delta | worst_temporal_ppg_relative_delta | all_leagues_screen_pass | all_leagues_one_se_near_best | tier_sensitivity_guardrail | phase_b_finalist |
| --- | --- | --- | --- | --- | --- | --- |
| rb_dual_role_w100 | -0.000007 | 0.000022 | False | True | True | False |
| rb_dual_role_w050 | 0.000003 | 0.000309 | False | True | True | False |
| rb_scoring_role_w050 | 0.000029 | 0.000716 | False | True | True | False |
| rb_passing_down_w050 | 0.000031 | -0.000106 | False | True | True | False |

## RB core and depth slices

| league | scope | period | method | n | ppg_crps_delta | ppg_crps_relative_delta | contribution_crps_delta | contribution_crps_relative_delta | played_crps_delta | played_crps_relative_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dk | rb_core_main | development_2017_2022 | rb_dual_role_w050 | 216 | 0.001224 | 0.000498 | -0.019053 | -0.000605 | -0.000185 | -0.000114 |
| dk | rb_core_main | development_2017_2022 | rb_dual_role_w100 | 216 | 0.001223 | 0.000498 | -0.011706 | -0.000372 | 0.002687 | 0.001655 |
| dk | rb_core_main | development_2017_2022 | rb_passing_down_w050 | 216 | 0.001144 | 0.000466 | 0.014042 | 0.000446 | 0.001661 | 0.001023 |
| dk | rb_core_main | development_2017_2022 | rb_scoring_role_w050 | 216 | 0.002006 | 0.000817 | 0.022237 | 0.000706 | 0.000852 | 0.000525 |
| dk | rb_core_main | temporal_2023_2025 | rb_dual_role_w050 | 108 | 0.002521 | 0.000944 | -0.039286 | -0.001137 | -0.004346 | -0.003184 |
| dk | rb_core_main | temporal_2023_2025 | rb_dual_role_w100 | 108 | 0.000077 | 0.000029 | -0.006488 | -0.000188 | -0.000408 | -0.000299 |
| dk | rb_core_main | temporal_2023_2025 | rb_passing_down_w050 | 108 | -0.000915 | -0.000343 | -0.071874 | -0.002079 | -0.006666 | -0.004884 |
| dk | rb_core_main | temporal_2023_2025 | rb_scoring_role_w050 | 108 | 0.002729 | 0.001022 | 0.028279 | 0.000818 | 0.000143 | 0.000105 |
| dk | rb_depth_main | development_2017_2022 | rb_dual_role_w050 | 314 | 0.000552 | 0.000302 | -0.004187 | -0.000290 | -0.004986 | -0.002668 |
| dk | rb_depth_main | development_2017_2022 | rb_dual_role_w100 | 314 | 0.000837 | 0.000459 | -0.009008 | -0.000624 | -0.005473 | -0.002928 |
| dk | rb_depth_main | development_2017_2022 | rb_passing_down_w050 | 314 | -0.000317 | -0.000174 | -0.001885 | -0.000131 | -0.003503 | -0.001874 |
| dk | rb_depth_main | development_2017_2022 | rb_scoring_role_w050 | 314 | 0.001995 | 0.001093 | 0.003784 | 0.000262 | -0.003925 | -0.002100 |
| dk | rb_depth_main | temporal_2023_2025 | rb_dual_role_w050 | 150 | -0.000325 | -0.000171 | 0.006350 | 0.000445 | -0.000305 | -0.000135 |
| dk | rb_depth_main | temporal_2023_2025 | rb_dual_role_w100 | 150 | -0.001006 | -0.000529 | 0.005618 | 0.000394 | 0.000869 | 0.000385 |
| dk | rb_depth_main | temporal_2023_2025 | rb_passing_down_w050 | 150 | -0.001676 | -0.000882 | -0.003388 | -0.000237 | 0.000069 | 0.000030 |
| dk | rb_depth_main | temporal_2023_2025 | rb_scoring_role_w050 | 150 | 0.001882 | 0.000990 | 0.021689 | 0.001520 | 0.003531 | 0.001562 |
| beta | rb_core_main | development_2017_2022 | rb_dual_role_w050 | 216 | -0.000902 | -0.000413 | -0.023244 | -0.000850 | 0.003563 | 0.002195 |
| beta | rb_core_main | development_2017_2022 | rb_dual_role_w100 | 216 | -0.001039 | -0.000475 | -0.013224 | -0.000484 | 0.003423 | 0.002109 |
| beta | rb_core_main | development_2017_2022 | rb_passing_down_w050 | 216 | -0.000491 | -0.000225 | -0.000917 | -0.000034 | 0.000233 | 0.000144 |
| beta | rb_core_main | development_2017_2022 | rb_scoring_role_w050 | 216 | -0.001178 | -0.000539 | -0.007649 | -0.000280 | 0.002490 | 0.001534 |
| beta | rb_core_main | temporal_2023_2025 | rb_dual_role_w050 | 108 | -0.000426 | -0.000177 | -0.018959 | -0.000638 | -0.002954 | -0.002176 |
| beta | rb_core_main | temporal_2023_2025 | rb_dual_role_w100 | 108 | 0.000144 | 0.000060 | -0.009325 | -0.000314 | -0.007800 | -0.005744 |
| beta | rb_core_main | temporal_2023_2025 | rb_passing_down_w050 | 108 | -0.000713 | -0.000296 | -0.015275 | -0.000514 | -0.009300 | -0.006849 |
| beta | rb_core_main | temporal_2023_2025 | rb_scoring_role_w050 | 108 | 0.004803 | 0.001994 | 0.041991 | 0.001412 | -0.000111 | -0.000081 |
| beta | rb_depth_main | development_2017_2022 | rb_dual_role_w050 | 314 | -0.000494 | -0.000296 | 0.000632 | 0.000050 | -0.002756 | -0.001475 |
| beta | rb_depth_main | development_2017_2022 | rb_dual_role_w100 | 314 | -0.001149 | -0.000689 | -0.014247 | -0.001135 | -0.003586 | -0.001919 |
| beta | rb_depth_main | development_2017_2022 | rb_passing_down_w050 | 314 | -0.000760 | -0.000456 | -0.005915 | -0.000471 | -0.000980 | -0.000525 |
| beta | rb_depth_main | development_2017_2022 | rb_scoring_role_w050 | 314 | 0.000648 | 0.000389 | 0.005100 | 0.000406 | -0.002894 | -0.001549 |
| beta | rb_depth_main | temporal_2023_2025 | rb_dual_role_w050 | 150 | 0.001979 | 0.001074 | 0.010849 | 0.000832 | -0.001482 | -0.000653 |
| beta | rb_depth_main | temporal_2023_2025 | rb_dual_role_w100 | 150 | 0.001161 | 0.000630 | 0.022846 | 0.001752 | -0.000111 | -0.000049 |
| beta | rb_depth_main | temporal_2023_2025 | rb_passing_down_w050 | 150 | 0.000600 | 0.000326 | -0.014225 | -0.001091 | -0.011928 | -0.005255 |
| beta | rb_depth_main | temporal_2023_2025 | rb_scoring_role_w050 | 150 | -0.001090 | -0.000592 | 0.011148 | 0.000855 | 0.002066 | 0.000910 |
