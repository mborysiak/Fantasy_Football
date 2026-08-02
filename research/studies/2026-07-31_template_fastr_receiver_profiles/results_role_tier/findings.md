# Role-tier findings

| method | mean_development_ppg_relative_delta | worst_temporal_ppg_relative_delta | all_leagues_screen_pass | all_leagues_one_se_near_best | tier_sensitivity_guardrail | phase_b_finalist |
| --- | --- | --- | --- | --- | --- | --- |
| usage_depth_w100 | 0.000095 | 0.000415 | False | False | True | False |
| usage_air_value_w100 | 0.000130 | 0.000128 | False | True | True | False |
| usage_air_value_disp_w100 | 0.000205 | -0.000205 | False | True | True | False |
| usage_air_value_disp_w150 | 0.000264 | 0.000325 | False | True | True | False |

## Directly affected position slices

| league | scope | period | method | n | ppg_crps_delta | ppg_crps_relative_delta |
| --- | --- | --- | --- | --- | --- | --- |
| dk | wr_core_main | development_2017_2022 | usage_air_value_disp_w100 | 288 | 0.003200 | 0.001208 |
| dk | wr_core_main | development_2017_2022 | usage_air_value_disp_w150 | 288 | 0.003332 | 0.001257 |
| dk | wr_core_main | development_2017_2022 | usage_air_value_w100 | 288 | 0.003335 | 0.001258 |
| dk | wr_core_main | development_2017_2022 | usage_depth_w100 | 288 | 0.001427 | 0.000538 |
| dk | wr_core_main | temporal_2023_2025 | usage_air_value_disp_w100 | 144 | -0.005237 | -0.002065 |
| dk | wr_core_main | temporal_2023_2025 | usage_air_value_disp_w150 | 144 | -0.006926 | -0.002732 |
| dk | wr_core_main | temporal_2023_2025 | usage_air_value_w100 | 144 | -0.007386 | -0.002913 |
| dk | wr_core_main | temporal_2023_2025 | usage_depth_w100 | 144 | -0.007992 | -0.003152 |
| dk | te_core_main | development_2017_2022 | usage_air_value_disp_w100 | 108 | -0.001279 | -0.000581 |
| dk | te_core_main | development_2017_2022 | usage_air_value_disp_w150 | 108 | -0.000730 | -0.000332 |
| dk | te_core_main | development_2017_2022 | usage_air_value_w100 | 108 | 0.000149 | 0.000068 |
| dk | te_core_main | development_2017_2022 | usage_depth_w100 | 108 | -0.002757 | -0.001253 |
| dk | te_core_main | temporal_2023_2025 | usage_air_value_disp_w100 | 54 | -0.008343 | -0.003805 |
| dk | te_core_main | temporal_2023_2025 | usage_air_value_disp_w150 | 54 | -0.011032 | -0.005032 |
| dk | te_core_main | temporal_2023_2025 | usage_air_value_w100 | 54 | -0.009154 | -0.004175 |
| dk | te_core_main | temporal_2023_2025 | usage_depth_w100 | 54 | -0.004946 | -0.002256 |
| beta | wr_core_main | development_2017_2022 | usage_air_value_disp_w100 | 288 | 0.000301 | 0.000172 |
| beta | wr_core_main | development_2017_2022 | usage_air_value_disp_w150 | 288 | 0.000483 | 0.000275 |
| beta | wr_core_main | development_2017_2022 | usage_air_value_w100 | 288 | -0.001112 | -0.000635 |
| beta | wr_core_main | development_2017_2022 | usage_depth_w100 | 288 | 0.001191 | 0.000680 |
| beta | wr_core_main | temporal_2023_2025 | usage_air_value_disp_w100 | 144 | 0.000046 | 0.000026 |
| beta | wr_core_main | temporal_2023_2025 | usage_air_value_disp_w150 | 144 | 0.002326 | 0.001325 |
| beta | wr_core_main | temporal_2023_2025 | usage_air_value_w100 | 144 | 0.001493 | 0.000851 |
| beta | wr_core_main | temporal_2023_2025 | usage_depth_w100 | 144 | 0.002201 | 0.001254 |
| beta | te_core_main | development_2017_2022 | usage_air_value_disp_w100 | 108 | -0.001119 | -0.000736 |
| beta | te_core_main | development_2017_2022 | usage_air_value_disp_w150 | 108 | -0.000783 | -0.000515 |
| beta | te_core_main | development_2017_2022 | usage_air_value_w100 | 108 | -0.000666 | -0.000438 |
| beta | te_core_main | development_2017_2022 | usage_depth_w100 | 108 | -0.001544 | -0.001016 |
| beta | te_core_main | temporal_2023_2025 | usage_air_value_disp_w100 | 54 | -0.002878 | -0.001754 |
| beta | te_core_main | temporal_2023_2025 | usage_air_value_disp_w150 | 54 | -0.001834 | -0.001117 |
| beta | te_core_main | temporal_2023_2025 | usage_air_value_w100 | 54 | -0.002264 | -0.001379 |
| beta | te_core_main | temporal_2023_2025 | usage_depth_w100 | 54 | -0.000296 | -0.000180 |
