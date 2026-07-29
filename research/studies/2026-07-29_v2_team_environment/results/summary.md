# Team Environment and QB Style Results

Negative deltas favor the challenger.

## Pooled OOF

| Method | RMSE |
|---|---:|
| `qb_style_equal_thirds` | 3.0928 |
| `all_environment_equal_thirds` | 3.0946 |
| `trajectory_base_equal_thirds` | 3.0949 |
| `qb_yardage_equal_thirds` | 3.0949 |
| `team_rush_scoring_equal_thirds` | 3.0950 |
| `team_support_equal_thirds` | 3.0969 |
| `qb_tds_equal_thirds` | 3.0978 |
| `qb_style_tree_average` | 3.1085 |
| `trajectory_base_tree_average` | 3.1112 |
| `team_rush_scoring_tree_average` | 3.1117 |
| `all_environment_tree_average` | 3.1120 |
| `qb_yardage_tree_average` | 3.1128 |
| `qb_tds_tree_average` | 3.1129 |
| `team_support_tree_average` | 3.1158 |
| `qb_style_lightgbm` | 3.1164 |
| `team_rush_scoring_random_forest` | 3.1187 |
| `trajectory_base_random_forest` | 3.1203 |
| `qb_style_random_forest` | 3.1205 |
| `all_environment_random_forest` | 3.1210 |
| `trajectory_base_lightgbm` | 3.1210 |
| `qb_yardage_lightgbm` | 3.1210 |
| `qb_tds_random_forest` | 3.1217 |
| `qb_yardage_random_forest` | 3.1226 |
| `qb_tds_lightgbm` | 3.1228 |
| `all_environment_lightgbm` | 3.1228 |
| `team_support_lightgbm` | 3.1232 |
| `team_rush_scoring_lightgbm` | 3.1235 |
| `team_support_random_forest` | 3.1261 |
| `all_environment_lasso` | 3.1463 |
| `qb_style_lasso` | 3.1470 |
| `team_support_lasso` | 3.1471 |
| `team_rush_scoring_lasso` | 3.1486 |
| `qb_yardage_lasso` | 3.1497 |
| `trajectory_base_lasso` | 3.1523 |
| `qb_tds_lasso` | 3.1527 |

## Paired season comparisons

| Challenger | Reference | Delta | Recent | 95% interval | Wins | Sign-flip p |
|---|---|---:|---:|---:|---:|---:|
| `team_support_equal_thirds` | `team_support_tree_average` | -0.0188 | -0.0166 | [-0.0286, -0.0102] | 8/9 | 0.0078 |
| `qb_yardage_equal_thirds` | `qb_yardage_tree_average` | -0.0178 | -0.0149 | [-0.0278, -0.0091] | 8/9 | 0.0078 |
| `all_environment_equal_thirds` | `all_environment_tree_average` | -0.0173 | -0.0142 | [-0.0276, -0.0090] | 9/9 | 0.0039 |
| `team_rush_scoring_equal_thirds` | `team_rush_scoring_tree_average` | -0.0167 | -0.0146 | [-0.0266, -0.0081] | 8/9 | 0.0156 |
| `qb_style_equal_thirds` | `qb_style_tree_average` | -0.0158 | -0.0137 | [-0.0256, -0.0071] | 8/9 | 0.0117 |
| `qb_tds_equal_thirds` | `qb_tds_tree_average` | -0.0151 | -0.0128 | [-0.0243, -0.0069] | 8/9 | 0.0078 |
| `all_environment_lasso` | `trajectory_base_lasso` | -0.0060 | -0.0116 | [-0.0179, +0.0041] | 6/9 | 0.3164 |
| `qb_style_lasso` | `trajectory_base_lasso` | -0.0053 | -0.0107 | [-0.0103, -0.0009] | 8/9 | 0.0586 |
| `team_support_lasso` | `trajectory_base_lasso` | -0.0052 | -0.0040 | [-0.0131, +0.0010] | 5/9 | 0.2109 |
| `qb_style_lightgbm` | `trajectory_base_lightgbm` | -0.0045 | -0.0107 | [-0.0102, +0.0010] | 6/9 | 0.1836 |
| `team_rush_scoring_lasso` | `trajectory_base_lasso` | -0.0036 | -0.0130 | [-0.0111, +0.0029] | 5/9 | 0.3320 |
| `qb_style_tree_average` | `trajectory_base_tree_average` | -0.0027 | -0.0092 | [-0.0086, +0.0032] | 5/9 | 0.3828 |
| `qb_yardage_lasso` | `trajectory_base_lasso` | -0.0026 | -0.0096 | [-0.0092, +0.0045] | 5/9 | 0.5117 |
| `qb_style_equal_thirds` | `trajectory_base_equal_thirds` | -0.0021 | -0.0089 | [-0.0076, +0.0030] | 5/9 | 0.4258 |
| `team_rush_scoring_random_forest` | `trajectory_base_random_forest` | -0.0016 | -0.0068 | [-0.0088, +0.0049] | 5/9 | 0.6641 |
| `all_environment_equal_thirds` | `trajectory_base_equal_thirds` | -0.0003 | -0.0074 | [-0.0084, +0.0069] | 5/9 | 0.8984 |
| `qb_yardage_equal_thirds` | `trajectory_base_equal_thirds` | +0.0000 | -0.0093 | [-0.0064, +0.0058] | 4/9 | 0.9492 |
| `qb_yardage_lightgbm` | `trajectory_base_lightgbm` | +0.0001 | -0.0042 | [-0.0039, +0.0035] | 5/9 | 0.9453 |
| `team_rush_scoring_equal_thirds` | `trajectory_base_equal_thirds` | +0.0001 | -0.0085 | [-0.0074, +0.0064] | 5/9 | 0.9961 |
| `qb_style_random_forest` | `trajectory_base_random_forest` | +0.0002 | -0.0059 | [-0.0109, +0.0111] | 5/9 | 0.9844 |
| `qb_tds_lasso` | `trajectory_base_lasso` | +0.0005 | -0.0080 | [-0.0060, +0.0056] | 3/9 | 0.9062 |
| `team_rush_scoring_tree_average` | `trajectory_base_tree_average` | +0.0005 | -0.0080 | [-0.0072, +0.0071] | 4/9 | 0.9180 |
| `all_environment_random_forest` | `trajectory_base_random_forest` | +0.0007 | -0.0022 | [-0.0107, +0.0097] | 3/9 | 0.9453 |
| `all_environment_tree_average` | `trajectory_base_tree_average` | +0.0008 | -0.0073 | [-0.0063, +0.0072] | 4/9 | 0.8672 |
| `qb_tds_random_forest` | `trajectory_base_random_forest` | +0.0014 | -0.0071 | [-0.0077, +0.0100] | 5/9 | 0.8320 |
| `qb_yardage_tree_average` | `trajectory_base_tree_average` | +0.0015 | -0.0084 | [-0.0053, +0.0084] | 5/9 | 0.7227 |
| `qb_tds_tree_average` | `trajectory_base_tree_average` | +0.0017 | -0.0064 | [-0.0034, +0.0068] | 4/9 | 0.5898 |
| `qb_tds_lightgbm` | `trajectory_base_lightgbm` | +0.0018 | -0.0054 | [-0.0025, +0.0059] | 4/9 | 0.4297 |
| `all_environment_lightgbm` | `trajectory_base_lightgbm` | +0.0018 | -0.0111 | [-0.0063, +0.0090] | 5/9 | 0.7070 |
| `team_support_equal_thirds` | `trajectory_base_equal_thirds` | +0.0020 | +0.0030 | [-0.0026, +0.0061] | 4/9 | 0.3945 |
| `team_support_lightgbm` | `trajectory_base_lightgbm` | +0.0022 | +0.0013 | [-0.0019, +0.0061] | 2/9 | 0.3164 |
| `qb_yardage_random_forest` | `trajectory_base_random_forest` | +0.0023 | -0.0128 | [-0.0090, +0.0134] | 5/9 | 0.7461 |
| `team_rush_scoring_lightgbm` | `trajectory_base_lightgbm` | +0.0025 | -0.0085 | [-0.0069, +0.0103] | 3/9 | 0.6133 |
| `qb_tds_equal_thirds` | `trajectory_base_equal_thirds` | +0.0028 | -0.0051 | [-0.0017, +0.0073] | 4/9 | 0.2812 |
| `team_support_tree_average` | `trajectory_base_tree_average` | +0.0046 | +0.0056 | [-0.0003, +0.0094] | 2/9 | 0.1289 |
| `team_support_random_forest` | `trajectory_base_random_forest` | +0.0058 | +0.0100 | [-0.0059, +0.0159] | 3/9 | 0.3672 |
