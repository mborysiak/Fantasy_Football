# Deterministic Full-Column LightGBM Attribution

Negative deltas mean the addition improved RMSE. Full row/column sampling and deterministic LightGBM settings prevent an unavailable feature from changing which incumbent columns are sampled.

## family

| Variant | Added | RMSE | Pooled delta | Mean season delta | 95% season bootstrap | Wins | Ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| `family_opportunity_share` | 4 | 3.1414 | -0.0038 | -0.0038 | [-0.0135, +0.0048] | 4/9 | 0/9 |
| `family_market_room` | 5 | 3.1417 | -0.0035 | -0.0037 | [-0.0106, +0.0045] | 7/9 | 0/9 |
| `family_experience_context` | 3 | 3.1467 | +0.0015 | +0.0012 | [-0.0110, +0.0130] | 5/9 | 0/9 |
| `family_all_legacy` | 12 | 3.1491 | +0.0039 | +0.0035 | [-0.0154, +0.0236] | 5/9 | 0/9 |

## feature

| Variant | Added | RMSE | Pooled delta | Mean season delta | 95% season bootstrap | Wins | Ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| `feature_adp_mean_teammate_gap` | 1 | 3.1413 | -0.0039 | -0.0039 | [-0.0110, +0.0029] | 6/9 | 0/9 |
| `feature_team_receiving_yard_share` | 1 | 3.1421 | -0.0031 | -0.0031 | [-0.0127, +0.0069] | 5/9 | 0/9 |
| `feature_adp_teammates_better_count` | 1 | 3.1428 | -0.0024 | -0.0024 | [-0.0070, +0.0025] | 6/9 | 0/9 |
| `feature_adp_best_teammate_gap` | 1 | 3.1431 | -0.0021 | -0.0023 | [-0.0082, +0.0038] | 6/9 | 0/9 |
| `feature_adp_worst_teammate_gap` | 1 | 3.1435 | -0.0017 | -0.0019 | [-0.0093, +0.0062] | 6/9 | 0/9 |
| `feature_team_rush_attempt_share` | 1 | 3.1442 | -0.0010 | -0.0012 | [-0.0121, +0.0084] | 3/9 | 0/9 |
| `feature_expert_ppg_exp_diff` | 1 | 3.1450 | -0.0002 | -0.0003 | [-0.0098, +0.0090] | 5/9 | 0/9 |
| `feature_expert_ppg_exp_percentile` | 1 | 3.1450 | -0.0002 | -0.0002 | [-0.0095, +0.0113] | 6/9 | 0/9 |
| `feature_team_target_share` | 1 | 3.1452 | +0.0000 | +0.0000 | [+0.0000, +0.0001] | 0/9 | 8/9 |
| `feature_adp_room_strength_share` | 1 | 3.1470 | +0.0018 | +0.0016 | [-0.0049, +0.0087] | 4/9 | 0/9 |
| `feature_expert_ppg_exp_peer_mean` | 1 | 3.1516 | +0.0064 | +0.0063 | [-0.0005, +0.0142] | 2/9 | 0/9 |
| `feature_team_reception_share` | 1 | 3.1529 | +0.0077 | +0.0079 | [-0.0023, +0.0162] | 2/9 | 0/9 |
