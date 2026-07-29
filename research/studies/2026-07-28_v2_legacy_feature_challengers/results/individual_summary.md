# Individual LightGBM Feature Additions

Negative deltas mean that adding the one feature improved the direct LightGBM model versus the original 31-feature manifest.

| Added feature | RMSE | Pooled delta | Mean season delta | 95% season bootstrap | Wins |
|---|---:|---:|---:|---:|---:|
| `team_target_share` | 3.1405 | -0.0038 | -0.0037 | [-0.0079, +0.0022] | 7/9 |
| `adp_room_strength_share` | 3.1423 | -0.0020 | -0.0023 | [-0.0088, +0.0043] | 5/9 |
| `adp_worst_teammate_gap` | 3.1444 | +0.0001 | -0.0001 | [-0.0079, +0.0070] | 4/9 |
| `expert_ppg_exp_diff` | 3.1452 | +0.0010 | +0.0009 | [-0.0066, +0.0079] | 4/9 |
| `adp_best_teammate_gap` | 3.1453 | +0.0010 | +0.0009 | [-0.0067, +0.0079] | 3/9 |
| `adp_mean_teammate_gap` | 3.1457 | +0.0015 | +0.0015 | [-0.0044, +0.0071] | 4/9 |
| `team_rush_attempt_share` | 3.1470 | +0.0027 | +0.0025 | [-0.0047, +0.0110] | 4/9 |
| `team_receiving_yard_share` | 3.1483 | +0.0041 | +0.0042 | [-0.0011, +0.0092] | 3/9 |
| `adp_teammates_better_count` | 3.1490 | +0.0047 | +0.0047 | [-0.0023, +0.0107] | 2/9 |
| `expert_ppg_exp_percentile` | 3.1499 | +0.0056 | +0.0054 | [-0.0027, +0.0136] | 4/9 |
| `expert_ppg_exp_peer_mean` | 3.1510 | +0.0067 | +0.0061 | [-0.0040, +0.0212] | 4/9 |
| `team_reception_share` | 3.1527 | +0.0084 | +0.0083 | [+0.0012, +0.0155] | 1/9 |
