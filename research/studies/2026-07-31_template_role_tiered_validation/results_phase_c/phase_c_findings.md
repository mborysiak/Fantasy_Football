# Phase C findings

The Phase-B finalist was evaluated on 1,296 paired historical 20-player best-ball rosters per league. Each weekly lineup naturally uses roster depth to replace missed player-weeks.

**Promotion decision: DO NOT PROMOTE.**

The frozen roster-score CRPS non-inferiority margin is +0.5%. Individual played-games CRPS is not a Phase-C gate.

## Decision table

| league | period | score_crps_candidate | score_crps_baseline | score_crps_relative_delta | abs_missed_week_bias_degradation_per_player | zero_player_weeks_crps_relative_delta | zero_active_players_crps_relative_delta | roster_score_guardrail | missed_week_bias_guardrail | phase_c_joint_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dk | development_2017_2022 | 253.576438 | 251.789852 | 0.007096 | -0.005345 | 0.000907 | -0.001302 | False | True | False |
| dk | temporal_2023_2025 | 217.902586 | 216.668381 | 0.005696 | 0.016561 | -0.002852 | -0.017368 | False | True | False |
| beta | development_2017_2022 | 97.465410 | 97.439245 | 0.000269 | -0.014486 | -0.011135 | 0.014755 | True | True | False |
| beta | temporal_2023_2025 | 87.949130 | 87.846424 | 0.001169 | -0.003640 | -0.011202 | -0.000163 | True | True | False |

## Season-cluster roster-score intervals

| league | period | candidate | baseline | metric | n | season_clusters | candidate_minus_baseline | bootstrap_p025 | bootstrap_p975 | probability_candidate_better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dk | development_2017_2022 | flatter_w025_all | production | score_crps | 864 | 6 | 1.786586 | 0.000113 | 3.606752 | 0.023200 |
| dk | temporal_2023_2025 | flatter_w025_all | production | score_crps | 432 | 3 | 1.234205 | 0.200709 | 2.205494 | 0.000000 |
| beta | development_2017_2022 | flatter_w025_all | production | score_crps | 864 | 6 | 0.026165 | -0.545371 | 0.562355 | 0.463400 |
| beta | temporal_2023_2025 | flatter_w025_all | production | score_crps | 432 | 3 | 0.102706 | -0.527608 | 0.858835 | 0.400600 |
