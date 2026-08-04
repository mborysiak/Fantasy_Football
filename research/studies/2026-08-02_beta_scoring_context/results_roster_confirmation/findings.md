# Roster validation findings

| candidate | period | baseline_score_crps | candidate_score_crps | score_crps_relative_delta | gate_within_0_5_percent | roster_level_pass |
| --- | --- | --- | --- | --- | --- | --- |
| beta_context_only | development_2017_2022 | 93.32034034744872 | 92.86135935219878 | -0.004918338205165934 | True | True |
| beta_context_only | temporal_2023_2025 | 89.8236151465 | 90.06139029701788 | 0.002647134054113401 | True | True |
| beta_scored_full | development_2017_2022 | 93.32034034744872 | 94.16594233981644 | 0.009061282773073779 | False | False |
| beta_scored_full | temporal_2023_2025 | 89.8236151465 | 90.16405084376785 | 0.003790046712243858 | True | False |
| beta_scored_ppg_rank_w050 | development_2017_2022 | 93.32034034744872 | 94.1094204553518 | 0.008455606837321696 | False | False |
| beta_scored_ppg_rank_w050 | temporal_2023_2025 | 89.8236151465 | 90.14030409004317 | 0.0035256757705271423 | True | False |

No fully beta-scored representation passes the roster gate.
