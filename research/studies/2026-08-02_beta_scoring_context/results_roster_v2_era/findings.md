# Roster validation findings

| candidate | period | baseline_score_crps | candidate_score_crps | score_crps_relative_delta | gate_within_0_5_percent | roster_level_pass |
| --- | --- | --- | --- | --- | --- | --- |
| beta_context_only | development_2017_2022 | 91.06822635084505 | 90.43303513530654 | -0.006974893889899702 | True | True |
| beta_context_only | temporal_2023_2025 | 89.8236151465 | 90.06139029701788 | 0.002647134054113401 | True | True |
| beta_scored_full | development_2017_2022 | 91.06822635084505 | 91.89241086638154 | 0.009050187409616101 | False | False |
| beta_scored_full | temporal_2023_2025 | 89.8236151465 | 90.16405084376785 | 0.003790046712243858 | True | False |
| beta_scored_v2_era | development_2017_2022 | 91.06822635084505 | 93.67804498198431 | 0.02865783968477437 | False | False |
| beta_scored_v2_era | temporal_2023_2025 | 89.8236151465 | 90.28290107653481 | 0.005113198007959555 | False | False |

`beta_scored_full` fails the roster promotion gate.
