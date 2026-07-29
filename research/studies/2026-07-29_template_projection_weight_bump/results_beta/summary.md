# Projection-Weight Weekly-Template Replay (beta)

## Scope

- Strict rolling target seasons: 2017-2025.
- Held-out player-seasons: 1,620.
- Every weekly donor precedes its target season.
- All variants retain the production donor eligibility, pool size, kernel,
  12-season recency prior, and 5% donor cap.
- Production configuration and databases are unchanged.

## Period results

| period | method | ppg_crps | contribution_crps | played_crps | ppg_bias | played_bias |
| --- | --- | --- | --- | --- | --- | --- |
| all_2017_2025 | component_rank_w150 | 2.34394 | 26.76926 | 1.51417 | -1.65086 | 0.13486 |
| all_2017_2025 | ppg225_component_rank150 | 2.34544 | 26.79341 | 1.51762 | -1.65209 | 0.12849 |
| all_2017_2025 | ppg225_component_rank150_raw100 | 2.34779 | 26.81223 | 1.52063 | -1.65274 | 0.11677 |
| all_2017_2025 | ppg300_component_rank200_raw150 | 2.34922 | 26.81021 | 1.52221 | -1.65335 | 0.11028 |
| all_2017_2025 | ppg_w225 | 2.34543 | 26.79370 | 1.51868 | -1.65250 | 0.12888 |
| all_2017_2025 | ppg_w300 | 2.34722 | 26.79744 | 1.52104 | -1.65306 | 0.12789 |
| all_2017_2025 | production | 2.34383 | 26.77231 | 1.51582 | -1.65107 | 0.13445 |
| all_2017_2025 | raw_component_w100 | 2.34622 | 26.79109 | 1.51974 | -1.65227 | 0.11726 |
| temporal_2023_2025 | component_rank_w150 | 2.33094 | 26.28222 | 1.52040 | -1.43346 | 0.01211 |
| temporal_2023_2025 | ppg225_component_rank150 | 2.33523 | 26.33562 | 1.52662 | -1.43467 | 0.00740 |
| temporal_2023_2025 | ppg225_component_rank150_raw100 | 2.33645 | 26.32909 | 1.52890 | -1.43527 | -0.00798 |
| temporal_2023_2025 | ppg300_component_rank200_raw150 | 2.33938 | 26.35564 | 1.53253 | -1.43643 | -0.01301 |
| temporal_2023_2025 | ppg_w225 | 2.33525 | 26.32698 | 1.52903 | -1.43494 | 0.00403 |
| temporal_2023_2025 | ppg_w300 | 2.33796 | 26.36542 | 1.53330 | -1.43570 | -0.00030 |
| temporal_2023_2025 | production | 2.33220 | 26.30861 | 1.52646 | -1.43356 | 0.01145 |
| temporal_2023_2025 | raw_component_w100 | 2.33450 | 26.31646 | 1.52912 | -1.43483 | -0.01205 |

## Position results

| method | pos | ppg_crps | contribution_crps | played_crps |
| --- | --- | --- | --- | --- |
| component_rank_w150 | QB | 2.39058 | 23.76563 | 1.33447 |
| component_rank_w150 | RB | 2.31029 | 26.84575 | 1.64469 |
| component_rank_w150 | TE | 2.05329 | 25.24165 | 1.53982 |
| component_rank_w150 | WR | 2.45331 | 28.21592 | 1.45676 |
| ppg225_component_rank150 | QB | 2.40491 | 23.94935 | 1.34798 |
| ppg225_component_rank150 | RB | 2.30953 | 26.85104 | 1.64587 |
| ppg225_component_rank150 | TE | 2.05447 | 25.19314 | 1.53987 |
| ppg225_component_rank150 | WR | 2.45254 | 28.22684 | 1.45988 |
| ppg225_component_rank150_raw100 | QB | 2.41758 | 24.07674 | 1.35863 |
| ppg225_component_rank150_raw100 | RB | 2.31023 | 26.86115 | 1.64800 |
| ppg225_component_rank150_raw100 | TE | 2.05633 | 25.16404 | 1.54181 |
| ppg225_component_rank150_raw100 | WR | 2.45299 | 28.23269 | 1.46142 |
| ppg300_component_rank200_raw150 | QB | 2.42714 | 24.11456 | 1.36758 |
| ppg300_component_rank200_raw150 | RB | 2.31026 | 26.86291 | 1.64713 |
| ppg300_component_rank200_raw150 | TE | 2.05840 | 25.14445 | 1.54172 |
| ppg300_component_rank200_raw150 | WR | 2.45266 | 28.22009 | 1.46314 |
| ppg_w225 | QB | 2.40999 | 24.02342 | 1.35585 |
| ppg_w225 | RB | 2.30925 | 26.83740 | 1.64965 |
| ppg_w225 | TE | 2.05265 | 25.16990 | 1.54073 |
| ppg_w225 | WR | 2.45164 | 28.22198 | 1.45648 |
| ppg_w300 | QB | 2.42191 | 24.13117 | 1.36455 |
| ppg_w300 | RB | 2.30801 | 26.81732 | 1.64840 |
| ppg_w300 | TE | 2.05607 | 25.12686 | 1.53967 |
| ppg_w300 | WR | 2.45204 | 28.22649 | 1.46087 |
| production | QB | 2.39388 | 23.82463 | 1.34049 |
| production | RB | 2.30967 | 26.83809 | 1.64518 |
| production | TE | 2.05288 | 25.21987 | 1.54262 |
| production | WR | 2.45259 | 28.21754 | 1.45752 |
| raw_component_w100 | QB | 2.41681 | 24.02309 | 1.35160 |
| raw_component_w100 | RB | 2.30864 | 26.83632 | 1.65087 |
| raw_component_w100 | TE | 2.05327 | 25.16246 | 1.54259 |
| raw_component_w100 | WR | 2.45165 | 28.21893 | 1.45888 |

## Paired candidate-minus-production results

| candidate_method | period | metric | candidate_minus_baseline | bootstrap_p025 | bootstrap_p975 | probability_candidate_better |
| --- | --- | --- | --- | --- | --- | --- |
| ppg_w225 | all_2017_2025 | ppg_crps | 0.00160 | -0.00031 | 0.00358 | 0.05300 |
| ppg_w225 | all_2017_2025 | contribution_crps | 0.02139 | 0.00750 | 0.03620 | 0.00100 |
| ppg_w225 | all_2017_2025 | played_crps | 0.00287 | 0.00114 | 0.00451 | 0.00050 |
| ppg_w225 | temporal_2023_2025 | ppg_crps | 0.00306 | -0.00178 | 0.00678 | 0.03500 |
| ppg_w225 | temporal_2023_2025 | contribution_crps | 0.01837 | 0.00782 | 0.02377 | 0.00000 |
| ppg_w225 | temporal_2023_2025 | played_crps | 0.00257 | -0.00131 | 0.00495 | 0.03500 |
| ppg_w300 | all_2017_2025 | ppg_crps | 0.00339 | 0.00041 | 0.00656 | 0.01050 |
| ppg_w300 | all_2017_2025 | contribution_crps | 0.02513 | -0.00951 | 0.06021 | 0.07750 |
| ppg_w300 | all_2017_2025 | played_crps | 0.00523 | 0.00127 | 0.00920 | 0.00500 |
| ppg_w300 | temporal_2023_2025 | ppg_crps | 0.00576 | -0.00174 | 0.01130 | 0.03500 |
| ppg_w300 | temporal_2023_2025 | contribution_crps | 0.05682 | 0.03907 | 0.06803 | 0.00000 |
| ppg_w300 | temporal_2023_2025 | played_crps | 0.00684 | -0.00447 | 0.01472 | 0.03500 |
| component_rank_w150 | all_2017_2025 | ppg_crps | 0.00011 | -0.00077 | 0.00091 | 0.36200 |
| component_rank_w150 | all_2017_2025 | contribution_crps | -0.00306 | -0.02105 | 0.01159 | 0.61600 |
| component_rank_w150 | all_2017_2025 | played_crps | -0.00164 | -0.00442 | 0.00084 | 0.89200 |
| component_rank_w150 | temporal_2023_2025 | ppg_crps | -0.00126 | -0.00301 | -0.00003 | 1.00000 |
| component_rank_w150 | temporal_2023_2025 | contribution_crps | -0.02639 | -0.05940 | 0.01419 | 0.96950 |
| component_rank_w150 | temporal_2023_2025 | played_crps | -0.00605 | -0.00848 | -0.00182 | 1.00000 |
| raw_component_w100 | all_2017_2025 | ppg_crps | 0.00239 | 0.00036 | 0.00472 | 0.00650 |
| raw_component_w100 | all_2017_2025 | contribution_crps | 0.01877 | 0.00053 | 0.03697 | 0.02100 |
| raw_component_w100 | all_2017_2025 | played_crps | 0.00392 | 0.00123 | 0.00618 | 0.00350 |
| raw_component_w100 | temporal_2023_2025 | ppg_crps | 0.00230 | -0.00182 | 0.00850 | 0.25950 |
| raw_component_w100 | temporal_2023_2025 | contribution_crps | 0.00786 | -0.01624 | 0.02001 | 0.26250 |
| raw_component_w100 | temporal_2023_2025 | played_crps | 0.00266 | -0.00478 | 0.00829 | 0.24500 |
| ppg225_component_rank150 | all_2017_2025 | ppg_crps | 0.00161 | -0.00039 | 0.00375 | 0.06100 |
| ppg225_component_rank150 | all_2017_2025 | contribution_crps | 0.02110 | -0.00289 | 0.04432 | 0.04300 |
| ppg225_component_rank150 | all_2017_2025 | played_crps | 0.00181 | 0.00038 | 0.00348 | 0.00300 |
| ppg225_component_rank150 | temporal_2023_2025 | ppg_crps | 0.00303 | -0.00161 | 0.00777 | 0.14800 |
| ppg225_component_rank150 | temporal_2023_2025 | contribution_crps | 0.02702 | 0.00556 | 0.04452 | 0.00000 |
| ppg225_component_rank150 | temporal_2023_2025 | played_crps | 0.00017 | -0.00110 | 0.00094 | 0.26250 |
| ppg225_component_rank150_raw100 | all_2017_2025 | ppg_crps | 0.00396 | 0.00099 | 0.00743 | 0.00300 |
| ppg225_component_rank150_raw100 | all_2017_2025 | contribution_crps | 0.03992 | 0.01237 | 0.07453 | 0.00200 |
| ppg225_component_rank150_raw100 | all_2017_2025 | played_crps | 0.00481 | 0.00243 | 0.00714 | 0.00000 |
| ppg225_component_rank150_raw100 | temporal_2023_2025 | ppg_crps | 0.00425 | -0.00159 | 0.01046 | 0.03500 |
| ppg225_component_rank150_raw100 | temporal_2023_2025 | contribution_crps | 0.02049 | -0.00957 | 0.06297 | 0.14950 |
| ppg225_component_rank150_raw100 | temporal_2023_2025 | played_crps | 0.00244 | -0.00182 | 0.00809 | 0.14800 |
| ppg300_component_rank200_raw150 | all_2017_2025 | ppg_crps | 0.00540 | 0.00134 | 0.01002 | 0.00400 |
| ppg300_component_rank200_raw150 | all_2017_2025 | contribution_crps | 0.03790 | 0.00565 | 0.07525 | 0.01050 |
| ppg300_component_rank200_raw150 | all_2017_2025 | played_crps | 0.00639 | 0.00258 | 0.01067 | 0.00000 |
| ppg300_component_rank200_raw150 | temporal_2023_2025 | ppg_crps | 0.00718 | -0.00462 | 0.01874 | 0.14800 |
| ppg300_component_rank200_raw150 | temporal_2023_2025 | contribution_crps | 0.04703 | -0.02059 | 0.09289 | 0.03500 |
| ppg300_component_rank200_raw150 | temporal_2023_2025 | played_crps | 0.00607 | -0.00170 | 0.01675 | 0.14800 |

Lower CRPS is better. Raw component magnitudes are scoring-aligned component
PPG estimates on the same `/10` scale as absolute projected PPG.

Runtime: 59.5 seconds.
