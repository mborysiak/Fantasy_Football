# Projection-Weight Weekly-Template Replay (dk)

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
| all_2017_2025 | component_rank_w150 | 2.34418 | 26.75854 | 1.51286 | -1.65409 | 0.13842 |
| all_2017_2025 | ppg225_component_rank150 | 2.34544 | 26.76738 | 1.51625 | -1.65532 | 0.13073 |
| all_2017_2025 | ppg225_component_rank150_raw100 | 2.34741 | 26.78605 | 1.52165 | -1.65597 | 0.11396 |
| all_2017_2025 | ppg300_component_rank200_raw150 | 2.34983 | 26.78928 | 1.52305 | -1.65659 | 0.10334 |
| all_2017_2025 | ppg_w225 | 2.34655 | 26.78267 | 1.51882 | -1.65555 | 0.12912 |
| all_2017_2025 | ppg_w300 | 2.34828 | 26.78937 | 1.52143 | -1.65646 | 0.12202 |
| all_2017_2025 | production | 2.34329 | 26.75515 | 1.51372 | -1.65419 | 0.13657 |
| all_2017_2025 | raw_component_w100 | 2.34747 | 26.77735 | 1.51838 | -1.65531 | 0.11689 |
| temporal_2023_2025 | component_rank_w150 | 2.34481 | 26.20711 | 1.52168 | -1.40582 | 0.03687 |
| temporal_2023_2025 | ppg225_component_rank150 | 2.34920 | 26.23394 | 1.52871 | -1.40738 | 0.02700 |
| temporal_2023_2025 | ppg225_component_rank150_raw100 | 2.35163 | 26.26379 | 1.53707 | -1.40817 | 0.00789 |
| temporal_2023_2025 | ppg300_component_rank200_raw150 | 2.35356 | 26.26401 | 1.53754 | -1.40941 | -0.00024 |
| temporal_2023_2025 | ppg_w225 | 2.35219 | 26.26825 | 1.53422 | -1.40764 | 0.02260 |
| temporal_2023_2025 | ppg_w300 | 2.35417 | 26.29070 | 1.53767 | -1.40873 | 0.00872 |
| temporal_2023_2025 | production | 2.34574 | 26.23136 | 1.52544 | -1.40589 | 0.03347 |
| temporal_2023_2025 | raw_component_w100 | 2.35192 | 26.26104 | 1.53298 | -1.40742 | 0.00889 |

## Position results

| method | pos | ppg_crps | contribution_crps | played_crps |
| --- | --- | --- | --- | --- |
| component_rank_w150 | QB | 2.39349 | 23.74246 | 1.32626 |
| component_rank_w150 | RB | 2.30946 | 26.86411 | 1.64196 |
| component_rank_w150 | TE | 2.04220 | 25.23425 | 1.54253 |
| component_rank_w150 | WR | 2.45735 | 28.18402 | 1.45759 |
| ppg225_component_rank150 | QB | 2.40238 | 23.82930 | 1.34010 |
| ppg225_component_rank150 | RB | 2.30717 | 26.86766 | 1.64208 |
| ppg225_component_rank150 | TE | 2.04361 | 25.19801 | 1.54470 |
| ppg225_component_rank150 | WR | 2.45898 | 28.18631 | 1.46061 |
| ppg225_component_rank150_raw100 | QB | 2.41238 | 23.91293 | 1.36238 |
| ppg225_component_rank150_raw100 | RB | 2.30804 | 26.89045 | 1.64514 |
| ppg225_component_rank150_raw100 | TE | 2.04444 | 25.17726 | 1.54647 |
| ppg225_component_rank150_raw100 | WR | 2.45954 | 28.19302 | 1.46355 |
| ppg300_component_rank200_raw150 | QB | 2.42126 | 23.95390 | 1.37768 |
| ppg300_component_rank200_raw150 | RB | 2.30868 | 26.90605 | 1.64635 |
| ppg300_component_rank200_raw150 | TE | 2.04591 | 25.14614 | 1.54718 |
| ppg300_component_rank200_raw150 | WR | 2.46162 | 28.18480 | 1.46072 |
| ppg_w225 | QB | 2.40960 | 23.92301 | 1.34964 |
| ppg_w225 | RB | 2.30736 | 26.86367 | 1.64676 |
| ppg_w225 | TE | 2.04260 | 25.17324 | 1.54243 |
| ppg_w225 | WR | 2.45952 | 28.20487 | 1.46072 |
| ppg_w300 | QB | 2.41928 | 24.00258 | 1.36776 |
| ppg_w300 | RB | 2.30571 | 26.85276 | 1.64344 |
| ppg_w300 | TE | 2.04520 | 25.17746 | 1.54662 |
| ppg_w300 | WR | 2.46112 | 28.20279 | 1.46259 |
| production | QB | 2.39493 | 23.77835 | 1.32929 |
| production | RB | 2.30757 | 26.84980 | 1.64371 |
| production | TE | 2.04219 | 25.20760 | 1.54307 |
| production | WR | 2.45621 | 28.18438 | 1.45710 |
| raw_component_w100 | QB | 2.41333 | 23.88600 | 1.34584 |
| raw_component_w100 | RB | 2.30775 | 26.85409 | 1.64546 |
| raw_component_w100 | TE | 2.04267 | 25.17001 | 1.54281 |
| raw_component_w100 | WR | 2.46022 | 28.21296 | 1.46185 |

## Paired candidate-minus-production results

| candidate_method | period | metric | candidate_minus_baseline | bootstrap_p025 | bootstrap_p975 | probability_candidate_better |
| --- | --- | --- | --- | --- | --- | --- |
| ppg_w225 | all_2017_2025 | ppg_crps | 0.00326 | 0.00066 | 0.00667 | 0.00500 |
| ppg_w225 | all_2017_2025 | contribution_crps | 0.02753 | 0.00995 | 0.04640 | 0.00100 |
| ppg_w225 | all_2017_2025 | played_crps | 0.00509 | 0.00099 | 0.00900 | 0.00700 |
| ppg_w225 | temporal_2023_2025 | ppg_crps | 0.00646 | -0.00087 | 0.01420 | 0.03500 |
| ppg_w225 | temporal_2023_2025 | contribution_crps | 0.03690 | -0.00873 | 0.08077 | 0.03050 |
| ppg_w225 | temporal_2023_2025 | played_crps | 0.00878 | -0.00111 | 0.01595 | 0.03500 |
| ppg_w300 | all_2017_2025 | ppg_crps | 0.00499 | 0.00212 | 0.00892 | 0.00000 |
| ppg_w300 | all_2017_2025 | contribution_crps | 0.03423 | 0.00151 | 0.06598 | 0.01900 |
| ppg_w300 | all_2017_2025 | played_crps | 0.00771 | 0.00159 | 0.01445 | 0.00650 |
| ppg_w300 | temporal_2023_2025 | ppg_crps | 0.00844 | 0.00038 | 0.01818 | 0.00000 |
| ppg_w300 | temporal_2023_2025 | contribution_crps | 0.05934 | 0.01087 | 0.08765 | 0.00000 |
| ppg_w300 | temporal_2023_2025 | played_crps | 0.01223 | -0.00649 | 0.02780 | 0.03500 |
| component_rank_w150 | all_2017_2025 | ppg_crps | 0.00089 | -0.00077 | 0.00268 | 0.14500 |
| component_rank_w150 | all_2017_2025 | contribution_crps | 0.00339 | -0.01515 | 0.02098 | 0.34250 |
| component_rank_w150 | all_2017_2025 | played_crps | -0.00086 | -0.00328 | 0.00160 | 0.76200 |
| component_rank_w150 | temporal_2023_2025 | ppg_crps | -0.00093 | -0.00145 | -0.00032 | 1.00000 |
| component_rank_w150 | temporal_2023_2025 | contribution_crps | -0.02425 | -0.04710 | -0.00593 | 1.00000 |
| component_rank_w150 | temporal_2023_2025 | played_crps | -0.00375 | -0.00619 | -0.00237 | 1.00000 |
| raw_component_w100 | all_2017_2025 | ppg_crps | 0.00418 | 0.00099 | 0.00792 | 0.00300 |
| raw_component_w100 | all_2017_2025 | contribution_crps | 0.02220 | 0.00568 | 0.04080 | 0.00150 |
| raw_component_w100 | all_2017_2025 | played_crps | 0.00466 | 0.00072 | 0.00854 | 0.01050 |
| raw_component_w100 | temporal_2023_2025 | ppg_crps | 0.00618 | -0.00027 | 0.01560 | 0.03500 |
| raw_component_w100 | temporal_2023_2025 | contribution_crps | 0.02969 | 0.00018 | 0.06915 | 0.00000 |
| raw_component_w100 | temporal_2023_2025 | played_crps | 0.00754 | 0.00189 | 0.01595 | 0.00000 |
| ppg225_component_rank150 | all_2017_2025 | ppg_crps | 0.00215 | 0.00024 | 0.00421 | 0.01300 |
| ppg225_component_rank150 | all_2017_2025 | contribution_crps | 0.01224 | 0.00140 | 0.02490 | 0.01150 |
| ppg225_component_rank150 | all_2017_2025 | played_crps | 0.00252 | 0.00009 | 0.00473 | 0.02100 |
| ppg225_component_rank150 | temporal_2023_2025 | ppg_crps | 0.00346 | -0.00176 | 0.00811 | 0.03500 |
| ppg225_component_rank150 | temporal_2023_2025 | contribution_crps | 0.00258 | -0.00477 | 0.01326 | 0.29000 |
| ppg225_component_rank150 | temporal_2023_2025 | played_crps | 0.00328 | -0.00129 | 0.00693 | 0.03500 |
| ppg225_component_rank150_raw100 | all_2017_2025 | ppg_crps | 0.00412 | 0.00054 | 0.00818 | 0.00800 |
| ppg225_component_rank150_raw100 | all_2017_2025 | contribution_crps | 0.03090 | 0.00859 | 0.05760 | 0.00300 |
| ppg225_component_rank150_raw100 | all_2017_2025 | played_crps | 0.00792 | 0.00247 | 0.01420 | 0.00200 |
| ppg225_component_rank150_raw100 | temporal_2023_2025 | ppg_crps | 0.00589 | -0.00277 | 0.01595 | 0.14800 |
| ppg225_component_rank150_raw100 | temporal_2023_2025 | contribution_crps | 0.03244 | 0.00064 | 0.05248 | 0.00000 |
| ppg225_component_rank150_raw100 | temporal_2023_2025 | played_crps | 0.01163 | -0.00171 | 0.02831 | 0.03500 |
| ppg300_component_rank200_raw150 | all_2017_2025 | ppg_crps | 0.00654 | 0.00194 | 0.01146 | 0.00200 |
| ppg300_component_rank200_raw150 | all_2017_2025 | contribution_crps | 0.03413 | 0.00830 | 0.06255 | 0.00400 |
| ppg300_component_rank200_raw150 | all_2017_2025 | played_crps | 0.00933 | 0.00336 | 0.01624 | 0.00000 |
| ppg300_component_rank200_raw150 | temporal_2023_2025 | ppg_crps | 0.00783 | -0.00523 | 0.02180 | 0.14800 |
| ppg300_component_rank200_raw150 | temporal_2023_2025 | contribution_crps | 0.03266 | -0.01221 | 0.07267 | 0.03500 |
| ppg300_component_rank200_raw150 | temporal_2023_2025 | played_crps | 0.01211 | -0.00153 | 0.03148 | 0.03500 |

Lower CRPS is better. Raw component magnitudes are scoring-aligned component
PPG estimates on the same `/10` scale as absolute projected PPG.

Runtime: 59.4 seconds.
