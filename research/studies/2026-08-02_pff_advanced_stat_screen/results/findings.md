# PFF advanced-stat screen findings

## Read this first

This is a strictly prior-season predictive screen, not a causal-effect estimate. The baseline is the locked production PPG forecast, which already contains expert projection, ADP, experience, historical production, projected role, room context, and projection trajectory. Negative deltas favor the PFF rate challenger over a prior-PFF-opportunity-only control.

## Screening decision

Advance `TE rec_mtf_per_reception` to a full locked-model and template-matcher test, but do not promote it from this screen. It is the only DK row whose season-bootstrap intervals clear zero for both point PPG and q90 Brier.

Its DK RMSE delta versus the opportunity control is -0.0196, including -0.0269 in 2023-2025; its q90 Brier delta is -0.0013, including -0.0027 recently. Beta agrees directionally (PPG -0.0133; q90 Brier -0.0016). The result is not a one-season artifact: it improved PPG and q90 Brier in six of eight test seasons in each scoring system, including all three recent seasons for both outcomes.

The signal is moderately persistent year to year (Spearman 0.330 across 832 qualifying consecutive-season pairs) and unusually nonredundant: its largest absolute correlation with the selected existing baseline features is only 0.094. The next-best related TE candidate is YAC/route, but its DK PPG and q90 intervals cross zero.

Because this was a broad screen, the intervals do not correct for selecting the best of many candidates. Treat TE missed tackles per reception as a prespecified follow-up candidate, not confirmatory evidence or a production change.

DK rows with a season-bootstrap PPG interval entirely below zero: **1**.
DK rows with a season-bootstrap q90 Brier interval entirely below zero: **1**.
Rows directionally improving both PPG and q90 Brier in both DK and beta: **5**.

## Best DK PPG screens

| position | candidate | candidate_delta_vs_opportunity_control | temporal_2023_2025_delta_vs_control | season_bootstrap_low | season_bootstrap_high | year_to_year_spearman | max_existing_spearman |
|---|---|---:|---:|---:|---:|---:|---:|
| QB | rush_first_downs_per_attempt | -0.0265 | 0.0033 | -0.0822 | 0.0347 | 0.2289 | 0.2883 |
| TE | rec_mtf_per_reception | -0.0196 | -0.0269 | -0.0381 | -0.0015 | 0.3300 | 0.0943 |
| TE | rec_yac_per_route | -0.0118 | -0.0169 | -0.0251 | 0.0039 | 0.3646 | 0.4612 |
| TE | rec_yards_per_target | -0.0073 | 0.0067 | -0.0187 | 0.0051 | 0.2275 | 0.3116 |
| TE | rec_yprr | -0.0051 | 0.0027 | -0.0124 | 0.0019 | 0.4858 | 0.7135 |
| TE | rec_route_grade | -0.0028 | 0.0029 | -0.0089 | 0.0019 | 0.4579 | 0.7119 |
| RB | rec_yprr | -0.0026 | -0.0072 | -0.0106 | 0.0087 | 0.4023 | 0.4616 |
| RB | rec_first_downs_per_route | -0.0025 | -0.0051 | -0.0118 | 0.0072 | 0.3896 | 0.4702 |
| WR | rec_route_rate | -0.0025 | 0.0028 | -0.0107 | 0.0052 | 0.1881 | 0.0831 |
| RB | rec_yac_per_route | -0.0024 | -0.0013 | -0.0128 | 0.0104 | 0.3867 | 0.4465 |
| RB | rec_yards_per_target | -0.0014 | -0.0040 | -0.0141 | 0.0092 | 0.1211 | -0.2424 |
| RB | rec_route_grade | -0.0004 | 0.0026 | -0.0195 | 0.0226 | 0.3274 | -0.3723 |

## Best DK q90-upside screens

| position | candidate | candidate_brier_delta_vs_opportunity_control | temporal_2023_2025_brier_delta_vs_control | candidate_ap_delta_vs_control | candidate_brier_bootstrap_low | candidate_brier_bootstrap_high |
|---|---|---:|---:|---:|---:|---:|
| TE | rec_yac_per_route | -0.0019 | -0.0043 | 0.0158 | -0.0043 | 0.0008 |
| WR | rec_first_downs_per_route | -0.0013 | -0.0001 | 0.0072 | -0.0032 | 0.0007 |
| QB | rush_mtf_per_attempt | -0.0013 | -0.0001 | 0.0133 | -0.0051 | 0.0039 |
| TE | rec_mtf_per_reception | -0.0013 | -0.0027 | 0.0073 | -0.0024 | -0.0002 |
| TE | rec_wide_rate | -0.0010 | -0.0000 | 0.0141 | -0.0026 | 0.0005 |
| WR | rec_route_grade | -0.0005 | 0.0006 | 0.0048 | -0.0022 | 0.0015 |
| RB | rec_targeted_qb_rating | -0.0003 | 0.0016 | 0.0012 | -0.0015 | 0.0014 |
| TE | rec_yards_per_target | -0.0003 | -0.0018 | 0.0050 | -0.0017 | 0.0011 |
| WR | rec_targets_per_route | -0.0003 | -0.0007 | -0.0044 | -0.0016 | 0.0010 |
| WR | rec_yprr | -0.0003 | 0.0011 | -0.0021 | -0.0024 | 0.0018 |
| TE | rec_yprr | -0.0002 | -0.0009 | 0.0037 | -0.0016 | 0.0012 |
| RB | rush_ypa | -0.0001 | -0.0002 | -0.0089 | -0.0005 | 0.0004 |

## Cross-scoring directional replication

| position | candidate | candidate_delta_vs_opportunity_control_dk | candidate_delta_vs_opportunity_control_beta | candidate_brier_delta_vs_opportunity_control_dk | candidate_brier_delta_vs_opportunity_control_beta |
|---|---|---:|---:|---:|---:|
| TE | rec_mtf_per_reception | -0.0196 | -0.0133 | -0.0013 | -0.0016 |
| TE | rec_yac_per_route | -0.0118 | -0.0099 | -0.0019 | -0.0026 |
| TE | rec_yards_per_target | -0.0073 | -0.0086 | -0.0003 | -0.0007 |
| TE | rec_yprr | -0.0051 | -0.0054 | -0.0002 | -0.0009 |
| RB | rush_ypa | -0.0003 | -0.0014 | -0.0001 | -0.0002 |

## Coverage

| domain | position | eligible_rows | identity_coverage | positive_opportunity_coverage | median_positive_opportunity |
|---|---|---:|---:|---:|---:|
| receiving | RB | 1011 | 1.0000 | 0.7656 | 138.0000 |
| receiving | WR | 1503 | 1.0000 | 0.8004 | 347.0000 |
| receiving | TE | 802 | 1.0000 | 0.8217 | 207.0000 |
| rushing | RB | 1011 | 1.0000 | 0.7676 | 86.5000 |
| rushing | QB | 380 | 1.0000 | 0.8474 | 33.5000 |

## High within-PFF redundancy

There are 30 position-specific candidate pairs with |Spearman| >= 0.75. See `pff_pair_correlations.csv` for the full matrix. Highly correlated rates should enter later model/template tests as alternative representatives, not as a bundle.

## Promotion rule

This screen does not change production. A candidate should advance only if it has reasonable coverage and persistence, adds directionally consistent PPG or upside value beyond the opportunity control, survives the 2023-2025 slice, and is not merely a duplicate of an existing projection/history feature. Advanced candidates then need the full locked model or template validation rather than direct promotion.
