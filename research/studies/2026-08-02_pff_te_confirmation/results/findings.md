# PFF TE confirmation findings

## Decision

The two tracks diverge. Prior-season PFF TE efficiency/tackle-breaking adds a small, repeatable signal to the point-projection model, but it does not justify changing weekly-template matching.

Advance `te_pff_mtf` as the primary **projection-only** implementation candidate, routed to TE predictions so tree-fit spillover cannot change QB/RB/WR outputs. `te_pff_yac` is a separately passing projection sensitivity, not a bundle recommendation. Production remains unchanged because the broad screen and confirmation reuse the same historical origins; this is strong retrospective evidence, not a genuinely new-origin confirmation.

Reject both PFF features for template matching. The primary 0.25 tackle-breaking arm worsens development TE PPG and q90 Brier in both leagues. The YAC/route sensitivity clears the mechanical player-level screen but fails roster transport: all three DK roster metrics worsen in development and 2023-2025, while beta is mixed.

## Projection: TE-routed avoided tackles/reception

| league | period | rmse_delta_vs_production | rmse_delta_vs_opportunity_control | q90_brier_delta_vs_production | q90_brier_delta_vs_opportunity_control | season_wins_vs_control | bootstrap_low | bootstrap_high |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dk | development_2017_2022 | -0.005756 | -0.005561 | -0.000182 | -0.000120 | 8 | -0.009136 | -0.003099 |
| dk | temporal_2023_2025 | -0.008752 | -0.007804 | -0.000815 | -0.000931 | 8 | -0.009136 | -0.003099 |
| beta | development_2017_2022 | -0.003503 | -0.003728 | -0.000097 | -0.000066 | 6 | -0.006164 | -0.000232 |
| beta | temporal_2023_2025 | -0.003969 | -0.002537 | -0.000777 | -0.000789 | 6 | -0.006164 | -0.000232 |

Negative deltas favor the challenger. Bootstrap intervals compare the TE rate challenger with the prior-PFF-opportunity control across all nine seasons.

## Template mechanical screen

| league | method | te_development_ppg_delta | te_recent_ppg_delta | te_development_q90_brier_delta | te_recent_q90_brier_delta | advance_to_roster |
|---|---|---:|---:|---:|---:|---:|
| dk | te_pff_mtf_w025 | 0.000576 | -0.003199 | 0.000530 | -0.000208 | False |
| beta | te_pff_mtf_w025 | 0.000249 | -0.000661 | 0.000159 | -0.000405 | False |
| dk | te_pff_yac_w025 | -0.000982 | -0.003794 | -0.000330 | -0.000838 | True |
| beta | te_pff_yac_w025 | -0.001111 | -0.001095 | -0.000161 | -0.000312 | True |

## Template roster transport for the YAC finalist

| league | period | score_crps_delta | championship_brier_delta | championship_log_loss_delta |
|---|---|---:|---:|---:|
| dk | development_2017_2022 | 0.100974 | 0.000415 | 0.025797 |
| dk | temporal_2023_2025 | 0.241732 | 0.000769 | 0.041188 |
| beta | development_2017_2022 | -0.031568 | -0.000381 | -0.016030 |
| beta | temporal_2023_2025 | -0.251308 | 0.000157 | 0.008205 |

## Governance note

No database, production feature manifest, projection lock, template weight, or app objective was changed by this study.
