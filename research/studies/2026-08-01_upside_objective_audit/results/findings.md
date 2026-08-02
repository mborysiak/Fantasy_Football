# Findings

## Decision

Keep the production matcher and both app objectives unchanged. Adopt the new rare-upside and championship metrics as secondary validation objectives for future tests.

## Player level

`wr_ppg225_both025` is the first challenger exposed by the new objective. On the primary q90 core event it improves Brier score, log loss, continuous tail-utility CRPS, and contribution CRPS in all four league-by-period cells. The absolute gains are small and their season-bootstrap intervals generally cross zero. DK rare-event probabilities are also materially underpredicted, so raw absolute tail probabilities should not drive app decisions yet.

## Roster level

The player-level tail signal does not transport to 12-team championship probability. `wr_ppg225_both025` worsens championship Brier/log loss in DK development and recent periods and beta development, improving only recent beta. The prior flatter-distance arm also fails joint replication. All season-bootstrap intervals are wide. This rejects both matcher promotions.

## Recommended downstream objective

Use a constrained, lexicographic tilt rather than a weighted or distorted forecast: retain ordinary calibrated scenario draws; require expected-score non-inferiority; among candidates within 0.25% of the best expected roster score, prefer the highest paired championship-probability lower bound. Auction should compare Buy versus Pass on the same scenario rooms; Snake should compare forced current-pick candidates on the same future-draft and weekly scenario banks.

## Bootstrap evidence

Player q90 candidate rows:

| league | period | scope | severity | method | metric | delta | season_bootstrap_low | season_bootstrap_high | seasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dk | development_2017_2022 | core | q90 | wr_ppg225_both025 | league_winner_brier | -0.000116 | -0.000573 | 0.000292 | 6.000000 |
| dk | development_2017_2022 | core | q90 | wr_ppg225_both025 | average_precision | 0.000887 | -0.004227 | 0.005414 | 6.000000 |
| dk | development_2017_2022 | core | q90 | wr_ppg225_both025 | tail_utility_crps | -0.007422 | -0.023045 | 0.007886 | 6.000000 |
| dk | development_2017_2022 | core | q90 | wr_ppg225_both025 | ppg_crps | -0.000624 | -0.002513 | 0.001241 | 6.000000 |
| dk | development_2017_2022 | core | q90 | wr_ppg225_both025 | contribution_crps | -0.022687 | -0.054792 | 0.000180 | 6.000000 |
| dk | temporal_2023_2025 | core | q90 | wr_ppg225_both025 | league_winner_brier | -0.000282 | -0.000370 | -0.000197 | 3.000000 |
| dk | temporal_2023_2025 | core | q90 | wr_ppg225_both025 | average_precision | 0.007903 | 0.000108 | 0.014007 | 3.000000 |
| dk | temporal_2023_2025 | core | q90 | wr_ppg225_both025 | tail_utility_crps | -0.012170 | -0.034814 | 0.012860 | 3.000000 |
| dk | temporal_2023_2025 | core | q90 | wr_ppg225_both025 | ppg_crps | 0.002191 | 0.001390 | 0.003325 | 3.000000 |
| dk | temporal_2023_2025 | core | q90 | wr_ppg225_both025 | contribution_crps | -0.006221 | -0.016468 | 0.012476 | 3.000000 |
| beta | development_2017_2022 | core | q90 | wr_ppg225_both025 | league_winner_brier | -0.000019 | -0.000184 | 0.000146 | 6.000000 |
| beta | development_2017_2022 | core | q90 | wr_ppg225_both025 | average_precision | -0.000170 | -0.002565 | 0.002970 | 6.000000 |
| beta | development_2017_2022 | core | q90 | wr_ppg225_both025 | tail_utility_crps | -0.002898 | -0.012187 | 0.006897 | 6.000000 |
| beta | development_2017_2022 | core | q90 | wr_ppg225_both025 | ppg_crps | -0.000917 | -0.001651 | -0.000076 | 6.000000 |
| beta | development_2017_2022 | core | q90 | wr_ppg225_both025 | contribution_crps | -0.018444 | -0.035757 | -0.001446 | 6.000000 |
| beta | temporal_2023_2025 | core | q90 | wr_ppg225_both025 | league_winner_brier | -0.000115 | -0.000240 | 0.000004 | 3.000000 |
| beta | temporal_2023_2025 | core | q90 | wr_ppg225_both025 | average_precision | 0.001562 | 0.000249 | 0.004414 | 3.000000 |
| beta | temporal_2023_2025 | core | q90 | wr_ppg225_both025 | tail_utility_crps | -0.009737 | -0.012344 | -0.005313 | 3.000000 |
| beta | temporal_2023_2025 | core | q90 | wr_ppg225_both025 | ppg_crps | -0.000436 | -0.001893 | 0.000614 | 3.000000 |
| beta | temporal_2023_2025 | core | q90 | wr_ppg225_both025 | contribution_crps | -0.007664 | -0.012707 | 0.000978 | 3.000000 |

Roster candidate rows:

| league | period | method | metric | delta | season_bootstrap_low | season_bootstrap_high | seasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dk | development_2017_2022 | wr_ppg225_both025 | score_crps | -0.493762 | -1.859995 | 0.740248 | 6.000000 |
| dk | development_2017_2022 | wr_ppg225_both025 | championship_brier | 0.000297 | -0.000355 | 0.001033 | 6.000000 |
| dk | development_2017_2022 | wr_ppg225_both025 | championship_log_loss | 0.009290 | -0.023655 | 0.048708 | 6.000000 |
| dk | temporal_2023_2025 | wr_ppg225_both025 | score_crps | -0.361833 | -0.888508 | 0.511046 | 3.000000 |
| dk | temporal_2023_2025 | wr_ppg225_both025 | championship_brier | 0.000831 | -0.000488 | 0.002176 | 3.000000 |
| dk | temporal_2023_2025 | wr_ppg225_both025 | championship_log_loss | 0.049093 | -0.039166 | 0.133826 | 3.000000 |
| beta | development_2017_2022 | wr_ppg225_both025 | score_crps | 0.215742 | -0.325413 | 0.780214 | 6.000000 |
| beta | development_2017_2022 | wr_ppg225_both025 | championship_brier | 0.000160 | -0.000632 | 0.000972 | 6.000000 |
| beta | development_2017_2022 | wr_ppg225_both025 | championship_log_loss | 0.020303 | -0.028104 | 0.065355 | 6.000000 |
| beta | temporal_2023_2025 | wr_ppg225_both025 | score_crps | 0.320418 | 0.053543 | 0.482312 | 3.000000 |
| beta | temporal_2023_2025 | wr_ppg225_both025 | championship_brier | -0.000137 | -0.001200 | 0.001526 | 3.000000 |
| beta | temporal_2023_2025 | wr_ppg225_both025 | championship_log_loss | -0.019423 | -0.094184 | 0.088432 | 3.000000 |
