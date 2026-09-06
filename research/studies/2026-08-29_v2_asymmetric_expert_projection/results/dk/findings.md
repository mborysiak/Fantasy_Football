# Asymmetric expert projection study - DK

## Conditional-PPG mean

| Surface | Challenger | Delta RMSE | Recent | Wins | Season 95% | Player 95% | Pos nonworse |
|---|---|---:|---:|---:|---:|---:|---:|
| controlled_equal_thirds | `max_minus_median_raw` | -0.00548 | +0.00069 | 5/9 | [-0.01378, +0.00208] | [-0.01346, +0.00264] | 2/4 |
| controlled_equal_thirds | `max_minus_median_fraction` | +0.00075 | +0.00286 | 1/9 | [-0.00316, +0.00328] | [-0.00318, +0.00469] | 1/4 |
| controlled_equal_thirds | `asymmetric_robust_stack` | +0.00049 | +0.00297 | 4/9 | [-0.00378, +0.00372] | [-0.00378, +0.00483] | 1/4 |
| equal_thirds | `max_minus_median_raw` | -0.00526 | -0.00090 | 6/9 | [-0.01336, +0.00256] | [-0.01273, +0.00239] | 2/4 |
| equal_thirds | `max_minus_median_fraction` | +0.00160 | +0.00401 | 3/9 | [-0.00231, +0.00463] | [-0.00236, +0.00563] | 1/4 |
| equal_thirds | `asymmetric_robust_stack` | +0.00086 | +0.00260 | 4/9 | [-0.00384, +0.00482] | [-0.00362, +0.00536] | 1/4 |

## Upper residual events

| Event | Challenger | Delta Brier | Recent | Delta log loss | AUC | Wins | Season 95% |
|---|---|---:|---:|---:|---:|---:|---:|
| plus3 | `tail_bullish` | +0.002176 | +0.001775 | +0.007068 | 0.6166 -> 0.6132 | 3/8 | [+0.000240, +0.004424] |
| plus3 | `tail_asymmetric` | +0.002561 | +0.002005 | +0.009250 | 0.6166 -> 0.6120 | 2/8 | [+0.000343, +0.005244] |
| plus5 | `tail_bullish` | +0.000329 | +0.000142 | +0.002510 | 0.6001 -> 0.5851 | 2/8 | [+0.000000, +0.000682] |
| plus5 | `tail_asymmetric` | +0.000261 | +0.000114 | +0.001440 | 0.6001 -> 0.5918 | 3/8 | [-0.000106, +0.000705] |

## Weekly-template replay

| Period | Method | PPG CRPS | Contribution CRPS | +5 Brier | +5 AUC | Impact Brier | Impact AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| full_2017_2025 | `asymmetric_add_w025` | 2.34173 | 26.81597 | 0.154233 | 0.5478 | 0.136488 | 0.7361 |
| full_2017_2025 | `bull_add_w050` | 2.34100 | 26.80615 | 0.154124 | 0.5520 | 0.136488 | 0.7365 |
| full_2017_2025 | `bull_replace_symmetric_w075` | 2.34184 | 26.81524 | 0.154058 | 0.5544 | 0.136508 | 0.7367 |
| full_2017_2025 | `incumbent` | 2.34375 | 26.75495 | 0.153660 | 0.5511 | 0.134939 | 0.7467 |
| recent_2020_2025 | `asymmetric_add_w025` | 2.30655 | 26.19961 | 0.144230 | 0.5503 | 0.135506 | 0.7340 |
| recent_2020_2025 | `bull_add_w050` | 2.30468 | 26.18243 | 0.143934 | 0.5582 | 0.135340 | 0.7349 |
| recent_2020_2025 | `bull_replace_symmetric_w075` | 2.30550 | 26.19049 | 0.143964 | 0.5575 | 0.135210 | 0.7382 |
| recent_2020_2025 | `incumbent` | 2.30686 | 26.14125 | 0.143622 | 0.5534 | 0.133863 | 0.7450 |
| temporal_2023_2025 | `asymmetric_add_w025` | 2.34751 | 26.30702 | 0.151829 | 0.5798 | 0.135609 | 0.7069 |
| temporal_2023_2025 | `bull_add_w050` | 2.34600 | 26.29014 | 0.151423 | 0.5917 | 0.135265 | 0.7091 |
| temporal_2023_2025 | `bull_replace_symmetric_w075` | 2.34630 | 26.30113 | 0.151201 | 0.5980 | 0.135102 | 0.7133 |
| temporal_2023_2025 | `incumbent` | 2.34572 | 26.21922 | 0.151351 | 0.5769 | 0.133641 | 0.7225 |

## Gates and scope

- Point primary passes all gates: `False`.
- Tail primary passes all gates: `False`.
- Template primary passes all gates: `False`.
- A point pass advances only to nested retuning; this receipt cannot promote production.
- No production table, model lock, or template contract was changed.

## Lineage

- Lock: `v2_conditional_ppg_2026_candidate_v1`
- Model run: `v2_locked_final_dk_20260828T220817Z_cdd32491`
- Feature run: `milestone_3_20260828T220116Z_117cfe38`
- Read-only database SHA-256: `56E0D46C885F3498A52A6FB2C267F8A40E53CA195F35A6A43257D8B95822F342`
