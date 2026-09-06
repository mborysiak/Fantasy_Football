# Asymmetric expert projection study - BETA

## Conditional-PPG mean

| Surface | Challenger | Delta RMSE | Recent | Wins | Season 95% | Player 95% | Pos nonworse |
|---|---|---:|---:|---:|---:|---:|---:|
| controlled_equal_thirds | `max_minus_median_raw` | -0.00568 | +0.00395 | 4/9 | [-0.01781, +0.00391] | [-0.01438, +0.00314] | 3/4 |
| controlled_equal_thirds | `max_minus_median_fraction` | -0.00035 | +0.00211 | 3/9 | [-0.00611, +0.00450] | [-0.00474, +0.00423] | 1/4 |
| controlled_equal_thirds | `asymmetric_robust_stack` | -0.00075 | +0.00111 | 4/9 | [-0.00703, +0.00462] | [-0.00576, +0.00469] | 1/4 |
| equal_thirds | `max_minus_median_raw` | -0.00565 | +0.00412 | 5/9 | [-0.01818, +0.00459] | [-0.01380, +0.00244] | 3/4 |
| equal_thirds | `max_minus_median_fraction` | -0.00070 | +0.00217 | 4/9 | [-0.00720, +0.00516] | [-0.00505, +0.00383] | 1/4 |
| equal_thirds | `asymmetric_robust_stack` | -0.00009 | +0.00113 | 4/9 | [-0.00630, +0.00532] | [-0.00505, +0.00498] | 1/4 |

## Upper residual events

| Event | Challenger | Delta Brier | Recent | Delta log loss | AUC | Wins | Season 95% |
|---|---|---:|---:|---:|---:|---:|---:|
| plus3 | `tail_bullish` | +0.003075 | +0.003084 | +0.008050 | 0.6123 -> 0.6056 | 1/8 | [+0.000800, +0.005491] |
| plus3 | `tail_asymmetric` | +0.002997 | +0.003733 | +0.007730 | 0.6123 -> 0.6024 | 1/8 | [+0.000733, +0.005603] |
| plus5 | `tail_bullish` | +0.000598 | +0.000418 | +0.004372 | 0.6973 -> 0.6760 | 1/8 | [-0.000069, +0.001296] |
| plus5 | `tail_asymmetric` | +0.001405 | +0.001907 | +0.006751 | 0.6973 -> 0.6708 | 1/8 | [+0.000049, +0.002803] |

## Weekly-template replay

| Period | Method | PPG CRPS | Contribution CRPS | +5 Brier | +5 AUC | Impact Brier | Impact AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| full_2017_2025 | `asymmetric_add_w025` | 1.91285 | 20.61976 | 0.069153 | 0.5949 | 0.097704 | 0.6809 |
| full_2017_2025 | `bull_add_w050` | 1.91308 | 20.62261 | 0.069346 | 0.5875 | 0.097496 | 0.6844 |
| full_2017_2025 | `bull_replace_symmetric_w075` | 1.91302 | 20.62214 | 0.069438 | 0.5817 | 0.097532 | 0.6829 |
| full_2017_2025 | `incumbent` | 1.91477 | 20.64256 | 0.069240 | 0.5927 | 0.097641 | 0.6812 |
| recent_2020_2025 | `asymmetric_add_w025` | 1.91473 | 20.53520 | 0.063526 | 0.5975 | 0.096131 | 0.6732 |
| recent_2020_2025 | `bull_add_w050` | 1.91434 | 20.53337 | 0.063751 | 0.5871 | 0.095748 | 0.6806 |
| recent_2020_2025 | `bull_replace_symmetric_w075` | 1.91402 | 20.52770 | 0.063941 | 0.5756 | 0.095663 | 0.6808 |
| recent_2020_2025 | `incumbent` | 1.91541 | 20.56233 | 0.063617 | 0.5931 | 0.095831 | 0.6784 |
| temporal_2023_2025 | `asymmetric_add_w025` | 1.99470 | 20.97597 | 0.069072 | 0.6018 | 0.099450 | 0.6153 |
| temporal_2023_2025 | `bull_add_w050` | 1.99548 | 20.98701 | 0.069222 | 0.5923 | 0.099328 | 0.6158 |
| temporal_2023_2025 | `bull_replace_symmetric_w075` | 1.99441 | 20.97567 | 0.069267 | 0.5869 | 0.099433 | 0.6153 |
| temporal_2023_2025 | `incumbent` | 1.99410 | 20.98898 | 0.068984 | 0.6030 | 0.098513 | 0.6291 |

## Gates and scope

- Point primary passes all gates: `False`.
- Tail primary passes all gates: `False`.
- Template primary passes all gates: `False`.
- A point pass advances only to nested retuning; this receipt cannot promote production.
- No production table, model lock, or template contract was changed.

## Lineage

- Lock: `v2_conditional_ppg_2026_candidate_beta_v1`
- Model run: `v2_locked_final_beta_20260828T221727Z_3b720936`
- Feature run: `milestone_3_20260828T220505Z_09190cec`
- Read-only database SHA-256: `F7D0D3EBAA98D86DCCE7B01FE86C45BE4927594BA7FF5E072789AB052ECB7F68`
