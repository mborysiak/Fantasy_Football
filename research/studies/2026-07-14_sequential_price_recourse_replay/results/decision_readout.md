# Sequential recourse readout

This is a non-anticipating fixed-price stress test, not a reconstruction of the historical nomination room. Future prices and target outcomes were hidden from every Buy/Pass decision.

- Paths: 1,024
- Completed legal rosters: 821/1,024
- Runtime generic-repair paths: 0 (the replay never invents a player)
- Prefix-invariance check: True

## Decision

**No buffer is selected by this replay.** The primary comparison fails the predeclared completion, discordance, sign-stability, and randomization-precision requirements.

- `$5` completed 44/72 primary paths; `$10` completed 42/72.
- Both completed in 38/72 pairs; 10/72 pairs had discordant completion.
- Only 15/72 pairs were clean enough for the prespecified point comparison.
- No primary order family had clean observations in every development origin, so the equal-origin effect and its randomization error are undefined.

## Primary: strict p+1, 2022-2024, by order family

The table is intentionally separated by order family; no probabilities are assigned to the synthetic regimes. Tier-early, uniform, and position-run are the primary families; star-late is adversarial sensitivity.

| order_regime | paired_paths | completion_rate_5 | completion_rate_10 | completion_rate_diff_5_minus_10 | paired_completion_rate | paired_clean_rate | mean_points_diff_clean_equal_origin_5_minus_10 | randomization_se_clean_equal_origin | minimum_clean_paths_per_origin | mean_points_diff_completed_5_minus_10 | mean_failure_penalized_diff_5_minus_10 | mean_relax_events_5 | mean_relax_events_10 | mean_top_n_relax_events_5 | mean_top_n_relax_events_10 | generic_repair_path_rate_5 | generic_repair_path_rate_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| position_run | 24 | 0.417 | 0.417 | 0.0 | 0.375 | 0.167 |  |  | 0 | -7.278 | 5.69 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| star_late | 24 | 0.958 | 0.792 | 0.167 | 0.75 | 0.208 |  |  | 0 | -1.277 | 264.931 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| tier_early | 24 | 0.5 | 0.458 | 0.042 | 0.333 | 0.125 |  |  | 0 | -39.952 | 62.078 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| uniform | 24 | 0.917 | 0.875 | 0.042 | 0.875 | 0.333 |  |  | 0 | -24.758 | 48.629 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Positive point differences favor `$5`; negative differences favor `$10`. Points are decision-worthy only when paired-clean completion is effectively complete. The failure-penalized difference assigns an incomplete draft zero points; it is a deliberately harsh policy-invalid sensitivity, not an observed season score.

Operational mode (`+$5 -> +$10 -> no nominal row`, or `+$10 -> no nominal row`, followed by Top-N relaxation) and recorded-price `p` are sensitivities in `summary_development.csv`. No runtime fallback relaxes the real `$298` cap, roster size, or position limits.

The replay begins with an empty personal roster after all league keepers are removed. It does not validate a universal buffer for a fixed personal keeper state.

If completion, order family, or price convention changes the conclusion, this study does not identify a buffer choice.

2025 is a temporal sensitivity rather than a fresh holdout because its results were inspected during earlier tuning.