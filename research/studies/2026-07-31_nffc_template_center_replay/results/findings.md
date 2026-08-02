# NFFC Template Center Replay Findings

## Conclusion

Retain the scoring-matched expert donor center; do not promote the locked OOF donor center.

The recommendation follows the prespecified gate without post-hoc weighting.
All target centers are the same locked OOF NFFC forecasts, and both arms use
identical scoring-matched V2 match context and donor pools.

## Pooled calibration

| center_policy | n | ppg_crps | ppg_bias | ppg_80_coverage | contribution_crps | contribution_80_coverage | played_crps | played_80_coverage | extended_absence_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expert_donor_center | 540.000000 | 2.087718 | 0.229029 | 0.733333 | 26.033694 | 0.764815 | 1.655135 | 0.914815 | 0.092362 |
| locked_oof_donor_center | 540.000000 | 2.090619 | 0.228396 | 0.720370 | 26.080665 | 0.753704 | 1.655135 | 0.914815 | 0.092362 |

## Locked minus expert

Negative CRPS deltas favor the locked donor center.

| metric | n | expert | locked | locked_minus_expert | relative_delta |
| --- | --- | --- | --- | --- | --- |
| ppg_crps | 540 | 2.087718 | 2.090619 | 0.002901 | 0.001390 |
| contribution_crps | 540 | 26.033694 | 26.080665 | 0.046970 | 0.001804 |
| played_crps | 540 | 1.655135 | 1.655135 | 0.000000 | 0.000000 |

## Clustered uncertainty

| cluster | metric | n_rows | n_clusters | samples | locked_minus_expert | bootstrap_p025 | bootstrap_p975 | probability_locked_better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| player_key | ppg_crps | 540 | 271 | 2000 | 0.002901 | -0.004914 | 0.010748 | 0.248500 |
| player_key | contribution_crps | 540 | 271 | 2000 | 0.046970 | -0.051702 | 0.143198 | 0.173500 |
| player_key | played_crps | 540 | 271 | 2000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| season | ppg_crps | 540 | 3 | 2000 | 0.002901 | 0.000578 | 0.005788 | 0.000000 |
| season | contribution_crps | 540 | 3 | 2000 | 0.046970 | -0.000015 | 0.089673 | 0.035500 |
| season | played_crps | 540 | 3 | 2000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Position safety

| center_policy | pos | ppg_crps | contribution_crps | played_crps |
| --- | --- | --- | --- | --- |
| expert_donor_center | QB | 2.526246 | 35.008384 | 1.785742 |
| expert_donor_center | RB | 2.261741 | 26.386157 | 1.787391 |
| expert_donor_center | TE | 1.514947 | 20.510926 | 1.545553 |
| expert_donor_center | WR | 1.987445 | 24.589335 | 1.537915 |
| locked_oof_donor_center | QB | 2.542243 | 35.279879 | 1.785742 |
| locked_oof_donor_center | RB | 2.263481 | 26.457029 | 1.787391 |
| locked_oof_donor_center | TE | 1.515560 | 20.504151 | 1.545553 |
| locked_oof_donor_center | WR | 1.987712 | 24.559461 | 1.537915 |

## Season consistency

| center_policy | season | ppg_crps | contribution_crps | played_crps |
| --- | --- | --- | --- | --- |
| expert_donor_center | 2023 | 2.094773 | 25.292237 | 1.453472 |
| expert_donor_center | 2024 | 2.140031 | 26.718791 | 1.595954 |
| expert_donor_center | 2025 | 2.028349 | 26.090056 | 1.915980 |
| locked_oof_donor_center | 2023 | 2.097111 | 25.381910 | 1.453472 |
| locked_oof_donor_center | 2024 | 2.145819 | 26.770043 | 1.595954 |
| locked_oof_donor_center | 2025 | 2.028926 | 26.090041 | 1.915980 |

## Prespecified gates

| gate | passed | observed | threshold |
| --- | --- | --- | --- |
| pooled_ppg_crps_improves | 0 | 0.002901 | 0.000000 |
| player_cluster_ppg_upper_at_or_below_zero | 0 | 0.010748 | 0.000000 |
| ppg_season_wins_at_least_two | 0 | 0.000000 | 2.000000 |
| contribution_crps_within_0_25pct | 1 | 0.001804 | 0.002500 |
| played_crps_within_0_25pct | 1 | 0.000000 | 0.002500 |
| all_coverage_within_one_point | 0 | -0.012963 | -0.010000 |
| all_event_briers_within_0_001 | 1 | 0.000314 | 0.001000 |
| absolute_ppg_bias_within_0_10 | 1 | -0.000633 | 0.100000 |
| position_composite_within_0_5pct | 1 | 0.004696 | 0.005000 |
| position_metric_within_1pct | 1 | 0.007755 | 0.010000 |

## Scope

- Target seasons: 2023-2025.
- Donors: 2021 through the season immediately before each target.
- Horizon: 17 weeks.
- Positions: QB, RB, WR, TE.
- Contribution uses the inherited managed-auction replacement baselines and is
  secondary to PPG and played-games calibration.
- Three target-season clusters are not enough for strong season-level
  asymptotics; player-cluster uncertainty and directional safety are reported.
- Production code and live databases were not changed.

Runtime: 4.5 seconds.
