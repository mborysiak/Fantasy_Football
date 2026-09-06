# V2 stabilized bullish expert-gap follow-up

This is a read-only, post-primary follow-up to
`2026-08-29_v2_asymmetric_expert_projection`. The earlier normalized feature
used `max(abs(median PPG), 1.0)` and elevated fringe players with very small
consensus projections. This study tests whether a football-scale denominator
stabilizer preserves useful one-sided disagreement without that explosion.

No production database, model lock, feature contract, or template table is
changed.

## Frozen protocol

Provider eligibility and grouping are unchanged from the parent study:

- use finite `provider_points_per_team_game` only when
  `configured_points_complete=1`;
- compute within `(player_key, season, position)` from distinct providers;
- require at least three providers, otherwise retain a missing feature plus an
  explicit availability flag;
- retain the locked model's existing `projection_provider_count` control.

The prespecified primary is:

```text
bull_gap_smooth_k5 =
    (max_provider_ppg - median_provider_ppg)
    / sqrt(median_provider_ppg^2 + 5^2)
```

The denominator is approximately `5` for fringe projections and approaches
the absolute median for established starters, without a hard discontinuity.
`k=5` was selected before inspecting this follow-up's outcomes.

The following are sensitivities only and cannot replace a failed primary by
post-hoc selection:

- smooth `k=3` and `k=8`;
- hard floor `gap / max(abs(median), 5)`;
- additive stabilizer `gap / (abs(median) + 5)`;
- unscaled raw gap.

Each feature receives a position-season percentile for template matching.

## Tests and gates

The study reuses the exact parent-study surfaces and gates:

1. locked 2017-2025 DK/beta conditional-PPG replay with inherited per-origin
   hyperparameters;
2. strictly-prior +3 and +5 PPG residual logistic models against the current
   symmetric spread controls;
3. 1,620-target-per-league weekly-template replay with each candidate added at
   fixed weight `0.50` and its availability flag at `0.25`.

Only `smooth_k5` is promotion-eligible. A point pass would still advance only
to nested retuning, not production.

## Run

```powershell
python research/studies/2026-08-29_v2_stabilized_bull_gap/run_study.py --league all
```

Durable outputs are written below `results/`.

## Result

The smooth-`k=5` primary is not promotion-worthy on any surface.

- Controlled point RMSE improves by `0.00317` DK and `0.00409` beta, but both
  recent-period deltas are worse, only two of four positions are nonworse,
  and both season- and player-cluster intervals cross zero. Most of the pooled
  gain remains a QB effect.
- Strictly-prior +3 Brier worsens by `0.001421` DK and `0.002763` beta; +5
  Brier worsens by `0.000332` and `0.000294`. All four AUC comparisons also
  worsen.
- The fixed-weight weekly-template addition fails cross-league replication.
  It improves full-period PPG CRPS slightly in both leagues but worsens DK
  contribution/impact behavior and does not preserve beta recent-period
  PPG, +5, or impact behavior.
- The stabilizer reduces the mathematical explosion at small projection
  denominators, but it does not create a ceiling relationship. The top
  smooth-`k=5` quartile still has lower +3 rates than the bottom quartile
  (`13.79%` versus `15.90%` DK; `13.58%` versus `14.56%` beta), and +5 is
  also flat-to-lower (`6.57%` versus `6.79%`; `5.73%` versus `6.11%`).
- None of the declared `k=3`, `k=8`, hard-floor, additive, or raw-gap
  sensitivities provides a stable cross-league rescue.

Decision: retain every stabilized bullish-gap form outside production. The
run used 2,000 season/player bootstrap iterations for the point and tail
comparisons and 2,000 bootstrap iterations for the template comparisons.
Both source database hashes match their pre-run manifests.
