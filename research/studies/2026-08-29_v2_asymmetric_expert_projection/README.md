# V2 asymmetric expert projection study

This read-only study tests whether one-sided expert projection disagreement
contains information that the locked symmetric spread features miss.  It does
not write to a production SQLite database, change a feature contract, or alter
a model/template lock.

## Frozen feature protocol

The source is `player_season_projection_values`.  A provider observation is
eligible only when `configured_points_complete=1` and
`provider_points_per_team_game` is finite.  Features are computed within
`player_key, season, position` from distinct providers.

The primary one-sided feature is defined only with at least three eligible
providers:

- `expert_ppg_bull_gap = max(provider PPG) - median(provider PPG)`
- `expert_ppg_bear_gap = median(provider PPG) - min(provider PPG)`
- `expert_ppg_top2_gap = mean(top two provider PPG) - median(provider PPG)`
- fractional forms divide by `max(abs(median provider PPG), 1.0)`
- position-season percentiles are computed only among available rows
- `expert_ppg_bull_gap_available` and the eligible provider count preserve
  the missingness/coverage distinction

Two-provider rows are deliberately treated as unavailable: with two values,
`max - median` and `median - min` are mechanically equal and cannot identify
asymmetry.  The locked conditional-PPG model already controls for
`projection_provider_count`; the availability flag is added with each gap
challenger.

## Prespecified tests

1. **Conditional-PPG mean:** replay the locked DK and beta rolling-origin
   model with its per-origin selected hyperparameters.  Compare the incumbent
   with raw bullish gap, normalized bullish gap, and an asymmetric robust
   stack.  This is attribution only; any pass advances to nested retuning.
2. **Upper residual events:** use strictly prior logistic models to predict
   incumbent residuals of at least +3 and +5 PPG.  Compare the current
   symmetric projection-spread controls with bullish and asymmetric gap
   additions using Brier score, log loss, AUC, calibration bias, recent-period
   behavior, and season/player clustered intervals.
3. **Weekly templates:** replay the current strictly-prior donor matcher for
   DK and beta.  Compare the incumbent with a fixed 0.50 bullish-gap
   percentile weight, a symmetric 0.25/0.25 bullish/bearish addition, and a
   replacement of the current 0.75 projection-disagreement weight.  Score PPG,
   contribution and availability distributions plus +3, +5, and impact-event
   discrimination.

The primary point and tail candidate is the normalized bullish gap.  The
primary template candidate is the fixed 0.50 bullish-gap percentile addition.
All other variants are sensitivities, not a post-hoc promotion path.

## Run

```powershell
python research/studies/2026-08-29_v2_asymmetric_expert_projection/run_study.py --league all
```

Durable outputs are written below `results/`.

## Result

Retain every asymmetric-gap feature outside production.

- The prespecified normalized bullish gap missed every cross-league point and
  upper-tail promotion path.  Controlled RMSE changed by `+0.00075` DK and
  `-0.00035` beta, while 2023-2025 worsened in both.  +3 PPG Brier worsened by
  `+0.002176/+0.003075` DK/beta and +5 PPG Brier by
  `+0.000329/+0.000598`; AUC declined for both events in both leagues.
- The literal raw `max - median` sensitivity reduced pooled controlled RMSE by
  `0.00548` DK and `0.00568` beta, but it worsened recent controlled RMSE,
  won only `5/9` and `4/9` seasons, improved only `2/4` and `3/4` positions,
  and both season- and player-cluster intervals crossed zero.  Most of the
  position signal came from QB (`-0.0379/-0.0383` RMSE DK/beta), so this is a
  post-slice lead rather than promotion evidence.
- Adding the fixed bullish-gap matcher weight improved full-period PPG CRPS by
  `0.00276` DK and `0.00169` beta, but DK contribution, +5, and impact scores
  worsened.  Beta +3 and impact were directionally better over the full period,
  while +5 worsened and the 2023-2025 PPG/impact directions reversed.  Replacing
  symmetric disagreement also failed the multi-outcome gates.
- Descriptively, the top bullish-gap quartile did not behave like a ceiling
  group.  Its +3/+5 residual rates were `12.7%/5.6%` DK and `11.9%/4.9%` beta,
  below the bottom quartile's `17.7%/8.1%` and `16.5%/6.9%`.  Large normalized
  gaps are frequently fringe players with very small consensus denominators.

The only reasonable follow-up is a separately prespecified, projection-floor
or projection-tier-controlled **raw absolute QB gap** experiment with nested
retuning.  Do not promote a generic max-minus-median point feature or template
weight from this result.
