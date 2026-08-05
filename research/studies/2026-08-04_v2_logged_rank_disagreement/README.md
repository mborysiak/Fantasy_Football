# Logged Expert-Rank Disagreement Study

## Question

Does cross-provider expert-rank disagreement add out-of-sample signal after
controlling for the player's consensus within-position rank and the number of
rank sources that covered the player?

The motivating scale is a logged common-depth position rank. A 30-to-50 split
is intentionally larger than a 130-to-170 split, while unanimous number-one
ranks have zero disagreement.

## Frozen feature definition

For each scoring objective, season, source, and position:

1. Convert the published overall rank to a within-source position percentile.
2. Map that percentile to a common position depth from the V2 feature universe.
3. Transform the common-depth rank with `log1p`.
4. Take the median absolute deviation (MAD) across providers.

`expert_rank_logged_mad` is missing, not zero, when fewer than two sources rank
a player. The model also receives consensus rank level, observed source count,
and publication coverage so missingness or rank depth cannot masquerade as
agreement.

DK replaces the half-PPR ETR table with the pinned full-PPR ETR files for 2025
and 2026. Beta retains the half-PPR ETR table. Every other source and identity
rule is inherited from the frozen 2026-07-30 scoring-specific rank study.

The prespecified sensitivity is `expert_rank_logged_mad_excess`: observed
logged MAD minus a strictly-prior expected MAD. Expected MAD is estimated only
from earlier seasons with hierarchical medians by position, consensus-rank
decile, and source-count bucket. It is never allowed to use the target season
or future seasons.

## Point-model test

Variants:

- `incumbent`: locked production features.
- `rank_level`: incumbent plus consensus within-position rank percentile.
- `rank_level_logged`: rank level plus logged MAD, source count, and coverage.
- `rank_level_excess`: rank level plus causal excess MAD, source count, and
  coverage.

All variants reuse the locked, strictly-prior hyperparameters. The primary
attribution surface forces the random forest to consider all columns; the
locked equal-thirds production surface is reported as a sensitivity. The main
incremental comparison is `rank_level_logged` against `rank_level`, not against
the incumbent.

The logged feature advances only to a nested retune if, in both DK and beta:

- controlled pooled RMSE improves by at least 0.001;
- controlled 2023-2025 RMSE is nonworse;
- at least 6 of 9 seasons improve;
- season- and player-cluster 95% interval upper bounds are at most zero;
- production-surface pooled and recent RMSE are nonworse; and
- at least three of four positions are nonworse on the controlled surface.

The excess feature is a sensitivity and cannot independently advance the raw
logged feature.

## Residual-uncertainty test

To isolate scale signal, all uncertainty variants use the identical
`rank_level` controlled point prediction. For each target season, a fixed
ridge model is trained only on earlier out-of-fold seasons to predict
`log1p(abs(point residual))`. Its predicted MAE is calibrated on the same
strictly-prior training rows and converted to Gaussian sigma.

Scale variants:

- `scale_rank_level`: consensus rank level, rank level squared, position,
  source count, and coverage.
- `scale_logged`: baseline scale features plus logged MAD.
- `scale_excess`: baseline scale features plus causal excess MAD.

The logged scale feature advances only to downstream residual/template
validation if, in both leagues:

- pooled Gaussian CRPS improves by at least 0.25%;
- recent CRPS is nonworse;
- at least 5 of 8 scale-evaluation seasons improve;
- season- and player-cluster 95% interval upper bounds are at most zero; and
- absolute 80% interval coverage error is nonworse.

## Governance

- This study reads current V2 databases in SQLite read-only mode.
- Database SHA-256 hashes are checked before and after each league run.
- Results are written only below this study's `results/` directory.
- Passing either gate is an invitation to a nested/downstream validation, not
  production promotion.

Run both leagues with:

```powershell
.\.venv_ff_312\Scripts\python.exe research\studies\2026-08-04_v2_logged_rank_disagreement\run_study.py --league all
```
