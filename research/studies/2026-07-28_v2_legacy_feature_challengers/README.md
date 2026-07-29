# V2 Legacy-Inspired Feature Challengers

## Question

Do a small set of leakage-safe constructions retained from the legacy feature
pipeline improve the V2 conditional-PPG model beyond its original governed
31-feature manifest?

## Challenger Families

- `experience_context`: same-season, same-position experience-cohort preseason
  PPG peer mean, difference, and percentile. Experience is uncapped in the mart
  and capped only to an 8+ cohort for this comparison.
- `market_room`: self-excluded best, worst, and mean same-position teammate ADP
  gaps; number of better-drafted teammates; and inverse-square-root ADP room
  strength share.
- `opportunity_share`: projected targets, receptions, rush attempts, and
  receiving yards divided by same-team preseason projection totals.

All constructions use only same-season preseason covariates. They do not use
target-season NFL outcomes, forward filling, or future seasons.

## Method

`run_validation.py` reruns the original 31-feature direct Ridge and shallow
LightGBM baselines, adds each family separately, and adds all families together.
Every experiment uses the same deterministic five-fold assignments within each
2017-2025 validation season. Every held player-season is fit only on prior
seasons, with the same four-trial hyperparameter surface as V2 Milestone 4A.

## Run

```powershell
python research/studies/2026-07-28_v2_legacy_feature_challengers/run_validation.py
```

Durable score, slice, hyperparameter, and paired season summaries are written
to `results/`.

## Results

The original 31-feature direct models reproduced their prior OOF scores:

- Ridge: 3.1731 RMSE
- Shallow LightGBM: 3.1443 RMSE

On the original pipeline, opportunity shares lowered LightGBM RMSE by 0.0028,
while experience context and teammate ADP were neutral to worse. Adding all 12
features worsened LightGBM by 0.0080 and Ridge by 0.0131.

An individual pass exposed an attribution trap: `team_target_share` appeared to
improve seven of nine seasons even though the source is entirely unavailable
through 2023. LightGBM's fixed 80% column subsampling changed which incumbent
columns were sampled when even an all-null column was added. Those stochastic
comparisons remain saved as audit evidence, but they are not used for feature
promotion.

`run_deterministic_validation.py` therefore reruns the original manifest, every
family, and every individual feature with full row/column sampling and
deterministic LightGBM settings. Under that controlled attribution:

- experience-context family: +0.0015 RMSE;
- teammate-ADP family: -0.0035 RMSE;
- opportunity-share family: -0.0038 RMSE;
- all 12 features: +0.0039 RMSE.

Negative is better. Every family season-bootstrap interval crosses zero.
The strongest individual point estimate, self-excluded mean teammate ADP gap,
improves RMSE by only 0.0039 with a season interval
`[-0.0110, +0.0029]`; it is neutral in the 2023-2025 provider era. The legacy
projection-versus-experience difference improves RMSE by only 0.0002 with a
wide interval crossing zero. Projected target share is exactly tied through
eight origins and has only one learnable validation season.

No challenger is promoted. All 12 remain in the separate
`residual_legacy_challenger_v1` manifest for audit and future deeper-history
retesting; the incumbent `residual_candidate_v1` remains at 31 features.
