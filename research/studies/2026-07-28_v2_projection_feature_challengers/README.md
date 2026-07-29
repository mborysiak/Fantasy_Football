# V2 Projection Feature Challengers

## Question

After standardizing provider scoring, do compact projection-shape,
provider-specific, or component-disagreement features improve direct
conditional-PPG forecasts beyond the governed 31-feature incumbent?

## Scoring Contract

- A provider enters the primary point consensus only when its required
  position-specific raw components are complete and can be scored under the
  configured league.
- When exactly one required component is missing, it is filled only when at
  least two other providers supply that player-season component; the imputation
  count remains explicit. Rows missing more than one required component remain
  unscored.
- Provider-published fantasy-point totals and PPG are retained only as raw
  source audit fields. They never enter the consensus, provider features, or
  provider room/team context.
- Provider room/team point context uses only component-scored values.

## Challenger Families

- `projection_shape`: ten volume/rate constructions, including total touches,
  total opportunities, yards per attempt/reception, TD rates, interception
  rate, and catch rate.
- `projection_disagreement`: eight across-provider standard deviations for
  volume, receiving yards, and passing/rushing/receiving touchdowns.
- `provider_projection`: eight provider-specific component-scored team-game
  PPG values. Missingness is explicit. The provider-specific column remains
  masked until the provider has three prior projection seasons, preventing a
  learned provider adjustment from being estimated from one recent season.
  New providers still contribute equally to the configured-score consensus.

## Method

The study runs direct Lasso and deterministic full-column shallow LightGBM on
the incumbent, each family separately, and all 26 additions together. Every
comparison uses identical five-fold 2017-2025 assignments and trains a held
player-season only on earlier seasons.

```powershell
python research/studies/2026-07-28_v2_projection_feature_challengers/run_validation.py
```

This is shadow research. It does not modify production projections, templates,
or optimizer inputs.

Results and the promotion decision are in
[`results/findings.md`](results/findings.md).
