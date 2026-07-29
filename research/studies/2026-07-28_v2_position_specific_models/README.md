# V2 Position-Specific Conditional-PPG Models

## Question

Do separate QB, RB, WR, and TE models improve conditional-PPG prediction
relative to the pooled V2 models because the relevant projection, history,
market, room, and team relationships differ by position?

## Prespecified comparison

Fit two deterministic shallow LightGBM specifications under a four-model
QB/RB/WR/TE split, a three-model QB/RB/WR+TE role split, and a two-model
QB/RB+WR+TE split:

1. Projection core: the 22 projection consensus/component fields used in the
   projection consensus ladder.
2. Full: the governed 31-feature `residual_candidate_v1` manifest.

Constant position indicators are omitted inside a one-position model. The
multi-position models retain their applicable position indicators. Otherwise
the input sets and LightGBM search space match the pooled comparators. Every
component model:

- uses the same conditional-PPG eligibility contract;
- predicts the same 2017-2025 player-season population as its pooled
  comparator;
- trains each target season only on earlier seasons;
- uses five deterministic within-component folds; and
- selects hyperparameters from the other four rolling OOF folds.

The component outputs are stitched together only after prediction.
Comparisons are paired by player-season and uncertainty is clustered by
season. This is isolated research and does not modify the V2 database,
production projections, templates, or optimizers.

```powershell
python research/studies/2026-07-28_v2_position_specific_models/run_validation.py
```

See [`results/findings.md`](results/findings.md) for the decision readout.
