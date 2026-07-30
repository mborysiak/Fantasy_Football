# QB Passing/Rushing Target Decomposition

This study tests whether quarterback conditional PPG is easier to forecast by
modeling realized passing and rushing fantasy points per opportunity game
independently and then summing them.

The comparison is deliberately same-sample and strictly rolling:

- leagues: DK and beta;
- validation seasons: 2017-2025;
- training for an origin uses seasons strictly before that origin;
- hyperparameters are selected using only strictly prior annual validation
  predictions;
- the same QB rows, features, model families, grids, and forecast origins are
  used for the direct-total and decomposed targets.

The study compares raw expert projections, the locked production candidates, a
QB-only direct-total Lasso/RF/LightGBM blend, and the corresponding independently
fit passing-plus-rushing blend. Because configured total QB outcomes also
contain small receiving, fumble, two-point, and special-teams components, the
strict pass-plus-rush sum and a version with a strictly-prior mean other-points
adjustment are both reported.

Run from the repository root:

```powershell
python research/studies/2026-07-29_v2_qb_component_targets/run_validation.py
```

Outputs are written to `results/` and `results_beta/`. This is a research-only
study and does not update production databases or model locks.
