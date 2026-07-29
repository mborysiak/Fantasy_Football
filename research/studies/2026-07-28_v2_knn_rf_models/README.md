# V2 KNN and Random-Forest Model Families

## Question

Do K-nearest-neighbor or random-forest regressors improve conditional-PPG
prediction, or add useful diversity to deterministic shallow LightGBM, on the
projection-core and governed full feature sets?

## Prespecified comparison

- KNN uses median imputation with missingness indicators, standardization,
  Manhattan/Euclidean distance, 15/35/75 neighbors, and uniform/distance
  weights.
- Random forest uses 250 trees, depth 6/10, minimum leaf size 5/15, and
  50%/100% feature sampling.
- Both use the exact V2 2017-2025 conditional-PPG population, five
  deterministic folds, and strictly prior-season training.
- Projection core and the 31-feature full manifest are tested separately.
- Fixed 50/50 averages with the corresponding LightGBM prediction are
  secondary diversity diagnostics, not tuned ensembles.

This is isolated research. The default M4A model surface, production
projections, templates, optimizers, and databases are unchanged.

```powershell
python research/studies/2026-07-28_v2_knn_rf_models/run_validation.py
```

See [`results/findings.md`](results/findings.md) for the decision readout.

