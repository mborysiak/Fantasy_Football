# SKM fold-ensemble sealed holdout

This study compares the current locked V2 single-fit methodology with the
fold-parameter and seed ensembling used by the legacy S1/S2 workflow.

## Sealed test

- Seasons 2023-2025 are excluded from every fit, fold assignment, and
  hyperparameter decision.
- Both methods use the same current V2 conditional-PPG population, primary
  feature set, median-plus-missing-indicator preprocessing, compact model
  grids, and equal-third Lasso/RF/LightGBM family blend.
- Pre-2023 hyperparameter evidence is generated from rolling season forecasts
  beginning in 2013. No model fit for an origin uses that origin or a later
  season.
- Each 2023-2025 feature row is the legal preseason vintage for that season.
  Thus 2024 may use information known before the 2024 season, but no
  2023-2025 outcome is used to fit or select a model.

## Methods

- `current_single`: one configuration per family minimizes mean season RMSE
  over complete pre-2023 rolling forecasts, then one model is fit on all data
  through 2022.
- `current_seed_bag`: repeats the current selected configuration across five
  predetermined estimator seeds and averages predictions.
- `skm_fold_param_bag`: for each family, the 2013-2022 development rows are
  assigned to five shuffled season-stratified folds. Each member selects its
  configuration from the other four folds using the legacy pooled-MSE
  objective, refits on all data through 2022, and the five predictions are
  averaged. Estimator seed stays fixed.
- `skm_fold_seed_bag`: the same fold-specific configurations are refit with
  five predetermined estimator seeds before averaging.

Fold-assignment seeds differ by model family. Under the frozen production
pipeline, seed variation affects random forest; Lasso is deterministic and
LightGBM uses deterministic full-row/full-column fitting. This decomposition
therefore distinguishes parameter bagging from the incremental RF seed bag.

## Run

```powershell
.venv_ff_312\Scripts\python.exe research/studies/2026-08-02_skm_fold_ensemble_holdout/run_validation.py --league dk
.venv_ff_312\Scripts\python.exe research/studies/2026-08-02_skm_fold_ensemble_holdout/run_validation.py --league beta
.venv_ff_312\Scripts\python.exe research/studies/2026-08-02_skm_fold_ensemble_holdout/summarize.py
```

Production databases are read-only and no production model or projection is
changed.

## Result

Retain the current single-fit methodology. The full fold-plus-seed bag is
effectively tied in DK (`+0.000213` RMSE) and worse in beta (`+0.004413`),
where its paired player-cluster interval is fully above zero. Seed-bagging the
current parameters is neutral. Fold-specific RF improves as a component in
both leagues, but fold-specific Lasso destabilizes beta and makes the complete
SKM blend worse. A post-hoc current-Lasso/current-LightGBM plus fold-bagged-RF
hybrid is retained only as a future prespecified hypothesis.

See `results/findings.md` for the decision and compact diagnostics.
