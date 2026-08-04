# LightGBM and CatBoost grid expansion

## Question

Does a broader, causally selected learning-rate search improve the locked
LightGBM member or rescue CatBoost as a complementary fourth member?

## Prespecified design

- Preserve the locked 40-feature conditional-PPG population and median
  imputation pipeline.
- Select every hyperparameter only on rolling 2013-2022 forecasts, ranked by
  mean seasonal RMSE; refit through 2022 and confirm on 2023-2025.
- Retain all eight incumbent LightGBM candidates, then add eight schedules
  spanning learning rates 0.01-0.10 with compensating tree counts under two
  structural profiles.
- Retain all eight prior CatBoost candidates, then add four learning-rate/tree
  schedules around the prior winner and four deeper/less-regularized boundary
  candidates.
- Do not use early stopping on a scored origin or on 2023-2025.
- Reuse the exact current Lasso/RF/LightGBM and research Extra Trees holdout
  predictions from the prior model-family screen.

Primary comparisons:

1. Replace incumbent LightGBM in the equal-third blend with expanded-grid
   LightGBM.
2. Add expanded-grid CatBoost to the current blend at equal-four weight.

Secondary comparisons measure whether expanded LightGBM changes the prior
Extra Trees result and whether CatBoost helps after the LightGBM replacement.
DK and beta are alternate scoring views of substantially the same
player-seasons, not independent samples. The 2023-2025 block has been reused
and is a historical temporal confirmation block rather than a pristine
holdout.

## Commands

```powershell
.venv_ff_312\Scripts\python.exe research/studies/2026-08-02_boosting_grid_expansion/run_validation.py --league dk
.venv_ff_312\Scripts\python.exe research/studies/2026-08-02_boosting_grid_expansion/run_validation.py --league beta
.venv_ff_312\Scripts\python.exe research/studies/2026-08-02_boosting_grid_expansion/summarize.py
.venv_ff_312\Scripts\python.exe research/studies/2026-08-02_boosting_grid_expansion/verify_results.py
```

## Result

DK retained the exact incumbent LightGBM 0.05 / 100-tree configuration. Beta
selected the new 0.01 / 500-tree schedule on pre-2023 forecasts, but the
three-way blend worsened pooled 2023-2025 RMSE by 0.000583; its interval crosses
zero. Retain the existing LightGBM grid and parameters.

Both scorings retained the original CatBoost 0.03 / 300-iteration candidate,
reproducing the prior small DK gain and beta loss. Expanded tuning does not
rescue CatBoost. The beta LightGBM replacement also slightly weakens the prior
Extra Trees blend. No production files were changed. See `results/summary.md`.
