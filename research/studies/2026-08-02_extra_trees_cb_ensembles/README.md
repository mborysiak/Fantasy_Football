# Extra Trees and CatBoost ensemble screen

## Question

Do Extra Trees or CatBoost add complementary conditional-PPG signal to the
current equal-weight Lasso / random-forest / LightGBM projection blend?

## Prespecified design

- Use the locked 40-feature V2 conditional-PPG feature set and eligibility
  rules without adding new predictors.
- Use the same median imputation with missing-value indicators as the current
  models.
- Select one of eight compact configurations for each challenger using only
  rolling pre-2023 origins (2013-2022), ranked by mean seasonal RMSE.
- Refit the selected challenger through 2022 and evaluate on 2023-2025.
- Run independently against `Projection_V2.sqlite3` (DK) and
  `Projection_V2_beta.sqlite3` (half-PPR beta).
- Reuse the exact current-model holdout predictions from the sealed-methodology
  study so the comparison baseline is bit-for-bit identical.

Primary comparisons are fixed before looking at results:

1. Current equal thirds vs. Lasso / RF / LightGBM / Extra Trees equal fourths.
2. Current equal thirds vs. Lasso / RF / LightGBM / CatBoost equal fourths.

The equal-fifths blend containing both challengers is secondary and is only a
promotion candidate if both challengers add value independently. Standalone
scores and component residual correlations are diagnostic.

The 2023-2025 block has been reused by earlier research and is therefore a
historical temporal confirmation block, not a pristine holdout. The model
families, candidate grids, and ensemble weights above were specified before
this study's results were observed.

## Commands

```powershell
.venv_ff_312\Scripts\python.exe research/studies/2026-08-02_extra_trees_cb_ensembles/run_validation.py --league dk
.venv_ff_312\Scripts\python.exe research/studies/2026-08-02_extra_trees_cb_ensembles/run_validation.py --league beta
.venv_ff_312\Scripts\python.exe research/studies/2026-08-02_extra_trees_cb_ensembles/run_seed_robustness.py
.venv_ff_312\Scripts\python.exe research/studies/2026-08-02_extra_trees_cb_ensembles/summarize.py
.venv_ff_312\Scripts\python.exe research/studies/2026-08-02_extra_trees_cb_ensembles/verify_results.py
```

## Result

Extra Trees improves the equal-four blend by 0.003461 RMSE in DK and 0.006270
in beta. It wins all six scoring-system/season cells, every non-QB position
cell, and all ten estimator-seed robustness cells. The beta player-cluster
interval is below zero; the DK interval crosses zero. Advance Extra Trees only
as a research shadow candidate because the gain is small and the 2023-2025
confirmation block has been reused.

CatBoost improves DK by only 0.000686 and worsens beta by 0.002758. Reject it
as an equal-weight fourth member. The secondary five-way blend is also rejected
because both challengers did not pass independently. No production files were
changed. See `results/summary.md` for the full comparison.
