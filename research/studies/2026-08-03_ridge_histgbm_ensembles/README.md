# Ridge and histogram-gradient-boosting ensemble screen

## Question

Do a wider Ridge regularization range or scikit-learn histogram gradient
boosting add useful conditional-PPG signal on the exact locked 40-feature V2
surface?

## Prespecified design

- Use the locked `PRIMARY_PPG_FEATURES`, eligibility population, median
  imputation with missing-value indicators, and exact current
  Lasso/random-forest/LightGBM predictions.
- Select every challenger only on rolling 2013-2022 origins, ranked by mean
  seasonal RMSE. Refit the selected configuration through 2022 and evaluate
  fixed comparisons on 2023-2025.
- Run independently for DK and beta. These are alternate scoring views of
  substantially the same player-seasons, not independent samples.
- Treat 2023-2025 as a reused historical confirmation block. Do not tune
  ensemble weights or grids after observing it.

Ridge uses standardized post-imputation inputs and the following alpha grid:

`0.001, 0.01, 0.1, 1, 10, 100, 1000`

Histogram gradient boosting uses squared-error loss, disables internal early
stopping, and crosses two conservative schedules with
`l2_regularization` values `0, 0.1, 1, 10, 100`:

1. 150 iterations, learning rate 0.03, depth 3, seven leaves, minimum leaf 20.
2. 100 iterations, learning rate 0.05, depth 4, 15 leaves, minimum leaf 20.

Primary comparisons:

1. Replace Lasso with selected Ridge while retaining equal thirds.
2. Add selected histogram gradient boosting as a fixed equal-weight fourth
   member.

Diagnostics:

- split the existing one-third linear sleeve equally between Lasso and Ridge;
- add Ridge as an equal-weight fourth member;
- replace LightGBM with histogram gradient boosting at equal thirds; and
- report standalone scores, seasonal/position slices, component correlations,
  and paired player-cluster intervals.

A challenger advances only as a research shadow if its primary comparison
improves pooled RMSE in both scorings and wins at least four of six
scoring-season cells. This screen cannot authorize a production change.

## Commands

```powershell
.venv_ff_312\Scripts\python.exe research/studies/2026-08-03_ridge_histgbm_ensembles/run_validation.py --league dk
.venv_ff_312\Scripts\python.exe research/studies/2026-08-03_ridge_histgbm_ensembles/run_validation.py --league beta
.venv_ff_312\Scripts\python.exe research/studies/2026-08-03_ridge_histgbm_ensembles/summarize.py
.venv_ff_312\Scripts\python.exe research/studies/2026-08-03_ridge_histgbm_ensembles/verify_results.py
```

## Result

Both scoring systems selected Ridge `alpha=10`, which was already inside the
older narrow grid; neither expanded boundary helped. Replacing Lasso with
Ridge improved the equal-third blend by 0.002443 RMSE in DK and 0.006348 in
beta, won all six scoring-season cells, improved MAE, and reduced positive
bias. Beta's player-cluster interval is below zero; DK's crosses zero. The
result is a research shadow only because Lasso/Ridge errors are correlated at
0.9986 DK and 0.9961 beta, RB worsens slightly in both scorings, and 2023-2025
has been reused.

DK selected the shallow HistGBM schedule with zero L2; beta selected the
deeper schedule with L2 10. Adding HistGBM as a fourth member worsened RMSE by
0.003646 DK and 0.001022 beta. Replacing LightGBM improved pooled RMSE slightly
but won only three of six scoring-season cells, and every interval crossed
zero. Reject HistGBM on this surface.

No production files changed. See `results/summary.md` for all fixed
comparisons.
