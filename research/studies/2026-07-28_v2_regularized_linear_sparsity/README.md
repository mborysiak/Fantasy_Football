# V2 Regularized Linear Sparsity

## Question

Can Lasso or Elastic Net simplify the conditional-PPG model without weakening
strict rolling OOF performance, and can either method safely absorb the 12
legacy-inspired challenger features that did not earn direct promotion?

## Design

- Direct conditional PPG, matching the current linear and nonlinear finalists.
- Validation seasons 2017-2025.
- Five deterministic player folds per season.
- Every held player-season prediction is fit only on earlier seasons.
- The hyperparameter search uses the existing `SciKitModel.time_series_cv`
  protocol and never observes the held outer fold.
- Two feature variants:
  - `incumbent`: the governed 31-feature `residual_candidate_v1` manifest plus
    four position indicators;
  - `expanded`: incumbent plus the 12-feature
    `residual_legacy_challenger_v1` manifest.
- Three standardized linear estimators: Ridge, Lasso, and Elastic Net.

The study records OOF scores, identical-fold season comparisons, selected
hyperparameters, and coefficients from the exact 45 season-fold training
populations used by each specification. Selection frequency is reported both
for raw inputs and imputation indicators.

## Run

```powershell
python research/studies/2026-07-28_v2_regularized_linear_sparsity/run_validation.py
```

All artifacts are written to `results/`. This is shadow research: it does not
modify the V2 database or any projection, template, or optimizer output.

## Result

Lasso is the strongest linear challenger. On the incumbent feature set it
improves pooled RMSE by 0.0091 versus fold-identical Ridge while selecting a
mean 23.6 of 35 raw inputs, but the nine-season interval crosses zero. Adding
all 12 legacy-inspired challengers improves Lasso by only another 0.0041 with
an interval crossing zero and weaker rookie/second-year results. Elastic Net is
slightly worse and selects penalties close to Lasso. Direct shallow LightGBM
remains the overall PPG leader.

See `results/findings.md` for the decision and feature-stability interpretation.
