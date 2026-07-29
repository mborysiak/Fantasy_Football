# V2 Next-Year Expert-Residual Modeling

This study models following-season conditional PPG as a residual around the
origin-season expert team-game PPG consensus and separately models the
probability of any following-season appearance.

The origin-to-target horizon is one season. Every fit for origin `t` uses
training origins no later than `t-2`, and hyperparameter selection only scores
inner origins no later than `t-2`. Confirmed identities with no following-season
game evidence are participation zeros; unresolved provisional identities remain
unlabeled. Conditional PPG is never filled for a non-participant.

Run DK:

```powershell
python research/studies/2026-07-29_v2_next_year_residual/run_validation.py
```

Run beta:

```powershell
python research/studies/2026-07-29_v2_next_year_residual/run_validation.py `
  --league beta `
  --output-db Data/Databases/Projection_V2_beta.sqlite3 `
  --results-dir research/studies/2026-07-29_v2_next_year_residual/results_beta
```

The validation runs remain shadow-only. Production publication is a separate,
explicit step through `python -m Scripts.V2.production_handoff`.

Test the DK next-year fields as incremental weekly-template matching context:

```powershell
python research/studies/2026-07-29_v2_next_year_residual/run_template_validation.py
```

Test beta:

```powershell
python research/studies/2026-07-29_v2_next_year_residual/run_template_validation.py `
  --league beta `
  --v2-db Data/Databases/Projection_V2_beta.sqlite3 `
  --results-dir research/studies/2026-07-29_v2_next_year_residual/template_results_beta
```

## Result

The equal-third Lasso/random-forest/LightGBM residual blend improves
following-season conditional-PPG RMSE from 5.2070 to 3.9003 in DK and from
4.6685 to 3.6718 in beta. It wins all eight validation origins in both scoring
systems. The separate LightGBM appearance model improves Brier score from
0.2648 to 0.1604 in DK and 0.1623 in beta.

The incremental template replay does not support adding these fields to the
weekly matcher. The residual rank improves weekly-PPG CRPS by only
0.0010-0.0023 depending on weight and league, while season-contribution CRPS is
generally worse. The production handoff therefore keeps them out of matching
and consumes them explicitly as a two-part forecast: conditional PPG given an
appearance plus the probability of any following-season appearance. Keeper
value applies the appearance draw and sets future market value to zero on no
appearance.

The production guardrail and rejected historical donor-recentering result are
documented in `results/production_handoff_findings.md`.
