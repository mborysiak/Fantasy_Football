# V2 Locked Final Validation

## Scope

This study executes the four pre-production steps for the 2026 V2 shadow
candidate:

1. freeze the reviewed feature sets, compact model grids, and fixed blend;
2. replay 2017-2025 as complete forecast seasons, with hyperparameters selected
   only from earlier seasons;
3. fit the selected specifications through 2025 and publish 2026 shadow
   conditional-PPG and participation predictions; and
4. publish a template handoff that uses the V2 point forecast as the donor
   residual center and explicitly prohibits a second independent residual draw.

The primary conditional-PPG candidate is a fixed equal-third blend of pooled
Lasso, random forest, and deterministic shallow LightGBM using the 31 reviewed
features, five preseason projection-trajectory fields, and four position
controls. Raw ADP remains in the primary. The participation primary is pooled
deterministic shallow LightGBM using the reviewed 19 fields and four position
controls.

Prespecified secondary checks are:

- logged ADP for Lasso only;
- projection-anchored history gaps only for players with no observed history;
- a strictly prior-season projection-only/full router for limited-history
  players; and
- QB rushing-share models only for WR/TE.

## Temporal contract

For each forecast origin:

- fitted coefficients and trees use only seasons before the origin;
- hyperparameters minimize mean whole-season validation error using only
  earlier origins, beginning with 2013;
- routing decisions use only earlier 2017-2025 whole-season forecast errors;
- residual intervals use only earlier whole-season residuals; and
- 2026 settings use no 2026 outcome.

This is stricter than the five-fold donor-generation study, whose fitted
models were prior-season-only but whose hyperparameters could use the other
four player folds from the target season.

## Run

```powershell
python research/studies/2026-07-29_v2_locked_final_validation/run_validation.py
python research/studies/2026-07-29_v2_locked_final_validation/analyze_calibration.py
python research/studies/2026-07-29_v2_locked_final_validation/audit_template_handoff.py
```

The beta-scored parallel lineage uses:

```powershell
python -m Scripts.V2.build_milestone_3 --league beta --output-db Data/Databases/Projection_V2_beta.sqlite3
python research/studies/2026-07-29_v2_locked_final_validation/run_validation.py --league beta --output-db Data/Databases/Projection_V2_beta.sqlite3 --results-dir research/studies/2026-07-29_v2_locked_final_validation/results_beta
python research/studies/2026-07-29_v2_locked_final_validation/analyze_calibration.py --output-db Data/Databases/Projection_V2_beta.sqlite3 --results-dir research/studies/2026-07-29_v2_locked_final_validation/results_beta
python research/studies/2026-07-29_v2_locked_final_validation/audit_template_handoff.py --league beta --output-db Data/Databases/Projection_V2_beta.sqlite3 --results-dir research/studies/2026-07-29_v2_locked_final_validation/results_beta
```

Durable results are written to `results/` and shadow tables prefixed `locked_`
are published to `Data/Databases/Projection_V2.sqlite3`.

Production projections, best-ball weekly tables, and optimizer inputs are not
changed.

See `results/findings.md` for the DK lock and `results_beta/findings.md` for
the beta lock. Weekly templates and current player maps now carry canonical
V2 `player_key`; both audits join on that key rather than display names.
