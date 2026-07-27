# Projection Validation Residual Backfill

This study upgrades the existing `Validations.Model_Validations_Resid` rows
without rerunning the point-projection models.

It:

- calibrates residual quantiles within each homogeneous model slice;
- uses only realized outcomes strictly before each forecast origin;
- applies the extra outcome embargo required by `next` models;
- retains target-season and donor-cutoff provenance;
- builds `Final_Validations_Resid` with the same component/current/next family
  weights used by the production final ensemble; and
- compares the persisted final point means with an independent reconstruction
  of the prior historical ensemble logic.

Run from the repository root:

```powershell
.venv_ff_312\Scripts\python.exe research\studies\2026-07-16_projection_validation_residual_backfill\run_backfill.py --apply
```

The script makes a timestamped SQLite backup under `artifacts/local/` before
mutating the database. Audits are written to `results/`.

