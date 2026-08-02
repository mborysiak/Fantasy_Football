# Weekly-Template Height/Weight Ablation

This study tests whether basic player size from the nflverse player master
improves weekly-template matching before adding a separate combine-data source.

The source is the same nflverse `players.csv` snapshot already used by
Projection V2 identity construction. Rows join to the governed V2
`player_identity` table by `gsis_id`, with `pfr_id` used only as an exact
fallback, and then join to templates by canonical `player_key`. No display-name
join is allowed.

Height and weight are converted to season-position percentiles so their
distance scale matches the existing template features. Missing measurements
receive the neutral `0.5` profile only for distance calculation and remain
explicitly unavailable in the coverage audit.

The predeclared primary comparison adds height and weight to QB/RB/WR/TE at
weight `0.25` each. Diagnostic arms isolate height and weight, test both at
weight `0.50`, and remove QB from the primary size specification.

Every arm retains:

- strictly prior donors;
- the production top-80 donor pool;
- the production adaptive distance kernel;
- the 12-season recency prior;
- the 5% donor probability cap; and
- the centered joint PPG-residual/weekly-path outcome.

Run DK:

```powershell
.venv_ff_312\Scripts\python.exe `
  research\studies\2026-07-30_template_height_weight_ablation\run_validation.py
```

Run beta:

```powershell
.venv_ff_312\Scripts\python.exe `
  research\studies\2026-07-30_template_height_weight_ablation\run_validation.py `
  --league beta `
  --v2-db Data\Databases\Projection_V2_beta.sqlite3 `
  --results-dir research\studies\2026-07-30_template_height_weight_ablation\results_beta
```

The study is read-only with respect to production databases and matcher
configuration. Production code remains unchanged.

## Result

The player master covered every rolling target and more than 99.8% of
historical templates. The primary arm changed about 9% of donors overall and
12% for WR, so the size fields materially affected matching.

The outcome did not transport across leagues. Beta improved modestly across
PPG, contribution, played-games, and impact metrics, while DK slightly worsened
PPG, contribution, and impact discrimination. Height alone was essentially
neutral in DK and only weakly favorable in beta. Production remains unchanged,
and this result does not justify adding a lower-coverage combine source for
height and weight. See `results/findings.md`.
