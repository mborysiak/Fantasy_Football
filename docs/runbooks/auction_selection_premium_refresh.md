# Auction Selection Premium Refresh

Run this after the active season's salary predictions, keeper table, projection
residuals, and managed weekly templates have been rebuilt. It is also the
refresh step when those inputs materially change before the auction.

The live 2026 beta prerequisite is salary method
`current_locked_spec_v6_v2_population_11f`: exactly 323 canonical player keys,
all from the governed `model_inputs_projonly` population, and 14 keyed keepers.
The highest 142 non-keeper point salaries must total the `$2,979` available
market budget before the reserve is built.

From the Fantasy Football model repo:

```powershell
.\.venv_ff_312\Scripts\python.exe Scripts\Modeling\s5_Auction_Selection_Premium.py `
  --year 2026 --league beta --trials 1000 --workers 8
```

`--year` defaults to `Scripts/config.py::YEAR`; it is shown explicitly above
to make the run receipt unambiguous. The historical bootstrap is currently
validated for `beta`, so another league requires its own persisted seed history
instead of silently reusing beta rates.

The default policy is the validated half-strength reserve. The command:

1. loads or bootstraps the durable historical seed table;
2. attaches newly available prior-season actual auction prices;
3. clears only the staged active year/league premium slice, then runs one
   premium-free 1,000-roster Target seed under the active keeper market and
   live organic construction policy;
4. fits the ridge model only on seasons before the target year and records any
   governed historical/current salary-surface transfer;
5. writes `Salary_Selection_Seeds` and `Salary_Selection_Calibrator` to
   `Validations.sqlite3`;
6. replaces the active `Salary_Selection_Premium` slice in source
   `Simulation.sqlite3`; and
7. synchronizes that slice to `Fantasy_Football_App/app/Simulation.sqlite3`.

Use `--reuse-current-seed` to change only the shrinkage strength or republish
the table without paying for another Target seed. Use `--no-app-sync` for a
source-only diagnostic run.

The scoped clear in step 3 is required because the simulation constructor
loads the published table before the seed explicitly disables selection
premiums. It prevents stale players from a prior refresh from blocking or
influencing the clean seed; other seasons and leagues are preserved, and the
new active slice is published only after the seed and calibration succeed.

Do not use `--reuse-current-seed` across a material roster-construction policy
change. The current keeper-aware organic policy is versioned as
`app_target_selection_only_keeper_portfolio_v3`; a method change requires a
fresh seed.

After a normal refresh, verify:

- Target success equals requested trials;
- each season's selection rates sum to roster size 13;
- the active seed uses `app_target_selection_only_keeper_portfolio_v3`;
- the training cutoff is strictly earlier than the target year;
- the source and app premium slices have the same canonical player keys;
- the active 2026 beta slice has 309 non-keeper rows and no missing or duplicate
  `player_key`;
- `salary_method_version` is
  `current_locked_spec_v6_v2_population_11f`;
- the historical-v5/current-v6 transfer is labeled
  `historical_v5_selection_surface_to_current_v6_v1`; and
- keepers and entered/explicit prices remain deterministic in app smoke tests.

The reserve is conditional on the seed configuration: 12 teams, `$298` cap,
13-player roster, current starter/FLEX structure, current position maxima,
Top-12 requirement, five salary draws, projected waiver baselines, zero generic
bench-upside weight, and the soft whole-bench keeper portfolio described in the
auction app docs. Rerun the workflow if those baseline rules materially change.

The 1,000-roster default is intentional. A two-seed QA check found that 250
rosters stabilized the aggregate reserve but left avoidable noise in individual
player premiums; the larger offline seed materially improves table stability
without adding any live-app cost.

The current v6-population refresh completed 1,000/1,000 Target rosters across
309 non-keepers and produced an expected 13-player reserve of `$8.8068`. The
transfer retains v5 historical calibration because v5/v6 current common-player
point salaries are nearly unchanged (correlation `0.99957`, MAE `$0.274`);
this is explicit transfer evidence, not a claim that historical v6 seeds were
reconstructed.
