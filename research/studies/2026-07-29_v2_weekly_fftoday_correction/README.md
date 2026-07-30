# V2 Weekly Scoring and FFToday Vintage Correction

## Purpose

This study records two follow-up corrections to the 2026-07-29 V2 rebuild:

1. historical beta weekly templates were labeled beta but scored through the
   default DK dictionary; and
2. the 50 FFToday QB projection rows stored under 2018 match the provider's
   official 2019 archive and were leaking future-vintage evidence into 2018.

The correction does not change model architecture, template matching weights,
or app sampling semantics. It repairs the scoring and preseason-data lineage,
then reruns the locked current, calibration, following-season, weekly-template,
handoff, and publication gates.

## Reproduction

Build isolated league databases:

```powershell
python -m Scripts.V2.build_milestone_3 --league dk --output-db <staging>\Projection_V2.sqlite3
python -m Scripts.V2.build_milestone_3 --league beta --output-db <staging>\Projection_V2_beta.sqlite3
```

Run the locked and following-season validations against each staged V2
database, then publish the production handoff into a copied
`Simulation.sqlite3`. Rebuild both weekly slices explicitly:

```powershell
python Scripts\Modeling\s4_Best_Ball_Weekly.py --league beta --simulation-db <staging>\Simulation.sqlite3 --v2-db <staging>\Projection_V2_beta.sqlite3 --no-app-sync
python Scripts\Modeling\s4_Best_Ball_Weekly.py --league dk --simulation-db <staging>\Simulation.sqlite3 --v2-db <staging>\Projection_V2.sqlite3 --no-app-sync
```

Run the template handoff audit and strict rolling weekly replays with the same
league-specific V2 paths. Promote only after SQLite integrity, quarantine,
model, handoff, cross-league scoring, idempotence, and app-copy gates pass.
See `docs/runbooks/best_ball_weekly_build.md`.

## Durable outputs

- `results/findings.md`: correction rationale, accepted evidence, and
  publication evidence.
- `results/run_metadata.json`: staged build/model lineage.
- `results/validation_metrics.csv`: compact current and following-season
  validation results.
- `results/artifact_audit.csv`: quarantine and artifact-count evidence.
