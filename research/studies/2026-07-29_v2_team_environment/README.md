# V2 Team Environment and QB Style

## Question

Do preseason team-environment features improve conditional PPG after the
current full model already knows the player's own projection, market context,
history, room context, and preseason projection trajectory?

The study specifically separates QB passing production from QB rushing value,
because one QB fantasy-PPG feature can have different implications for RBs and
pass catchers.

## Design

The reference is the 31-feature incumbent plus the five-field preseason
projection-trajectory family. Four position controls remain model inputs.

Six prespecified additions are tested:

- `qb_yardage`: QB1 projected passing and rushing yards;
- `qb_tds`: QB1 projected passing and rushing touchdowns;
- `qb_style`: QB1 projected rushing fantasy-point share;
- `team_support`: capped core-skill projection, its within-season team
  percentile, and self-excluded supporting-cast projection;
- `team_rush_scoring`: core-plus-QB1 rushing yards/TDs and non-duplicated
  offensive TDs; and
- `all_environment`: all 11 fields.

The core skill group is capped at two RBs, three WRs, and one TE per team.
Offensive TDs add QB1 passing TDs to core-plus-QB1 rushing TDs; receiving TDs
are not added again. No target-season actual, injury, result, or Vegas field
enters any feature.

Lasso, random forest, deterministic shallow LightGBM, their tree average, and
their equal-third blend use the same 3,696 OOF rows and folds as the trajectory
reference. Primary inference covers the six equal-third blend comparisons.
Position, history depth, projection history, ADP, QB-style, and team-strength
slices are diagnostics.

This is isolated shadow research. Production projections, templates, and
optimizers remain unchanged.

```powershell
python research/studies/2026-07-29_v2_team_environment/run_validation.py --variant qb_yardage
python research/studies/2026-07-29_v2_team_environment/run_validation.py --variant qb_tds
python research/studies/2026-07-29_v2_team_environment/run_validation.py --variant qb_style
python research/studies/2026-07-29_v2_team_environment/run_validation.py --variant team_support
python research/studies/2026-07-29_v2_team_environment/run_validation.py --variant team_rush_scoring
python research/studies/2026-07-29_v2_team_environment/run_validation.py --variant all_environment
python research/studies/2026-07-29_v2_team_environment/run_validation.py --compile
```

Each variant runs in a fresh process and persists its batch before compilation.
This avoids cumulative Windows/joblib worker instability during the 18-model
study and makes the execution resumable.

See [`results/findings.md`](results/findings.md) for the decision readout.
