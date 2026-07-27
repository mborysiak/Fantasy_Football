# Keeper Option Bonus Replay

This study tests a bench-only keeper option alongside current-season managed
roster value. It uses the league's confirmed rules:

- two keepers per team;
- keeper salary rises by `$10` for each season kept; and
- a player can be kept for at most three future seasons.

At every frozen preseason origin, the study fits separate monotone QB/RB/WR/TE
curves from current projected PPG to the sampled current Market `$`. Frozen
next-year PPG draws are transformed through those curves. For a player bought
at price `p`, a draw with future market value `v` has three-year option payoff:

```text
max(v - (p + 10), 0)
+ max(v - (p + 20), 0)
+ max(v - (p + 30), 0)
```

This is a persistent-hit proxy: the same next-year value draw is treated as the
player's established value during the possible keeper term. First-year keeper
surplus is reported separately and is the primary realized keeper outcome.
Later realized seasons are supplementary because only older origins have two
or three observable future seasons.

Only non-QBs below the league positional impact threshold and with a genuine
origin-frozen next-year forecast can receive an option bonus. The augmented ILP
can activate at most two keeper options. Current-year forecast evaluation keeps
the waiver baseline unchanged and never includes the keeper bonus as fantasy
points.

Policies:

- `current_bench025`: current weekly-P90 bench heuristic and no keeper bonus.
- `bench0`: no weekly-P90 bench heuristic and no keeper bonus.
- `keeper_engine0`: no bench or keeper bonus, but the same shortlist/exact-swap
  refiner used by the keeper policies. This is the clean optimizer-engine
  control.
- `keeper_tiebreak`, `keeper_0p01`, `keeper_1p0`, and `keeper_10p0`: `bench0`
  plus weights of `0.0001`, `0.01`, `1`, and `10` current-season objective
  points per expected future keeper-surplus dollar. The wide grid tests whether
  keeper value acts only as a free bench tie-break or eventually sacrifices
  current lineup value.

Run the mechanics check:

```powershell
python research/studies/2026-07-19_keeper_option_bonus/verify_keeper_mechanics.py
```

Run a smoke replay:

```powershell
python research/studies/2026-07-19_keeper_option_bonus/run_replay.py `
  --years 2024 --trials 4 --contexts 20 `
  --projection-draws 200 --salary-draws 500 `
  --output-dir research/studies/2026-07-19_keeper_option_bonus/artifacts/local/smoke
```

Run the full replay:

```powershell
python research/studies/2026-07-19_keeper_option_bonus/run_replay.py
```

Interrupted full runs can be continued without repeating complete year-policy
blocks:

```powershell
python research/studies/2026-07-19_keeper_option_bonus/run_replay.py --resume
```

Build the same-engine effect and keeper-selection concentration tables:

```powershell
python research/studies/2026-07-19_keeper_option_bonus/analyze_results.py
```

Durable outputs belong in `results/`. Local smoke outputs are ignored.
