# 2025 Beta Auction Historical Replay

## Question

Can the current Auction app review the 2025 Beta draft using causal preseason
projections, salaries, keepers, and weekly-template donors without changing the
live 2026 database?

## Interpretation

This is a current-method rolling-origin replay. Projection and salary training
and calibration stop at 2024, but the reviewed method specification is as of
2026. It is useful for app behavior, roster construction, and hindsight salary
review; it is not a fresh claim about which modeling method would have been
selected before the 2025 draft.

Next-year keeper discovery and optimizer-selection premiums are disabled because
no causal app-ready 2025 publication exists for either signal.

## Run

From the model repository root:

```powershell
.\.venv_ff_312\Scripts\python.exe `
  Scripts/Modeling/build_historical_auction_replay.py `
  --year 2025 --league beta --replace-stage
```

The ignored staged databases are written under `staging/databases/`. The build
copies every SQLite input it consumes and never writes or synchronizes the live
model or app databases.

## App

Point `AUCTION_SIMULATION_DB` at the absolute staged `Simulation.sqlite3` path,
set `FF_CURRENT_SEASON=2025`, and start `Fantasy_Football_App/app/ffapp.py`.
Both predicted and actual 2025 Beta salary contexts should appear.

See [results/summary.md](results/summary.md) for the completed build receipt and
limitations.
