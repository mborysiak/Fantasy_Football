# Session Notes Landing

Last updated: 2026-07-09

## Project Objective

Maintain and improve the fantasy football modeling pipeline that builds player
projections, residual distributions, simulation inputs, and best-ball weekly
template tables consumed by downstream draft apps.

## Current Focus

- Current active workstream: best-ball weekly template generation and Snake app
  integration.
- The modeling repo owns the source `Simulation.sqlite3` and copies it to the app
  repos when `Scripts/Modeling/s4_Best_Ball_Weekly.py` runs.
- Recent best-ball changes added multi-feature template matching, weighted
  template sampling, non-QB zero-active filtering, template residual context,
  ADP audit support, league-aware table slices, and app-side ILP/runtime
  updates.
- Auction salary predictions now expose bootstrap-averaged empirical residual
  quantiles in `Salaries_Pred`; `Fantasy_Football_App` auction ILP sampling
  now uses residual quantiles for both projections and salaries.
- Fantasy_Football_App has a managed-league weekly auction ILP that uses weekly
  templates, lineup decision scores, waiver baselines, candidate marginal
  managed value, and an exact one-swap roster refinement pass.

## Recent Durable Decisions

- Template matching now uses projection strength plus position-specific role and
  team context, not only projected-points buckets.
- RB rush/rec room shares and WR/TE receiving shares are based on projected
  fantasy points, not raw attempts alone.
- Non-QB zero-active historical templates are excluded from pools; QB pools can
  keep zero-active outcomes for backup/fringe-starter context.
- Template pools expose `template_sample_prob` so apps can use all selected
  templates while favoring closer matches.
- `Best_Ball_ADP_Audit` is the durable place to review missing or fallback ADP
  joins for draftable players.
- Best-ball weekly table rebuilds should replace only the active league slice
  and preserve other league slices already present in `Simulation.sqlite3`.
- Auction salary uncertainty should use bootstrap-averaged historical
  out-of-fold residual quantiles by position and predicted salary, and the
  auction app should sample projections/salaries from residual quantile columns
  instead of truncated-normal or upside/top probability branches.
- Managed auction weekly scoring should value startable weekly lineup points
  above waiver baseline and avoid best-ball-style hindsight lineup decisions.
- Managed auction ILP rosters should receive one exact, vectorized one-player
  swap pass to reduce fixed-base additive roster interaction bias.

## Key Links

- Module tracker: `MODULE_TRACKER.md`
- Decision log: `DECISION_LOG.md`
- Cross-repo context: `CROSS_REPO_CONTEXT.md`
- Best-ball table contract: `../docs/data_contracts/best_ball_weekly_tables.md`
- Best-ball build runbook: `../docs/runbooks/best_ball_weekly_build.md`
- Research index: `../research/README.md`
- Latest chronological log: `Session_Notes/2026-07.md`

## Working Defaults

- Keep numbered scripts notebook-friendly and import-safe where practical.
- Favor surgical changes over broad refactors.
- Treat name cleaning, projection joins, ADP joins, and table schemas as
  first-class risk areas.
- Update app-facing contracts when Simulation tables change.
