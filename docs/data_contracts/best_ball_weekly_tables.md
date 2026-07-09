# Best-Ball Weekly Tables Contract

Last updated: 2026-07-07

## Owner

`Scripts/Modeling/s4_Best_Ball_Weekly.py` builds the best-ball weekly tables in
the `Simulation` database.

## Consumer

`Fantasy_Football_Snake/app/zSim_Helper.py` reads these tables from the app copy
of `Simulation.sqlite3` for best-ball drafting.
`Fantasy_Football_App/app/zSim_Helper.py` also reads them for managed auction
weekly roster scoring.

## Tables

### `Best_Ball_Weekly_Templates`

Historical player-season weekly profile rows.

Important columns:
- identity: `league`, `template_id`, `template_local_id`, `player`, `pos`,
  `team`, `season`
- projection context: `avg_proj_points`, `preseason_proj_ppg`,
  `historical_pred_fp_per_game`, `projection_rank_pct`,
  `projection_decile`, `projection_tier`
- player context: `avg_pick`, `year_exp`, `year_exp_bucket`, `exp_bucket`
- QB context: `qb_team_rank`, `qb_team_rank_bucket`
- quality/context: `active_games`, `active_ppg`, `season_points`,
  `active_ppg_resid`, `profile_total`
- weekly multipliers: `week_1` through `week_16`

### `Best_Ball_Weekly_Template_Pools`

Mapping from each current player pool to selected historical templates.

Important columns:
- `template_pool_key`
- `template_id`
- `league`, `template_league`
- `pool_player`, `pool_year`, `pool_version`, `pool_dataset`, `pos`
- `template_distance`
- `match_rank`
- `template_sample_weight`
- `template_sample_prob`
- target/current matching features and template feature values

The Snake app should sample templates using `template_sample_prob` when present.
The current intent is broad pool coverage with closer matches sampled more often,
not hard selection of only the top match.

### `Best_Ball_Weekly_Player_Map`

Current prediction rows enriched with template-pool context.

Important columns:
- identity: `player`, `pos`, `year`, `version`, `dataset`, `team`
- projection: `pred_fp_per_game`, residual quantile columns prefixed
  `pred_resid_`
- context: `current_avg_proj_points`, `avg_pick`, `year_exp`,
  `year_exp_bucket`, `exp_bucket`
- pool: `template_pool_key`, `template_pool_level`, `template_pool_size`

### Audit Tables

- `Best_Ball_Weekly_Template_Audit`: historical template quality checks.
- `Best_Ball_Weekly_Player_Pool_Audit`: current-player pool quality checks.
- `Best_Ball_Weekly_Bucket_Audit`: current vs historical bucket comparability.
- `Best_Ball_ADP_Audit`: current-player ADP availability/join review.

## Contract Rules

- Keep `template_pool_key` stable across `Player_Map`, `Template_Pools`, and
  app queries.
- Treat `template_id` as unique across league slices in a generated database.
  Use `template_local_id` only for within-league diagnostics.
- Join template pools to templates with both `template_id` and league context
  (`Template_Pools.template_league`/`pool_version` to `Templates.league`) when
  the app or audit logic does not rely on globally offset IDs.
- Preserve `week_1` through `week_16` as profile multipliers, not raw points.
- Preserve `active_ppg_resid` as template active-game PPG minus historical
  predicted PPG.
- Do not remove `template_sample_prob` without updating app sampling logic.
- Rebuilds should replace only the active league/year/dataset slice and preserve
  other league slices already present in the table.
- Treat role-share features as projected fantasy-point shares unless a column
  name explicitly says attempts, targets, or another raw volume unit.
- Update this document when app-consumed columns are renamed, removed, or
  semantically changed.
