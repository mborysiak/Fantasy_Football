# Best-Ball Weekly Tables Contract

Last updated: 2026-07-29

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
- identity: `league`, `template_id`, `template_local_id`, canonical
  `player_key`, `player`, `pos`, `team`, `season`, and
  `player_key_match_method`
- projection context: `avg_proj_points`, `preseason_proj_ppg`,
  `historical_pred_fp_per_game`, `projection_rank_pct`,
  `projection_decile`, `projection_tier`, absolute projected PPG, projection
  disagreement, and market-vs-projection gap match fields
- center audit: `legacy_historical_pred_fp_per_game`,
  `v2_historical_pred_fp_per_game`, `v2_point_center_source`,
  `v2_template_center_available`,
  `v2_template_center_unavailable_reason`,
  `v2_template_center_position`,
  `v2_template_center_position_mismatch`,
  `v2_template_center_position_mismatch_reason`,
  `historical_center_policy`, and `v2_recenter_promoted`
- player context: `avg_pick`, uncapped `year_exp`, `source_year_exp`,
  `year_exp_source`, `year_exp_uncapped_delta`, `year_exp_bucket`, `exp_bucket`
- workload context: projected role shares plus within-room rank, gap to the next
  player, and room concentration for RBs and pass catchers
- QB context: `qb_team_rank`, `qb_team_rank_bucket`
- quality/context: `active_games`, `played_games`, `active_ppg`, `season_points`,
  `active_ppg_resid`, `profile_total`, `managed_profile_total`,
  `template_eligible`, `template_exclusion_reason`
- weekly multipliers: `week_1` through `week_16`
- managed weekly multipliers: `managed_week_1` through `managed_week_16`
- weekly played evidence: `played_week_1` through `played_week_16`

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
- `season`, `template_season_gap`, `template_recency_multiplier`
- `template_sample_prob`
- target/current matching features and template feature values

The Snake app should sample templates using `template_sample_prob` when present.
The probability uses a position-specific distance kernel, blends back toward
uniform when even the nearest donor is not locally close, applies a
`0.5 ** (template_season_gap / 12)` recency multiplier, renormalizes, and caps
any donor at 5%. Pool summaries persist kernel bandwidth, local-weight
fraction, the 12-season recency half-life, weighted donor-season gap, recent
and old donor weight, effective sample size, and probability range. The current
intent is broad pool coverage with meaningfully closer and more recent matches
sampled more often, not hard selection of only the top match.

### `Best_Ball_Weekly_Player_Map`

Current prediction rows enriched with template-pool context.

Important columns:
- identity: canonical `player_key`, `player`, `pos`, `year`, `version`,
  `dataset`, `team`, and `player_key_match_method`
- projection: current `pred_fp_per_game`, conditional next-year
  `pred_fp_per_game_ny`, `pred_appear_current`, `pred_appear_ny`, and residual
  quantile columns prefixed `pred_resid_`
- V2 provenance: current/next model version, scoring hash, handoff version,
  projection/uncertainty sources, and
  `independent_current_residual_draw_allowed`
- context: `current_avg_proj_points`, `avg_pick`, uncapped `year_exp`,
  `source_year_exp`, `year_exp_source`, `year_exp_uncapped_delta`,
  `year_exp_bucket`, `exp_bucket`
- pool: `template_pool_key`, `template_pool_level`, `template_pool_size`

### Audit Tables

- `Best_Ball_Weekly_Template_Audit`: historical template quality checks,
  including mask/count agreement, `played_only_games`, active-evidence subset,
  and non-QB active/played equality.
- `Best_Ball_Weekly_Player_Pool_Audit`: current-player pool quality checks.
- `Best_Ball_Weekly_Bucket_Audit`: current vs historical bucket comparability.
- `Best_Ball_ADP_Audit`: current-player ADP availability/join review.

## League-Scoring Boundary

Historical weekly points are scored explicitly for the requested league before
they become templates. `load_weekly_points(..., league=...)` passes that league
to the QB, RB, WR, and TE scoring paths and attaches a transient
`scoring_league` marker. `build_weekly_templates(..., league=...)` derives the
template league from that marker and fails if the requested league differs.
The marker is an in-memory build guard; the durable template row stores the
result as `league`, and its globally offset `template_id` uses the same league.

This boundary is methodologically important because beta and DK use different
reception, touchdown, sack, and yardage-bonus rules. Historical `active_ppg`,
`season_points`, centered residuals, and weekly trajectories are all
league-scored outcomes. Yardage bonuses therefore flow through the realized
weekly-upside path and `active_ppg`; they are not inferred from season totals.
A beta build must never reuse a DK-scored weekly frame even when the source
statistics and player identities are otherwise identical.

## Validated V2 Identity/Scoring Refresh

The 2026-07-29 V2 correction was republished without changing required consumer
fields or sampling semantics; additive audit columns were added. A follow-up
correction binds weekly scoring explicitly to the requested league; the earlier
beta template slice had been labeled beta after receiving the default DK
scoring dictionary. Both league slices must therefore be rebuilt from weekly
source statistics rather than relabeled or copied.

The upstream V2 refresh also quarantines the 50 FFToday QB rows stored as 2018
that match the provider's 2019 archive. Native 2019 rows remain. This changes
historical beta QB provider completeness where the quarantined rows were the
only apparent sack donor; it does not authorize a zero-sack fill. See
`v2_identity_outcomes.md` and `v2_feature_mart.md`.

These are source, scoring, and lineage corrections. Consumer joins and runtime
sampling semantics are unchanged. In the production rebuild, 5,120 of 5,298
paired historical `active_ppg` values and 5,147 normalized weekly paths differ
between beta and DK. Mean absolute PPG difference is 1.3508 and the maximum is
6.8.

The promoted source and both V2 databases match their staged artifacts
byte-for-byte. All 11 generated auction-app tables match source row-for-row and
hash-for-hash without changing app-owned content, and the Snake database
SHA-256 equals the source Simulation database. All five source/app databases
pass SQLite integrity with zero foreign-key errors.

## Contract Rules

- Keep `template_pool_key` stable across `Player_Map`, `Template_Pools`, and
  app queries.
- `player_key` is the permanent V2 identity for historical templates and
  current player-map rows. It is required and non-null. Resolve it from the
  governed V2 alias/career-window tables, prioritizing a unique confirmed
  identity and using team only for true same-name collisions. Pre-play rookies
  retain their stable provisional key. Never replace this contract with a
  fuzzy display-name join.
- `player_key_match_method` records the exact canonical resolution route.
  Generated app exports must fail closed if either handoff table lacks
  `player_key` or contains a null key.
- Treat `template_id` as unique across league slices in a generated database.
  Use `template_local_id` only for within-league diagnostics.
- Derive the durable template `league` and league ID offset from the weekly
  frame's `scoring_league` marker. Reject mixed markers and any explicit
  requested-league mismatch.
- Join template pools to templates with both `template_id` and league context
  (`Template_Pools.template_league`/`pool_version` to `Templates.league`) when
  the app or audit logic does not rely on globally offset IDs.
- Preserve `week_1` through `week_16` as profile multipliers, not raw points.
- Preserve `managed_week_1` through `managed_week_16` as the auction app's
  managed-season profile multipliers. They match `week_*` for workload-qualified
  outcomes and additionally retain scores from short QB appearances. The Snake
  best-ball app continues to use only `week_*`.
- Preserve `played_week_1` through `played_week_16` as separate 0/1 masks.
  A value of `1` means the source weekly play-by-play table contained a
  player-week row; it must not be inferred from the fantasy-point value. For
  QBs, this evidence is captured before the existing greater-than-15-play
  performance-profile filter, so short and injury-truncated appearances remain
  played weeks even though they do not affect the template's `active_ppg`.
  This is strong participation evidence for a target, carry, or QB play, but it
  is not a comprehensive snap-count or pre-kickoff game-status flag.
- For every template, the sum of `played_week_1` through `played_week_16` must
  equal `played_games`. `active_games` remains the performance-profile
  denominator and can be lower for QBs because of the greater-than-15-play
  filter. Played zero and negative fantasy-point outcomes remain valid outcomes
  rather than being reclassified as missed games.
- A short QB week has `played_week_N = 1`, retains the existing filtered
  `week_N = 0` best-ball multiplier, and stores its unfiltered score relative to
  the template PPG denominator in `managed_week_N`. This preserves the managed
  outcome without injecting small-workload QB games into the best-ball profile.
- Preserve `active_ppg_resid` as template active-game PPG minus historical
  predicted PPG.
- Production donor residuals remain centered on the previously validated
  historical OOS projection (`historical_center_policy =
  legacy_validated_oos`). The strict-OOS V2 donor center is retained in
  `v2_historical_pred_fp_per_game` for audit, but
  `v2_recenter_promoted = 0`. A strict rolling replay on 2017-2025 rows found
  that V2 recentering worsened PPG CRPS in both DK and beta; do not switch the
  active center without clearing that replay again.
- A missing V2 diagnostic center normally fails the build. The only governed
  exception is a beta 2018 QB locked-handoff row with
  `template_center_available = 0` when the active FFToday quarantine receipt
  proves that no valid beta sack donor exists. Keep
  `v2_historical_pred_fp_per_game` null, retain the complete
  `legacy_validated_oos` active center, and record
  `legacy_validated_oos_fallback:fftoday_qb_stored_2018_2019_vintage_quarantine_v1:no_valid_beta_qb_sack_donor`
  in `v2_template_center_unavailable_reason`. A missing locked row, stale
  policy receipt, other season/position/league, or any other unavailable center
  remains an error.
- A template/V2-center position mismatch also fails closed except for three
  reviewed hybrid-role rows: Cordarrelle Patterson's 2019 and 2021 templates
  are WR while the locked center position is RB, and Ty Montgomery's 2022
  template is RB while the locked center position is WR. Persist the locked
  position, mismatch flag, and exact
  `canonical_hybrid_role_shift:<player>` reason. No name-only, position-family,
  or generalized hybrid exception is allowed.
- Keep structurally non-transferable outcomes in the template and audit tables,
  but set `template_eligible = 0` and record a declared reason. Le'Veon Bell's
  2018 contract holdout is currently the only exclusion. Ordinary zero-active
  and low-active seasons remain eligible because they represent real downside.
- `year_exp` is template-matching tenure reconstructed from the most recent
  plausible draft year at or before the row season, preferring matching draft
  team and using first recorded season only when draft identity is unavailable.
  Known overlapping same-name careers use explicit team/career overrides.
  `source_year_exp` retains the compiled model-input value for audit; it may be
  capped and must not replace `year_exp` for template matching.
- `year_exp_scaled` is `year_exp / 10` with no upper clip. Values above ten
  therefore continue to separate template distances instead of collapsing into
  one veteran endpoint.
- `projection_x_exp` remains persisted for backward-compatible diagnostics but
  is not an active match-distance feature. Direct projected PPG and uncapped
  experience remain independently weighted.
- Apply the fixed 12-season recency prior only after the adaptive
  distance/local-uniform probability is formed. Donors must precede the target
  year; recency changes their sampling prevalence without excluding older
  archetypes. Renormalize and reapply the 5% donor cap afterward.
- Do not remove `template_sample_prob` without updating app sampling logic.
- Rebuilds should replace only the active league/year/dataset slice and preserve
  other league slices already present in the table.
- Rebuild beta and DK separately with explicit league arguments. As a
  cross-league audit, paired historical rows should not have identical
  `active_ppg` and weekly paths throughout the full population; complete
  equality indicates a scoring-dictionary routing failure.
- The live source `Simulation.sqlite3` may read only the configured V2 database
  for its active league. A copied/staged Simulation database must use an
  explicit staged V2 copy and `--no-app-sync`; it may not read a live V2
  database. These asymmetric guards prevent staged evidence from reaching live
  output and live evidence from contaminating a staged audit.
- Treat role-share features as projected fantasy-point shares unless a column
  name explicitly says attempts, targets, or another raw volume unit.
- Compute current room shares/ranks/concentration on the complete preseason
  projection universe before pruning to final model predictions.
- For the managed auction consumer, a sampled donor jointly supplies its
  centered `active_ppg_resid` and its `managed_week_*` trajectory. Apply both to
  the current calibrated point forecast; do not draw a second independent PPG
  residual or rescale the historical residual to the model-residual spread.
- In the V2 production handoff, current residual quantiles are deliberately
  zero, `independent_current_residual_draw_allowed = 0`, and
  `current_uncertainty_source = joint_weekly_template_only`. Snake and auction
  consumers must use the centered donor residual directly with the same donor
  path; scaling it to the zeroed legacy spread collapses variance.
- `pred_fp_per_game_ny` is conditional on a following-season appearance.
  Consumers must draw its conditional residual and then apply a separate
  Bernoulli draw from `pred_appear_ny`. A no-appearance draw has zero future
  market/keeper value and must not be resurrected by a weekly residual.
- Update this document when app-consumed columns are renamed, removed, or
  semantically changed.
