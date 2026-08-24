# Best-Ball Weekly Tables Contract

Last updated: 2026-08-13

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
  disagreement, market-vs-projection gap match fields,
  `projection_context_source`, `projection_context_scoring_hash`,
  `projection_context_run_id`, `scoring_context_available`,
  `scoring_context_unavailable_reason`, and the audit-only `model_input_*`
  projection-context columns
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
- auction managed normalization: `managed_profile_ppg`,
  `managed_residual_center_ppg`, `managed_active_ppg_resid`, and
  `managed_center_policy`
- weekly multipliers: `week_1` through the league horizon (`week_16` for
  DK/beta/NV and `week_17` for NFFC in the approved 2026 cycle)
- managed weekly multipliers: `managed_week_1` through the same league horizon
- weekly played evidence: `played_week_1` through the same league horizon

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
  `year_exp_bucket`, `exp_bucket`, and `current_team_source`
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

This boundary is methodologically important because beta, NV, DK, and NFFC have
distinct configured reception, touchdown, sack, and yardage-bonus rules.
Historical `active_ppg`, `season_points`, centered residuals, and weekly
trajectories are all league-scored outcomes. Configured yardage bonuses
therefore flow through the realized weekly-upside path and `active_ppg`; they
are not inferred from season totals. No league build may reuse another
league's scored weekly frame even when source statistics and player identities
are otherwise identical.
These league-scored weekly outcomes feed optimizer selection but are not forced
into the preseason salary point estimate. Two-point conversions and
special-teams touchdowns remain omitted by decision.

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

The promoted source and the then-live DK/beta V2 databases match their staged
artifacts byte-for-byte, and a second handoff leaves all governed table hashes
unchanged across all eight governed tables. All 20 generated auction-app
tables match staging while all six
app-owned tables are unchanged, and every Snake table matches staging. All six
source/app databases pass SQLite integrity with zero foreign-key errors.
`Model_Inputs.sqlite3` was preserved, and pre-promotion copies live under
`research/studies/2026-07-30_canonical_adp_handoff/results/pre_promotion/`.

## Canonical Current Market Contract

`Scripts/V2/production_handoff.py` is the sole owner of the current
`Avg_ADPs` publication. It replaces the current DK, NFFC, and ETR slices from
the explicitly bound `Season_Stats_New.sqlite3` snapshot while preserving other
years/leagues. Staged handoffs must pass that frozen source path explicitly.
The legacy s3 workflow calls this publisher; it must not append a name-only
table.

The current grain is `(year, league, draft_entity_key)`. Every QB/RB/WR/TE row
has a unique, non-null governed `player_key`, and consumers join that key
directly without re-resolving the display label. Source position must agree
with the current `player_season_features.position`, falling back to
`player_identity.position` only when current-season position is unavailable;
the authority and source are retained on each row. This permits governed
current roles such as Travis Hunter at WR while still auditing his DB identity
position. NFFC `TK` and `TDSP` rows are draft units rather than asserted NFL
player identities: all are retained with a deterministic `draft_entity_key`
and null `player_key`. This includes source labels such as `Ghost` and `Jeff
Holder`; the publication neither drops them nor invents player/team identities
for them.

For offensive rows, `team` is the canonical `player_identity.latest_team`.
The weekly context builder may use that same keyed `Avg_ADPs.team` only when
the compiled Model Inputs or V2 feature row has no assigned current team; it
never overrides an already assigned team. `Best_Ball_Weekly_Player_Map` stores
`current_team_source` as `model_inputs`, `v2_player_season_features`,
`canonical_avg_adps`, or `unassigned` so this fallback remains auditable.

ETR rows preserve exact source `etr_rank` and `etr_pos_rank`; `avg_pick` equals
`etr_rank` only for backward-compatible overall ordering. The source
`etr_adp`, `etr_adp_pos_rank`, and `etr_adp_diff` fields are also retained.
`Avg_ADPs_Publication_Audit` stores row-level source labels, match methods, and
row/snapshot digests. `Avg_ADPs_Publication_Receipt` records source/published
row counts, SHA-256 digests, version, snapshot ID, and UTC publication time for
each feed. An unchanged source snapshot reuses its prior publication time.
Canonicalization removes legacy DK/NFFC/ETR rows whose year is null,
non-numeric, non-finite, or fractional while preserving valid historical
years and unrelated leagues; the per-feed removal count is retained in both
the row audit and receipt. The initial migration removes 476 duplicated
year-null ETR rows. Raw `ETR_Ranks` remains the source authority, so generated
`Avg_ADPs` junk is not retained as historical evidence.

NFFC `avg_pick`, `min_pick`, and `max_pick` are equal averages of Best Ball
Overall and Best Ball $25/$50 where both contain the player. `std_dev` pools
each feed's `(max_pick-min_pick)/5` within-feed spread with between-feed ADP
disagreement. A fringe player present in only one of the two feeds retains
that feed's center, bounds, and within-feed spread with `source_count=1` and a
null `feed_gap`; zero is reserved for observed agreement between two feeds.
`source_count`, `feed_gap`, aggregation/bounds/SD policy labels, and the ADP
policy version remain on source `ADP_Averages`; the keyed publication preserves
the runtime values. DraftKings keeps its direct API center, scales these NFFC
bounds and pooled SD to that center, and labels the distribution as synthetic
in the source provenance.

Publication fails closed on unresolved/duplicate offensive
keys, missing draft-entity keys, invalid ranks/picks, unsupported positions,
lost source rows, or insufficient offensive depth. All three tables are
generated/source-owned tables copied to Auction; Snake receives the full
source database.

## Approved 2026 NFFC Offense-Only Candidate

The approved 2026 production cycle adds an independently scored NFFC V2,
handoff, weekly-template, and Snake-app path. NFFC production eligibility is
the union of the core projection population and enough canonical NFFC ADP rows
after filtering to QB/RB/WR/TE to cover all 360 draft slots after reviewed
protected-market exclusions. Every projection and weekly-map row still
requires a canonical `player_key`. The `TK` and `TDSP` rows remain in
`Avg_ADPs` as audited draft entities but do not enter the model population,
template pools, or Snake player pool.

The canonical ADP table remains complete even when a fringe market-only player
cannot produce a valid current or next-year V2 center. Core players and keepers
always fail closed. New market-only gaps also fail closed through the first
five-sixths of the draft surface (`200` DK, `300` NFFC, and `150` beta/NV) unless
separately accepted in the annual explicit-exclusion review. An incomplete row
beyond that protected pick depth may be omitted from projections and weekly maps only
under exclusion policy `v2_market_only_incomplete_buffer_exclusion_v3`; the row
and reason remain in `V2_Production_Eligibility_Audit`, and the handoff still
requires at least `240`/`360`/`180`/`180` complete players for
DK/NFFC/beta/NV. Legacy
current or next-year projections must not fill the missing V2 center.

NFFC uses a 17-week template horizon and modern 2021-forward historical donors.
Its `week_17`, `managed_week_17`, and `played_week_17` values must be populated;
those columns are null for the 16-week DK, beta, and NV slices and consumers drop
only a wholly null trailing horizon for the selected league.

NFFC historical and current scoring-sensitive matcher context is authoritative
from the NFFC-scored V2 `player_season_features` preseason consensus. This
includes total points and team-game PPG, projection uncertainty, pass/rush/
receiving point shares, and team-QB context. The builder reconstructs component
points on that same scoring scale and fails closed on missing, inconsistent, or
wrong-position context. The older `Model_Inputs.avg_proj_points` context is
DK-scored and remains audit-only for NFFC; it cannot fill or override an NFFC
matcher field.

For WR/TE environment matching, `qb_avg_proj_pass_points` is the selected
team QB1's NFFC-scored passing component
(`expert_points_median * projected_pass_point_share`). Total QB fantasy PPG is
retained as `team_qb1_ppg` context but must not be relabeled as passing points;
doing so would overstate mobile-quarterback passing environments.

The NFFC historical donor center is the scoring-matched preseason expert
consensus, with
`historical_center_policy = 'nffc_scored_expert_consensus'`. The strict-OOS
locked NFFC center remains a diagnostic and is not promoted. A strict
2023-2025 rolling replay on 540 targets found locked-minus-expert PPG CRPS of
`+0.002901`; the locked arm lost all three seasons and its player-cluster 95%
interval was `[-0.004914, +0.010748]`. It passed six of ten safety gates but
failed all three promotion gates. This NFFC-only decision does not change the
separately governed DK or beta historical-center contracts.

The live NFFC build contains 1,509 historical templates from 2021-2025, all
with 17 populated weeks, and a 385-player current map.

This is not a complete NFFC contest implementation. The Snake NFFC selector is
an offense-only 3RR mode; it does not project or draft K/DST and does not cover
alternate NFFC contest formats. The governed adapter is live after the full
season-registered refresh, NFFC model-acceptance and template gates, both app
smokes, and explicit promotion.

## Registered 2026 NV Auction Candidate

The approved 2026 cycle registers NV as a fourth independent V2/handoff/weekly
objective for the Auction app. NV scoring is identical to beta for rushing,
receiving, interceptions, sacks, fumbles, and yardage bonuses; passing
touchdowns are four points in NV and five in beta. NV must therefore be scored
and modeled from its own `Projection_V2_nv.sqlite3` database and cannot reuse or
relabel beta weekly rows.

NV eligibility is the union of the core projection population, the first 180
canonical ETR overall ranks, and all governed NV keepers. The active 2026
keeper slice contains 16 unique canonical keys and is a required hash-bound
refresh input. NV uses a 16-week
horizon, 2008-forward donor history, exact
`v2_nv_scoring_matched_preseason` context, and
`nv_scored_expert_consensus` as the authoritative center for context-available
historical donors. The quarantined 2018 QB rows without sack-aware preseason
context retain an audit-only preseason fallback and are donor-ineligible.

The production gate requires at least 300 NV projection rows, including
40 QB/75 RB/105 WR/38 TE, at least 180 complete auction-market rows, exact
projection/weekly/salary key parity, 80 donors per player, and a fully populated
16-week map. The active promoted surface has 324 projections, weekly maps, and
salary rows and passed the complete staged refresh, app smoke, review, and
explicit-promotion gates.

## Current Live 2026 Population and Context Contract

The current production population is selected before weekly context and donor
construction:

- DK contains 343 players: 51 QB, 101 RB, 140 WR, and 51 TE. The governed
  eligibility audit preserves every reviewed core, market-tail, and exclusion
  decision.
- NFFC contains 385 players: 59 QB, 110 RB, 156 WR, and 60 TE.
- Beta contains 324 players: 50 QB, 94 RB, 130 WR, and 50 TE. It is the union
  of the core population, top-180 ETR overall-rank population, and all governed
  keepers.
- NV contains 324 players with the same position counts as beta and an
  independently scored V2/weekly lineage.

Current and following-season centers come from the locked league-specific V2
shadows. Legacy current/next values are retained only for audit. DK donor
residuals use the validated legacy OOS historical center where available and
the DK preseason projection fallback otherwise. Beta keeps the validated
legacy OOS center where available but uses the beta-scored expert consensus
for fallback rows; its matcher fields come from the beta-scored V2 preseason
context. NFFC uses the independently validated
`nffc_scored_expert_consensus` rule described above.

Beta eligibility remains ordered by ETR overall rank, but ETR does not define
the current template-match ADP. Beta matcher `avg_pick` uses the same governed
family-level `adp_median` as V2 training, falling back to the rebuilt canonical
Model Inputs value and only then to ETR rank when a player lacks consensus
coverage. This keeps the residual/upside comparison scale aligned between
historical donors and the current player without changing the beta app's
eligibility rule.

`V2_Production_Eligibility_Audit` retains the full reviewed eligibility union
for every league build, including governed exclusions and non-selected rows.

Every live player requires a canonical `player_key` before context joins and
receives exactly 80 historical donors. The August 23 release has zero generic
ADP defaults, high-impact unresolved rows, or review routes. Fresh providers
omit Jayden Higgins' current projection center while DK/NFFC ADP retains him
inside protected depth; he remains in market and eligibility audits but is
explicitly excluded from DK/NFFC handoff under
`market_only_without_current_projection_center`, without a legacy fill.

The current four-league surface was promoted by governed refresh
`20260823T162821Z_0490d19d`. All release gates, the 1,000/1,000 reserve trial,
and both app smokes passed; post-promotion hashes match the manifest, all three
live Simulation copies pass SQLite integrity, and Snake/Auction content parity
checks pass.

Team aliases are normalized only while calculating position-room features:
`LA`/`LAR` and `ARZ`/`ARI` share their corresponding room. Outward player/team
labels are preserved. Free agents retain their outward `FA` label and receive
zero room features rather than being grouped into a synthetic team room.

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
- Preserve `week_1` through the registered league horizon as profile
  multipliers, not raw points. The approved 2026 horizons are 16 for DK/beta/NV
  and 17 for NFFC.
- Preserve `managed_week_1` through the registered league horizon as the
  auction app's managed-season profile multipliers. They match `week_*` for
  workload-qualified outcomes and additionally retain scores from short QB
  appearances. The Snake best-ball app continues to use only `week_*`.
- Persist the auction-only managed normalization contract alongside those
  paths: `managed_profile_ppg` is the path denominator,
  `managed_residual_center_ppg` is the conditional center,
  `managed_active_ppg_resid` is `active_ppg` minus that center, and
  `managed_center_policy` records whether the V2 conditional center or the
  governed historical fallback was available. When a positive V2 center is
  available, it must be used for both the managed residual and any nonpositive-
  active-PPG fallback path. Keep the legacy `active_ppg_resid` unchanged for
  Snake.
- Preserve `played_week_1` through the registered league horizon as separate
  0/1 masks.
  A value of `1` means the source weekly play-by-play table contained a
  player-week row; it must not be inferred from the fantasy-point value. For
  QBs, this evidence is captured before the existing greater-than-15-play
  performance-profile filter, so short and injury-truncated appearances remain
  played weeks even though they do not affect the template's `active_ppg`.
  This is strong participation evidence for a target, carry, or QB play, but it
  is not a comprehensive snap-count or pre-kickoff game-status flag.
- For every template, the sum of `played_week_*` through that slice's
  registered horizon must equal `played_games`. `active_games` remains the
  performance-profile denominator and can be lower for QBs because of the
  greater-than-15-play filter. Played zero and negative fantasy-point outcomes
  remain valid outcomes rather than being reclassified as missed games.
- A short QB week has `played_week_N = 1`, retains the existing filtered
  `week_N = 0` best-ball multiplier, and stores its unfiltered score relative to
  `managed_profile_ppg` in `managed_week_N`. This preserves the managed outcome
  without injecting small-workload QB games into the best-ball profile. Builds
  and release validation must reject non-finite managed rows, managed totals or
  individual multipliers above the league horizon plus the governed `0.5`
  tolerance, and any row whose persisted center, denominator, residual, or
  policy is internally inconsistent.
- Preserve `active_ppg_resid` as template active-game PPG minus historical
  predicted PPG.
- DK production donor residuals use the previously validated historical OOS
  projection where it exists (`historical_center_policy =
  legacy_validated_oos`) and the same-season DK preseason projection otherwise
  (`historical_center_policy = preseason_projection_fallback`).
- Beta production uses exact `v2_beta_scoring_matched_preseason` matcher
  context. Donor residuals retain `legacy_validated_oos` for 2,696 rows and use
  the beta-scored expert consensus fallback for 2,602 rows
  (`historical_center_policy = beta_scored_expert_fallback`). The strict-OOS V2
  center remains audit-only for Snake's legacy `active_ppg_resid`, and
  `v2_recenter_promoted = 0`; the auction-only managed residual contract uses
  the conditional V2 center without changing that Snake decision.
- Strict rolling validation did not establish a predictive promotion: the full
  beta arm passed all player gates but worsened development roster CRPS by
  `+0.9061%` against the `0.5%` limit; 2023-2025 worsened `+0.3790%`. The
  2026-08-03 promotion is an explicit data-correctness override that removes
  mixed DK/beta units. Do not describe it as a performance improvement or
  retune matcher weights on this evidence.
- NFFC production candidates use the NFFC-scored preseason expert consensus for
  both matcher context and the donor center
  (`historical_center_policy = nffc_scored_expert_consensus`). The locked OOF
  center remains diagnostic after failing its prespecified replay gates.
- NV production candidates use exact `v2_nv_scoring_matched_preseason` matcher
  context and `nv_scored_expert_consensus` donor centers for every available
  row. The 2018 QB audit exception remains a preseason fallback and is never
  template-eligible.
- A missing beta or NV scoring context normally fails the build. The only governed
  exception is an auction-league 2018 QB locked-handoff row with both center and scoring
  context availability set to zero when the active FFToday quarantine receipt
  proves that no valid beta sack donor exists. Keep
  `v2_historical_pred_fp_per_game` null, retain the auditable
  `legacy_validated_oos` center, set `template_eligible = 0`, and record
  `legacy_validated_oos_fallback:fftoday_qb_stored_2018_2019_vintage_quarantine_v1:no_valid_beta_qb_sack_donor`
  in both unavailable-reason fields. A missing locked row, stale policy
  receipt, other season/position/league, or any other unavailable context
  remains an error. The live table contains exactly 39 such 2018 QB rows.
- A template/V2-center position mismatch also fails closed except for three
  reviewed hybrid-role rows: Cordarrelle Patterson's 2019 and 2021 templates
  are WR while the locked center position is RB, and Ty Montgomery's 2022
  template is RB while the locked center position is WR. Persist the locked
  position, mismatch flag, and exact
  `canonical_hybrid_role_shift:<player>` reason. No name-only, position-family,
  or generalized hybrid exception is allowed.
- Keep structurally non-transferable outcomes in the template and audit tables,
  but set `template_eligible = 0` and record a declared reason. The live beta
  slice has 39 scoring-context exclusions plus Le'Veon Bell's 2018 contract
  holdout. Ordinary zero-active and low-active seasons remain eligible because
  they represent real downside.
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
- Template and template-audit rebuilds replace the active league. Pool, map,
  summary, and player-audit rebuilds replace every retained year for the active
  league/dataset, while preserving other leagues. Template IDs are regenerated
  from the current donor bank, so retaining an older player map against a new
  unversioned template table is prohibited. The Snake year selector exposes
  only prediction slices present in the current weekly player map.
- Rebuild beta, NV, DK, and NFFC separately with explicit league arguments. As a
  cross-league audit, paired historical rows should not have identical
  `active_ppg` and weekly paths throughout the full population; complete
  equality indicates a scoring-dictionary routing failure.
- The live source `Simulation.sqlite3` may read only the configured V2 database
  for its active league. A copied/staged Simulation database must use an
  explicit staged V2 copy and `--no-app-sync`; it may not read a live V2
  database. These asymmetric guards prevent staged evidence from reaching live
  output and live evidence from contaminating a staged audit.
- After every governed production build finishes writing Simulation outputs,
  the staged source `Simulation.sqlite3` is vacuumed before validation, app
  copies, and promotion. The compaction receipt must report SQLite integrity
  and foreign keys `ok` with `freelist_count=0`; only that compacted artifact
  may become the live repository database.
- Treat role-share features as projected fantasy-point shares unless a column
  name explicitly says attempts, targets, or another raw volume unit.
- Compute current room shares/ranks/concentration on the complete preseason
  projection universe before pruning to final model predictions.
- For the managed auction consumer, a sampled donor jointly supplies its
  centered `managed_active_ppg_resid` and its `managed_week_*` trajectory.
  Apply both to the current calibrated point forecast; do not draw a second
  independent PPG residual or rescale the historical residual to the
  model-residual spread. Older databases without the managed contract may be
  repaired at load time, but newly generated databases must pass the persisted
  source-owned contract without runtime rescaling.
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
