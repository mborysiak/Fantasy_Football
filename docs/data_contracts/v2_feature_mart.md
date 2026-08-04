# V2 Preseason Feature Mart Contract

## Scope

V2 Milestone 3 builds a leakage-safe feature layer on the Milestone 2
player-season spine. It does not train a projection model and does not replace
production projections, templates, or optimizer inputs.

Run the complete isolated build with:

```powershell
python -m Scripts.V2.build_milestone_3
```

The command rebuilds Milestones 1 and 2, normalizes historical preseason
projection and market values, constructs the reviewed feature mart, and
atomically publishes every table to
`Data/Databases/Projection_V2.sqlite3`.

Scoring-dependent V2 databases remain separate because
`player_season_projection_values` is keyed by player/season/provider rather
than league:

- `Projection_V2.sqlite3`: DK scoring; and
- `Projection_V2_beta.sqlite3`: beta scoring.

The two builds must share the same canonical `player_key` identity and
player-season population, but outcomes, reconstructed provider points,
consensus features, model fits, and scoring hashes are rebuilt independently.
Never run one league into the other league's database or copy a fitted point
center across scoring systems.

## Source Boundaries

Milestone 3 uses two distinct kinds of information:

- same-season preseason projections, ADP, and expert ranks;
- realized NFL history from seasons strictly earlier than the projected
  season.

Same-season NFL actuals are never features. The current source database has
season-aligned snapshots but not a common immutable capture timestamp for every
provider. A source must therefore be known to represent preseason information;
Milestone 3 does not claim a common calendar-day cutoff.

Here, same-season means the governed effective source season. The physical
stored label remains auditable. Known mislabeled snapshots are corrected
before identity resolution, candidate windows, and value aggregation under
the source-season ledger in `v2_identity_outcomes.md`.

Known duplicate or invalid vintages use the separate
`SOURCE_ROW_EXCLUSIONS` quarantine policy. Quarantine is applied to the
physical stored season before any effective-season override or identity join.
The raw source remains unchanged, but excluded rows cannot enter identity,
candidate, normalized-value, feature, or template-key lineage. A foundation
may be reused only when its source-manifest policy hash matches the current
configured quarantine policy.

Mutable historical team fields use the separate
`SOURCE_TEAM_TRUST_POLICIES` ledger. Historical team labels from
`FFA_RawStats`, `FFA_Projections`, and `FantasyPros_Best_Ball_ADP` are
discarded while their projection/market values remain eligible. Normalized
rows inherit team only from unconflicted trusted same-season aliases;
otherwise team and team/room context remain null. An equal-count team tie is
fail-closed, and otherwise-identical value rows that differ only by team are
rejected unless the source-season scope is explicitly governed. Foundation
reuse also requires a matching source-team trust policy receipt.
Recognized `TEAM_MAP` aliases are canonicalized before conflict detection,
provider grouping, alias-context voting, and room construction. The receipt
hash includes both the trust rules and `TEAM_MAP`, so a changed alias mapping
invalidates foundation reuse.

Legacy `Rookie_RB_Stats` and `Rookie_WR_Stats` fields remain excluded. Rookie
coverage comes from same-season expert/market evidence, canonical draft
metadata, age/experience availability, and explicit no-history flags.

## Normalized Long Tables

### `player_season_projection_values`

One row exists per `player_key, season, provider`.

The table retains normalized passing, rushing, and receiving projections,
provider totals, projected games, provider uncertainty, and provider-specific
team/position-room context. Component projections are scored under the
configured league rules when the position's dominant required components are
complete.

The governed provider estimand is
`core_offensive_season_components_v1`: linear season-total passing, rushing,
and receiving points. Beta QB sacks are included because beta assigns them a
nonzero coefficient. DK does not require sacks because its coefficient is
zero. Weekly yardage bonuses are excluded because season totals cannot
identify weekly threshold crossings. Projected fumbles, two-point conversions,
and return TDs also remain excluded until their source quality and league
coefficients are explicitly governed. Never infer a weekly bonus from a
season-yardage total.

Consensus passing, rushing, and receiving point shares preserve the sign of
the configured components whenever their sum is nonzero. This matters for beta
fringe QBs: sack penalties can make a passing component or even the configured
total negative, so valid shares may fall outside `[0, 1]` while still summing
to one. Do not zero, clip, or renormalize those signed shares through a
positive-only denominator. The 2026 beta weekly matcher consumes this signed
contract in production through the exact-lineage
`v2_beta_scoring_matched_preseason` context. It must not fall back to the DK
component representation.

If exactly one required component is absent, the default imputation requires
at least two other providers for the same
`player_key, season, position`. Beta QB sacks are the sole one-donor exception:
FFToday is the only modeled sack source in several historical seasons, with
PFF adding recent coverage. A donor from another position is never valid.
Rows missing multiple required components, or beta QB rows with no valid sack
donor, remain unscored. FantasyData's `fdta_sack` is a defensive statistic and
is not a QB sack projection. Provider-published fantasy-point totals and PPG
remain raw audit fields and are never substituted into configured scoring.
The FFToday QB rows stored under 2018 are not valid sack donors: they match the
provider's 2019 archive and are quarantined while the native 2019 rows remain.

Important fields include:

| Field | Meaning |
|---|---|
| `configured_projected_points` | Points reconstructed from normalized components |
| `configured_points_complete` | Required position components were available |
| `configured_points_imputed_component_count` | Number of required components filled by the guarded cross-provider median policy |
| `configured_points_imputed_components` | Name of the imputed required component |
| `configured_points_imputation_donor_providers` | Sorted provider lineage used for the imputation |
| `configured_points_imputation_donor_count` | Number of same-position donor rows |
| `provider_projected_points` | Configured component score; null when the row cannot be standardized |
| `provider_points_estimand` | Versioned season-component scoring estimand |
| `points_method` | `configured_components`, `configured_components_imputed`, or `insufficient` |
| `provider_points_per_team_game` | Season value divided by the 16/17-game team schedule |
| `provider_points_per_projected_game` | Value divided by provider-projected games when available |
| provider room fields | Same-provider team/position share, rank, gap, and HHI |
| `sources`, `source_tables` | Exact contributing source lineage |
| source-season provenance fields | Stored seasons plus override IDs, reasons, and archive references |

`sources` and `source_tables` describe the original provider row.
The imputation fields separately disclose its cross-provider dependencies.
The scoring hash versions league coefficients; it does not replace the
explicit provider-estimand version.

Only QB, RB, WR, and TE rows survive canonical-position recovery. Kicker and
defense rows from positionless sources cannot enter the feature universe.

### `player_season_market_values`

One row exists per `player_key, season, source`. It retains normalized ADP,
expert rank, and provider position rank plus source table and identity context.

NFFC contributes one modeled ADP row per player-season:
`adp_average_nffc`, the equal-center composite of Best Ball Overall and Best
Ball $25/$50. Raw `NFFC_ADP` contest rows remain candidate/identity evidence
but do not enter `player_season_market_values` as separate model votes.

The canonical `adp_median` first reduces observations to one vote per approved
provider family, then takes the player-season median. Families are MFL through
2024 only, FantasyPros redraft, FantasyPros best ball `AVG`, DraftKings, and
the two-feed NFFC aggregate. The canonical `ADP_Averages` DraftKings row takes
precedence over the legacy `DraftKings_ADP` row when both exist. FFA,
FantasyPoints, ETR, and component FantasyPros best-ball fields remain raw
audit/challenger evidence and do not silently expand the consensus. The active
policy version and SHA-256 are written to the V2 source manifest.

Neither long table contains realized NFL outcomes. These tables are the
rebuildable source for future feature challengers even when those challengers
are not admitted to the reviewed wide mart.

## `player_season_features`

This table has exactly the same `player_key, season, league` population as
`player_season_spine`. It retains the spine's target and provenance fields for
training-table convenience, but target fields are never cataloged or
manifested as model inputs.

The reviewed mart contains exactly 160 features across:

- projection level and role composition;
- provider coverage and disagreement;
- ADP/rank level and projection-versus-market disagreement;
- age, experience, rookie status, and draft capital;
- strictly prior career, last-observed, exact-prior-year, and three-year
  history;
- provider-level and consensus room share/rank/gap/concentration;
- team and projected QB/pass-catcher context;
- controlled experience-relative, teammate-ADP, and team-opportunity research
  challengers;
- controlled projection-shape, component-disagreement, and historical-provider
  research challengers;
- controlled projection-anchored history-gap, availability, recency, and
  opportunity-game reliability challengers;
- controlled preseason projection-trajectory and logged-ADP challengers;
- controlled QB pass/rush, supporting-cast, and team scoring-environment
  challengers; and
- explicit identity, history-depth, and source-availability indicators.

Adding a derived column in code does not add it to the mart. Every published
feature must be listed in the reviewed `FEATURE_MART_FEATURES` contract.

### History semantics

- Every target season has `feature_cutoff_season = season - 1`.
- Career and last-observed features use only outcome rows with
  `outcome.season < target.season`.
- `prior_year_*` means exactly `target.season - 1`; a missing calendar year is
  not forward-filled.
- `last_observed_*` may reach farther back and is paired with
  `seasons_since_observed`.
- Rookies and other no-history players retain the row with
  `has_prior_outcome = 0`; their unavailable history values stay null.
- Prior participation and prior residual fields use only the shifted prior
  spine row, never the current target.

The separate 13-feature `residual_history_gap_challenger_v1` manifest provides
a model-safe alternative to median-imputing absolute historical PPG. Its
current expert anchor uses active-game consensus when available and otherwise
team-game consensus. Prior-year, prior-three-year, and career PPG are expressed
as historical PPG minus that anchor. When the corresponding history is
unavailable, the gap is exactly zero: no history means no adjustment to the
player's own current projection. Availability, log opportunity games, neutral
prior projection residual, and neutral recency remain explicit. Secondary
reliability fields multiply each gap by `games / (games + 8)`.

The original absolute-history fields remain in the 31-feature incumbent until
the gap challenger clears temporal and sparse-history routing validation.

### Projection trajectory semantics

The separate five-feature
`residual_projection_trajectory_challenger_v1` manifest compares preseason
projection evidence with earlier preseason projection evidence. It never uses
realized PPG, games, injury outcomes, or target-season actuals.

- `projection_trajectory_change_1year` is current consensus team-game PPG
  minus exact prior-year consensus team-game PPG.
- `projection_trajectory_change_3year` is current consensus team-game PPG
  minus the 3/2/1 recency-weighted mean of projections from `t-1`, `t-2`, and
  `t-3` that exist.
- Exact-prior availability, prior-three-year count, and prior-projection
  volatility distinguish unavailable history from an observed zero change.
- A player with no prior projection receives zero change and zero
  availability/count rather than a pooled fill.

Team-game PPG is used on both sides because active-game projection fields do
not have consistent historical coverage. The separate
`residual_adp_transform_challenger_v1` manifest publishes `adp_log =
log1p(adp_median)` for model-family-specific comparison with raw ADP.

### Consensus and room semantics

`expert_points_median` and PPG consensus features summarize available
historical providers, not a single preferred provider. Provider counts and
dispersion remain explicit so the later model can shrink or ignore thin
consensus.

Every provider with a configured score may contribute equally to consensus
immediately. Provider-specific PPG columns are more conservative: they remain
null until that provider has three prior projection seasons. This allows
future learning of persistent provider differences without fitting a
provider adjustment from one recent season.

Team and room features use only same-season preseason point estimates. They do
not use realized depth charts, games, targets, or touches from the target
season. Provider-specific room summaries and consensus-room summaries are
both retained as controlled challengers; the correlation audit makes their
near-redundancy visible.

### Team-environment semantics

The 11-feature `residual_team_environment_challenger_v1` manifest separates
QB fantasy value into passing and rushing context and builds compact team
strength measures without Vegas data:

- QB1 passing/rushing yards, passing/rushing TDs, and rushing fantasy-point
  share;
- projected points for a capped core of two RBs, three WRs, and one TE;
- that core total's within-season team percentile;
- core supporting-cast points after subtracting the player when the player is
  part of the core;
- core-plus-QB1 rushing yards and TDs; and
- offensive TDs defined as QB1 passing TDs plus core-plus-QB1 rushing TDs.

QB1 remains the same team's highest projected fantasy-point QB. Receiving TDs
are not added to offensive TDs because they duplicate QB passing TD events.
The fixed core limits reduce sensitivity to how many fringe players a provider
publishes. Projected target aggregates remain excluded because broad historical
coverage begins only in 2024.

### Legacy-inspired challenger semantics

The 12 `residual_legacy_challenger_v1` features reproduce useful concepts from
the legacy feature pipeline without its forward filling or target-informed
global selection:

- Experience context compares expert team-game PPG with self-excluded,
  same-season position/experience peers. Exact experience seasons zero through
  seven remain distinct; eight and above share an `8+` cohort. A singleton
  cohort falls back to self-excluded same-position peers in that season.
- Market-room context compares player ADP with self-excluded same-position
  teammates on the same NFL team. Missing ADP remains null. Room strength share
  uses normalized inverse-square-root ADP so earlier selections carry more
  weight without allowing one top pick to dominate raw inverse ADP.
- Opportunity context divides projected targets, receptions, rush attempts, or
  receiving yards by the same team's preseason projection total. Missing source
  components remain null and are never reconstructed from target-season actuals.

These features are published for controlled research but are not part of the
31-feature incumbent residual manifest.

## Feature Governance Tables

### `feature_catalog`

One row exists per published feature. It records family, dtype, collinearity
group, manifest eligibility, and whether the feature is audit-only.

### `feature_manifests`

The initial manifests are candidate sets, not selected final models:

| Manifest | Features | Purpose |
|---|---:|---|
| `residual_candidate_v1` | 31 | Predict conditional-PPG residual around expert consensus |
| `residual_legacy_challenger_v1` | 12 | Test experience, teammate-market, and opportunity-share additions without changing the incumbent |
| `residual_projection_challenger_v1` | 26 | Test standardized provider, projection-shape, and component-disagreement additions without changing the incumbent |
| `residual_projection_trajectory_challenger_v1` | 5 | Test current-versus-prior preseason projection level, availability, depth, and volatility |
| `residual_adp_transform_challenger_v1` | 1 | Test logged ADP as a replacement for raw ADP |
| `residual_team_environment_challenger_v1` | 11 | Test QB pass/rush context, capped supporting cast, team rushing, and non-duplicated scoring environment |
| `participation_candidate_v1` | 19 | Predict probability/opportunity of appearing |
| `template_challenger_v1` | 12 | Test compact conceptual dimensions for donor matching |

The template manifest assigns one fixed budget per conceptual family. Multiple
features in one family compete inside that budget rather than receiving
independent full-weight votes.

### `feature_audit`

Stores all-row, training-row, and current-season coverage; unique counts; and
zero-variance flags for every cataloged feature.

### `feature_correlations`

Stores within-family Spearman pairs with at least 100 shared rows and
`abs(spearman) >= 0.90`. It is a pruning queue, not an automatic deletion rule.
Rolling-origin ablation remains the authority when correlated features encode
different useful semantics.

### `feature_source_resolution_audit`

Stores eligible input rows, resolved identity rows, and resolution rate by
source table and source kind. Ambiguous positionless rows remain unresolved;
the builder does not force fuzzy matches to improve coverage.

The audit also stores `excluded_rows`,
`source_row_exclusion_ids`, `source_row_exclusion_reasons`, and
`source_row_exclusion_references`. Any nonzero exclusion count requires all
three metadata fields. The build's `source_manifest` publishes a matching
`source_quarantine` receipt with the governed rule ID/reference and excluded
row count.

## Required Invariants

1. Projection values are unique by `player_key, season, provider`.
2. Market values are unique by `player_key, season, source`.
3. Every normalized value row belongs to a Milestone 2 spine player-season.
4. Feature rows exactly match spine keys and counts.
5. The reviewed feature contract contains exactly 160 features.
6. No target or target-availability column appears in a feature manifest.
7. Pending rows expose no participation or production label.
8. NFL-history features use only seasons strictly before the target.
9. Missing prior calendar years are never forward-filled.
10. Every published long-table position is QB, RB, WR, or TE.
11. Feature, spine, and foundation run IDs form an explicit lineage.
12. Template family budgets do not exceed one in total.
13. Every complete beta QB provider row has a non-null sack projection.
14. Every imputed value equals the median of its recorded same-position
    donors; component, donor-provider, and donor-count lineage is complete.
15. Every projection-value row carries exactly one governed provider-estimand
    version.
16. DK and beta share canonical identity, alias, spine, and normalized value
    populations, while configured-score completeness may differ by league.
17. No configured source-row quarantine survives into normalized values or
    features, and every nonzero exclusion audit has rule ID, reason, and
    reference.
18. Reused foundations carry exactly one quarantine-policy receipt whose hash
    matches the current `SOURCE_ROW_EXCLUSIONS` policy.
19. Reused foundations carry exactly one team-trust-policy receipt whose hash
    matches `SOURCE_TEAM_TRUST_POLICIES`; governed historical source-team
    labels never reach aliases, normalized values, or team/room features.

## Current Validation Snapshot

The corrected and fully replayed 2026-07-29 builds contain, per league:

- 6,655 identities and 55,914 aliases, identical between DK and beta;
- 13,909 spine and feature rows;
- 31,798 normalized provider projection rows;
- 28,801 normalized market/rank rows after enforcing one NFFC-family ADP vote;
- 160 cataloged features;
- 31 incumbent residual, 13 history-gap residual challenger, 12
  legacy-inspired residual challenger, 19 participation, 26 projection
  residual challenger, five projection-trajectory challenger, one ADP-transform
  challenger, 11 team-environment challenger, and 12 template features;
- 82 DK and 84 beta high-correlation within-family pairs;
- 745 pending 2026 candidates, all with null outcome labels, including 715 DK
  and 673 beta rows with configured-score expert consensus; and
- 102 current rookies: DK has 97 with expert consensus, beta has 89, and both
  have 79 with ADP and 80 with draft-capital features.

Positioned sources remain effectively complete after quarantine. Positionless
unresolved names remain excluded and auditable. The
`FFToday_Projections` feature-source receipt records 6,308 eligible/resolved
rows and 50 excluded rows under
`fftoday_qb_stored_2018_2019_vintage_quarantine_v1`. No stored-2018 FFToday QB
row remains in aliases or normalized values, while all 50 native 2019 provider
rows remain. There are still zero complete beta QB rows with null sacks. When
the quarantined row was the only apparent 2018 sack donor, the affected beta
QB provider row is now correctly incomplete rather than scored with leaked
2019 evidence or an invented zero. The production weekly build consequently
marks the 39 affected 2018 QB contexts unavailable and donor-ineligible; it
does not substitute a DK context or zero sacks.

The legacy-inspired rolling OOF study did not promote any of the 12 features.
No family materially improved Ridge or shallow LightGBM, every deterministic
season-bootstrap interval crossed zero, and the full 12-feature addition
worsened both models. The features therefore remain a separate research
manifest rather than expanding the incumbent.

The projection-anchored history-gap study confirms that all 3,696 validation
rows receive complete gap/reliability fields and that no-history gaps are
neutral zero. It does not support a global incumbent replacement: raw-gap
Lasso is 0.0018 RMSE worse, while the raw-gap equal-third blend is 0.0029
better overall but worse in 2023-2025 and has a season interval crossing zero.
Rookie and other no-career-history point estimates improve, so the 13 fields
remain a governed sparse-history/router challenger.

The preseason trajectory study finds 84.3% exact prior-year projection
coverage on the 3,696 common OOF rows. Exact current-versus-last-year change
alone improves the equal-third blend by only 0.0016 RMSE. The three-year
context improves it by 0.0041, and the full five-field trajectory family by
0.0051 with a slight 2023-2025 improvement. Logged ADP materially improves
Lasso but is neutral for random forest and slightly worse for LightGBM.
Trajectory and logged ADP remain separate model-family challengers; neither
changes the 31-feature incumbent.

The team-environment study finds 98.0% coverage for all 11 fields on the common
OOF population. The full family is flat globally. QB1 rushing fantasy-point
share is the strongest compact representation, improving the trajectory blend
by 0.0021 RMSE and the 2023-2025 mean by 0.0089, but it wins only five of nine
seasons and does not survive the six-family correction. Its effect is isolated
to WR/TE rather than RB. Retain it as a pass-catcher whole-season/template
finalist; do not expand the global point-model manifest.

## Modeling Boundary

Milestone 4A is implemented in
`docs/data_contracts/v2_modeling_framework.md`. It:

1. freezes rolling-origin train/validation/test splits;
2. treats expert conditional PPG consensus as the incumbent baseline;
3. tests shrunk residual models against that incumbent out of sample;
4. fits participation separately from conditional production;
5. compares provider-era and history-depth slices, especially rookies and
   second-year players;
6. tests compact transformations and feature sets rather than relying on global
   correlation alone; and
7. publishes shadow OOF evidence without changing point estimates or
   weekly-template sampling.

Template integration remains a later milestone. The existing joint
residual/weekly-path templates should not be combined with an independent
second residual draw.
