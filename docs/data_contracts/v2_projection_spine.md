# V2 Player-Season Projection Spine Contract

## Scope

V2 Milestone 2 creates the population on which participation and conditional
fantasy-point models will be trained. It does not train a model or publish
production projections.

Run the complete foundation and spine build with:

```powershell
python -m Scripts.V2.build_milestone_2
```

The command rebuilds Milestone 1 first, then atomically publishes the
Milestone 2 tables in `Data/Databases/Projection_V2.sqlite3`. Rebuilding
Milestone 1 by itself removes the downstream spine so stale player keys cannot
be consumed accidentally.

## Candidate Rule

A player-season enters the spine only when the player was present in at least
one allowlisted source for that same projected season:

- preseason statistical projections;
- preseason ADP or market tables;
- preseason expert rankings; or
- the NFL draft table for a rookie's draft season.

Observed NFL stats are never used to create a candidate row. A player who
appeared during a season but had no allowlisted preseason evidence therefore
remains in `player_season_outcomes` but not in the projection spine.

The legacy `Rookie_RB_Stats` and `Rookie_WR_Stats` tables remain identity
evidence only. Their college-stat fields do not establish eligibility and do
not enter the V2 spine.

## `player_season_sources`

This is the auditable long-form candidate evidence table. It contains one row
per `player_key, season, source`.

| Column | Meaning |
|---|---|
| `player_key`, `season` | Candidate identity and projected NFL season |
| `source` | Normalized provider/source name |
| `source_kind` | `projection`, `market`, `ranking`, or `draft` |
| `source_player_name` | Provider's player name |
| `source_position`, `source_team` | Provider season-specific context |
| `match_method` | Identity-resolution method or methods |
| `record_count` | Resolved alias rows collapsed into the observation |
| `run_id` | Milestone 2 build provenance |

Position-less market rows are retained only when they resolve to an existing
identity and another source or the canonical identity supplies an eligible
fantasy position.

Historical team labels covered by `SOURCE_TEAM_TRUST_POLICIES` are discarded
before this table is aggregated. The source observation remains present, but
its `source_team` is null. The season-level spine team is selected only from
remaining trusted same-season evidence; an unresolved equal-count tie also
produces null rather than an alphabetical winner. See
`v2_identity_outcomes.md` for the governed source scopes.
Recognized `TEAM_MAP` aliases are canonicalized before both the per-source and
cross-source votes, so values such as `JAC` and `JAX` count as one team while
`JAC` and `BUF` remain a true conflict.

Rows matching `SOURCE_ROW_EXCLUSIONS` are partitioned before candidate
aggregation. They remain in the raw source database but cannot contribute a
`player_season_sources` record, establish a candidate, or influence the
position/team consensus. In particular, the quarantined FFToday QB slice
stored under 2018 is not shifted to 2019 because a native 2019 slice already
exists. See `v2_identity_outcomes.md` for the governed rule.

## `player_season_spine`

This table contains one row per `player_key, season, league`.

### Candidate and identity fields

| Column | Meaning |
|---|---|
| `position`, `team` | Season-specific source consensus; position is always QB/RB/WR/TE |
| `identity_status` | `confirmed` or `provisional`; identity metadata is not a model feature |
| `draft_year`, `rookie_season` | Retained only when not later than the projected season |
| `year_exp` | Projected season minus known rookie/draft season; nullable when unknown |
| `experience_known` | A rookie/draft season is known as of the projected season |
| `is_rookie` | Known rookie/draft season equals projected season; null when experience is unknown |
| `candidate_rule` | `preseason_evidence` or `drafted_rookie_only` |
| source-count columns | Counts of distinct projection, market, ranking, and draft sources |
| `candidate_sources` | Sorted pipe-delimited audit list |
| `position_conflict`, `team_conflict` | Multiple preseason values were present |

Canonical GSIS identity and season-specific fantasy position are deliberately
separate. For example, a player classified as a defensive player by nflverse
can retain his GSIS identity while preseason WR evidence determines his
fantasy position. The same mapping is used when aggregating offensive weekly
outcomes.

### Cutoff fields

| Column | Meaning |
|---|---|
| `preseason_source_season` | Governed effective source season; must equal the projected `season` |
| `feature_cutoff_season` | Always `season - 1` for NFL-history features |
| `foundation_run_id` | Exact Milestone 1 identity/outcome build used |
| `run_id` | Milestone 2 build that published the row |

The historical source database has season-level snapshots, not immutable
capture timestamps for every provider. Milestone 2 therefore guarantees
season alignment and excludes NFL outcomes from candidate construction, but
does not claim a common calendar-day cutoff. A later feature builder must
version or exclude any source that cannot be shown to be preseason.
`player_aliases.source_stored_season` retains the physical source label;
candidate construction and all feature windows use the governed effective
season. Known source-season corrections are applied before the spine is built.
Configured source-row quarantines are applied even earlier, before either
stored/effective-season matching or candidate collapse.

### Outcome and target fields

The spine represents participation and conditional production as separate
targets.

| Situation | `appeared` | `opportunity_games` | `unconditional_season_points` | `conditional_ppg` |
|---|---:|---:|---:|---:|
| Completed, observed opportunity | 1 | observed | observed | observed |
| Completed, confirmed identity, no opportunity row | 0 | 0 | 0 | null |
| Completed, unresolved identity | null | null | null | null |
| Pending season | null | null | null | null |

Additional availability fields:

- `active_target_available = 1` only for completed candidate seasons with a
  confirmed identity or an observed outcome.
- `conditional_ppg_target_available = 1` only when an opportunity outcome was
  observed.
- `conditional_ppg_training_eligible = 1` only when the conditional target was
  observed and the player reached the configured useful-season threshold.
- `outcome_join_status` is `observed_opportunity`, `no_opportunity`,
  `unresolved_identity`, or `pending`.

An unresolved candidate cannot be safely interpreted as a nonparticipant.
Historical source aliases include duplicates and truncated names that failed
canonical identity resolution. Those rows remain available for identity/source
audits and current scoring fallbacks, but their missing outcome join is not a
zero label.

Low-sample observed seasons are retained. They have a real conditional PPG but
are not marked training-eligible until they meet the current four-game
threshold. This avoids replacing their outcomes while keeping the default
conditional model from fitting on extremely small samples.

## Required Invariants

1. Source observations are unique by `player_key, season, source`.
2. Spine rows are unique by `player_key, season, league`.
3. Every spine row has at least one allowlisted same-season source.
4. NFL outcomes alone cannot create a spine row.
5. `feature_cutoff_season` is strictly earlier than the target season.
6. Pending rows expose no participation or production label.
7. Completed rows expose a participation label only when identity resolution
   makes the outcome join auditable.
8. Confirmed completed candidates without an opportunity outcome receive an
   active target of zero and a null conditional-PPG target.
9. Unresolved completed identities expose neither a positive nor zero
   participation label.
10. Source-family counts reconcile exactly to the long-form source table.
11. Every published fantasy position is QB, RB, WR, or TE.
12. Quarantined source rows cannot create or contribute to a candidate source
    observation.
13. The active foundation's `source_manifest` contains exactly one current
    source-row-exclusion policy receipt; `--reuse-foundation` rejects a missing
    or stale receipt.

## Downstream Feature Boundary

Milestone 3 builds features against this spine:

- current-season projection and market consensus from same-season preseason
  sources;
- NFL-history features using only seasons at or before
  `feature_cutoff_season`;
- explicit history-depth and missingness fields for rookies, second-year
  players, injuries, and long absences;
- separate participation and conditional-PPG estimators before combining them
  into unconditional season value.

See `docs/data_contracts/v2_feature_mart.md` for the implemented normalized
source tables, reviewed feature set, manifests, and audit contracts.
