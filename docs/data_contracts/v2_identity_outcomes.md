# V2 Player Identity And Outcome Contract

## Scope

V2 Milestone 1 owns canonical player identity and exact calendar-season
outcomes. It does not build projection features or train models.

The paired league builds are run with:

```powershell
python -m Scripts.V2.build_milestone_1 --league dk --output-db Data/Databases/Projection_V2.sqlite3
python -m Scripts.V2.build_milestone_1 --league beta --output-db Data/Databases/Projection_V2_beta.sqlite3
```

It reads:

- the nflverse canonical `players` release;
- nflverse weekly player-stat releases for completed seasons;
- narrow identity fields from existing projection, ADP, draft, and legacy
  rookie tables.

Identity and aliases must be identical across the two outputs. Outcomes and
scoring hashes are league-specific. Existing production databases and tables
are not modified by the Milestone 1 builder.

## `player_identity`

`player_key` is the permanent V2 identifier. A confirmed NFL player carries a
`gsis_id`; a player who has not received one yet is retained with
`identity_status = provisional`.

When a provisional player later receives a GSIS ID, the builder reuses the
existing `player_key` when normalized name, position, and draft year resolve
uniquely. Team is identity metadata and is never part of the permanent key.

Draft year inferred from a source season is weak metadata. It must not
override a unique confirmed player. A directly supplied contradictory
draft/entry year and a source season before a known entry year are hard
incompatibilities. `last_season` is not a hard career endpoint: a unique
returning player may resolve after a multi-year gap. Career windows are used
only to break ties among namesakes, followed by normalized team evidence.
Exact draft-year matching is used as an identity discriminator only when the
source supplied the year directly.

After source resolution, a provisional identity is reconciled to a confirmed
same-name/position identity only when the compatible confirmed candidate is
unique across draft, rookie, and career-window evidence. Redundant provisional
rows are removed after their aliases are remapped. Legitimate same-name
players, such as the two historical WRs named Chris Harper, remain distinct
canonical identities.

Provider spellings that punctuation and suffix normalization cannot recover
use a narrow, reviewed `GOVERNED_NAME_ALIASES` ledger. Entries may be global
or source-scoped; there is no general fuzzy-name matcher. Reviewed examples
include:

- `Tet Mcmillan` to `Tetairoa McMillan`;
- FantasyPros `Amon Ra St` to `Amon-Ra St. Brown`; and
- FantasyPros `Equanimeous St` plus ADP-MFL `Brown St` to
  `Equanimeous St. Brown`.

Tetairoa McMillan's confirmed GSIS identity `00-0040124` retains the existing
production key `c16a5e67-fff0-57b9-838c-c8df91df7b9d`. This governed key
migration prevents a GSIS refresh from breaking downstream continuity.

Important fields:

| Column | Meaning |
|---|---|
| `player_key` | Immutable V2 UUID |
| `gsis_id` | nflverse/GSIS identifier, nullable before assignment |
| `pfr_id`, `pff_id`, `espn_id`, `nfl_id` | Provider crosswalk identifiers |
| `display_name`, `normalized_name` | Display and provider-neutral matching names |
| `position` | Canonical nflverse position group |
| `draft_year`, `draft_round`, `draft_pick`, `draft_team` | Draft identity metadata |
| `identity_status` | `confirmed` or `provisional` |
| `identity_source` | Source that established the identity |

## `player_aliases`

Every eligible accepted provider identity row is retained with its resolved
`player_key`, source name/ID, player name, position, team, effective season,
draft year, and `match_method`. Unresolved positionless rows and governed
aliases carrying a hard entry-year contradiction fail closed instead of
creating a knowingly incorrect identity.

Unmatched and ambiguous rows are not silently forced onto another player.
They receive a provisional identity and remain auditable.

Important season-provenance fields are:

| Column | Meaning |
|---|---|
| `source_stored_season` | Season label physically stored by the source table |
| `season` | Governed effective preseason season used for identity and feature windows |
| `source_season_override_id` | Declarative correction identifier, when applied |
| `source_season_override_reason` | Why the stored label is known to be wrong |
| `source_season_override_reference` | Archive/source evidence for the correction |
| `match_method` | Exact governed resolution path or provisional outcome |

### Governed effective seasons

Two archived FantasyPros WR snapshots have incorrect stored season labels:

| Source table | Position | Stored | Effective | Evidence |
|---|---|---:|---:|---|
| `FantasyPros_Projections` | WR | 2016 | 2018 | Wayback timestamp `20180808115212` |
| `FantasyPros_Projections` | WR | 2020 | 2021 | Wayback timestamp `20210728120136` |

The shared correction helper runs before identity matching, candidate windows,
alias joins, and projection-value construction. It preserves both stored and
effective seasons, is idempotent, and fails closed if native rows already
exist for the target source/position/effective-season combination.

### Governed source-row quarantines

A source-row quarantine is different from an effective-season override. It
removes a known-invalid stored slice from V2 lineage while leaving the raw
source database unchanged:

| Exclusion ID | Source table | Position | Stored season | Rows | Evidence |
|---|---|---|---:|---:|---|
| `fftoday_qb_stored_2018_2019_vintage_quarantine_v1` | `FFToday_Projections` | QB | 2018 | 50 | [FFToday official 2019 QB projection archive](https://www.fftoday.com/rankings/playerproj.php?Season=2019&PosID=10) |

The 50 stored-2018 rows match the provider's 2019 projection vintage, while a
native 2019 QB slice already exists. They are therefore excluded rather than
overridden to 2019, which would double count the same provider vintage. Native
2019 rows remain eligible.

`SOURCE_ROW_EXCLUSIONS` is applied before identity resolution, effective-season
handling, candidate aggregation, projection-value construction, and template
identity backfills. A quarantined row may not create an alias, candidate,
normalized value, feature, or weekly-template key. Excluded rows retain rule
ID, reason, and reference in build audit artifacts rather than in
`player_aliases`.

### Governed historical team-label trust

Some source rows retain valid historical projection or market values while
their `team` field is mutable and can be overwritten by a later destination.
`SOURCE_TEAM_TRUST_POLICIES` therefore discards only the historical team label
through the completed-season cutoff for `FFA_RawStats`, `FFA_Projections`, and
`FantasyPros_Best_Ball_ADP`. The player row and every non-team value remain
eligible. Current projection-season labels are not covered by these rules.

The policy is applied before identity disambiguation, alias publication,
candidate team voting, and normalized value construction. A trusted
same-season alias may supply the missing team downstream; if trusted evidence
is absent or tied, team remains null. It is never selected alphabetically.
Otherwise-identical value rows with multiple teams fail closed unless their
source-season scope is covered by this governed discard policy.
Recognized provider aliases are first canonicalized through `TEAM_MAP` (for
example, `JAX` and `JAC` both become `JAC`), so code variants do not create a
false tie; genuinely different franchises still fail closed.

## `player_season_outcomes`

Outcomes come from canonical nflverse weekly player stats and are grouped
directly by `player_key, season`. No `shift`, forward fill, or player-specific
week deletion is permitted.

Fantasy-week coverage matches the legacy modeling horizon:

- seasons before 2021: weeks 1–16;
- seasons from 2021 onward: weeks 1–17.

An opportunity game is:

- QB: more than 15 pass attempts + sacks suffered + carries;
- RB/WR/TE: at least one carry or target.

`conditional_ppg` is therefore offensive-opportunity-game PPG, not
conventional games-active PPG. Each qualifying weekly row is scored before
season aggregation. Actual points and PPG include configured:

- sacks suffered and lost fumbles;
- 300/400 passing-yard bonuses;
- 100/200 rushing-yard bonuses; and
- 100/200 receiving-yard bonuses.

When both thresholds have nonzero coefficients, the higher threshold is
cumulative with the lower one. Two-point and special-teams components score
only when the selected league dictionary supplies coefficients; the current
DK and beta dictionaries do not, so those component columns are presently
zero. Provider preseason scores use a separate season-total estimand documented
in `v2_feature_mart.md`; weekly bonus counts cannot be reconstructed from
season yardage totals.

Important fields:

| Column | Meaning |
|---|---|
| `opportunity_games` | Count of qualifying fantasy-week games |
| `season_points` | Configured fantasy points across opportunity games |
| `conditional_ppg` | `season_points / opportunity_games` for completed targets |
| `appeared` | At least one opportunity game |
| `useful_season` | At least four opportunity games; threshold is recorded in `build_runs` |
| `target_available` | Outcome is complete and may be used as a model target |
| `outcome_complete` | Season is at or before the declared completion cutoff |
| component point columns | Passing, rushing, receiving, fumble, two-point, and special-teams contributions |
| `scoring_hash` | Hash of the exact league scoring dictionaries |
| `run_id` | Build provenance |

The table contains observed opportunity seasons. In Milestone 2, the
preseason projection spine will be left-joined to this table. A player who was
forecast in a completed season but has no outcome row will then receive
`appeared = 0`, `useful_season = 0`, and a null conditional-PPG target. The
absence is not represented by copying a prior or later season.

## Provenance

`source_manifest` stores the source URI, SHA-256 checksum when the source is a
downloaded canonical release, and row count for every input.

Stored/effective-season corrections and their archive evidence remain on
`player_aliases` and are propagated into projection-value lineage. Paired DK
and beta builds must publish identical identity and alias tables.

Every Milestone 2 run also writes one `source_row_exclusion_policy` receipt
named `configured_source_row_exclusions`. Its `source_sha256` is the stable hash
of the complete configured quarantine policy and its `row_count` is the number
of rules. Reusing a foundation fails closed if that receipt is missing, stale,
or duplicated. Milestone 3 separately publishes a `source_quarantine` manifest
row for each source slice with excluded rows.

Milestone 2 also writes one `source_team_trust_policy` receipt named
`configured_source_team_trust`. Its hash covers the complete governed
historical team-label policy and the canonical `TEAM_MAP` alias mapping.
Reusing a foundation fails closed if either policy receipt is missing, stale,
or duplicated.

`build_runs` stores run time, league, season boundaries, useful-season
threshold, scoring hash, output row counts, and completion status.

## Required invariants

1. `player_identity.player_key` is unique and non-null.
2. Confirmed `gsis_id` values are unique.
3. All aliases and outcomes resolve to an existing `player_key`.
4. Outcomes are unique by `player_key, season, league`.
5. Point components reconcile exactly to `season_points`.
6. `useful_season` matches the recorded opportunity-game threshold.
7. Incomplete outcomes never expose a conditional-PPG training target.
8. Missing calendar seasons are absent rather than shifted or filled.
9. Inferred draft metadata cannot supersede a unique compatible confirmed
   match.
10. No redundant provisional identity may remain when exactly one compatible
    confirmed identity exists.
11. Stored/effective seasons and override ID, reason, and reference reconcile
    to the governed source-season ledger.
12. Corrected FantasyPros WR rows do not remain effective in 2016 or 2020, and
    a native effective-season collision aborts the build.
13. Identity and value loaders apply the same effective-season rules.
14. Paired DK/beta identity and alias tables are identical.
15. Governed retired player keys do not survive in aliases, outcomes,
    projection values, spine rows, or features.
16. No row matching a configured source quarantine survives in aliases,
    candidate sources, normalized projection/market values, or features.
17. The active Milestone 2 foundation carries exactly one current
    source-row-exclusion policy receipt.
18. FFToday's native 2019 QB rows remain eligible; the stored-2018 duplicate
    vintage is not relabeled into that native slice.

Season-specific fantasy position may come from resolved preseason aliases when
the canonical nflverse position group is not QB/RB/WR/TE. This preserves the
canonical GSIS identity for converted and two-way players without dropping
their offensive weekly production.
