# V2 Player Identity And Outcome Contract

## Scope

V2 Milestone 1 owns canonical player identity and exact calendar-season
outcomes. It does not build projection features or train models.

The build is run with:

```powershell
python -m Scripts.V2.build_milestone_1
```

It reads:

- the nflverse canonical `players` release;
- nflverse weekly player-stat releases for completed seasons;
- narrow identity fields from existing projection, ADP, draft, and legacy
  rookie tables.

It writes only `Data/Databases/Projection_V2.sqlite3`. Existing production
databases and tables are not modified.

## `player_identity`

`player_key` is the permanent V2 identifier. A confirmed NFL player carries a
`gsis_id`; a player who has not received one yet is retained with
`identity_status = provisional`.

When a provisional player later receives a GSIS ID, the builder reuses the
existing `player_key` when normalized name, position, and draft year resolve
uniquely. Team is identity metadata and is never part of the permanent key.

Draft year inferred from a source season is weak metadata. It must not
override a unique confirmed player's active career window or attach a current
source row to a retired same-name player. Exact draft-year matching is used as
an identity discriminator only when the source supplied the year directly.

After source resolution, a provisional identity is reconciled to a confirmed
same-name/position identity only when the compatible confirmed candidate is
unique across draft, rookie, and career-window evidence. Redundant provisional
rows are removed after their aliases are remapped. Legitimate same-name
players, such as the two historical WRs named Chris Harper, remain distinct
canonical identities.

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

Every narrow provider identity row is retained with its resolved `player_key`,
source name/ID, player name, position, team, season, draft year, and
`match_method`.

Unmatched and ambiguous rows are not silently forced onto another player.
They receive a provisional identity and remain auditable.

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
9. Inferred draft metadata cannot supersede a unique active career-window
   match.
10. No redundant provisional identity may remain when exactly one compatible
    confirmed identity exists.

Season-specific fantasy position may come from resolved preseason aliases when
the canonical nflverse position group is not QB/RB/WR/TE. This preserves the
canonical GSIS identity for converted and two-way players without dropping
their offensive weekly production.
