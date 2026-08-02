# Canonical ADP Handoff

## Objective

Publish one governed current ADP/rank surface, rebuild the locked 2026 DK and
beta production populations against it, and promote identical generated data
to the Auction and Snake apps without changing app-owned state.

## Source Snapshot

- DK: 416 rows from the live DraftKings feed, fetched 2026-07-30.
- NFFC: 497 rows from the latest local exports dated 2026-07-27: 431 offensive
  players, 33 `TK` units, and 33 `TDSP` units.
- ETR: 243 rows from the latest local export dated 2026-07-27.

The canonical publication is keyed by `(year, league, draft_entity_key)`.
Offensive rows require a governed `player_key`; NFFC team-kicker and
team-defense units keep deterministic draft-entity keys and null player keys.
ETR's exact overall and position ranks remain source-authoritative. NFFC
contributes one composite ADP vote rather than four correlated contest votes.
The migration also removes 476 invalid year-null ETR duplicates.

## Release Result

- DK production: 351 players (56 QB, 101 RB, 143 WR, 51 TE), with 343 exact
  canonical ADP matches and eight governed fallbacks.
- Beta production: 328 players (50 QB, 95 RB, 133 WR, 50 TE), with 238 exact
  canonical ADP matches and 90 governed fallbacks.
- DK no-center exclusions: Tyreek Hill, Joe Mixon, DeAndre Hopkins, Nick
  Chubb, Austin Ekeler, Kareem Hunt, Brandin Cooks, and Taysom Hill.
- A second handoff leaves all eight governed table hashes unchanged.
- All 20 generated Auction tables match staging; all six app-owned tables are
  unchanged.
- Every Snake table matches the promoted source.

## Verification

- Main repository: 187 tests passed.
- Strict release gate: 69 tests passed.
- Auction app: 49 tests passed.
- Snake app: 16 tests passed.
- Snake `AppTest`: zero exceptions.

Recoverable pre-promotion databases are under
[`results/pre_promotion`](results/pre_promotion/).

## Durable Results

- [`results/findings.md`](results/findings.md)
- [`results/live_validation.json`](results/live_validation.json)
- [`results/dk_source_receipt.json`](results/dk_source_receipt.json)
- [`results/promotion_receipt.json`](results/promotion_receipt.json)

