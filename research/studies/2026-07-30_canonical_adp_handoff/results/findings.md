# Findings

The canonical ADP handoff is live and app-ready.

## Sources

- DK publishes all 416 current rows from the live feed. NFFC supplies scaled
  distribution bounds for 396 rows; the other 20 use the explicit governed
  20% distribution fallback instead of being dropped.
- NFFC publishes 497 rows from the latest 2026-07-27 local exports: 431
  offensive players, 33 `TK` units, and 33 `TDSP` units.
- ETR publishes 243 rows from the latest 2026-07-27 local export, preserving
  exact overall and position ranks.
- NFFC contributes one composite market vote. Its four contest rows remain
  identity and candidate evidence only.
- Canonicalization removed 476 invalid year-null ETR duplicates.
- The 1,156-row current publication has no duplicate offensive
  `(league, player_key)` values. Its only 66 null player keys are the intended
  NFFC `TK` and `TDSP` draft units.

## Production

- DK: 351 rows (56 QB, 101 RB, 143 WR, 51 TE); 343 exact ADP matches and eight
  governed fallbacks.
- Beta: 328 rows (50 QB, 95 RB, 133 WR, 50 TE); 238 exact ADP matches and 90
  governed fallbacks.
- Every player has 80 weekly donors: 28,080 DK and 26,240 beta pool rows.
- The eight governed DK no-center exclusions are Tyreek Hill, Joe Mixon,
  DeAndre Hopkins, Nick Chubb, Austin Ekeler, Kareem Hunt, Brandin Cooks, and
  Taysom Hill.

## Promotion Gates

- All eight governed handoff hashes are unchanged on the second publish.
- All 20 generated Auction tables match staging.
- All six app-owned Auction tables are unchanged.
- Every Snake table matches the promoted source.
- Source, main, Auction, Snake, and validation SQLite integrity are `ok`.
- Recoverable pre-promotion copies are under
  `research/studies/2026-07-30_canonical_adp_handoff/results/pre_promotion/`.

## Tests

- Main repository: 187 passed.
- Strict release: 69 passed.
- Auction: 49 passed.
- Snake: 16 passed.
- Snake `AppTest`: zero exceptions.
