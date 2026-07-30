# Findings

## Decision

Accept and publish the corrected V2 identity/scoring lineage. The corrected DK
and beta foundations, locked current-season models, calibration audits,
template joins, following-season models, and production handoff all pass.

The provider estimand is
`core_offensive_season_components_v1`. Beta QB sacks are mandatory and may be
imputed from one same-position donor because FFToday is the sole historical
sack source in several seasons. Other one-component imputations still require
at least two same-position donors. Missing valid donors leave the row unscored.

Actual conditional PPG includes configured weekly yardage bonuses, sacks
suffered, and lost fumbles. Provider season-total projections intentionally do
not attempt to reconstruct weekly bonuses. Projected fumbles remain outside
the provider estimand pending a source-quality study. Two-point and return-TD
coefficients remain unresolved league-rule inputs and were not invented.

## Identity and source-season audit

- DK and beta each contain 6,655 identities, 55,964 aliases, 52,526 source
  observations, 9,934 outcomes, 31,848 projection-provider rows, 31,834 market
  rows, and 13,909 spine/features rows.
- Identity, alias, source, and spine populations are identical between leagues.
- Tetairoa McMillan is one confirmed identity on stable key
  `c16a5e67-fff0-57b9-838c-c8df91df7b9d` / GSIS `00-0040124`; all 43 aliases
  resolve there. His 2025 outcome feeds the 2026 prior-outcome features.
- Amon-Ra St. Brown and Equanimeous St. Brown each have one confirmed identity.
  All reviewed truncated aliases resolve to those keys; obsolete provisional
  keys are absent downstream.
- FantasyPros WR stored 2016 is effective 2018 for 253 rows, and stored 2020 is
  effective 2021 for 263 rows. Every corrected row retains override ID, reason,
  and archive reference; no affected WR row remains effective in 2016 or 2020.
- No current returning-player alias remains a provisional exact-name/position
  duplicate of a confirmed player. Governed draft namesakes remain separate.

## Scoring audit

- Every complete beta QB-provider row has non-null sacks.
- 2,180 beta QB rows use sack imputation: 1,407 FFToday-only, 181 PFF-only,
  and 592 FFToday/PFF medians.
- Every stored imputed value, component, donor list, donor count, and position
  reconciles exactly. Taysom Hill QB rows no longer borrow TE sack zeros.
- Current beta QB coverage is 380/437 provider rows and 66/111 candidate
  players. All 31 QBs with ADP at or before 200 are covered; 38/42 are covered
  through ADP 300.
- Actual outcome components reconcile exactly to season points in both leagues.
  Weekly yardage bonuses contributed 9,918 DK and 3,608 beta points over
  2017-2025 qualifying rows.

## Model gates

The corrected locked current-season primary remains accepted:

| League | Primary RMSE | Expert RMSE | Season wins |
|---|---:|---:|---:|
| DK | 3.1092 | 3.1939 | 9/9 |
| beta | 2.8993 | 2.9765 | 9/9 |

No strictly prior calibration overlay improves pooled RMSE. The no-history
route has a very small fully negative season interval in both leagues, but it
remains a secondary diagnostic rather than changing the primary during a
corrective data relock.

The corrected following-season primary also passes:

| League | Primary RMSE | Expert carry RMSE | Origin wins | Appearance Brier |
|---|---:|---:|---:|---:|
| DK | 3.9118 | 5.2297 | 8/8 | 0.1618 |
| beta | 3.5902 | 4.4274 | 8/8 | 0.1656 |

The corrected current and 2027 shadow tables contain 745 candidates per
league, 715 DK/673 beta conditional centers, and 745 appearance probabilities.

## Production verification

- Canonical and point-center coverage is 100% for 268 DK and 180 beta
  production rows; there are no unmatched or duplicate targets.
- The production publisher is idempotent. Repeating the exact publish yields
  zero current- and next-point deltas.
- Source, auction-app, and Snake simulation databases pass integrity and
  foreign-key checks. All 11 generated app tables match the source exactly;
  the Snake database is byte-identical to the source. App-owned table counts
  are unchanged.

## Remaining caveats

- Five historical ADP-at-or-before-200 QB seasons lack any valid beta sack
  donor and remain unstandardized: Ryan Tannehill and Tony Romo (2017), Tyrod
  Taylor (2018), Andrew Luck (2019), and J.J. McCarthy (2024).
- Eight complete non-QB hybrid provider rows have small passing projections
  with null sacks. This is a future conditional-component refinement, not a
  current QB coverage blocker.
- Actual two-point and special-teams fields currently score zero because the
  configured dictionaries contain no coefficients. Confirm league rules before
  adding them.
