# Verification Summary

The 2026 beta and DK rebuilds passed all production checks. The quantitative
pool metrics below describe the managed auction app's beta slice.

- The 2018 Le'Veon Bell holdout is retained for audit, marked
  `contract_holdout`, and used by zero pools.
- Ordinary zero-active seasons remain part of the downside distribution: 61
  distinct eligible zero-active donors appear in 134 current-player pool rows.
- All 180 current-player pools sum to probability 1 within floating-point
  tolerance, and no donor exceeds the 5% cap.
- Minimum/mean effective sample size is 58.5/72.3 for QB, 53.2/70.4 for RB,
  47.6/67.7 for TE, and 53.9/71.4 for WR.
- All current beta player rows have the new absolute PPG, projection
  disagreement, market gap, and applicable workload-room fields.
- Source and managed auction app generated tables return identical checks.
- Both retained beta and DK template slices have 5,298 rows, no null new match
  or eligibility fields, one declared Bell exclusion, and zero Bell pool uses.
- The copied Snake database loaded all 268 DK players with 80 donors and 16
  weekly profile columns through its unchanged runtime cache.
- The auction app's full suite passed 30 tests, including a direct assertion
  that one donor supplies both its centered PPG residual and weekly trajectory;
  a live database smoke loaded 180 players and sampled finite 4x12x16 contexts.
- A budget-120 Sequential Target parity run matched all 64 result rows exactly;
  four workers reduced runtime from 19.52 to 9.91 seconds (1.97x).
