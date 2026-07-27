# Managed Played-Week Mask

This study verifies that managed auction scoring keeps weekly fantasy outcomes
separate from weekly participation evidence. A source-observed zero or negative
performance must remain a valid lineup result, while a genuinely missed week
can use a bench or waiver replacement.

Run from the modeling repository after rebuilding the `beta` weekly-template
slice and updating the app database:

```powershell
python research/studies/2026-07-13_managed_played_mask/verify_managed_played_mask.py
```

The verification covers the modeling table contract, scalar and vectorized
lineup scoring, learned decision scores, batched marginal values, paired
template score/mask sampling, SQLite loader compatibility, and the rebuilt app
database. It also rescans the source weekly rows to distinguish exact zero and
negative scores from short-QB appearances excluded from the best-ball profile,
and verifies that the separate managed profile preserves their scores.

To compare the corrected Target path with the legacy score-threshold behavior
and optionally smoke-test process workers:

```powershell
python research/studies/2026-07-13_managed_played_mask/benchmark_played_mask.py --iterations 100 --parallel-workers 2
```

The serial timing comparison is counterbalanced: each availability method runs
once first and once second because this workload showed material process-order
timing effects. Seeded EV must remain identical within each method.
