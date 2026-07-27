# Soft Whole-Bench Keeper Portfolio

This study removes the hard one/two/three keeper-addition policy from the prior
full-roster sensitivity. Every nominal bench player can simultaneously provide:

- current-season managed fill-in value; and
- one-year keeper-option value.

The keeper objective is the expected best positive surplus across the complete
five-player bench:

```text
E[max(max(next_year_market_value_i - (price_i + 10), 0))]
```

It is not the additive sum of five player bonuses. The expected-best portfolio
creates diminishing returns when additional players cover the same outcome
space.

The current-only control is rebuilt on the complete cached-bank expected weekly
profile. The soft policy then searches for a better whole-bench option portfolio
and fully re-solves the remaining roster around each proposed candidate. There
are no age, role, or lottery-ticket-count quotas. A candidate is accepted only
when construction-bank mean and p10 remain at least as high as the control.

The number and concentration of options are outcomes. Diagnostics include
draw-level option winner shares, inverse-Herfindahl effective option count,
bench-player current marginal values, starter spend/strength, waiver use, and
realized keeper results. Search anchors are an implementation device rather than
designated keeper slots; final utility always scores all five bench players.

Run mechanics checks:

```powershell
python research/studies/2026-07-20_soft_keeper_portfolio/verify_mechanics.py
```

Run a smoke replay:

```powershell
python research/studies/2026-07-20_soft_keeper_portfolio/run_replay.py `
  --years 2024 --trials 4 --contexts 30 --gate-contexts 20 `
  --projection-draws 250 --candidate-shortlist 5 `
  --output-dir research/studies/2026-07-20_soft_keeper_portfolio/artifacts/local/smoke
```

Run or resume the full replay:

```powershell
python research/studies/2026-07-20_soft_keeper_portfolio/run_replay.py
python research/studies/2026-07-20_soft_keeper_portfolio/run_replay.py --resume
```

Build the durable diagnostics and decision readout:

```powershell
python research/studies/2026-07-20_soft_keeper_portfolio/analyze_results.py
```

The full 2022-2025 replay contains 1,000 paired trials. The soft policy improved
predicted/realized best one-year keeper surplus by `+$4.3`/`+$4.4` and average
independent current-year mean/p10 by `+4.2`/`+2.7`. It is not ready for live use:
only 46.2% of changed rosters improved both independent metrics, with 2024 the
clear weak origin. See `results/decision_readout.md`.
