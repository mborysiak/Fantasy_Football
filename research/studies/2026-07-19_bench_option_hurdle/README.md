# Bench Call-Option And Waiver-Hurdle Replay

This study tests whether a higher construction-only waiver hurdle is a useful
proxy for a studs-and-scrubs bench policy, and compares that proxy with a
sustained-breakout option utility.

The replay reuses the leakage-safe frozen 2022-2025 origins from
`2026-07-13_managed_auction_rolling_replay`. Every policy uses five salary
draws, the Top-12 constraint, a 13-player roster, and the current projected
waiver baseline for independent forecast evaluation. Target-season outcomes
are used only after roster construction.

Policies:

- `current_bench025`: current projected waiver baseline and bench weight 0.25.
- `hurdle_plus1`, `hurdle_plus2`, `hurdle_plus3`: add the named PPG hurdle to
  RB/WR/TE during construction and refinement only. Evaluation retains the
  current projected waiver baseline.
- `bench0`: remove the current weekly-P90 bench heuristic.
- `sustained_option025`, `sustained_option050`: replace the weekly-P90
  heuristic with 0.25x or 0.50x a sustained-breakout utility. A breakout must
  become causally detectable from a trailing three-week window, remain above a
  league starter-impact threshold for at least three of the next four weeks,
  and produce future points above that threshold.

The sustained option is intentionally a strategy-utility sensitivity, not a
claim that its bonus is literal expected lineup points. Ordinary lineup EV
already captures starts after an identified breakout; the option term tests a
preference for persistent difference-makers over steady replacement-level
depth.

Run the synthetic mechanics check:

```powershell
python research/studies/2026-07-19_bench_option_hurdle/verify_policy_mechanics.py
```

Run a smoke replay:

```powershell
python research/studies/2026-07-19_bench_option_hurdle/run_replay.py `
  --years 2025 --trials 4 --contexts 20 `
  --projection-draws 200 --salary-draws 500 `
  --output-dir research/studies/2026-07-19_bench_option_hurdle/artifacts/local/smoke
```

Run the full replay:

```powershell
python research/studies/2026-07-19_bench_option_hurdle/run_replay.py
```

Durable outputs belong in `results/`. Smoke outputs under `artifacts/local/`
are ignored.

