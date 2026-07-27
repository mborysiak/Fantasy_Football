# One-Year Keeper Portfolio Replay

This study revises the keeper call-option test around the actual managed-bench
strategy:

- one successful keeper option is a good outcome;
- misses have no negative future payoff and can be dropped for waivers;
- the current starting roster should not be rearranged to chase keeper value;
- keeper value is limited to the first keeper season at acquisition price plus
  `$10`; and
- the five bench slots form the option portfolio.

The construction utility is the expected best positive first-year keeper
surplus across the five nominal bench players:

```text
E[max(max(future_market_value_i - (acquisition_price_i + 10), 0))]
```

Historical next-year point predictions and residual quantiles come from the
current-method, out-of-sample `Model_Validations_Resid` rows with
`current_or_next_year='next'`. Their residual donor cutoffs are strictly prior
to each origin. Players without a dedicated historical next row use the
origin-year current point projection plus the position's next-model residual
quantiles and are explicitly flagged as proxies.

Policies start from the prior study's same-engine zero-option roster. The eight
nominal starters are fixed. At most two bench players can be swapped, and the
final exact current-season construction score must remain within the policy's
small tolerance of the baseline:

- `control`: unchanged current-only roster;
- `best1_lex0`: maximize expected best keeper surplus with no current-score
  loss; and
- `best1_lex2`: maximize expected best keeper surplus with at most two modeled
  season points of current-score loss.

The primary realized keeper outcomes are the best positive surplus across all
five bench players and indicators that at least one bench player produced
positive, `$10+`, or `$20+` surplus. Observed historical acquisition cost is a
supplementary audit; counterfactual modeled acquisition cost provides complete
policy coverage.

Run the mechanics check:

```powershell
python research/studies/2026-07-20_one_year_keeper_portfolio/verify_mechanics.py
```

Run a smoke replay:

```powershell
python research/studies/2026-07-20_one_year_keeper_portfolio/run_replay.py `
  --years 2024 --trials 4 --contexts 20 --projection-draws 250 `
  --output-dir research/studies/2026-07-20_one_year_keeper_portfolio/artifacts/local/smoke
```

Run or resume the full replay:

```powershell
python research/studies/2026-07-20_one_year_keeper_portfolio/run_replay.py
python research/studies/2026-07-20_one_year_keeper_portfolio/run_replay.py --resume
```

