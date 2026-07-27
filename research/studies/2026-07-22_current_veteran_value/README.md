# Current-Season Veteran Value

This study asks whether experienced RB, WR, and TE players are worse
current-season managed-auction values after accounting for both preseason
projection and market cost. Following-season outcomes and keeper value are
excluded.

Two market views are deliberately kept separate:

1. a long 2008-2025 history using preseason ADP as the market-cost proxy;
2. the 2022-2025 rolling `beta` auction backtest using the current v5 predicted
   salary surface.

Current-season player contribution is reconstructed from the same 16 weekly
managed profiles consumed by the auction app. A player's standalone managed
contribution is the sum of positive weekly points above the position waiver
baseline (`RB=7`, `WR=7`, `TE=5`). This is not a full roster replay; it is an
age-distribution diagnostic that credits replaceability when a player misses or
underperforms.

The primary comparison matches each above-threshold veteran to up to five
same-position, same-season younger players on projection, market, or both.
Primary thresholds remain RB year 7, WR year 9, and TE year 8; "above" means
strictly beyond that threshold. Player-cluster bootstraps and leave-one-season-
out ranges expose uncertainty.

Run from the modeling repository root:

```powershell
python research/studies/2026-07-22_current_veteran_value/run_current_veteran_value.py
```

Durable outputs are written to `results/`.

Key outputs:

- `summary.md`: decision readout;
- `matched_summary.csv`: matched deltas, player-cluster intervals, and
  leave-one-season-out ranges;
- `regression_models.csv` and `quantile_models.csv`: adjusted sensitivity;
- `auction_analysis_rows.csv`: exact rolling v5 auction sample and provenance;
- `current_named_market_context.csv`: Henry, Kamara, Kittle, and Kelce versus
  current younger projection peers.
