# Auction Expected-Excess Multi-Origin Replay

This study freezes the 2025 Auction objective definitions and evaluates the
same three construction policies on isolated 2022-2024 Beta replays:

- exact expected managed-score frontier;
- unconstrained expected winning-margin (`expected_excess`);
- a 50/50 blend of within-block standardized mean and expected excess.

Each replay uses current-method rolling-origin projections trained through the
prior season, historical keeper contracts, actual auction clearing prices, and
weekly-template donors ending at `target_year - 1`. Actual target-season weekly
outcomes are loaded only after every candidate roster and arm has been selected.
The model specification is as of 2026, so this is a frozen current-method
multi-origin replay rather than a pristine method-as-of-year backtest.

Run after building the three staged databases:

```powershell
python research/studies/2026-08-27_auction_excess_multi_origin/run_multi_origin.py
```

