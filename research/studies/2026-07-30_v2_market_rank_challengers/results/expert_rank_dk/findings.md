# Normalized Expert-Rank Challenger — DK

## Method

- Expert ranks are converted to within-source, within-season, within-position percentiles before taking the cross-source median.
- The incumbent, rank-level, and expert-minus-projection gap matrices reuse the incumbent's strictly-prior selected hyperparameters.
- Primary attribution replaces the random forest's 50% feature subsampling with full-column forests on both sides. This prevents the added column from changing which incumbent columns are sampled.
- Every forecast origin is fit only on earlier seasons. Fold-local median imputation and missing indicators are unchanged.
- This is an attribution study, not a production promotion.

## Results

| Variant | RMSE | Pooled delta | Recent delta | Mean season delta | Season wins | Season 95% | Player 95% |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rank_level` | 3.1057 | -0.0022 | -0.0043 | -0.0023 | 7/9 | [-0.0042, -0.0003] | [-0.0038, -0.0006] |
| `rank_gap` | 3.1074 | -0.0005 | -0.0013 | -0.0005 | 6/9 | [-0.0019, +0.0012] | [-0.0028, +0.0019] |

The production-surface sensitivity retains the locked 50% random-forest feature subsampling. It is reported separately because adding a column changes the sampled feature set:

| Variant | Production-surface RMSE | Delta |
|---|---:|---:|
| `rank_level` | 3.1058 | -0.0017 |
| `rank_gap` | 3.1080 | +0.0004 |

Negative deltas improve on the incumbent.

## Governance

- Feature run: `milestone_3_20260730T140041Z_e06ca8aa`
- Staged database: `C:\Users\borys\OneDrive\Documents\GitHub\Fantasy_Football\research\studies\2026-07-30_v2_market_rank_challengers\artifacts\local\Projection_V2_single_nffc.sqlite3`
- Locked-incumbent reproduction max delta: `3.55e-15`
- 2026 normalized rank providers: 6
- A dedicated ETR coefficient is not tested because ETR has only 2024-2026 half-PPR history and 2025-2026 full-PPR history. ETR instead contributes one normalized vote to the cross-provider rank consensus.
- Promotion requires a favorable pooled result without a recent reversal and uncertainty intervals that support a stable gain.
