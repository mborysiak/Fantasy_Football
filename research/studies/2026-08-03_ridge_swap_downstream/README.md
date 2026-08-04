# Ridge Swap Downstream Validation

## Question

Does the frozen conditional-PPG ensemble
`Ridge(alpha=10) + RandomForest + LightGBM`, with equal one-third weights,
improve enough on the active production `Lasso + RandomForest + LightGBM`
ensemble to use as the 2026 point center?

This is a shadow-only study. It must not mutate either V2 database or any
production, Auction, or Snake artifact.

## Frozen protocol

- Leagues: `dk` and `beta`, evaluated separately and jointly.
- Features: the exact locked 40-feature conditional-PPG surface.
- Baseline RF/LightGBM forecasts: the active corrected-lineage rolling-origin
  forecasts already persisted in each live V2 database.
- Challenger linear member: `SimpleImputer(strategy="median",
  add_indicator=True, keep_empty_features=True)`, `StandardScaler`, and
  `Ridge(alpha=10, max_iter=10000)`.
- Ridge is fit only on seasons before the forecast origin. Alpha is fixed from
  the completed 2013-2022 selection study and is not retuned here.
- Point-forecast origins: 2017-2025. The 2026 output is shadow-only and has no
  outcome score.
- Distribution/template/roster origins: 2018-2025. Origin 2017 is excluded
  because a new V2 forecast has no earlier held-out V2 residuals for strict-
  prior calibration.
- Residual knots: q05/q10/q25/q50/q75/q90/q95 from the first available pool
  with at least 100 strictly prior observations: position plus history depth,
  position, then global.
- Weekly-template matcher weights, donor eligibility, participation model,
  scoring rules, room seeds, rosters, and scenario count are held fixed.
  Baseline and challenger are evaluated on the same eligible player keys and
  the same drafted roster indices.
- No 2023-2025 result may be used to alter the frozen model or gates.

## Gates

The Ridge swap advances only if all mandatory gates pass.

1. Point PPG: lower pooled RMSE in both leagues over 2017-2025, at least five
   of six league-season cells won in 2023-2025, and no position worsens by more
   than 0.01 RMSE in both leagues.
2. Distribution: pooled player CRPS over 2018-2025 does not worsen in either
   league; 50% and 80% interval coverage remain within 2 percentage points of
   nominal or do not move farther from nominal than production.
3. Weekly-template transport: core and depth role-tier PPG CRPS do not worsen
   by more than 0.25% in either league-period cell. Contribution and played-
   week CRPS do not worsen by more than 0.25%.
4. Fixed-roster Snake replay: roster-score CRPS does not worsen by more than
   0.5% in any league-period cell, and pooled development and temporal score
   CRPS are no worse in at least three of four league-period cells.

Championship Brier/log loss and 2026 Auction/Snake shadow ranks are diagnostics,
not promotion gates. Historical auction prices do not provide a clean outcome
test of this point-center-only change, so Auction is limited to a 2026
non-mutation/feasibility shadow check after the historical player and Snake
gates pass.

## Outputs

- `results_projection_{league}/`: paired point forecasts, strict-prior
  residual calibration, distribution scores, 2026 shadows, and metadata.
- `results_template_{league}/`: player-level weekly-template replay.
- `results_roster_{league}/`: fixed-roster score/championship replay.
- `results/`: combined gate table, findings, and verification receipt.

## Decision

Do not promote the Ridge swap for 2026. It passes player distribution and
weekly-template transport, but fails the point-season replication gate (4/6
recent wins after losing 2025 in both leagues) and the fixed-roster gate (only
1/4 league-period cells improves; beta temporal is +0.527% versus the +0.5%
margin). Retain equal-third Lasso/RandomForest/LightGBM. See
`results/findings.md` and `results/verification.json`.
