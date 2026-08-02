# WR Template PPG/Profile Trade-off

This study tests the stated preference for tighter wide-receiver PPG matching
even if played-game calibration weakens. It combines the previously separate
PPG-weight and projected receiver-rate ablations.

The strict rolling replay uses 2017-2025 WR targets and only prior-season
donors. Candidate arms compare the production PPG weight of 1.50 with 2.25,
then add preseason projected yards/reception and/or touchdowns/reception at
weights of 0.25 or 0.50. The pool size, adaptive kernel, 12-season recency
prior, 5% donor cap, and centered joint outcome transport remain fixed.

The study also replays the current Ladd McConkey pool and records Terrelle
Pryor's rank and probability under every arm.

Run DK:

```powershell
.venv_ff_312\Scripts\python.exe `
  research\studies\2026-07-30_wr_template_ppg_profile_tradeoff\run_validation.py
```

Run beta:

```powershell
.venv_ff_312\Scripts\python.exe `
  research\studies\2026-07-30_wr_template_ppg_profile_tradeoff\run_validation.py `
  --league beta `
  --v2-db Data\Databases\Projection_V2_beta.sqlite3 `
  --results-dir research\studies\2026-07-30_wr_template_ppg_profile_tradeoff\results_beta
```

The study is read-only with respect to production code and databases.

## Result

No candidate is promoted. The 2.25 PPG weight tightens Ladd's current beta
donor PPG gap by 20.8% for 0.135 fewer expected games, but held-out WR PPG
calibration worsens in DK and improves only slightly in beta. Receiver-rate
terms do not remove Pryor; they generally move him up and increase his pool
weight because the preseason YPR and TD/rec projections are close. See
`results/findings.md`.
