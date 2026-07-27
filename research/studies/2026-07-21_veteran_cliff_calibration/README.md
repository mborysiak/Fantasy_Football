# Veteran Cliff Calibration

This study estimates the incremental risk associated with uncapped experience
for RB, WR, and TE players in the managed auction pipeline.

It separates three outcomes that are combined imperfectly in the live app:

1. a current-season PPG cliff while the player still appears in at least nine
   of the 16 modeled fantasy weeks;
2. an extended current-season absence of eight or fewer played weeks;
3. no useful following season, defined as four or fewer games or following-year
   PPG below 70% of the current preseason forecast.

Experience is reconstructed as years since draft, falling back to years since
the first recorded NFL season for undrafted/unmatched players. This avoids the
position-specific 95th-percentile cap in the production model inputs.

The study also audits the historical next-year validation target. It compares
the recorded `Model_Validations_Resid.y_act` with the following season in
`Season_Stats_New`. This catches retirement or disappearance rows that were
forward-filled before next-year model filtering.

Primary soft-penalty thresholds are seven completed years for RB, nine for WR,
and eight for TE. Sensitivity estimates move every threshold by one year.
Piecewise logistic models control for preseason PPG (quadratic) and origin
season, with player-clustered standard errors. The reported excess-experience
coefficient is the incremental effect of each year above the threshold.

Run from the modeling repository root:

```powershell
python research/studies/2026-07-21_veteran_cliff_calibration/run_veteran_cliff_calibration.py
```

Durable outputs are written to `results/`.

Key outputs:

- `summary.md`: compact readout and interpretation;
- `cohort_outcome_rates.csv`: unadjusted threshold cohorts;
- `excess_year_models.csv`: one-slope adjusted estimates and threshold
  sensitivity;
- `threshold_step_slope_models.csv`: separate threshold-crossing and
  further-year effects;
- `next_target_censoring_summary.csv`: true following-season attrition versus
  the recorded next-model target;
- `current_player_template_risk.csv`: live 2026 template diagnostics for the
  named veteran players.
