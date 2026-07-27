# Joint Template Rolling Validation Readout

## Design

- Held out 1,620 preseason player-seasons across nine forecast origins
  (2017-2025): QB24, RB60, WR72, and TE24 per origin.
- Used the causal out-of-sample `Final_Validations_Resid` final point forecast
  for each target and retained the live production template bank's historical
  forecast/residual fields for donors.
- Restricted every target to weekly-template donors from strictly earlier
  seasons. No realized target-season outcomes entered matching or weighting.
- Evaluated 8,100 target-method rows across the adaptive centered production
  matcher, its uncentered variant, uniform weighting, the legacy matcher, and a
  projection/experience-only matcher.
- Scored active PPG, the full managed weekly trajectory, contribution above
  position replacement, zero contribution, `+3 PPG`, `+5 PPG`, and a joint
  impact proxy (`+3 PPG` plus top-quintile position-season contribution).
- Used 2020-2025 as the primary recent window and season-clustered bootstrap
  comparisons so player rows from the same NFL season were not treated as
  independent.

The earlier CSVs directly under `results/` use the template builder's historical
forecast as both target and donor baseline. That field is not comparable to the
current final ensemble at QB and TE. Those files are retained only as a
diagnostic. Production conclusions below use `results/production_oos/`.

## Calibration Result

The pool-centered production method is well calibrated for the use that drives
managed roster scoring:

| Recent 2020-2025 metric | Result |
| --- | ---: |
| PPG bias | -0.01 PPG |
| P10-P90 PPG coverage | 80.8% |
| Managed contribution bias | +0.89 points/season |
| P10-P90 contribution coverage | 80.8% |
| `+3 PPG` predicted / actual | 17.9% / 18.5% |
| `+5 PPG` predicted / actual | 7.9% / 6.8% |
| Impact predicted / actual | 11.8% / 11.1% |

Centering is necessary, not cosmetic. Removing it worsened recent PPG CRPS by
0.271 (season-bootstrap 95% interval 0.196 to 0.373), worsened managed
contribution CRPS by 2.224 (1.583 to 3.039), changed PPG bias from -0.01 to
-1.16, and changed contribution bias from +0.89 to -10.22. A causal
position/year selector chose centering for every position in every 2020-2025
origin.

The new full matcher modestly improved recent PPG CRPS by 0.008 and managed
contribution CRPS by 0.091 versus projection/experience-only matching. Both
season-bootstrap intervals were positive. It did not improve impact AUC. The
adaptive production and legacy centered matchers were effectively tied, while
uniform weighting had a slightly better point estimate. These small differences
do not justify another production weighting change without a nested holdout.

## Upside Result

The joint paths now produce a coherent and reasonably calibrated *unconditional*
upside distribution, but they do not yet reliably identify which similarly
projected player will deliver the right-tail surprise:

- Recent overall AUC was 0.535 for `+3 PPG`, 0.559 for `+5 PPG`, and 0.655 for
  the joint impact event.
- Impact discrimination was useful at RB (0.679) and WR (0.672), weaker at TE
  (0.602), and near chance at QB (0.526).
- Mean realized contribution rose monotonically from 33.5 in the bottom impact-
  probability quintile to 90.2 in the top quintile. The model therefore finds
  high *absolute* impact profiles.
- Residual breakout did not rise monotonically. The fourth impact-probability
  quintile realized an 18.1% impact rate versus 15.3% in the top quintile, whose
  mean observed PPG residual was -0.11. Tail width itself had essentially zero
  correlation (-0.004) with realized contribution surprise.
- The highest probability ranges were overconfident: 20%-25% predicted impact
  realized at 12.2% (90 rows), and 25%-35% realized at 20.8% (24 rows).
- Young RBs (experience 0-2) had 11.2% predicted versus 4.6% actual `+5 PPG`
  hits and 13.0% predicted versus 8.1% actual impact hits. Young WRs had 6.7%
  versus 3.7% `+5 PPG` and 12.1% versus 6.8% impact. In this top-projection
  sample, seasons with 3-8 years of experience produced more residual hits.
  The 9+ samples were too small for a veteran-policy conclusion.

## Decision

- Keep the pool-centered joint residual/weekly-path implementation for managed
  mean, P10, and roster-EV scoring.
- Do not interpret the current donor tail width or raw impact probability as a
  calibrated player-specific breakout score. It mostly separates absolute
  workload/value, not residual outperformance.
- Do not infer a blanket age penalty from this replay. Its young-player result
  instead warns that the current analog distribution is overconfident about
  early-career residual upside.
- If an upside label or objective is added, first fit and roll-validate a
  separate residual-tail probability layer (with calibration shrinkage in the
  highest bins) using prospect pedigree, role-change opportunity, depth-chart
  movement, projection disagreement, and workload concentration. Preserve the
  current joint donor only for coherent weekly-path realization.

