# Decision Readout

## Finding

v2 reduces the salary model's average underprediction bias, but it does not improve ordinary absolute error consistently.

Across 644 common observed player-years, mean residual moved from -0.71 to -0.48, while MAE changed from 4.31 to 4.49 (+0.18).
In the 2025 temporal check, MAE changed from 3.73 to 4.20 (+0.48).

## Optimizer-relevant tail

On the frozen replay candidate universe, the strongest within-position value quintile's old-v1-selection-weighted residual changed from +4.82 to +3.91.
Across every recorded candidate, the old-v1-selection-weighted residual changed from +1.43 to +1.58.

## Paired optimizer replay

The identical-seed v2 chance-frontier replay completed all 4,000 cells and changed 79.1% of development rosters and 80.1% of 2025 rosters.
Across chance thresholds, development managed forecast EV changed +2.08 season points, held-out modeled affordability changed -1.02%, historical feasibility changed +0.00%, and historical overage changed $+1.09.
For 2025, managed forecast EV changed -1.79, held-out modeled affordability changed -0.43%, historical feasibility changed +1.20%, and historical overage changed $-0.46.

Season directions were unstable. In 2023, v2 changed managed forecast EV by +4.20 but changed historical roster spend by $+4.75; in 2022 those effects were -1.24 and $-2.18.

## Action

Do not promote v2 as the production salary method and do not discard its feature set. It improves mean bias and the apparent strongest-value residual tail, but worsens ordinary MAE and does not produce a stable affordability gain after optimizer reselection.

Keep v1 as the current comparison/default surface. The next inexpensive test should be a causally evaluated v1/v2 shrinkage blend or a restricted correction focused on the optimizer-relevant high-value/high-price tail. Any candidate should pass both point-error and selected-roster affordability gates before another full frontier.
