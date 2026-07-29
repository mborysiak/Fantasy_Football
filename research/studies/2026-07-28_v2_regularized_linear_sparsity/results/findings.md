# V2 Regularized Linear Sparsity Findings

## Outcome

Lasso is the strongest linear challenger in this study, but the evidence does
not support replacing the current V2 PPG finalist or deleting features from the
governed mart.

- Incumbent Lasso scores 3.1656 pooled OOF RMSE versus 3.1747 for the
  fold-identical Ridge reference, a 0.0091 improvement. It wins five of nine
  seasons, and the season-bootstrap interval `[-0.0238, +0.0027]` crosses zero.
- Expanded Lasso is the best linear point estimate at 3.1615, but adding all 12
  challengers improves Lasso by only 0.0041. Its season interval
  `[-0.0233, +0.0159]` crosses zero.
- Elastic Net is slightly worse than Lasso on both feature variants. All five
  selected incumbent fits use `l1_ratio` 0.7 or 0.9; the selected surface
  therefore leans toward Lasso rather than a broad grouped-retention solution.
- The prior direct shallow LightGBM result remains better at 3.1443 RMSE.
  Expanded Lasso trails it by about 0.017 pooled/mean-season RMSE despite
  winning five individual seasons.

No production model, feature manifest, projection, template, optimizer, or V2
database table changed.

## What "Fewer Features" Means Here

Lasso incumbent retains a mean 23.6 of 35 raw inputs per season-fold, ranging
from 17 to 30. Nineteen raw inputs are selected in at least 80% of the 45 exact
season-fold fits, while seven are selected in at most 20%.

The stable group is not merely prior NFL production. It includes:

- expert consensus level and disagreement;
- ADP and projection-versus-ADP disagreement;
- age, experience, draft capital, seasons since observation, and team change;
- prior-year and three-year production/volatility;
- position, pass/rush volume, team QB context, and room concentration.

This supports regularizing within the governed feature set rather than reducing
the mart to expert projection alone. The incumbent Lasso also retains an
average 8.6 missingness indicators, including stable indicators for ADP, pass
attempts, pass-point share, and room-leader gap. Missingness is part of the
rookie/no-history signal and should not be removed accidentally during manual
pruning.

The raw inputs selected at most 20% of the time are:

- `position_WR`;
- `proj_targets`;
- `proj_games`;
- `projected_rush_point_share`;
- `expert_ppg_active_median`;
- `consensus_room_share`;
- `expert_points_iqr`.

These are diagnostic pruning candidates, not deletion decisions. Several have
highly correlated retained proxies, and selection frequencies use the same OOF
period used to evaluate the model. A fixed subset chosen from these results
would therefore need a nested-origin or future-season confirmation.

## Expanded-Feature Interpretation

Expanded Lasso retains a mean 27.3 of 47 raw inputs, so it uses more substantive
inputs than incumbent Lasso even though it zeros a larger percentage of the
available matrix.

Among the 12 additions, the most stable selections are:

- `adp_room_strength_share`: 100%;
- `adp_teammates_better_count`: 100%;
- `team_receiving_yard_share`: 95.6%;
- `team_reception_share`: 84.4%;
- `adp_worst_teammate_gap`: 80.0%.

Projection-versus-experience difference is selected in 66.7% of Lasso fits,
while the experience peer mean is never selected. `team_target_share`, which
only starts in 2024, is selected in 4.4% of fits. This is internally coherent,
but the expanded model's aggregate gain is too small and temporally unstable to
promote the family.

Expanded Lasso improves on incumbent Lasso in 2017, 2018, 2020, 2021, and 2025,
but worsens in 2019, 2022, 2023, and 2024. It is also worse for rookies and
second-year players, while modestly improving veteran-with-history rows. That
is not the desired evidence for expanding the rookie/no-history projection
surface.

## Correctness

The coefficient audit refits the exact 45 season-fold training populations for
each of six specifications using each outer fold's selected hyperparameters.
All 22,206 refit predictions reproduce the stored OOF model predictions; the
maximum absolute difference is `3.55e-15`.

## Decision

- Keep direct shallow LightGBM as the current PPG leader.
- Retain incumbent Lasso as the preferred sparse linear challenger and a useful
  feature-stability diagnostic.
- Do not promote Elastic Net.
- Do not promote the 12-feature expanded manifest based on this result.
- Do not physically delete low-frequency features yet. If a static compact
  production manifest is desired, freeze its selection inside each historical
  training origin and test it on later untouched origins.
