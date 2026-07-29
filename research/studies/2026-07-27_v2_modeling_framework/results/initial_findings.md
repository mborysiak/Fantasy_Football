# V2 M4A Initial Findings

## Clean Run

- Model run: `milestone_4a_20260728T050818Z_86b06f1f`
- Feature run: `milestone_3_20260728T044840Z_88f9f8f3`
- OOF seasons: 2017-2025
- Conditional-PPG rows: 3,701
- Participation rows: 7,877
- Folds: five per season; every prediction trains only through `season - 1`

The run is shadow-only. No production projection, template, or optimizer table
was changed.

## Critical Target Correction

The first participation run exposed 314 provisional-identity rows labeled as
zeros. The rows included duplicate or truncated aliases such as Scotty Miller
and Robby Anderson, so failure to join an outcome was being treated as football
nonparticipation. The spine now assigns those rows
`outcome_join_status = unresolved_identity`, leaves participation null, and
excludes identity confirmation from the feature manifest.

The clean OOF participation population retains 3,092 confirmed zero-opportunity
rows and 4,785 observed-positive rows. The 314 unauditable rows remain in the
mart for identity/source audits and current fallback handling.

## Conditional PPG

Pooled OOF RMSE:

| Model | RMSE |
|---|---:|
| Direct shallow LightGBM | 3.1443 |
| Residual shallow LightGBM | 3.1567 |
| Direct full Ridge | 3.1731 |
| Residual full Ridge | 3.1827 |
| Position-aware consensus recalibration | 3.1996 |
| Active-game expert hybrid | 4.2244 |
| Raw team-game expert consensus | 4.2681 |

The raw expert comparison overstates model alpha because team-game consensus
has a -2.14 PPG bias against an opportunity-game target. The active-game hybrid
reduces that bias only slightly overall (-2.08); its source exists only in
2024-2026. It improves 2025 RMSE from 4.119 to 3.552 but worsens 2024 from
4.151 to 4.278.

The more relevant comparison is direct shallow LightGBM versus the simple
position-aware Ridge recalibration. LightGBM improves mean season RMSE by
0.0543, wins all nine seasons, and has a season-clustered bootstrap interval of
[-0.0827, -0.0353]. Against direct full Ridge, however, the mean gain is only
0.0285 and the interval crosses zero. Direct and residual LightGBM are also
statistically indistinguishable at this stage.

Direct LightGBM is preferable to residual LightGBM as the current nonlinear
challenger because it handles the recent change in provider denominator more
cleanly. In 2025 its RMSE is 2.949 versus 3.168 for residual LightGBM. It also
improves on recalibrated consensus for rookies (3.468 versus 3.516),
second-year players (3.242 versus 3.291), and veterans with history (3.056
versus 3.117). The other-no-history conditional slice has only 27 rows and does
not support a separate model.

## Participation

Pooled OOF results:

| Model | Brier | Calibration bias |
|---|---:|---:|
| Shallow LightGBM | 0.1222 | +0.0166 |
| Full logistic | 0.1366 | +0.0316 |
| Compact logistic | 0.1384 | +0.0419 |
| Prior-position rate | 0.2433 | +0.0633 |

LightGBM beats full logistic in eight of nine seasons. The mean season Brier
gain is 0.0135 with a season-clustered bootstrap interval of
[-0.0220, -0.0054]. The recent-era difference is much smaller (0.1342 versus
0.1364), so logistic remains the required calibration/stability benchmark.

This target is probability of at least one position-specific fantasy
opportunity game, not an injury or games-played forecast. It therefore avoids
claiming that the model can predict injuries, while still separating confirmed
zero-opportunity candidates from the conditional-PPG population.

## Features and Transformations

KBest, PCA, and feature agglomeration all underperform the corresponding raw
full pipeline for both targets. They should not be stacked into the next
production candidate.

The full residual Ridge beats the compact residual Ridge by 0.0793 mean season
RMSE, winning seven of nine seasons; its bootstrap interval is
[-0.1278, -0.0245]. Full versus compact logistic is much less certain
(-0.0016 mean Brier; interval crosses zero).

Fold-identical family dropouts show:

- Conditional PPG projection-level features are essential: removing five
  projection-level fields worsens RMSE by 0.1175, with a positive season
  interval [0.0657, 0.1703].
- Conditional history, room, availability, uncertainty, lifecycle, role, and
  team families are individually close to neutral in Ridge.
- Removing the two conditional market features improves pooled RMSE by 0.0147;
  market disagreement should be treated as a challenger, not automatically
  retained.
- Participation projection level is clearly useful (+0.0096 Brier when
  removed). History (+0.0037) and lifecycle (+0.0021) also have positive
  season intervals.
- Participation team change is neutral. Market, availability, and projection
  uncertainty have smaller or less stable incremental value.

These are linear-model ablations. They justify a lean next linear pipeline but
do not by themselves authorize deleting inputs from the nonlinear challenger
or the long normalized source tables.

## Initial Decision

Retain four benchmarks for the next stage:

1. active-game expert hybrid;
2. position-aware consensus recalibration;
3. direct full Ridge; and
4. direct shallow LightGBM.

For participation, retain full logistic and shallow LightGBM. Drop KBest, PCA,
agglomeration, and the broader model-family search from the primary path.

The next implementation step is to fit these finalists through 2025, generate
2026 shadow predictions with exact feature/model lineage, and use the OOF model
errors plus participation probabilities as inputs to the existing joint
weekly-path template framework. Model error must not be drawn independently a
second time if the selected weekly template already carries the joint residual.
