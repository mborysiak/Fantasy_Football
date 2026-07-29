# V2 Next-Year Residual Validation (DK)

## Scope

- Run: `v2_next_year_dk_20260729T130017Z_c37481f9`
- Feature run: `milestone_3_20260729T034246Z_ae57edb4`
- Target: following-season conditional PPG minus the origin-season expert
  team-game PPG consensus.
- Validation origins: 2017-2024.
- Each forecast origin uses training labels only through origin minus two; the
  latest usable target outcome is origin minus one.
- Production projections, templates, and optimizers remain unchanged.

## Primary results

| Target | Challenger | Reference | Challenger score | Reference score | Mean origin delta | Origin wins | 95% interval |
|---|---|---|---:|---:|---:|---:|---:|
| Conditional PPG RMSE | Equal-third residual blend | Expert carry-forward | 3.9003 | 5.2070 | -1.3022 | 8/8 | [-1.4349, -1.1815] |
| Conditional PPG RMSE | Position/experience aging | Expert carry-forward | 4.4847 | 5.2070 | -0.7180 | 8/8 | [-0.8248, -0.6105] |
| Appearance Brier | LightGBM | Position/experience prior | 0.1604 | 0.2648 | -0.1024 | 8/8 | [-0.1206, -0.0855] |
| Appearance Brier | LightGBM | Logistic | 0.1604 | 0.1732 | -0.0120 | 8/8 | [-0.0192, -0.0058] |

The primary conditional model has 3.0131 MAE, +0.2515 PPG bias, and 0.7229
Spearman correlation on 3,528 appearance-conditioned rows. The appearance
model has 0.4883 log loss, +0.0190 calibration bias, and 0.8452 ROC AUC on
7,048 labeled rows.

## 2027 shadow

- Candidate rows: 751
- Conditional-PPG centers: 720
- Appearance probabilities: 751
- Historical conditional training rows: 6,306
- Historical participation labels: 12,106

The conditional center is meaningful only if the player appears in 2027.
Keeper or optimizer integration must preserve the separate appearance
probability; it must not silently treat all 720 conditional centers as certain
participants.

Runtime: 153.8 seconds.
