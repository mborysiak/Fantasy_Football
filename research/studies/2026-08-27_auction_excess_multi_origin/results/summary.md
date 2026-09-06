# Frozen Expected-Excess Multi-Origin Results

The three policies were frozen from the 2025 experiment before these 2022-2024 outcomes were scored. Candidate construction uses preseason projections and donors through the prior year; actual auction prices define the retrospective cost surface.

## Annual results

| Year | Arm | Holdout EV | Holdout P90 | Expected excess | Actual score | Actual delta vs mean |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | Expected score | 1724.46 | 1907.43 | 11.67 | 1611.37 | +0.00 |
| 2022 | Pure expected excess | 1680.05 | 1861.54 | 9.05 | 1605.50 | -5.88 |
| 2022 | 50/50 mean + expected excess | 1701.04 | 1882.66 | 10.48 | 1637.37 | +26.00 |
| 2023 | Expected score | 1645.68 | 1828.33 | 9.71 | 1595.73 | +0.00 |
| 2023 | Pure expected excess | 1618.08 | 1793.34 | 8.23 | 1546.89 | -48.84 |
| 2023 | 50/50 mean + expected excess | 1632.77 | 1801.89 | 9.18 | 1536.17 | -59.56 |
| 2024 | Expected score | 1679.58 | 1864.94 | 10.25 | 1690.37 | +0.00 |
| 2024 | Pure expected excess | 1663.24 | 1843.71 | 10.45 | 1626.04 | -64.33 |
| 2024 | 50/50 mean + expected excess | 1670.96 | 1844.54 | 9.81 | 1607.14 | -83.23 |

## Cross-season readout

| Arm | Mean actual delta | Seasons positive | Mean holdout-EV delta | Mean holdout-P90 delta |
| --- | ---: | ---: | ---: | ---: |
| Expected score | +0.00 | 0/3 | +0.00 | +0.00 |
| Pure expected excess | -39.68 | 0/3 | -29.45 | -34.04 |
| 50/50 mean + expected excess | -38.93 | 1/3 | -14.98 | -23.87 |

## Most frequent roster changes

- **Pure expected excess adds:** Justin Jefferson 2024 (+50%), James Cook 2022 (+38%), Jerry Jeudy 2022 (+38%), Jaylen Warren 2024 (+38%), Zach Charbonnet 2024 (+38%), DJ Moore 2022 (+25%), Dallas Goedert 2022 (+25%), Dameon Pierce 2022 (+25%), George Pickens 2022 (+25%), Josh Allen 2022 (+25%)
- **Pure expected excess removes:** Drake London 2022 (-88%), Chase Brown 2024 (-62%), Tank Dell 2024 (-62%), Zay Flowers 2024 (-62%), DK Metcalf 2022 (-50%), Derrick Henry 2022 (-50%), Brandon Aiyuk 2023 (-50%), Breece Hall 2023 (-50%), Dameon Pierce 2023 (-50%), Damien Harris 2023 (-50%)
- **50/50 mean + expected excess adds:** Jerry Jeudy 2022 (+38%), Antonio Gibson 2023 (+38%), Jaylen Warren 2024 (+38%), Puka Nacua 2024 (+38%), A.J. Brown 2022 (+25%), Austin Ekeler 2023 (+25%), D'Andre Swift 2023 (+25%), Dalton Schultz 2023 (+25%), Devin Singletary 2023 (+25%), Drake London 2023 (+25%)
- **50/50 mean + expected excess removes:** Drake London 2022 (-62%), Zay Flowers 2024 (-62%), Brandon Aiyuk 2023 (-50%), Dameon Pierce 2023 (-50%), De'Von Achane 2023 (-50%), Kyle Pitts 2023 (-50%), Tank Dell 2024 (-50%), Jaylen Waddle 2022 (-38%), Breece Hall 2023 (-38%), Christian McCaffrey 2023 (-38%)

## Interpretation

- These are three season-level origins; eight construction blocks within a season measure seed sensitivity, not eight independent NFL seasons.
- Actual prices make this a hindsight cost replay. Projections and roster choices remain target-outcome blind.
- The 2026 model specification was applied to every origin, so positive results validate the frozen objective more than they validate a historically deployable 2022 method.
- Frozen rule identifier: `2026-08-27_pre_multi_origin_excess_v1`.
