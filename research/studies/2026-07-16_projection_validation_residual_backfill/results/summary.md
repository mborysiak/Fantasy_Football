# Backfill Summary

- Applied to database: `True`
- Base validation rows: `42,351`
- Final validation rows: `6,006`
- Model slices: `101`
- Final rows with calibrated residuals: `5,237` (87.2%)
- Earliest/latest forecast origins: `2017` / `2025`
- Maximum independent point-ensemble difference: `3.55e-15`
- Backup: `C:\Users\borys\OneDrive\Documents\GitHub\Fantasy_Football\research\studies\2026-07-16_projection_validation_residual_backfill\artifacts\local\Validations_pre_projection_resid_backfill_20260716_101155.sqlite3`

The first forecast origins intentionally retain unavailable intervals when fewer than 30 strictly prior realized residual rows exist. `next` rows use an additional horizon embargo and their terminal raw targets are flagged unavailable rather than treated as realized data.

## Season coverage

```text
version  model_spec_asof_year  season  rows  current_targets  next_targets  ensemble_targets  residual_rows  mean_resid_source_coverage  residual_row_rate
   beta                  2025    2017    33               33             0                33              0                    0.000000           0.000000
   beta                  2025    2018    28               28             0                28             28                    1.000000           1.000000
   beta                  2025    2019    26               26             0                26             26                    1.000000           1.000000
   beta                  2025    2020    27               27             0                27             27                    1.000000           1.000000
   beta                  2025    2021    34               34             0                34             34                    1.000000           1.000000
   beta                  2025    2022    37               37             0                37             37                    1.000000           1.000000
   beta                  2025    2023    35               35             0                35             35                    1.000000           1.000000
   beta                  2025    2024    35               35             0                35             35                    1.000000           1.000000
   beta                  2026    2017   315              309           238               315              0                    0.000000           0.000000
   beta                  2026    2018   307              295           234               307            295                    0.670469           0.960912
   beta                  2026    2019   302              292           230               302            302                    0.967364           1.000000
   beta                  2026    2020   305              293           235               305            305                    1.000000           1.000000
   beta                  2026    2021   298              292           234               298            298                    1.000000           1.000000
   beta                  2026    2022   307              299           239               307            307                    1.000000           1.000000
   beta                  2026    2023   313              309           249               313            313                    1.000000           1.000000
   beta                  2026    2024   314              305           248               314            314                    1.000000           1.000000
   beta                  2026    2025   305              305             0               100            305                    1.000000           1.000000
     dk                  2025    2017    30               30             0                30              0                    0.000000           0.000000
     dk                  2025    2018    26               26             0                26              0                    0.000000           0.000000
     dk                  2025    2019    26               26             0                26              0                    0.000000           0.000000
     dk                  2025    2020    29               29             0                29             29                    1.000000           1.000000
     dk                  2025    2021    22               22             0                22             22                    1.000000           1.000000
     dk                  2025    2022    28               28             0                28             28                    1.000000           1.000000
     dk                  2025    2023    28               28             0                28             28                    1.000000           1.000000
     dk                  2025    2024    30               30             0                30             30                    1.000000           1.000000
     dk                  2026    2017   315              309           238               315              0                    0.000000           0.000000
     dk                  2026    2018   307              295           234               307            295                    0.670469           0.960912
     dk                  2026    2019   302              292           230               302            302                    0.967364           1.000000
     dk                  2026    2020   305              293           235               305            305                    1.000000           1.000000
     dk                  2026    2021   298              292           234               298            298                    1.000000           1.000000
     dk                  2026    2022   307              299           239               307            307                    1.000000           1.000000
     dk                  2026    2023   313              309           249               313            313                    1.000000           1.000000
     dk                  2026    2024   314              305           248               314            314                    1.000000           1.000000
     dk                  2026    2025   305              305             0               100            305                    1.000000           1.000000
```
