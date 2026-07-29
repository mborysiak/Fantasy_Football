# Position-Specific Model Results

Negative deltas favor the separately fitted challenger.

## Pooled OOF

| Method | RMSE |
|---|---:|
| `pooled_full` | 3.1230 |
| `role_group_full` | 3.1231 |
| `qb_skill_full` | 3.1237 |
| `pooled_projection_core` | 3.1326 |
| `qb_skill_projection_core` | 3.1384 |
| `separate_full` | 3.1389 |
| `role_group_projection_core` | 3.1458 |
| `separate_projection_core` | 3.1752 |

## Paired season comparisons

| Challenger | Reference | Delta | 95% interval | Wins |
|---|---|---:|---:|---:|
| `separate_projection_core` | `pooled_projection_core` | +0.0425 | [+0.0192, +0.0693] | 1/9 |
| `role_group_projection_core` | `pooled_projection_core` | +0.0132 | [-0.0094, +0.0408] | 3/9 |
| `qb_skill_projection_core` | `pooled_projection_core` | +0.0058 | [-0.0178, +0.0288] | 2/9 |
| `separate_full` | `pooled_full` | +0.0160 | [-0.0147, +0.0449] | 4/9 |
| `role_group_full` | `pooled_full` | +0.0002 | [-0.0281, +0.0282] | 4/9 |
| `qb_skill_full` | `pooled_full` | +0.0007 | [-0.0108, +0.0138] | 5/9 |
| `role_group_projection_core` | `separate_projection_core` | -0.0294 | [-0.0412, -0.0188] | 9/9 |
| `role_group_full` | `separate_full` | -0.0158 | [-0.0230, -0.0086] | 8/9 |
| `qb_skill_projection_core` | `role_group_projection_core` | -0.0074 | [-0.0238, +0.0116] | 7/9 |
| `qb_skill_full` | `role_group_full` | +0.0006 | [-0.0213, +0.0231] | 4/9 |

## Position RMSE

| Position | Pooled projection | Separate projection | Role-group projection | QB/skill projection | Pooled full | Separate full | Role-group full | QB/skill full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QB | 3.4988 | 3.5960 | 3.5960 | 3.5960 | 3.5283 | 3.5254 | 3.5254 | 3.5254 |
| RB | 3.5455 | 3.5639 | 3.5639 | 3.5256 | 3.4987 | 3.4952 | 3.4952 | 3.4941 |
| TE | 2.3043 | 2.3577 | 2.3111 | 2.3030 | 2.3057 | 2.3242 | 2.3066 | 2.3001 |
| WR | 3.1222 | 3.1643 | 3.1100 | 3.1243 | 3.1249 | 3.1602 | 3.1284 | 3.1333 |
