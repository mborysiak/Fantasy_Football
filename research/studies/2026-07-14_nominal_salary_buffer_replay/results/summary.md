# Nominal Salary Buffer Replay Results

Run: 250 paired trials across 12 cells per origin, 250 construction plus independent evaluation contexts, seed 20260713.

Every cell retains the sampled-price $298 cap, Top-N on, projected waivers, and bench weight 0.25. Constrained cells add normalized point-price spend at or below $298 plus the named buffer.

## Development and temporal-check outcomes

Unqualified points include rosters that exceed historical final prices. Read them together with feasibility and overage; 2025 is a temporal check, not a pristine holdout.

| salary_draw_count | nominal_buffer | development_pareto | development_2022_2024_actual_points | development_2022_2024_cap_feasible_rate | development_2022_2024_mean_cap_overage | temporal_check_2025_actual_points | temporal_check_2025_cap_feasible_rate | temporal_check_2025_mean_cap_overage |
|---|---|---|---|---|---|---|---|---|
| 1 | none | True | 1628.243 | 0.057 | 57.469 | 1609.323 | 0.060 | 40.448 |
| 1 | 0 | False | 1572.054 | 0.268 | 21.571 | 1602.745 | 0.168 | 22.632 |
| 1 | 5 | False | 1579.536 | 0.217 | 24.833 | 1605.525 | 0.124 | 26.036 |
| 1 | 10 | False | 1584.088 | 0.167 | 28.247 | 1606.409 | 0.120 | 28.356 |
| 1 | 15 | False | 1590.557 | 0.140 | 31.180 | 1605.289 | 0.108 | 31.708 |
| 1 | 25 | True | 1598.726 | 0.084 | 36.769 | 1607.373 | 0.068 | 36.708 |
| 5 | none | True | 1593.105 | 0.208 | 27.380 | 1598.729 | 0.156 | 26.368 |
| 5 | 0 | True | 1573.150 | 0.276 | 20.595 | 1598.221 | 0.192 | 22.348 |
| 5 | 5 | True | 1579.844 | 0.253 | 22.659 | 1599.986 | 0.168 | 24.776 |
| 5 | 10 | True | 1587.007 | 0.232 | 23.865 | 1599.890 | 0.156 | 25.308 |
| 5 | 15 | False | 1586.170 | 0.227 | 24.905 | 1599.055 | 0.156 | 26.228 |
| 5 | 25 | True | 1590.715 | 0.212 | 26.241 | 1598.729 | 0.156 | 26.368 |

`development_pareto` requires no other tested cell to have at least as many unqualified points, at least as much realized-price feasibility, and no more mean overage. It is a descriptive frontier, not an automatic policy choice.

## Buffer-minus-no-constraint paired effects

Positive point and feasibility effects favor the named buffer; negative overage and spend effects favor it. Baseline violation is the share of unconstrained rosters that the nominal row would reject.

| draw_context | nominal_buffer | mean_actual_points_effect | mean_actual_cap_feasible_effect | mean_actual_cap_overage_effect | mean_actual_salary_spend_effect | mean_roster_changed | mean_baseline_nominal_violation | mean_candidate_near_nominal_cap_within_1 | mean_both_actual_cap_feasible | mean_joint_feasible_actual_points_effect | development_2022_2024_mean_actual_points_effect | temporal_check_2025_mean_actual_points_effect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | -43.786 | 0.185 | -31.378 | -33.813 | 0.939 | 0.898 | 0.110 | 0.049 | -13.090 | -56.189 | -6.578 |
| 1 | 5 | -37.480 | 0.136 | -28.080 | -29.670 | 0.893 | 0.838 | 0.088 | 0.048 | -13.221 | -48.707 | -3.798 |
| 1 | 10 | -33.845 | 0.097 | -24.940 | -25.959 | 0.827 | 0.759 | 0.114 | 0.050 | -14.115 | -44.155 | -2.914 |
| 1 | 15 | -29.273 | 0.074 | -21.902 | -22.659 | 0.745 | 0.665 | 0.113 | 0.055 | -7.554 | -37.686 | -4.034 |
| 1 | 25 | -22.625 | 0.022 | -16.460 | -16.638 | 0.543 | 0.488 | 0.083 | 0.058 | 3.658 | -29.517 | -1.949 |
| 5 | 0 | -15.093 | 0.060 | -6.094 | -7.214 | 0.697 | 0.556 | 0.089 | 0.167 | -9.953 | -19.955 | -0.508 |
| 5 | 5 | -9.632 | 0.037 | -3.939 | -4.560 | 0.461 | 0.357 | 0.052 | 0.179 | -1.990 | -13.261 | 1.257 |
| 5 | 10 | -4.283 | 0.018 | -2.901 | -3.144 | 0.275 | 0.211 | 0.047 | 0.191 | -0.115 | -6.098 | 1.160 |
| 5 | 15 | -5.120 | 0.014 | -1.891 | -1.993 | 0.164 | 0.135 | 0.031 | 0.193 | -0.586 | -6.935 | 0.326 |
| 5 | 25 | -1.793 | 0.003 | -0.854 | -0.858 | 0.080 | 0.067 | 0.015 | 0.195 | 0.600 | -2.390 | 0.000 |

## One-minus-five draw effects at each buffer

| nominal_buffer | mean_actual_points_effect | mean_actual_cap_feasible_effect | mean_actual_cap_overage_effect | mean_actual_salary_spend_effect | mean_roster_changed | development_2022_2024_mean_actual_points_effect | temporal_check_2025_mean_actual_points_effect |
|---|---|---|---|---|---|---|---|
| none | 29.002 | -0.137 | 26.087 | 27.667 | 0.990 | 35.138 | 10.593 |
| 0 | 0.309 | -0.012 | 0.803 | 1.068 | 0.639 | -1.096 | 4.524 |
| 5 | 1.154 | -0.038 | 1.946 | 2.557 | 0.788 | -0.308 | 5.539 |
| 10 | -0.559 | -0.058 | 4.048 | 4.852 | 0.911 | -2.919 | 6.519 |
| 15 | 4.849 | -0.077 | 6.076 | 7.001 | 0.950 | 4.387 | 6.234 |
| 25 | 8.170 | -0.118 | 10.481 | 11.887 | 0.981 | 8.012 | 8.644 |

## Buffer-by-draw interaction

This is (one-draw buffer effect) minus (five-draw buffer effect). Positive favors one draw for points and feasibility; negative favors one draw for overage and spend reductions. Joint-feasible interactions require all four underlying rosters to fit historical prices.

| nominal_buffer | actual_points_effect_interaction | actual_cap_feasible_effect_interaction | actual_cap_overage_effect_interaction | actual_salary_spend_effect_interaction | joint_feasible_actual_points_effect_interaction | mean_all_four_feasible_share | total_all_four_feasible_count |
|---|---|---|---|---|---|---|---|
| 0 | -28.693 | 0.125 | -25.284 | -26.599 | -4.340 | 0.034 | 34 |
| 5 | -27.848 | 0.099 | -24.141 | -25.110 | -10.567 | 0.036 | 36 |
| 10 | -29.561 | 0.079 | -22.039 | -22.815 | -7.306 | 0.036 | 36 |
| 15 | -24.153 | 0.060 | -20.011 | -20.666 | -3.574 | 0.038 | 38 |
| 25 | -20.832 | 0.019 | -15.606 | -15.780 | 5.594 | 0.040 | 40 |

## Validation and limits

Prior unconstrained controls reproduced: {'required': True, 'checked': True, 'matched': True, 'rows': 2000, 'mismatches': {'roster': 0, 'solve_status': 0, 'contains_top_n': 0, 'forecast_salary_spend': 0, 'actual_points': 0, 'drafted_only_points': 0, 'actual_salary_spend': 0, 'actual_cap_overage': 0, 'forecast_ev': 0, 'forecast_error': 0, 'actual_waiver_starts': 0}}.

- Historical final prices are exogenous and missing prices use the intentional $1 fallback, so realized affordability is diagnostic and optimistic.
- This tests the nominal guardrail on frozen historical salary laws. It does not rebuild the current empirical residual-quantile method walk-forward.
- Four seasons are four independent outcome units. Split halves measure Monte Carlo stability only.
- Waiver eligibility remains hindsight availability-filtered and frictionless, as in the parent replay.
