# Auction Salary Tables

## `League_Keepers`

`Scripts/Modeling/s4_Salaries_Injuries.py` owns this table in
`Data/Databases/Simulation.sqlite3`. The script replaces only the active
`year` and `league` slice, then the normal database copy makes the table
available to `Fantasy_Football_App`.

The active season comes from `Scripts.config.YEAR`, whose default can be set by
`FF_CURRENT_SEASON`; the approved production runner sets it from `--year`.
Keeper contracts are an explicit annual input, not source-code constants. The
script reads `FF_KEEPERS_FILE` when provided and otherwise requires
`Data/OtherData/Keepers/keepers_<year>_<league>.csv` with `player` and
`keeper_salary` columns. A missing file, blank player, or non-numeric salary
fails the build so a future season cannot silently inherit 2026 keepers.

| Column | Type | Meaning |
| --- | --- | --- |
| `year` | INTEGER | Auction season |
| `league` | TEXT | League identifier such as `beta` or `nv` |
| `player_key` | TEXT | Canonical V2 identity; required and unique within an active V2 league slice |
| `player` | TEXT | Cleaned player name used by simulation joins |
| `keeper_salary` | REAL | Actual salary committed to the keeper |

The primary key is `(year, league, player)`. All keepers in the active league
must be present before rebuilding salary predictions because the same input
drives keeper inflation and position-specific keeper-value features. The active
2026 beta slice has 14 keepers, all with canonical keys; a unique
`(year, league, player_key)` index prevents identity duplication.
The corresponding annual salary input remains
`Data/OtherData/Salaries/salaries_<year>_<league>.csv`; both files are hashed
at production-refresh snapshot and rechecked before promotion.

## `Salaries_Pred` Calibration

Salary point predictions and residuals use consistent non-keeper accounting:

- Historical OOF predictions are reconciled to the realized salary total of
  the exact represented non-keeper rows before residuals are calculated.
- Current non-keeper predictions are reconciled to league budget minus keeper
  spend across league roster slots minus keeper count.
- Point predictions use a common additive shift with a `$1` floor. The shift is
  solved so the highest open-slot non-keeper prices exactly equal the applicable
  market budget. This preserves salary differences above the floor instead of
  multiplying every discretionary dollar by one broad scale.
- Current keeper salaries are deterministic and have zero residual quantiles.
- Residual interval coverage is reviewed with leave-one-year-out quantile fits;
  the current market audit compares point and residual-implied totals with the
  keeper-adjusted available budget.

This table contains marginal player salary distributions. Exact draw-level
remaining-budget normalization belongs in the consuming auction app because
the available money and roster slots change as the auction proceeds.

## Historical Salary Validation Datasets

`Scripts/Modeling/s4_Salaries_Injuries.py` owns two tables in
`Data/Databases/Validations.sqlite3`. A build replaces only one
`(league, method_version, model_spec_asof_year)` slice. Both tables have a
unique row key of `(league, method_version, model_spec_asof_year, year, player)`.

### `Salary_Backtest_Predictions`

This is the replay-facing table. It contains the full joined salary candidate
pool for every backtest origin beginning in 2022, including players without an
observed auction result and keepers whose contract salaries are deterministic.

Important column groups are:

| Columns | Meaning |
| --- | --- |
| `player`, `pos`, `year`, `league` | Player and auction-origin identity |
| `pred_salary_raw` | Ensemble prediction before aggregate market calibration |
| `pred_salary` | Prediction after keeper-adjusted known-budget calibration |
| `base_salary`, `base_salary_observed` | Pre-auction source salary used as a model input and whether it was present in the manual ESPN copy; missing values use the existing zero/minimum convention |
| `actual_salary`, `actual_salary_observed` | Realized auction result and its availability flag; missing for unobserved pool players |
| `actual_resid`, `actual_resid_raw` | Realized salary minus normalized and raw predictions |
| `salary_resid_5` through `salary_resid_95` | Residual quantiles fitted only from earlier replay origins |
| `is_keeper`, `keeper_count`, `keeper_spend`, `available_slots`, `available_budget` | Keeper and remaining-market context known before bidding |
| `normalization_*`, `pred_salary_scale`, `pred_salary_shift`, `pre_normalized_total`, `post_normalized_total` | Explicit audit trail for market calibration; additive rows use `normalization_method = additive_floor`, `pred_salary_scale = 1`, and record the common dollar shift separately |
| `keeper_market_value`, `keeper_contract_discount`, `keeper_pool_inflation` | Pre-auction keeper market value, source-value discount versus deterministic contracts, and remaining-pool budget multiplier |
| `source_market_total`, `source_nonkeeper_market_total`, `source_market_scale` | Copied source-price market totals and the above-floor scale required to express the source curve in the known remaining market |
| `source_salary_floor`, `log_source_salary`, `keeper_adjusted_source_salary`, `budget_adjusted_source_salary`, related `*_diff` columns | Candidate-level source-price features used by the v4 salary model |
| `training_through_year`, `resid_training_through_year`, `resid_training_rows` | Data cutoffs and residual calibration sample size |
| `candidate_pool_rows`, `candidate_pool_covers_slots`, `candidate_pool_source` | Candidate-pool completeness checks |
| `method_version`, `model_spec_asof_year`, `prediction_mode`, `data_rolling_origin`, `fresh_method_holdout` | Model provenance and replay interpretation |
| `normalization_uses_target_actuals`, `date_modified` | Leakage guard and build timestamp |

For each origin, models are fit only on observed non-keeper salaries from prior
years. The normalized prediction uses the known league budget minus keeper
spend across the known open roster slots; it does not use that origin's
realized auction total. `pred_salary_raw` remains available when a study should
avoid aggregate normalization or apply a different market rule.

### `Salary_Validations_Resid`

This is the observed evaluation/calibration subset. It contains non-keeper rows
with realized auction prices beginning with the 2021 calibration origin and
adds `included_in_residual_evaluation = 1`. It otherwise preserves the same
prediction, normalization, cutoff, and provenance fields as the full pool so
raw versus normalized residuals can be evaluated without reconstructing joins.

The current method is a strict rolling-origin replay with respect to training
data, but it is not a fresh historical method holdout: model families and
hyperparameters are selected by the current `model_spec_asof_year` run. That
distinction is recorded by `data_rolling_origin = 1` and
`fresh_method_holdout = 0` and must be retained in downstream reporting.

Target-derived historical market totals (`total_spent` and `fraction_spent`)
are retained only as upstream audit context and are excluded from the model
matrix. The retained projection-versus-price rank gap is computed within
auctionable non-keepers, so deterministic keepers do not shift the bidding-pool
feature value. The candidate universe comes from the preseason `ProjOnly`
tables, not from the realized auction roster or the manually truncated ESPN
copy. Keeper contracts missing origin-year projection features are still added
to the backtest output with deterministic actual contract salaries and a
position inferred from other projection seasons; their discarded model
features cannot alter non-keeper predictions.

Set `SALARY_VALIDATION_DATASETS_ONLY=1` when running the salary script to build
these validation tables without rewriting the live `Simulation.Salaries`,
`Simulation.League_Keepers`, or `Simulation.Salaries_Pred` slices and without
copying the simulation database to the auction app.

## Experimental v3 Projection Features

`current_locked_spec_v3_resid_share_features` extends the v2 point-ensemble
feature surface with two causal preseason feature groups:

- projection residual quantiles from `Final_Validations_Resid` historically
  and `Final_Predictions_Resid` currently, including P75/P90/P95 ceilings,
  P10 downside, 50%/80%/90% interval widths, ceiling-versus-price ranks, and
  ceiling PPG per source dollar; and
- row-median team and position-room shares across the available projection
  sources for projected points, rush attempts, receptions, and receiving yards,
  plus within-row source disagreement for each share family.

Historical residual features use the causal cutoffs documented in
`projection_validation_residual_tables.md`. Deep current players outside the
rank-truncated final ensemble receive residuals interpolated from players at
the same position and similar projected PPG. Early seasons without causal
residual history receive neutral zero residuals rather than future information.
Missing role shares are filled only from the same year/position preseason pool,
then zero when a share family is not applicable, such as QB positional-room
share.

Point, residual, and share fallback flags are retained for audit but excluded
from the salary model matrix. The v3 search uses 15 Optuna iterations per model
family with a 45-second family timeout. This is an experimental specification;
promotion still requires player-error and optimizer-selected affordability
validation.

## Experimental v4 Keeper-Market Features and Additive Calibration

`current_locked_spec_v4_additive_keeper_market_features` retains the v3
projection residual and role-share surface and adds causal keeper-market price
features:

- `keeper_market_value` values each keeper at the copied preseason source price,
  with the deterministic contract used as a neutral fallback when that source
  value is missing;
- `keeper_contract_discount` is keeper market value minus committed keeper
  spend;
- `keeper_pool_inflation` is
  `available_budget / (league_budget - keeper_market_value)`;
- `keeper_adjusted_source_salary` applies that remaining-pool multiplier to
  each copied salary above the `$1` floor; and
- `budget_adjusted_source_salary` expresses the full copied auctionable price
  curve in the known keeper-adjusted remaining budget. Its
  `source_market_scale` is derived only from preseason source prices, open
  slots, and deterministic keeper contracts.

The raw copied salary remains available independently, along with
`log_source_salary` and the adjusted-minus-source gaps, so the ensemble can
select or shrink the market adjustment rather than being forced to reproduce
it.

After the ensemble predicts salaries, v4 replaces proportional point-market
normalization with an additive floor-constrained projection:

```text
pred_salary = max(1, pred_salary_raw + pred_salary_shift)
```

One shift is solved per origin so the highest `available_slots` non-keeper
predictions total exactly `available_budget`. Keeper rows remain deterministic
and are excluded from the shift. The preserved `pred_salary_raw`,
`pred_salary_shift`, and before/after totals permit downstream studies to apply
or compare another reconciliation rule without reconstructing model outputs.

The study under
`research/studies/2026-07-16_additive_salary_normalization/` holds the v3 raw
rolling predictions fixed and finds that additive normalization reduces
player-year MAE/RMSE slightly versus the prior proportional rule. A fresh v4
pipeline build and paired optimizer replay are still required to evaluate the
new input features and selected-roster affordability.

## Superseded v5 Compact Salary Features

`current_locked_spec_v5_compact_salary_features` retains v4 keeper-market
inputs and additive calibration but narrows the fitted feature surface from 155
nonconstant legacy features to 12:

- `budget_adjusted_source_salary` and `avg_pick_log` as complementary
  league-specific and broader-market anchors;
- `ensemble_pred_ppg`, `ensemble_vs_price_gap`, and
  `ensemble_pred_resid_90` for projection level, source-price disagreement,
  and upside;
- `pos_proj_points_share`, `rb_pos_rush_share`, `year_exp`, and `is_rookie`
  for role and breakout context; and
- `QB`, `RB`, and `TE` indicators, with WR as the reference position.

The two high-correlation relationships retained intentionally are adjusted
source salary versus log ADP and the RB-specific rush-share interaction versus
the RB indicator. Replacing log ADP with an adjusted-price rank gap removed the
first correlation but worsened rolling MAE, while centering the RB interaction
slightly worsened the fixed-model ensemble.

`year` and `game_date` remain in the input frame only to define rolling
training/test splits. The pipeline drops both before random feature sampling,
scaling, K-best selection, and model fitting, avoiding a duplicate calendar
feature. Random feature sampling remains enabled with compact 60%-100% column
fractions; K-best now searches 6, 8, 10, or all available features.

The reproducible study under
`research/studies/2026-07-16_salary_feature_reduction/` compares the compact
surface with the archived legacy feature construction across 526 strict rolling
player-years. A fixed six-model ensemble improved MAE from `$4.561` to `$4.312`
and RMSE from `$6.740` to `$6.461`.

The subsequent full rebuild and replay under
`research/studies/2026-07-16_salary_v5_replay/` found normalized all-year
MAE/RMSE of `$4.271`/`$6.197`, better than both preserved v1 and feature-rich
v3. In the identical-seed optimizer replay, v5 raised historical affordability
from 15.5% to 19.0% in development and from 12.0% to 18.5% in 2025 while
reducing average overage. It was the prior production salary surface.
However, selection-weighted actual-minus-point residual remains `+$1.35` per
selected player and the actual-minus-scenario roster gap remains approximately
`$25-$27`, so the next calibration experiment must still target
selection-conditioned error. These are rolling-data development comparisons,
not a fresh method holdout.

## Current v6 V2-Population Salary Surface

The live salary method is
`current_locked_spec_v6_v2_population_11f`. It keeps v5's additive
keeper-market calibration and all compact features except
`ensemble_pred_resid_90`, leaving 11 fitted fields. A strict rolling comparison
slightly favors the 11-field surface: MAE is `$4.2975` versus `$4.2991` for the
12-field version.

The removed feature mixed two incompatible uncertainty meanings. Historical
rows used legacy projection-residual p90, while current V2 uncertainty comes
from a centered matched weekly donor; current QB donor p90 standard deviation
was 9.35 times the historical projection-residual value. The centered donor p90
remains available as a diagnostic, not a salary-model input.

The current beta salary slice is keyed to exactly the 328-player production
population. It contains 326 direct `ProjOnly` rows plus governed V2 fallbacks
for Stefon Diggs and Deebo Samuel. All 14 keepers have canonical keys. After
keeper commitments, the highest 142 non-keeper point salaries total exactly
the `$3,071` available market budget.

`Salaries_Pred` retains canonical `player_key`,
`salary_population_source`, `ensemble_uncertainty_feature_source`, and
`salary_method_version` provenance. Name-only or independently pruned salary
populations are not valid production inputs.

League-scored yardage bonuses and beta sacks flow through weekly
`active_ppg`, matched donor paths, and optimizer selection. They are not forced
into the preseason salary point estimate. Two-point conversions and
special-teams touchdowns remain omitted by explicit modeling decision.

## Auction App Consumption

`Fantasy_Football_App` reads `League_Keepers` for the selected `year` and
`league`, pre-fills keeper costs, and removes unowned keepers from the auction
candidate pool. Keeper rows are excluded from auction-pace calculations because
their inflation is already represented in `Salaries_Pred`.

For each ILP iteration, the app calculates remaining league money and slots from
all keeper commitments plus entered non-keeper auction results. It then rescales
sampled non-keeper salary dollars above the `$1` floor so the top remaining-slot
market exactly equals the remaining league budget. Fixed players use their
entered actual salaries and are excluded from this normalization.

## Optimizer-Selection Premium Tables

`Scripts/Modeling/s5_Auction_Selection_Premium.py` owns a lightweight annual
second-stage calibration. It does not alter `Salaries_Pred`. The point salary
continues to represent the coherent market; the new premium is a separate
decision-price reserve for unresolved players that the optimizer selects more
often than the broader candidate pool.

### `Salary_Selection_Seeds`

This table lives in `Data/Databases/Validations.sqlite3` and has one row per
`(year, league, player)`. It stores the premium-free preseason Target selection
rate, the point salary at that origin, and the realized non-keeper salary once
available. Important fields are:

| Columns | Meaning |
| --- | --- |
| `point_salary`, `selection_rate`, `selection_slots` | Preseason point price and premium-free Target selection signal |
| `seed_trials`, `seed_success_trials`, `seed_random_seed` | Seed-run sample size and reproducibility |
| `actual_salary`, `actual_salary_recorded`, `salary_residual` | Later realized price and `actual_salary - point_salary`; current-season outcomes remain unavailable |
| `salary_method_version`, `seed_method_version`, `generated_at` | Salary and seed provenance |

The initial 2022-2025 history is reconstructed from the validated 250-trial
Target-style `baseline_298` replay. Each later annual refresh retains its saved
seed rates and attaches realized non-keeper prices from `Actual_Salaries` only
after that season is complete.

### `Salary_Selection_Calibrator`

This audit table also lives in `Validations.sqlite3`. It stores the active
ridge intercept and transformed-feature coefficients, target season, strict
training cutoff, alpha, cap, method version, and generation timestamp. The
model uses position, point salary, squared point salary, selection rate,
selection-rate-by-salary, and selection-rate-by-position interactions. A target
season is fit only from observed prior-season rows.

### `Salary_Selection_Premium`

This app-facing table lives in source `Simulation.sqlite3` and is synchronized
by active `(year, league)` slice to the auction app database. Its primary key is
`(year, league, player)`.

| Columns | Meaning |
| --- | --- |
| `point_salary`, `selection_rate`, `selection_slots` | Current premium-free seed inputs |
| `predicted_salary_residual` | Unclipped ridge prediction of actual-minus-point salary |
| `full_premium` | Positive prediction clipped to `$0-$10` |
| `half_premium` | Fixed 50% shrinkage diagnostic |
| `applied_premium`, `premium_strength` | Published reserve and its configured shrinkage |
| `training_through_year`, `training_rows`, `ridge_alpha`, `premium_cap` | Causal cutoff and model audit |
| `salary_method_version`, `seed_method_version`, `premium_method_version` | Full provenance |

The auction app first normalizes the market salary to live remaining money and
slots, then adds `applied_premium` in the salary-cap constraint. It does not
renormalize the premium back out. Entered purchases, keepers, and the explicit
price of a current nominee are deterministic and replace the premium. The UI
therefore keeps Market `$` separate from Reserve `$` and Decision `$`.
For Target candidate Buy/Pass rebasing, only the candidate's sampled Market `$`
reduces remaining league money; the personal Buy coefficient still uses Market
`$` plus Reserve `$`.
The default-off `Use Selection Reserve` control starts every active reserve at
zero. Enabling it applies the persisted reserve for an immediate comparison;
the control never modifies the premium table.

The 2026 v6 refresh completed 1,000/1,000 premium-free Target rosters and
published 314 non-keeper rows. Its expected 13-player roster reserve is
`$8.5598`. Historical calibrator rows remain on the validated v5 surface while
the current seed uses v6; this transfer is explicitly labeled
`historical_v5_selection_surface_to_current_v6_v1`. On common current players,
v5/v6 point salaries have correlation `0.99957` and MAE `$0.274`, supporting
the transfer as a closely aligned current-population update rather than a
claim of fresh historical v6 validation.

## Current Nomination Evaluation

The auction app evaluates a nominated player at one decision price with paired
Buy and Pass branches. Both branches remove the same salary and one roster slot
from the league market. Buy fixes the player on the user's roster; Pass removes
the player from availability. Projection residuals, salary residuals, weekly
templates, and managed-lineup outcomes are shared across each paired trial.

Branch rosters are constructed from managed values averaged across the cached
outcome scenarios; the scored weekly outcome is not available to the roster
optimizer, and organic convergence refinement is disabled. The resulting
rosters are then scored on paired weekly outcomes.

Position maxima are based on the user's roster before the nomination. A
pre-existing position-limit violation is grandfathered at its current count,
but forcing the nominee into the Buy branch cannot create another exception.
Every fixed roster, including a roster that fills all available slots, is sent
through the ILP so salary-cap, position, roster-size, and top-N constraints are
validated rather than bypassed.

The app reports exact managed-roster Buy EV, Pass EV, their season-point edge,
paired win rate, expected starts, and the most common available roster
alternative. Buy and Pass feasibility are counted independently. A price is
eligible for EV, recommendation, and Roster Max Bid only when every requested
trial produces both a valid Buy roster and a valid Pass roster; partial paired
samples are reported as infeasible instead of estimating EV on the surviving
trials. An unconstrained solve on the same decision-price scenarios provides
Fit Rate only when all requested Open-roster solves complete; otherwise Fit
Rate is reported as unavailable with its own completion count.

Roster Max Bid is the highest exact, fully paired integer price whose
non-increasing isotonic Buy Edge is non-negative under the current roster, cap,
market, and lineup settings. The search evaluates a global price grid, then
adaptively resolves both the fitted zero crossing and the full/partial
feasibility boundary to adjacent dollars. Once an evaluated price is only
partially feasible, no higher price is admissible. Roster Max Bid is not the
player's predicted market price. The entered decision price is still evaluated
exactly for the current recommendation, but it is not added to the deterministic
Max Bid search observations unless the search independently chooses that price;
changing only the decision-price input therefore cannot change Roster Max Bid.

Nomination output also includes season-point edge per week, a paired bootstrap
90% confidence interval for mean Buy Edge, and decision tiers (`STRONG BUY`,
`BUY`, `LEAN BUY`, `NEUTRAL`, `PASS`, or `INFEASIBLE`). A decision price above
the fitted Roster Max Bid cannot retain a Buy recommendation from a raw noisy
edge. The app reruns decision-price analysis after separately removing the two
most common alternatives at their market prices to show how scarcity and
league-budget changes affect the nominee.

## Target Board Roster Contribution

The Target Board combines selection frequency with managed roster value. Each
50-trial block builds a fresh bank of 50 managed-value contexts. Every ILP trial
bootstraps five contexts from that active bank with replacement and averages
them. This preserves stochastic build variation without letting a single
realized weekly outcome determine the roster or heavily reusing a small context
bank. For each selected
non-fixed player, the app
rebuilds candidate-rebased forced-Buy and Pass ILPs on the same sampled salary
market, allowing the entire roster to change. Both branch rosters are scored on
a separate weekly holdout draw.

The app reports:

- `Fit Rate`: share of successful simulated builds containing the player.
- `Gain When Selected`: mean holdout season-point advantage over the best
  feasible replacement, conditional on selection.
- `Expected Roster Gain`: mean contribution across every successful build,
  with zero contribution when the player is not selected. This equals Fit Rate
  as a fraction multiplied by Gain When Selected.

The Target Board is ordered by Expected Roster Gain rather than Fit Rate. It is
still a sampled-build option metric: contribution is zero in builds where the
player is not selected and conditional on the salary markets where he fits.
The Current Nomination evaluator remains the source of truth for forced Buy/Pass
decisions at one actual auction price across every scenario.

Current Nomination displays only exact evaluated prices. Raw Buy Edge and Win
Rate remain visible for fully paired prices, while non-increasing isotonic fits
provide the decision curve and determine the highest supported bid despite
Monte Carlo noise and discrete ILP roster switches. Feasibility is never
interpolated between evaluated prices. If the exact decision price is not one
of the deterministic search prices, its fitted chart value may be interpolated
after the Max Bid fit is frozen; that display value does not enter the fit or
the bid calculation.

Nomination roster construction averages managed values across 250 unique
weekly contexts independently of the 100 salary trials. Each salary-trial Buy
and Pass roster is scored across five spread-out weekly contexts and averaged.
This reduces sensitivity to one small objective bank and makes mean Buy Edge a
more stable expected-roster comparison; the paired trial count remains the
number of independently sampled salary markets.
