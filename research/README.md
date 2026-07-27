# Research Index

Purpose: keep standalone validation, calibration, and audit work in dated,
reproducible bundles.

## Layout
- `studies/YYYY-MM-DD_<slug>/`: runnable or reviewable investigations.

## Study Rules
- Each study should include a short `README.md`.
- Durable outputs should live in `results/`.
- Local scratch artifacts should live in `artifacts/local/` when needed.
- Promote lasting conclusions into `Agent_Notes/DECISION_LOG.md` or the relevant
  data contract/runbook.

## Current Study Types
- Best-ball weekly template calibration.
- Template residual blend calibration.
- ADP availability and pruning audits.
- Projection/ADP/name-cleaning join audits.
- Auction Target roster-construction and runtime validation.
- Managed auction played-week and replacement-scoring validation.
- Managed auction rolling-origin policy, salary-risk, and affordability replay.
  The 2026-07-14 nominal-buffer follow-up found that `+$5` improved affordability
  more than `+$10`, but later feasibility-first work supersedes any default choice
  from unconditional points on unaffordable rosters. The sequential fixed-price
  recourse follow-up also selected neither buffer: primary paired-clean completion
  was only 15/72, completion was discordant in 10/72 pairs, and order/year signs
  were unstable.
- Auction salary validation datasets. The 2026-07-14 build stores full
  projection-defined rolling-origin pools and observed non-keeper residual rows,
  preserving raw and keeper-budget-normalized predictions with explicit data and
  method provenance.
- Current-method auction salary buffer replay. The 2026-07-14 paired five-draw
  study confirms that `+$5` consistently buys about 6.5 historical feasibility
  points over `+$10`, but its unconditional point comparison rewards rosters
  that could not be purchased and cannot select a default.
- Salary chance-constraint frontier. The 2026-07-14 exact 60%-90% study separates
  preseason managed forecast EV from salary risk and evaluates affordability on
  unseen normalized five-draw markets. It finds a stable modeled frontier but a
  persistent roughly $29 actual-minus-modeled spend gap for selected rosters, so
  the marginal-residual market model needs roster-level calibration before a
  buffer or chance threshold is deployed.
- Selected-roster salary residual diagnostics. The 2026-07-14 follow-up
  reconciles all 52,000 chance-frontier roster slots and finds that unique
  ever-selected players resemble the full pool, while repeated selection weights
  concentrate positive residuals: frequent/core and strongest value-rank players
  drive the bias. It also decomposes the roughly `$29` gap into point-salary error
  and an additional selected-roster discount from normalized five-draw scenarios.
- Projection validation residual backfills. The 2026-07-16 study causally
  calibrates existing OOS projection rows without retraining point models,
  applies the required next-horizon outcome embargo, and persists a
  production-style `Final_Validations_Resid` roll-up with explicit source and
  target-availability provenance.
- Joint weekly-template rolling validation. The 2026-07-22 production-scale
  replay holds out 1,620 player-seasons and restricts every weekly donor to a
  strictly earlier season. Pool centering delivers near-zero recent PPG and
  managed-contribution bias with 80.8% P10-P90 coverage, but player-level
  residual-upside discrimination remains weak: the joint paths identify
  absolute contribution better than `+3`/`+5 PPG` surprise.
- Weekly-template context ablation. The 2026-07-23 strict rolling replay tests
  experience-decayed NFL draft capital, a causal projected supporting-cast
  percentile, and recency across 1,620 held-out player-seasons. The full
  combination is safe but does not consistently beat recency alone. Eight- and
  twelve-season recency priors improve played-games CRPS with neutral
  PPG/contribution calibration; eight seasons cuts 10+-year donor weight from
  32.3% to 20.5%, while twelve is slightly safer in the temporal checks. A
  four-season half-life is too aggressive. Draft capital and team environment
  do not earn global template weights and remain candidates for a separate
  upside layer or diagnostics. See
  `studies/2026-07-23_template_context_ablation/`.
- Weekly-template feature pruning. The 2026-07-23 paired backward-ablation
  replay evaluates 30 feature/recency specifications on the same 1,620 strict
  rolling targets. Aggregate and position-level guardrails select removal of
  the redundant projection-by-experience interaction together with the
  conservative 12-season recency prior. Direct projected PPG and uncapped
  experience remain. The specification modestly improves PPG, managed
  contribution, and played-games CRPS in untouched 2023-2025; nested rolling
  selection chooses it at every 2023-2025 origin. Component ranks and the
  remaining market, disagreement, room, concentration, and team-context
  families stay in the matcher. See
  `studies/2026-07-23_template_feature_pruning/`.
- Weekly-template weight sensitivity. The 2026-07-23 strict rolling replay
  perturbs each retained feature family by +/-25% around the pruning
  recommendation and sweeps overall distance sharpness from near-uniform to
  1.5x across 58,320 target-method rows. Individual family changes are
  essentially flat: the best development improvement is 0.012%. Near-uniform
  weighting improves older development origins but reverses slightly
  temporally, worsens 2023-2025 played-games CRPS with its clustered interval
  above zero, and fails exact nested-repeatability and temporal position
  guardrails. Retain the reference relative weights for the pending update and
  treat sampling-kernel sharpness as a separate future calibration problem. See
  `studies/2026-07-23_template_weight_sensitivity/`.
- Additive salary normalization. The 2026-07-16 audit holds v3 raw rolling
  predictions fixed and compares the prior proportional-above-floor market
  reconciliation with an exact additive `$1`-floor projection. The additive
  rule modestly improves player-year MAE/RMSE while preserving high-end dollar
  differences; the v4 keeper-market features still require a fresh full build
  and optimizer replay.
- Auction salary feature reduction. The 2026-07-16 audit reduces the v4-era
  matrix from 155 nonconstant legacy features to 12 causal market, projection,
  role, development, and position inputs. The compact fixed-model ensemble
  improves rolling MAE/RMSE, while retaining two intentional high-correlation
  pairs whose decorrelated substitutes weakened accuracy.
- Salary v5 full replay. The 2026-07-16 study rebuilds player-price metrics,
  reruns the identical-seed 4,000-cell chance frontier, and diagnoses the actual
  v5-selected rosters. v5 improves v1/v3 player error and v1 historical
  affordability, but leaves a roughly `$25-$27` actual-minus-scenario roster
  gap concentrated in frequent/core and strongest-value selections.
- Blind sequential salary-bias replay. The 2026-07-23 paired rolling study
  decomposes static selected-roster error into a `$14.4` full-surface
  scenario-shopping discount and a `$16.8` actual-minus-point residual. Blind
  initial plans remove the first component but retain a similar `$17.3`
  residual. The half reserve raises initial historical feasibility to 52.3%;
  live recourse is cap-legal but currently strands roughly `$45` per completed
  roster, making spending efficiency the next calibration target.
- Optimizer selection surcharge. The 2026-07-16 rolling-origin study fits a
  heavily regularized position/salary/preseason-selection correction and
  applies positive residual estimates as a separate decision-price reserve.
  Half shrinkage improves affordability with negligible 2025 realized-point
  change; full shrinkage is more targeted than a flat `$285` cap. The app now
  uses the half-strength correction through a standalone annual premium-free
  Target seed, while the study's low absolute feasibility remains a warning
  that the reserve is not complete salary-risk coverage.
- Target staged evidence. The 2026-07-17 production smoke separates 40-trial
  discovery from fresh 60-trial confirmation, applies a position-neutral
  empirical-Bayes effect model with block-aware uncertainty, and requires
  fresh independent support for confirmed targets. Two selection-only broad
  pilot scenarios scan the remaining eligible pool and append four discoveries
  to 20 protected heuristic names;
  pilot values are discarded before evidence. Final rows use tier-local A/B
  forced LCB80 ranks and a separate C Organic Gain watchlist. Fresh confirmation
  preserves eight evidence leaders and appends up to four highest-market-salary
  usable preliminary candidates without changing promotion gates. The same
  study verifies market-anchor visibility, dynamic Top-N Pass replacement, tier
  contracts, and exact one-versus-eight-worker invariance.
- Target roster convergence. The 2026-07-19 paired live-database study repeats
  the exact-scored organic correction until no swap improves, capped at 12. The
  final calibrated 64-trial replay raised holdout season EV by 42.0 points,
  lifted both distribution tails, averaged 3.78 accepted swaps, and never hit
  the cap. Forced Buy/Pass and Current Nomination remain unrefined; the annual
  premium-free selection seed follows the converged organic policy.
- Sequential Target seed stability. The 2026-07-20 live-state study confirms
  that a shared root seed is not player-nested when the remaining pool changes:
  current AJ-on/off Bijan evidence correlated only 0.25, while a research-only
  full-universe keyed sampler raised correlation to 1.00 and reduced the paired
  AJ effect SD from 23.95 to 1.03. Across 16 independent nested banks, however,
  Bijan gain still had 21.07 seed SD. Decomposition attributes more variation to
  the 32-context policy construction (16.49 SD) and 64 weekly seasons (12.69)
  than to 48 auction paths (3.95). Production v4 now uses full-universe keyed
  evidence and four partial-crossed blocks; each block has 32 balanced mean-PPG
  construction templates, 12 realized auction paths, and 64 complete seasons.
  A matched 16-root replay raised AJ-on/off correlation to 0.993, held the AJ
  state delta to +0.37 mean / 0.96 SD, and reduced Bijan seed SD from 21.07 to
  7.99-8.15. Between-block disagreement is included in the action SE. See
  `studies/2026-07-20_sequential_seed_stability/`.
- Managed bench call-option and waiver-hurdle replay. The 2026-07-19 frozen
  four-origin study rejects a higher construction waiver threshold as a proxy
  for studs-and-scrubs: `+1` bought a more expensive bench while lowering
  common-bank forecast mean and p10, and larger hurdles discarded substantial
  lineup value. Removing the current P90 bench heuristic cheaply improved the
  roster shape without a forecast decline, while a 0.25 sustained-breakout
  utility improved forecast and playoff scoring but needs a stricter 15+ PPG
  strike and current salary-risk replay before promotion.
- Keeper-aware bench option replay. The 2026-07-19 study encodes two keeper
  slots, annual `+$10` escalation, and a three-year maximum through
  position-specific projected-PPG-to-market-dollar curves. In 7,000 paired
  solves, a tiny keeper tie-break raised independent forecast mean and p10 in
  all four origins but reduced average realized next-year keeper surplus versus
  the same-engine zero-option control. Stronger weights were unstable and the
  selected signal became highly player-concentrated, so no keeper bonus was
  promoted. See `studies/2026-07-19_keeper_option_bonus/`.
- One-year keeper portfolio replay. The 2026-07-20 revision uses causal
  historical next-model residuals, scores the best first-year surplus across
  all five bench slots, fixes all eight starters, and permits at most two
  bench-only no-loss swaps. It improved best realized keeper surplus by
  `$17.9`/`$12.5`/`$6.4` in the three observable origins, increased `$20+` hit
  probability, and kept four-origin forecast mean/p10 neutral to positive.
  Advance the zero-loss policy to a current v5-plus-reserve replay; no live app
  behavior changed. See `studies/2026-07-20_one_year_keeper_portfolio/`.
- Full-roster keeper reinvestment sensitivity. The 2026-07-20 follow-up rebuilds
  the current-only control on the full cached-bank expected profile, forces up
  to one, two, or three proposed options to remain on the nominal bench, and
  re-solves every other slot. K1 shifted `$5.7` from bench to starter spend and
  improved starter-only mean/p10 by `+4.6`/`+3.6`, but full-roster mean/p10
  improved only `+0.6`/`+1.2` and realized best keeper surplus only `+$2.1`.
  K1/k2/k3 count incremental forced additions, not total lottery tickets or
  exclusive bench roles. Keep production unchanged; next score every bench
  player on both current fill-in and keeper value, protect roster mean/p10 and
  aggregate bench coverage, and let composition emerge without hard age/slot
  quotas. See `studies/2026-07-20_keeper_reinvestment_sensitivity/`.
- Soft whole-bench keeper portfolio. The 2026-07-20 replay removes incremental
  keeper-count and age/role quotas, gives all five bench players dual current and
  expected-best first-year keeper-option value, and fully re-solves the roster
  subject to construction mean/p10 gates. It improved predicted/realized best
  keeper surplus by `+$4.3`/`+$4.4`, shifted `$2.1` to starters, and kept average
  independent mean/p10 positive. The control already averaged 4.07 effective
  options, so a fixed ticket count is not the right output. Keep production
  unchanged: only 46.2% of changed rosters improved both held-out current-year
  metrics, and 2024 declined. Next strengthen or cross-fit the gate and require
  material keeper utility. See `studies/2026-07-20_soft_keeper_portfolio/`.
- Veteran cliff calibration. The 2026-07-21 rolling historical study uses
  uncapped experience and separates current PPG cliffs, extended absence, and
  following-season disappearance. It does not support a uniform current-year
  veteran tax: RB/TE current risk is weak after projection and season controls,
  while WR current cliff risk is materially positive. The dominant missing
  effect is next-season attrition, especially additional WR/TE years. The audit
  also finds that the next-model target forward-fills many no-appearance rows
  with prior PPG, understating above-threshold no-useful rates by 15-24 points.
  Fix that target and rebuild next-year calibration before deploying any
  residual penalty. See `studies/2026-07-21_veteran_cliff_calibration/`.
- Current-season veteran value. The 2026-07-22 study excludes keeper and
  following-year outcomes, reconstructs the auction app's weekly contribution
  above waiver, and matches above-threshold veterans to younger same-position,
  same-season peers on preseason PPG plus market ADP or rolling v5 salary. Across
  4,206 historical seasons, premium RB mean value was neutral with mildly fewer
  misses and upside hits; premium WRs averaged 6.2 fewer managed points and a
  10.2-point higher miss rate in Top-100 matches, directionally stable to every
  leave-one-season-out slice but uncertain across player clusters. Premium TE
  matches and the 2022-2025 `$5+` auction cells were too sparse for calibration.
  Do not deploy a blanket age tax or alter point forecasts. If desired, test a
  transparent uncapped-experience ceiling preference and separately validate a
  premium-WR current-outcome mixture. See
  `studies/2026-07-22_current_veteran_value/`.
- Joint weekly template outcomes. The 2026-07-22 production rebuild adds
  absolute PPG, projection-disagreement, and full-room workload matching; uses
  capped adaptive local donor weights; retains ordinary zero-active downside;
  and excludes only the declared 2018 Le'Veon Bell holdout. The managed auction
  app now draws the matched donor's centered PPG residual and weekly trajectory
  together, with no independent current-year residual blend. See
  `studies/2026-07-22_joint_template_outcomes/`.
