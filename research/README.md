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
- Projection V2 NFFC-market and expert-rank challengers. The 2026-07-30 replay
  reduces the NFFC family from its composite plus four correlated contest rows
  to the single existing `ADP_Averages` composite. The live market/rank mart
  falls from 31,834 to 28,801 rows, and locked RMSE improves slightly from
  3.10783 to 3.10756 DK and from 2.88446 to 2.88411 beta. A strictly-prior
  expert-rank study converts every source to a season-position percentile
  before taking a cross-source median. A full-column-forest attribution control
  removes the locked forest's feature-subsampling confound: normalized rank
  level improves the corresponding blend by 0.00221 DK and 0.00186 beta with
  seven of nine season wins. DK's two intervals exclude zero; beta's season
  interval excludes zero and its player interval ends essentially at zero. The
  locked-production-surface sensitivity is smaller and uncertain, and the
  expert-minus-projection gap is neutral. A scoring-matched raw-rank follow-up
  rejects percentile-after-median plus coverage: controlled gains are only
  0.00138 DK and 0.00016 beta, all intervals cross zero, and it trails the
  normalized comparator by 0.00173/0.00213. `log1p(raw median)` is not
  distinguishable from normalized rank in this study's direct paired
  comparisons, so it remains a diagnostic rather than a post-hoc promotion.
  Promote the one-vote NFFC
  contract; retain normalized expert rank as a challenger and raw rank as
  audit-only.
  ETR overall rank remains the beta top-180 eligibility ordering. See
  `studies/2026-07-30_v2_market_rank_challengers/`.
- Projection V2 QB target decomposition. The 2026-07-29 strict rolling study
  fits the same QB-only Lasso/RF/LightGBM surface to total conditional PPG and
  to realized passing/rushing PPG separately, then sums the component
  forecasts. The component sum plus a causal other-points adjustment worsens
  DK RMSE by 0.0414 versus the same-model direct-total blend and improves beta
  by 0.0270; both player-cluster intervals cross zero, recent 2023-2025 results
  worsen in both leagues, and the directions disagree. Rookie-QB and beta
  high-rush slices are exploratory only. Retain the direct total target and
  preserve the component route as a diagnostic. See
  `studies/2026-07-29_v2_qb_component_targets/`.
- Projection V2 weekly-scoring and FFToday-vintage correction. The follow-up
  binds every historical weekly position scorer to an explicit league, carries
  a scoring marker into template construction, and rejects league/marker or
  locked-V2-objective mismatches. It also quarantines 50 FFToday QB rows stored
  under 2018 that match the provider's native 2019 vintage. Rebuilt DK/beta
  foundations have 6,655 identities, 55,914 aliases, 31,798 projection values,
  and 13,909 feature rows each. Locked primary RMSE is 3.1078 DK and 2.8845
  beta versus 3.1951/2.9600 expert recalibration, both 9/9; next-year RMSE is
  3.9137/3.5538 versus 5.2351/4.3653 expert carry, both 8/8. In the rebuilt
  weekly slices, 5,120/5,298 paired active-PPG values and 5,147 paired paths
  now differ across leagues; 39 beta 2018 QB rows retain an audited legacy
  center with the unavailable V2 diagnostic left explicit. The byte-identical
  production cutover passes handoff idempotence, all 11 auction-table parity
  checks, Snake/source SHA-256 equality, and integrity/foreign-key checks. See
  `studies/2026-07-29_v2_weekly_fftoday_correction/`.
- Projection V2 identity/scoring revalidation (superseded foundation). The
  earlier 2026-07-29 corrective
  study consolidates governed aliases, fixes two mislabeled FantasyPros WR
  seasons with stored/effective provenance, removes `last_season` as a hard
  identity endpoint, and requires same-position sack estimates for beta QBs.
  Its identity, source-season, and beta-sack methods remain accepted, but its
  exact counts and model metrics are superseded by the FFToday quarantine
  follow-up. See
  `studies/2026-07-29_v2_identity_scoring_revalidation/`.
- Projection V2 following-season residual and appearance modeling. The
  2026-07-29 study predicts `t+1` conditional PPG as a residual around the
  origin-`t` expert projection and separately predicts any `t+1` appearance.
  A one-origin embargo means forecasts at `t` train only through origins
  `t-2`. The latest quarantined-lineage replay improves RMSE from
  5.2351/4.3653 expert carry to 3.9137/3.5538 and wins all eight origins.
  Both publish 745 canonical 2027 shadow rows. In the corrected strict
  1,620-target template replay, next-residual rank improves DK PPG CRPS by
  0.002006 but worsens contribution CRPS by 0.031825 with both intervals
  excluding zero; beta's 0.001110 PPG gain and 0.000771 contribution loss are
  uncertain. The signals remain outside production matching even though their
  two-stage outputs feed keeper valuation. The original study is retained as
  historical evidence; corrected metrics live in the weekly/FFToday correction
  bundle.
- Projection V2 locked whole-season and 2026 shadow validation. The 2026-07-29
  study freezes exact feature hashes, compact grids, and the fixed pooled
  Lasso/RF/LightGBM blend, then forecasts every 2017-2025 season using only
  earlier-season fitting, selection, routing, and calibration evidence. The
  latest one-vote-NFFC replay scores 3.1076 DK and 2.8841 beta RMSE
  versus 3.1951/2.9600 expert recalibration, both with nine of nine wins;
  prior-only point calibration remains rejected. Each fit-through-2025 run
  publishes 745
  unique 2026 shadow rows. Canonical
  `player_key` now covers 100% of both league template/map audit populations,
  and reconstruction passes with one joint donor residual/path.
  The original study is retained as historical evidence; corrected metrics
  live in the weekly/FFToday correction bundle.
- Projection V2 rolling OOF modeling. The 2026-07-27 M4A study uses the
  existing five-fold per-season `SciKitModel` scheme while fitting every held
  player-season only on prior years. It compares expert/hybrid baselines,
  Ridge/logistic, shallow LightGBM, KBest, PCA, and agglomeration on 2017-2025,
  fixes unresolved identity joins that had been false participation zeros, and
  runs fold-identical feature-family dropouts. Direct shallow LightGBM is the
  conditional-PPG leader but only modestly beats simple recalibration/full
  Ridge; participation LightGBM beats logistic, while transformed pipelines
  lose to raw full features. See
  `studies/2026-07-27_v2_modeling_framework/`.
- Projection V2 legacy-inspired feature challengers. The 2026-07-28 study
  rebuilds 12 leakage-safe experience-context, self-excluded teammate-ADP, and
  team-opportunity features from concepts in the legacy compile script. Neither
  Ridge nor shallow LightGBM materially improves, the full addition worsens
  both, and the old projection-versus-experience difference is flat. A
  deterministic full-column follow-up catches a false attribution caused by
  LightGBM column subsampling when an unavailable feature expands the matrix.
  No feature is promoted; all 12 remain in a separate research manifest. See
  `studies/2026-07-28_v2_legacy_feature_challengers/`.
- Projection V2 regularized linear sparsity. The 2026-07-28 study compares
  direct Ridge, Lasso, and Elastic Net on identical 2017-2025 rolling OOF folds
  using both the 31-feature incumbent and incumbent plus 12 legacy
  challengers. Lasso is the strongest linear challenger, selecting a mean 23.6
  of 35 raw incumbent inputs and improving RMSE by 0.0091 versus fold-identical
  Ridge, but the nine-season interval crosses zero. Expanded Lasso improves
  only another 0.0041, is weaker for rookies/second-year players, and remains
  behind direct shallow LightGBM. Keep it as a sparse diagnostic; no model or
  feature manifest is promoted. See
  `studies/2026-07-28_v2_regularized_linear_sparsity/`.
- Projection V2 standardized provider challengers. The 2026-07-28 study
  reconstructs provider fantasy points from configured DK raw-component
  scoring, permits one guarded cross-provider component imputation, removes
  published provider totals/PPG from modeled inputs, and requires three prior
  projection seasons before provider-specific columns become learnable. It
  tests 10 rate/opportunity, eight disagreement, and eight provider features.
  Only the deterministic LightGBM provider family improves its same-model
  point estimate (-0.0085 RMSE), with an interval crossing zero; FFToday
  supplies most of the exploratory result. No additions help rookies and none
  are promoted. See
  `studies/2026-07-28_v2_projection_feature_challengers/`.
- Projection V2 consensus ladder. The final 2026-07-28 projection-only study
  tests a strictly causal constrained provider stack, compact room
  disagreement, active-PPG alignment, projection shape, and all additions
  together in both projection-only and full residual models. None improves the
  full model. The raw provider stack gain is small and unstable, while the
  projection-only consensus core finishes only 0.0097 RMSE behind the full
  LightGBM. A causal position-by-history router is the remaining finalist,
  improving the full-model point estimate by 0.0055-0.0097 RMSE with intervals
  crossing zero. See
  `studies/2026-07-28_v2_projection_consensus_ladder/`.
- Projection V2 position-specific models. The 2026-07-28 study fits the
  projection-core and governed full LightGBM independently as QB/RB/WR/TE and
  as QB/RB/WR+TE components. Four complete position splits worsen both models;
  QB/RB full slices are effectively tied while independent WR/TE lose useful
  shared structure. The three-role full model ties pooled full over 2017-2025
  and has a favorable but three-season-only 2023-2025 point estimate. Keep
  pooled models primary and carry only the three-role full fit into temporal
  robustness validation. See
  `studies/2026-07-28_v2_position_specific_models/`.
- Projection V2 position-aware feature families. The 2026-07-28 study tests
  experience-relative projection, teammate ADP, role-specific team
  opportunity share, and richer room-clarity additions independently for QB,
  RB, WR, and TE. No family survives correction across 16 prespecified tests.
  QB room clarity is the strongest overall direction; rookie QB room clarity,
  rookie WR experience context, and young-TE teammate ADP are exploratory
  slice signals only. Opportunity shares are redundant to harmful for mean
  PPG. Promote nothing and preserve the slice hypotheses for whole-season and
  template-distribution analysis. See
  `studies/2026-07-28_v2_position_feature_families/`.
- Projection V2 KNN and random forest. The 2026-07-28 study runs scaled KNN
  and compact random-forest grids on projection core and the governed full
  feature set. KNN is decisively worse in standalone and fixed-blend tests.
  Full RF ties full LightGBM, while the prespecified 50/50 RF/LightGBM average
  improves RMSE from 3.1230 to 3.1143 with a season interval narrowly crossing
  zero. Carry RF and the untuned blend into whole-season/calibration
  validation; do not promote yet. See
  `studies/2026-07-28_v2_knn_rf_models/`.
- Projection V2 linear/tree blends. The 2026-07-28 current-lineage study
  refits governed full Lasso beside RF and LightGBM. Lasso is weaker alone but
  less error-correlated with either tree than the trees are with each other.
  A fixed full Lasso/RF/LightGBM equal-third average improves RMSE from 3.1143
  for the tree average to 3.1000, wins all nine season comparisons, and has an
  interval fully below zero. A causal prior-season-weighted blend confirms the
  signal at 3.1027. Carry both into whole-season/calibration and joint-template
  validation; production remains unchanged. See
  `studies/2026-07-28_v2_linear_tree_blends/`.
- Projection V2 projection-anchored history gaps. The 2026-07-28 study replaces
  missing absolute historical PPG with a zero adjustment to each player's own
  current expert baseline and adds explicit availability, recency, and
  opportunity-game reliability. The construction fixes the intended
  no-history semantics and improves rookie/no-career point estimates, but
  Lasso is flat-worse overall and the 0.0029 equal-third blend gain reverses
  recently with an interval crossing zero. Keep the 13-feature family as a
  governed sparse-history/router challenger rather than globally replacing
  the incumbent. See `studies/2026-07-28_v2_history_gap_features/`.
- Projection V2 preseason trajectory and logged ADP. The 2026-07-28 study
  compares each current consensus projection with the same player's exact
  prior-year and recency-weighted prior-three-year preseason projections.
  Exact one-year change alone adds little; the three-year context is stronger,
  and the combined five-field family improves the equal-third Lasso/RF/LightGBM
  blend by 0.0051 RMSE with a slight recent-period gain. Replacing raw ADP with
  logged ADP materially improves Lasso but not the trees. The pooled-best
  combined blend reverses slightly in 2023-2025, so keep trajectory and
  model-specific logged ADP as whole-season/template finalists rather than
  changing the incumbent. See
  `studies/2026-07-28_v2_projection_trajectory_adp/`.
- Projection V2 team environment and QB style. The follow-up study separates
  QB1 passing and rushing projection context, capped supporting-cast strength,
  team rushing, and non-duplicated offensive-TD environment. The full
  11-feature family is flat globally. QB1 rushing fantasy-point share is the
  strongest compact field: it improves the trajectory blend by 0.0021 RMSE
  and helps WR/TE by 0.0055/0.0068 while remaining neutral for RB. A
  same-evidence pass-catcher-only route improves 0.0033, wins seven of nine
  seasons, and has an interval narrowly crossing zero. Carry it into
  whole-season/template validation rather than changing the point model. See
  `studies/2026-07-29_v2_team_environment/`.
- Best-ball weekly template calibration. The 2026-07-31 role-tiered study
  replaces the equal-third core-player selection objective with hierarchical
  active-PPG CRPS, contribution tie-breaking, aggregate availability
  guardrails, and replacement-aware roster scoring. A fresh 2,647-target per
  league replay selected the historical 0.25x distance candidate as its only
  one-SE finalist: development core PPG improved 0.007901 DK and 0.005511
  beta. It nevertheless failed downstream 20-player roster CRPS
  non-inferiority, worsening DK 0.7096% in development and 0.5696% in
  2023-2025 while beta was slightly worse. Missed-week calibration stayed
  within margin, so retain production matching but use the role-tiered
  validation policy going forward. See
  `studies/2026-07-31_template_role_tiered_validation/`.
- nflfastR receiver-profile matching. The 2026-07-31 causal replay attaches
  only prior-season target share, air-yards share, aDOT, red-zone target share,
  and targeted-week dispersion; opportunity shrinkage makes rookies and
  missing histories neutral. None of four normalized WR/TE bundles improves
  development core PPG CRPS in both DK and beta. The profiles reduce the
  visible Ladd/Pryor mismatch but do not improve predictive scoring, so
  production remains unchanged. TE-only usage/depth is a post-hoc follow-up,
  not a promotion candidate. See
  `studies/2026-07-31_template_fastr_receiver_profiles/`.
- nflfastR RB role matching. The 2026-07-31 causal replay tests prior-season
  red-zone/goal-line carry room share and third/fourth-down target room share
  with opportunity shrinkage and neutral rookie handling. No global RB arm
  improves development core PPG in both DK and beta, so production and roster
  simulation remain unchanged. Passing-down share improves the depth-player
  composite in all four development/temporal league cells, but that depth-only
  slice is post-hoc and requires a separately frozen confirmation. See
  `studies/2026-07-31_template_fastr_rb_roles/`.
- NFFC weekly-template center calibration. The 2026-07-31 strict rolling replay
  uses NFFC-scored V2 preseason matcher context, 2021-forward donors, a 17-week
  horizon, and 540 held-out targets from 2023-2025. The locked OOF donor center
  worsens PPG CRPS by 0.002901, loses all three seasons, and has a
  player-cluster interval of `[-0.004914, +0.010748]`; it passes six of ten
  gates but fails all three promotion gates. Retain
  `nffc_scored_expert_consensus`; keep the DK-scored `Model_Inputs` context and
  locked OOF center diagnostic only. See
  `studies/2026-07-31_nffc_template_center_replay/`.
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
- Weekly-template projection-weight sensitivity. The 2026-07-29 strict rolling
  replay tests higher absolute PPG, component-rank, and scoring-aligned raw
  component-magnitude weights in DK and beta. The explicit-league corrective
  replay retains production weights: DK's 2.25 PPG weight worsens PPG CRPS by
  0.003261 with interval above zero; beta improves PPG CRPS by only 0.000351
  with interval crossing zero while worsening played-games CRPS by 0.002868
  with interval above zero. The earlier DK-routed beta claim that every global
  bump worsened every metric is superseded. See
  `studies/2026-07-29_v2_weekly_fftoday_correction/` and the original
  `studies/2026-07-29_template_projection_weight_bump/`.
- Weekly-template receiver-rate ablation. The 2026-07-30 strict rolling study
  tests preseason projected receiving yards/reception and TDs/reception on
  1,620 targets per league. The primary WR/TE arm changes roughly nine of 80
  donors and modestly improves PPG/contribution point estimates, but fails
  played-games or impact guardrails; RB does not benefit. Do not add the rates
  globally. TE yards/reception is the useful same-evidence hypothesis, with
  full-period contribution gains and intervals below zero in both leagues, but
  DK recent played-games safety fails, so require independent TE-only
  confirmation. See
  `studies/2026-07-30_template_receiver_rate_ablation/`.
- Weekly-template height/weight ablation. The 2026-07-30 strict rolling study
  joins the existing nflverse player master by exact governed IDs and tests
  season-position height and weight percentiles on 1,620 targets per league.
  Coverage is complete for rolling targets and 5,291/5,298 historical
  templates. The primary arm changes about 9% of donors overall and 12% for
  WR, but beta's small broad improvements reverse to DK PPG, contribution, and
  impact losses. Height alone is essentially neutral in DK and weakly
  favorable in beta. Keep production unchanged and defer combine acquisition
  unless a separate athletic-testing hypothesis is prespecified. See
  `studies/2026-07-30_template_height_weight_ablation/`.
- WR template PPG/profile trade-off. The 2026-07-30 follow-up jointly tests a
  WR-only 2.25 PPG weight with projected YPR and TD/reception on 648 held-out
  WR targets per league. The tighter weight reduces Ladd McConkey's current
  beta donor PPG gap by 20.8% for 0.135 fewer expected games, but worsens
  held-out DK PPG CRPS and does not transport with the rate terms. Terrelle
  Pryor remains a top-three donor and gains probability because his preseason
  YPR/TD-rate projections are close to Ladd's. Keep production unchanged; a
  historically complete direct role signal such as aDOT or alignment is the
  cleaner next archetype test. See
  `studies/2026-07-30_wr_template_ppg_profile_tradeoff/`.
- Weekly-template projection trajectory. The 2026-07-30 WR-only study tests
  signed current-minus-prior preseason projection gaps at 0.25/0.50 weights,
  including one-year, recency-weighted three-year, combined, and explicit
  history variants. Trajectory demotes Pryor from rank 3 to 12 in Ladd's beta
  pool at the one-year 0.50 weight, but every arm worsens held-out WR PPG CRPS
  in both leagues over both full and recent periods. Three-year 0.25 improves
  several contribution/impact diagnostics but fails the PPG-first requirement;
  history availability/depth is harmful. Retain trajectory as an explanation
  field, not a production distance criterion. See
  `studies/2026-07-30_template_projection_trajectory/`.
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
