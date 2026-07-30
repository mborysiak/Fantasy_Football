# Session Notes Landing

Last updated: 2026-07-29

## Project Objective

Maintain and improve the fantasy football modeling pipeline that builds player
projections, residual distributions, simulation inputs, and best-ball weekly
template tables consumed by downstream draft apps.

## Current Focus

- A strict 2017-2025 QB target-decomposition study now compares identical
  QB-only Lasso/RF/LightGBM models fit directly to total conditional PPG versus
  fit separately to passing and rushing PPG. The component sum plus a
  strictly-prior other-points adjustment worsens DK RMSE by 0.0414 and improves
  beta by 0.0270, but both player-cluster intervals cross zero and 2023-2025
  worsens in both leagues. Retain the direct total target; rookie-QB and beta
  high-rush slice gains remain exploratory.
- The V2 identity/scoring correction is active and fully replayed. DK and beta
  now share exactly 6,655 identities, 55,914 aliases, and 13,909
  player-season feature keys. Tetairoa McMillan keeps production key
  `c16a5e67-fff0-57b9-838c-c8df91df7b9d`; Amon-Ra and Equanimeous St. Brown
  truncations resolve to one confirmed identity each; returning players are
  not split by a hard `last_season` bound. FantasyPros WR stored 2016/2020
  snapshots are governed as effective 2018/2021 with complete provenance.
  Beta standardized QB scores require sacks and disclose same-position donor
  lineage. A policy-hashed quarantine now removes the 50 FFToday QB rows
  stored as 2018 that match the provider's native 2019 vintage from every V2
  identity, candidate, value, and feature path while preserving the raw source
  and native 2019 rows. The latest durable evidence is
  `research/studies/2026-07-29_v2_weekly_fftoday_correction/`.
- Weekly template scoring is now explicitly league-bound. Each positional
  scorer receives the requested league, the weekly frame carries a transient
  scoring marker, and template construction rejects mixed or mismatched
  markers before stamping the league and template-ID offset. Staged builds
  require an explicit matching V2 database and disable app sync. This
  supersedes the prior beta weekly slice, which was labeled beta after using
  the default DK scoring dictionary. Realized yardage bonuses continue to
  flow through weekly `active_ppg` and upside paths. The production rebuild has
  5,120/5,298 paired PPG values and 5,147 paired weekly paths differing across
  leagues. Beta has 2,657/2,696 V2 historical diagnostic centers; the 39
  unavailable 2018 QB diagnostics retain an audited legacy active center and
  exact quarantine-linked reason rather than importing DK or zero-sack values.
  Corrected 1,620-target/league replays retain current match weights and keep
  next-year fields out of the matcher: their small PPG gains do not survive the
  joint contribution/played-games gates. The promoted V2/Simulation artifacts
  match staging byte-for-byte; all 11 auction generated tables match source,
  Snake equals source by SHA-256, and app-owned auction content is unchanged.
- The V2 production handoff is active for 2026 DK (268 players) and beta (180).
  Current `pred_fp_per_game` uses the locked league-specific V2 center; current
  residual quantiles are zero and the joint matched donor supplies the only
  PPG residual plus weekly/played path. Next-year `pred_fp_per_game_ny` is
  conditional on appearing and `pred_appear_ny` supplies the separate
  Bernoulli risk; the auction keeper path zeros future market value on no
  appearance. Historical donor residuals retain the validated legacy OOS
  centers. V2 donor centers remain diagnostic with
  `v2_recenter_promoted = 0`: recentering worsened PPG CRPS by 0.0057 DK and
  0.0051 beta, with both player-cluster intervals above zero. Final source,
  auction, and Snake databases have 448 keyed current rows and 10,596 keyed
  templates with no handoff mismatch.
- Weekly template and current player-map tables now require canonical V2
  `player_key` plus match provenance. Governed source aliases resolve first,
  confirmed identities beat redundant provisional aliases, team disambiguates
  true same-name collisions, and pre-play rookies retain stable provisional
  keys. Coverage is 10,596/10,596 historical beta+DK templates and 448/448
  current rows in the source, auction-app, and Snake databases. DK and beta V2
  audits now join both historical and current league populations at 100%.
- Beta has a separate rebuilt `Projection_V2_beta.sqlite3` lineage with exactly
  the same 6,655 identities and 13,909 player-season keys as DK but independent
  outcomes, provider scoring, consensus features, fits, and hyperparameters.
  `v2_conditional_ppg_2026_candidate_beta_v1` scores 2.8845 RMSE versus 2.9600
  for expert recalibration and wins 9/9 seasons. The no-history route remains
  an unpromoted secondary diagnostic; its pre-quarantine exact metric is
  superseded. Prior-only point calibration is rejected. The beta fit publishes
  745 shadow rows, 673 PPG centers, and 745 participation probabilities.
- Projection V2 now has a versioned DK shadow lock:
  `v2_conditional_ppg_2026_candidate_v1`. A complete-season, strictly-prior
  2017-2025 replay scores 3.1078 RMSE versus 3.1951 for expert recalibration
  and wins all nine seasons. The primary remains fixed pooled
  Lasso/RF/LightGBM equal thirds with the five preseason trajectory fields;
  participation remains pooled LightGBM. Point calibration is rejected because
  all tested prior-only overlays worsen pooled RMSE. The genuinely-no-history
  gap route remains a locked secondary component rather than a post-hoc blend;
  its pre-quarantine exact metric is superseded.
  Final fitting through 2025 publishes 745 unique 2026 shadow players, 715 PPG
  centers, and 745 participation probabilities. The league-specific primary
  centers now feed the production handoff for the app population. Canonical
  weekly IDs remain mandatory; do not name-join or reuse a center across
  leagues.
- Projection V2 Milestone 4A is complete in shadow mode.
  `Projection_V2.sqlite3` contains the reviewed 160-feature mart; its earlier
  M4A model run and original 2026-07-29 lock evidence are superseded because
  the identity, source-season, and beta provider-scoring lineage changed.
  The clean feature manifests contain 31 residual, 19 participation, and 12
  template challengers, plus a separate 12-feature legacy-inspired residual
  research manifest, a separate 26-feature projection research manifest,
  five preseason projection-trajectory fields, one logged-ADP transform, and
  the 13-feature projection-anchored history-gap and 11-feature
  team-environment families.
  The original direct shallow LightGBM scored 3.144 RMSE and was only 0.055
  ahead of position-aware consensus recalibration;
  shallow LightGBM leads participation at 0.122 Brier versus 0.137 for full
  logistic. KBest/PCA/agglomeration are rejected. The corrected foundation has
  now been fully relocked, recalibrated, replayed through the following season,
  and republished. Current uncertainty still comes from exactly one joint
  weekly donor residual/path; no independent second residual is allowed.
- A 2026-07-28 fold-identical study tests the 12 legacy-inspired features:
  projection versus experience peers, self-excluded same-position teammate ADP,
  and team opportunity shares. No family materially improves Ridge or
  LightGBM, all 12 together worsen both, and the exact old
  projection-versus-experience difference is flat. A deterministic full-column
  replay identifies and removes false tiny gains caused by LightGBM column
  subsampling when unavailable features expand the matrix. The incumbent
  31-feature manifest remains unchanged; the 12 constructions stay separately
  governed for audit and future deeper-history retesting.
- A second 2026-07-28 study adds V2 execution support for Lasso and Elastic Net
  without expanding the default model surface. Incumbent Lasso scores 3.1656
  RMSE versus 3.1747 for fold-identical Ridge while selecting a mean 23.6 of 35
  raw inputs, but its nine-season interval crosses zero. Expanded Lasso scores
  3.1615 but adds only a small, unstable gain and is weaker for rookies and
  second-year players. Elastic Net is slightly weaker and chooses L1-heavy
  penalties. Direct shallow LightGBM remains the leader; production and
  manifests remain unchanged.
- Provider point estimates now carry the governed
  `core_offensive_season_components_v1` estimand. It scores linear season-total
  offense and never substitutes provider-published totals or PPG. Weekly
  yardage bonuses and projected fumbles/two-point/return-TD components remain
  outside that estimand. One missing component normally requires two
  same-player/season/position donors; beta QB sacks are the sole one-donor
  exception because FFToday is the only sack source in several historical
  seasons. Every imputation records component, donor providers, and donor
  count. Provider-specific columns require three
  prior projection seasons, so one-year FantasyPoints/FFF/FanDuel evidence and
  two-year PFF evidence cannot receive learned weights. Ten rate/opportunity,
  eight component-disagreement, and eight provider PPG additions remain in the
  separate `residual_projection_challenger_v1` manifest. Deterministic
  LightGBM improves from 3.1230 to 3.1145 with the provider family, but the
  season interval crosses zero; FFToday supplies most of the exploratory point
  estimate. Shape features are neutral and disagreement is harmful. None helps
  rookies, so no challenger is promoted and the 31-feature incumbent remains
  unchanged.
- A final 2026-07-28 projection consensus ladder closes the broad
  projection-only search. A causal nonnegative provider stack improves raw
  realized team-game PPG by 0.0096 RMSE versus the configured median, but wins
  only five of nine seasons, has an interval crossing zero, and worsens the
  conditional-PPG model. Room disagreement and active-PPG alignment produce
  only tiny projection-only point gains and do not improve the full model;
  active fields have only 2024-2025 history. Projection-only consensus core is
  nevertheless just 0.0097 RMSE behind the full model. A causal
  position-by-history router improves the full-model point estimate by
  0.0055-0.0097 RMSE across minimum-sample rules, but remains an unpromoted
  finalist because every interval crosses zero. The stable recent route is
  projection-only for limited-history QB/TE/WR and full LightGBM for
  limited-history RB and all veterans.
- A position-model follow-up fits the same projection-core and full LightGBM
  as four QB/RB/WR/TE models and as three QB/RB/WR+TE role models. Four-way
  splitting worsens projection core by 0.0425 RMSE and full by 0.0160.
  Independent full QB/RB slices are only 0.003 better, while WR/TE lose
  materially from smaller samples. The three-role full model is an overall tie
  at 3.1231 versus 3.1230 and improves the 2023-2025 point estimate by 0.0233,
  but that recent evidence spans only three seasons and reverses in 2024.
  Pooled models remain primary; only the three-role full fit advances as an
  unpromoted temporal-robustness finalist.
- The requested QB-versus-all-skill split also ties rather than improves full
  LightGBM (3.1237 versus 3.1230) and is worse in the recent window; its
  projection-core version is 0.0058 worse. A position-family ladder then tests
  experience peers, teammate ADP, role opportunity shares, and richer room
  clarity across 16 prespecified QB/RB/WR/TE comparisons. None survives
  false-discovery correction. QB room clarity is directionally best at
  -0.0126 versus separate QB, while opportunity shares worsen QB/RB/WR mean
  PPG. Rookie-QB room clarity, rookie-WR experience context, and young-TE
  teammate ADP remain same-evidence template/distribution hypotheses, not
  promoted point features.
- A final pooled model-family check rejects KNN but advances random forest.
  Scaled projection/full KNN worsen LightGBM RMSE by 0.1111/0.1964 and their
  fixed blends also lose. Full RF is effectively tied with full LightGBM at
  3.1242 versus 3.1230. Their prespecified untuned 50/50 average scores 3.1143,
  a -0.0086 delta with interval `[-0.0182, +0.0005]`. The blend improves
  QB/RB/WR and limited-history slices while TE is essentially flat. RF and the
  equal blend advance as whole-season/calibration finalists; KNN does not.
- A follow-up on the current 3,696-row lineage shows that the weaker standalone
  full Lasso is meaningfully complementary to the tree models. Its error
  correlation is 0.953 with RF and 0.962 with LightGBM versus 0.988 between
  the trees. A fixed full Lasso/RF/LightGBM equal-third average scores 3.1000,
  improves the RF/LightGBM average in all nine seasons, and has interval
  `[-0.0225, -0.0075]`. A strictly prior-season-weighted Lasso/tree blend
  confirms the result at 3.1027. Both advance to whole-season, calibration,
  and joint-template validation; production is unchanged.
- Projection-anchored history gaps now provide the governed alternative to
  pooled-median historical PPG imputation. Missing prior-year, three-year, and
  career PPG becomes a zero adjustment to the player's own current expert
  baseline, with explicit availability, recency, and opportunity-game
  reliability. All 3,696 OOF rows have complete gap fields. The representation
  improves rookie and other no-career point estimates, but raw/shrunken Lasso
  worsen overall by 0.0018/0.0027 RMSE; the raw equal-third blend improves only
  0.0029, reverses in 2023-2025, and has interval
  `[-0.0091, +0.0035]`. Keep the 31-feature incumbent primary and carry the
  13 gap fields only as a sparse-history/router challenger.
- A preseason-only projection-trajectory follow-up now compares this year's
  consensus team-game PPG with the same player's exact prior-year and
  recency-weighted prior-three-year projections. Exact one-year change alone
  improves the equal-third Lasso/RF/LightGBM blend by only 0.0016 RMSE; the
  three-year context improves 0.0041, and all five trajectory fields improve
  0.0051 with a slight favorable 2023-2025 mean. `log1p(ADP)` improves Lasso
  by 0.0319 but does not help the trees. The pooled-best combined blend scores
  3.0930 versus 3.1001, but reverses slightly in 2023-2025 and weakens
  missing-ADP/some sparse-history slices. Carry raw-ADP trajectory and
  logged-ADP Lasso as unpromoted finalists; production is unchanged.
- A compact team-environment follow-up separates QB1 passing/rushing yards and
  TDs, QB1 rushing fantasy-point share, capped core-skill and self-excluded
  supporting-cast strength, and non-duplicated team rushing/offensive TDs.
  The full 11-feature family is flat globally. QB rushing share is the best
  compact signal, improving the trajectory blend from 3.0949 to 3.0928 and
  the 2023-2025 mean by 0.0089. It improves WR/TE by 0.0055/0.0068, is neutral
  for RB, and worsens QB. A same-OOF WR/TE-only route scores 3.0916, wins seven
  seasons, and has interval `[-0.0070, +0.0002]`. Carry that route only as a
  whole-season/template finalist; the global point model remains unchanged.
- Current active workstream: best-ball weekly template generation and Snake app
  integration.
- An isolated NFFC Snake setup preview can be generated from the stable DK app
  database while NFFC projections are still running. It uses real NFFC ADP but
  cloned DK projections/templates and must not be treated as calibrated output.
- The modeling repo owns the source `Simulation.sqlite3`. The weekly builder
  syncs generated best-ball tables to the auction app without replacing its
  keeper/salary scenarios, and copies the complete database to Snake.
- Weekly template matching now uses absolute PPG, projection/market
  disagreement, uncapped experience, and full-universe workload-room structure.
  Adaptive capped donor weights retain broad ESS, ordinary zero-active outcomes
  remain eligible, and the declared Bell 2018 holdout is audit-only.
- The managed auction app now samples the matched donor's centered active-PPG
  residual and managed weekly path jointly around the current calibrated point
  forecast. It no longer combines an independent current-year model residual
  with a scaled 30% template residual.
- The auction Streamlit app now includes a cached Weekly Template Comp Explorer
  below the draft grid. It exposes each player's exact weighted donor rows,
  predicted/actual PPG, raw residual, participation, experience, ADP, match
  distance, and position-specific workload-room context, plus the target-side
  match profile and the pool-centering explanation.
- A strict 1,620-target context ablation found that draft capital and projected
  supporting-cast environment do not earn global weekly-template weights.
  Light 8-12-season recency is the supported candidate: both preserve point and
  managed-contribution calibration and improve played-games CRPS; eight seasons
  reduced 10+-year donor weight from 32.3% to 20.5%, while twelve was slightly
  safer temporally. Production now uses the conservative 12-season prior.
- A paired backward-ablation replay across the same 1,620 strict rolling
  targets recommends combining the 12-season recency prior with removal of the
  redundant projection-by-experience interaction. Direct projected PPG and
  uncapped experience remain. The choice passed aggregate, position, coverage,
  tail, zero-week, and absence guardrails; it was selected at each 2023-2025
  nested origin and modestly improved all three untouched-period CRPS measures
  versus production. Component ranks, ADP/market context, disagreement, room
  hierarchy/concentration, and pass-catcher team context remain supported.
  This specification is active in the rebuilt beta and DK template tables;
  `projection_x_exp` remains diagnostic only and no longer affects distance.
- A 36-method local weight-sensitivity replay supports retaining the remaining
  reference weights for that update. No individual feature-family +/-25%
  perturbation improved development composite by more than 0.012%. Much flatter
  within-pool weighting improved 2017-2022, but the near-uniform development
  winner reversed slightly in 2023-2025, breached the temporal position metric
  guardrail, worsened temporal played-games CRPS by 0.00284 with its clustered
  interval above zero, and repeated in only one of three recent nested origins.
  Lower sampling sharpness remains a separate hypothesis; none of the 35
  alternatives cleared every promotion rule. All retained production weights
  therefore remain unchanged.
- Auction salary predictions now expose bootstrap-averaged empirical residual
  quantiles in `Salaries_Pred`; `Fantasy_Football_App` auction ILP sampling
  now uses residual quantiles for both projections and salaries.
- `Validations.sqlite3` now stores `Salary_Backtest_Predictions` full historical
  pools and `Salary_Validations_Resid` observed non-keeper rows under explicit
  rolling-data/current-method provenance. Historical pools are defined by
  preseason projection availability rather than the realized roster or manual
  ESPN copy; raw and keeper-budget-normalized predictions are both retained.
- The experimental salary surface is now
  `current_locked_spec_v5_compact_salary_features`: it keeps v4 keeper-market
  inputs and additive calibration but reduces the fitted matrix to 12
  substantive features. The full rebuild produced the best v1/v3/v5 rolling
  MAE/RMSE. Its identical-seed 4,000-cell replay improved historical
  affordability and reduced the selected-roster salary gap, so v5 remains the
  leading/current surface; selection-conditioned residual bias is still the
  next salary-methodology target.
- A strictly rolling second-stage v5 selection-surcharge replay now tests that
  target directly. A ridge model uses only prior-origin actual-minus-v5 residuals
  with position, point salary, and preseason optimizer selection frequency.
  Half shrinkage improved development/2025 historical feasibility by 6.8 points
  in both periods, reduced mean overage by `$7.02`/`$8.06`, and changed raw
  season points by only `-1.31`/`-0.05`. Full shrinkage improved feasibility by
  15.6/21.2 points and outperformed a flat `$285` cap on affordability, but
  cost more forecast EV and still reached only 21.6%/29.2% absolute
  feasibility. The half-strength version is now published by a standalone
  annual workflow after a premium-free current-season Target seed; it remains
  a targeted reserve rather than complete salary-risk coverage.
- A paired 2022-2025 replay now separates full-surface scenario shopping from
  blind Sequential Target selection. Static rosters spent `$291.8` on their
  selected sampled rows but `$306.2` at point prices and `$323.0` historically.
  Blind initial plans removed the `$14.4` scenario discount but retained a
  similar `$17.3` actual-minus-point residual; historical-cap feasibility rose
  from 14.1% to 43.0%, or 52.3% with the half reserve. Live recourse made every
  completed roster legal, but left `$45.4` unused on average, so current work
  should improve spending efficiency rather than declare salary bias solved.
- `s5_Auction_Selection_Premium.py` persists premium-free selection seeds and
  ridge coefficients, refreshes realized prior-season salaries, and publishes
  `Salary_Selection_Premium` to both source and auction-app databases. The 2026
  beta convergence-policy refresh completed 1,000/1,000 rosters, trained on 518
  observed 2022-2025 rows, and assigned RJ Harvey a `$3.79` half-strength
  reserve under seed method `app_target_selection_only_converged_v2`.
- Fantasy_Football_App has a managed-league weekly auction ILP that uses weekly
  templates, lineup decision scores, waiver baselines, bootstrapped managed
  values, keeper-table inputs, draw-level remaining-market salary normalization,
  Target Board candidate-rebased Buy/Pass contribution, and paired Current
  Nomination evaluation with monotone price curves. Current Nomination now uses
  lean NumPy salary workspaces, caches repeated exact roster scores across
  prices, solves and counts Buy and Pass feasibility independently, and excludes
  incomplete paired samples from EV and Max Bid. Organic Target simulations now
  default to 320 trials in eight balanced 40-trial logical blocks. Both forced
  stages retain eight logical
  scenario blocks for uncertainty regardless of whether one or eight worker
  processes execute them, and each Target player contribution averages five
  shared holdout seasons. Each organic Target outer roster now repeats the
  constraint-safe exact-scored swap correction on the full cached-bank expected
  weekly profile until no improvement remains, with a 12-swap cap, before any
  holdout is drawn. Target candidate evaluation now uses lean salary arrays,
  safe paired salary-row reuse, exact per-trial roster-score caching, vectorized base-lineup
  thresholds, batched candidate/context marginal values, and an exact
  partial-selection weekly P90. Target results are keyed to the full draft and
  simulation state and expose adjusted market salary plus conditional
  downside/sample-size context. An adaptive forced Buy/Pass search now follows
  the organic run with four exact broad pilot scenarios over players outside the
  protected heuristic set. The
  pilot protects all 20 heuristic candidates and appends four fully paired
  discoveries to a 24-player, 64-pair preliminary cohort; its values are
  discarded before evidence estimation. The ten evidence-priority candidates
  plus any missing members of the four highest-market-salary usable preliminary
  candidates receive a fresh 96-pair confirmation stage, for 10-14 unique
  players. Market anchors never displace evidence leaders or alter evidence
  labels, and remain visible on the 30-row board after confirmation. The stages
  remain statistically and visibly separate. Forced effects use a
  position-neutral leave-one-player-out
  random-effects prior learned from the other preliminary candidates, with
  logical-block-aware standard errors; Organic Gain is not blended numerically
  because it is a different, selection-weighted estimand. Preliminary and
  confirmation disagreement now adds a continuous two-stage random-effects
  variance before the posterior is recomputed. Every usable forced screen ranks
  on posterior LCB80 (`mean - 0.842 * SE`) regardless of confirmation label;
  confirmed, preliminary, mixed, negative, conflicting, and low-fit-pivot labels
  describe evidence maturity without changing the score. Forced rows appear
  first in continuous LCB80 order, followed by a separate Organic Gain watchlist.
  Organic paired contribution coverage is explicit, and under-covered organic
  rows sort last.
  Forced-Buy position mix, WR/TE decision-dollar change versus Pass, and common
  completion players expose the roster tradeoff. The default Target Board now
  presents eight decision fields—numeric rank, player, position, evidence, labeled
  score, roster fit, combined decision price, and roster impact—while the full
  staged and composition diagnostics remain under a collapsed Evidence Details
  section. Candidate-rebased Target markets now remove only sampled Market `$`
  from league money; the selection reserve remains solely a personal
  affordability charge. If Pass removes an original Top-N candidate, salary rank
  N+1 replaces that player so Buy and Pass retain equal policy breadth.
  Current
  Nomination defers alternative sensitivity until requested, validates nominee
  position and fully fixed roster constraints, and searches exact global and
  adaptive prices until the isotonic EV and feasibility boundaries are resolved
  to adjacent dollars.
- The app's default Target workflow is now a non-anticipating Sequential Policy;
  the previous staged roster Target is retained as `Legacy Oracle`. The default
  sequential bank expands 24 heuristic candidates to 64 with capped local peers
  and QB/RB/WR/TE Premium/Mid/Value coverage. All 64 receive exact lightweight
  managed-season scoring. Confirmed and discovery rows now share one global
  LCB80 comparison table with explicit evidence-stage labels; discovery remains
  visually shaded, exploratory, and ineligible for max-bid guidance. Fresh confirmation keeps 18
  exact market candidates but caps a peer group at four and a position at eight,
  then profiles low/high prices for the leading 10. Hidden auction and validation
  banks are deterministic across unchanged settings, and allocation/cache
  versioning no longer rerolls the evidence bank. Candidate evaluation now uses
  four spawned processes by default with stage barriers, isolated GLPK/scoring
  state, local roster-score caches, and deterministic result assembly. Exact
  one-versus-four-worker parity held for all 64 rows and price curves; the
  default 320-budget empty-roster benchmark improved from 25.33 to 10.25 seconds
  (2.47x). The aggressive `$108` Bijan signal remained supported while player,
  position, and salary-tier exploration stayed broad.
  A subsequent 16-seed live-state audit shows that this single-bank signal is
  not robust enough for production interpretation. Pool-shaped random draws
  made matched AJ-on/off Bijan evidence correlate only 0.25; drawing on the full
  canonical player universe before masking raised the correlation to 1.00.
  Even then, seed-level Bijan gain SD was 21.07. Fixed-component tests locate
  more variation in 32-context policy construction (16.49 SD) and 64-season
  validation (12.69) than in 48 auction paths (3.95). Production v4 now draws
  the canonical full-player universe before masking, keys balanced templates by
  player, and uses four partial-crossed blocks. Each block has 32 mean-PPG
  construction templates, 12 realized confirmation paths, and 64 complete
  seasons; block disagreement enters the action SE. A matched 16-root replay
  raised AJ-on/off correlation to 0.993, reduced the AJ-state delta to +0.37
  mean / 0.96 SD, and reduced Bijan seed SD from 21.07 to 7.99-8.15. The same
  3,072 confirmation cells are retained; a warm live board ran in 15.2 seconds.
  The three highest-salary remaining players who are personally affordable and
  legal at their positions are now protected exact confirmations: they always
  appear on the 18-player main board, with `INCOMPLETE` shown when evidence is
  weak, and consume existing confirmation slots rather than adding runtime.
  Sequential Target now also screens up to eight first-year keeper-upside
  prospects and reserves up to three fresh confirmations for keeper evidence.
  Each completed Buy/Pass auction branch is scored on common next-year draws
  using actual simulated acquisition prices and expected-best positive surplus
  across the nominal bench. `LOTTERY` requires positive keeper LCB80 while
  current-season mean and season-score p10 deltas remain within two points and
  the named player is usually on the Buy bench and sometimes its best positive
  option; indirect gains are labeled `PORTFOLIO EDGE`.
  Keeper labels do not alter the current-season auction policy, primary rank,
  or Approx Max Bid, and impose no age, tenure, or ticket-count quota.
- Managed scoring now carries source-observed played-week masks independently
  from weekly fantasy scores. Played zero and negative outcomes count normally,
  missed weeks can still use bench/waiver replacement, and learned lineup
  expectations exclude missed weeks while retaining played downside outcomes.
  QB participation is captured before the existing greater-than-15-play
  performance-profile filter, with `played_games` kept separate from
  `active_games`; separate `managed_week_*` profiles retain short-QB scores for
  the auction app without changing Snake's best-ball profiles.
- Managed auction settings expose starter, bench, roster-position maximum, and
  waiver-replacement PPG dropdowns. All sidebar controls are organized into six
  collapsed sections for league, roster slots, position maxima, simulation and
  value, Top-N strategy, and managed scoring. The default TE roster maximum is
  two, and infeasible roster/waiver configurations are blocked before
  optimization.
- A four-origin, 16-cell managed-auction rolling replay completed 16,000 optimal
  roster solves with frozen preseason inputs and raw target-season scoring. It
  retains the current five-draw/projected-waiver/Top-N/bench-0.25 profile as the
  short-term default, but only 19.5% of exact-profile rosters fit historical final
  prices; selected rosters cost about $30 more than forecast. The next auction
  study is current-method walk-forward salary risk with cap haircuts, coherent
  market scenarios, and feasible recourse rather than another scoring-weight tune.
- A 7,000-solve four-origin bench-policy replay rejects raising the construction
  waiver baseline as a studs-and-scrubs proxy. Even `+1` lowered common-bank
  forecast mean/p10 and shifted about `$2.4-$2.9` from starters to the bench;
  larger hurdles discarded much more lineup value. Removing the current P90
  bench heuristic made the bench `$1.4-$3.2` cheaper without lowering forecast
  mean or p10, but the effect is small. A 0.25 sustained-breakout utility
  improved independent forecast and playoff scoring, yet used lineup-impact
  strikes around 10-12 PPG, lost literal 15+ PPG hits in 2025, and worsened
  frozen-price affordability. Keep the live baseline/weight pending a
  price-aware, stricter-ceiling replay on v5 plus the selection reserve.
- A second 7,000-row replay models keeper upside as future market-dollar surplus
  under the confirmed two-slot, annual-`+$10`, three-year rules. The tiny
  same-engine tie-break improved independent forecast mean and p10 in all four
  origins, but realized first-year keeper surplus averaged `$30.37` versus
  `$32.16` for the zero-option control. Stronger weights were unstable, and the
  top two names consumed 57%-96% of identified option slots by origin. Do not
  promote a keeper bonus; next require current residual calibration, complete
  counterfactual-cost scoring, true lexicographic multi-start selection, and a
  concentration guard.
- A revised 3,000-row keeper replay now targets the actual one-hit bench
  strategy: causal next-validation residuals, first-year `price + $10` surplus,
  expected best payoff across all five bench slots, eight fixed starters, and
  at most two no-loss bench swaps. `best1_lex0` improved best realized keeper
  surplus by `$17.9`/`$12.5`/`$6.4` and `$20+` hit probability in all three
  observable origins. Four-origin forecast mean/p10 averaged `+1.1`/`+0.6`,
  although the 2024 current-season slice declined. Advance it to the current
  v5-plus-reserve/converged replay rather than production.
- A 4,000-row full-roster keeper reinvestment sensitivity confirms that cheap
  options can finance starters: the one-option cap shifted `$5.7` from bench to
  starter spend and improved starter-only forecast mean/p10 by `+4.6`/`+3.6`.
  Full-roster mean/p10 improved only `+0.6`/`+1.2`, realized best keeper surplus
  only `+$2.1`, and playoff scoring was flat because the rebuilt current-only
  control already drafted most marquee hits. The option limits counted
  incremental forced additions, not total lottery tickets or exclusive roles.
  Do not promote a broad bonus; next give every bench player both current
  fill-in and keeper-option value and let composition emerge under roster
  mean/p10 and aggregate coverage gates.
- A 2,000-row soft whole-bench replay removed the k1/k2/k3 and age/role quotas.
  Expected-best first-year keeper surplus across all five bench players improved
  predicted/realized best surplus by `+$4.3`/`+$4.4`, with realized gains in all
  three observable next seasons. It shifted `$2.1` to starters, raised
  independent roster mean/p10 by `+4.2`/`+2.7` on average, and preserved top-two
  fill-in value. The control already contained 4.07 effective keeper options;
  the soft policy reached 4.19, so option concentration is more meaningful than
  a fixed ticket count. Keep production unchanged: only 46.2% of changed
  rosters improved both mean and p10 on the independent bank, with 2024 negative.
  Next strengthen or cross-fit the current-year gate before testing live use.
- The agreed production implementation now removes the generic 0.25 bench-P90
  bonus and raw next-year blend while leaving waiver baselines unchanged.
  Organic Target and the annual premium-free seed use expected-best positive
  first-year `price + $10` surplus across the whole nominal bench as a secondary
  objective. Each search step requires at least `$1` more keeper utility and no
  decline in current-season mean or p10 on the complete live construction bank;
  there are no age, role, or option-count quotas. Next-year residuals are
  separate from current draws and use player common-factor `rho=0.25`. Forced
  Target Buy/Pass and Current Nomination remain current-year estimands. The v3
  selection reserve must be refreshed before treating live affordability as
  aligned with the new organic policy.
- The fresh keeper-aware annual reserve completed 1,000/1,000 optimal Target
  seeds in 133.5 seconds across eight workers. The 166 active player rows sum to
  13 roster slots, train only through 2025, and match exactly between source and
  app databases. The republished half-strength reserve averages `$0.68` per
  selection, `$8.79` per roster, and caps at `$3.71`.
- The sequential `$5` versus `$10` fixed-price recourse replay completed 1,024
  policy paths but did not pass its buffer-selection gate. In the primary strict
  `p + 1` development slice, `$5` completed 44/72 paths, `$10` completed 42/72,
  only 38/72 pairs both completed, 10/72 were discordant, and 15/72 were clean.
  No order family had clean data in every origin, so the earlier static `+$5`
  preference remains provisional rather than sequentially validated.
- The current-method five-draw salary replay completed 3,000 paired cells, but
  its unconditional historical point comparison rewards rosters that could not
  be purchased and therefore cannot choose between `+$5` and `+$10`. Its valid
  finding is limited to the affordability tradeoff: `$5` gained about 6.5
  historical-price feasibility points and reduced overage by $3.3-$3.6 versus
  `$10`.
- A 4,000-cell exact salary chance-constraint frontier then compared 60%, 70%,
  80%, and 90% construction thresholds on independent normalized five-draw
  market banks. The rule produced a stable modeled risk/forecast-EV frontier,
  but actual selected rosters cost about $29 more than the independent scenario
  mean at every threshold. Moving from 60% to 90% raised development historical
  feasibility only from 12.1% to 18.1% and 2025 from 8.4% to 14.8%. Do not wire
  a threshold into production until roster-level salary calibration or robust
  recourse addresses this persistent center bias; exact chance solves were also
  about 3.2x slower per cell than the simple-buffer replay.
- A selected-roster residual diagnostic reconciled all 52,000 chance-frontier
  roster slots and 96.0% recorded-price coverage. The observed auctionable pool
  averaged `-$0.39` actual-minus-point-predicted residual and the unique
  ever-selected set averaged `-$0.40`, but selection-weighted roster slots
  averaged `+$1.43`. Frequent 25%-50% selections averaged `+$2.38`, three core
  above-50% selections averaged `+$12.32`, and the strongest value-over-price
  quintile averaged `+$4.82` per selected slot. The prior roughly `$29` gap
  decomposes into about `$16` development actual-minus-point error plus about
  `$13` of selected-roster scenario discount relative to the point row; 2025
  components were roughly `$22-$23` and `$9-$10`. Anchor future risk rules to
  the point salary row and learn a prior-only, shrinkage selection-propensity
  surcharge rather than hardcoding a `$29` haircut.
- The salary model now has an experimental
  `current_locked_spec_v2_ensemble_features` feature path that joins the
  optimizer-aligned OOS validation point ensemble for historical seasons and
  `Final_Predictions_Resid` for the current season. It adds positional ensemble
  strength, ensemble-versus-consensus and ensemble-versus-ESPN-price rank gaps,
  ensemble PPG per source dollar, and selected position interactions. The join
  covers all 2,696 validation-era projection rows exactly; current deep players
  outside the 180-row final ensemble retain a consensus-PPG fallback. Preserve
  the v1 validation slice.
- The preserved v1/v2 salary ablation found that v2 reduced mean normalized
  residual bias from `-$0.71` to `-$0.48` across 644 common observed
  player-years and reduced the old optimizer's strongest-value-quintile
  selection-weighted residual from `+$4.82` to `+$3.91`, but worsened MAE from
  `$4.31` to `$4.49` and 2025 MAE from `$3.73` to `$4.20`. An identical-seed
  4,000-cell v2 chance-frontier replay changed about 80% of rosters without a
  stable affordability gain: development gained 2.08 managed forecast points
  across thresholds but historical feasibility was unchanged and overage
  worsened `$1.09`; 2025 lost 1.79 forecast points while feasibility improved
  1.2 percentage points and overage improved `$0.46`. Keep v1 as the current
  comparison/default surface and test a causal shrinkage blend or restricted
  optimizer-tail correction before another full replay.
- Projection validation rows now carry horizon-aware, strict-prior-origin
  empirical residual quantiles and explicit donor/target provenance. A one-time
  backfill preserved all 42,351 `Model_Validations_Resid` rows and created 6,006
  `Final_Validations_Resid` player-origin rows; point means reproduce the prior
  ensemble to `3.55e-15`. Terminal next-season forecasts remain usable but their
  not-yet-realized targets are flagged unavailable.
- The experimental salary feature surface is now
  `current_locked_spec_v4_additive_keeper_market_features`. It retains v3's
  causal projection ceilings and role shares, adds keeper market value,
  remaining-pool inflation, and keeper/budget-adjusted source salary features,
  and replaces proportional point normalization with one additive dollar shift
  above a `$1` floor. The current keeper inputs imply a `1.0875` keeper-only
  multiplier and a `1.1573` fully coherent source-market scale. A fixed-v3-raw
  historical audit improved MAE/RMSE by `$0.070`/`$0.093`, but a fresh v4
  pipeline build and paired optimizer replay are still required.

## Recent Durable Decisions

- Template matching now uses projection strength plus position-specific role and
  team context, not only projected-points buckets.
- RB rush/rec room shares and WR/TE receiving shares are based on projected
  fantasy points, not raw attempts alone.
- Non-QB zero-active historical templates are excluded from pools; QB pools can
  keep zero-active outcomes for backup/fringe-starter context.
- Template pools expose `template_sample_prob` so apps can use all selected
  templates while favoring closer matches.
- `Best_Ball_ADP_Audit` is the durable place to review missing or fallback ADP
  joins for draftable players.
- Best-ball weekly table rebuilds should replace only the active league slice
  and preserve other league slices already present in `Simulation.sqlite3`.
- Auction salary uncertainty should use bootstrap-averaged historical
  out-of-fold residual quantiles by position and predicted salary, and the
  auction app should sample projections/salaries from residual quantile columns
  instead of truncated-normal or upside/top probability branches.
- Auction salary residual calibration must reconcile only like-for-like
  non-keeper dollars: represented OOF spend historically and keeper-adjusted
  remaining league budget for current predictions.
- Historical salary replay pools must come from preseason projection inputs,
  left-join manual base values/actual outcomes, flag minimum-filled base values,
  exclude target-derived aggregate spend from model features, and label current-
  spec retrospective rows as non-fresh method holdouts.
- Managed auction weekly scoring should value startable weekly lineup points
  above waiver baseline and avoid best-ball-style hindsight lineup decisions.
- Weekly participation must come from a separate source-observation mask, not
  from whether the realized fantasy score exceeds zero.
- Pool-center historical active-PPG residuals before transporting them onto the
  current final point forecast. Rolling validation supports the centered joint
  residual/weekly path for managed mean, P10, and contribution scoring, but not
  as a player-specific breakout ranker: high absolute-impact discrimination is
  useful at RB/WR while `+3`/`+5 PPG` residual discrimination remains weak.
- Organic Target managed auction ILP rosters should repeat the constraint-safe,
  exact-scored one-player swap correction on the full cached-bank expected
  weekly profile until no improvement remains, capped at 12 accepted swaps, to
  reduce fixed-base additive interaction bias without selecting on a few
  realized or holdout seasons. Forced Buy/Pass and Current Nomination retain
  their separate unrefined construction policies.
- Auction app salary draws should load the model's `League_Keepers` slice and
  reconcile sampled dollars above the `$1` floor to live remaining money/slots.
- Current Nomination decisions should compare Buy and Pass on shared stochastic
  draws, construct branch rosters without outcome hindsight, and derive Roster
  Max Bid from exact managed-roster EV rather than selection rate.
- Nomination EV and Max Bid should use only prices with complete paired Buy/Pass
  coverage, and Max Bid should come from a globally anchored non-increasing edge
  fit with exact adjacent-dollar refinement rather than raw-edge binary search.
- Strong auction recommendations should require both positive mean Buy Edge and
  supporting paired confidence/win-rate evidence; broad player comparisons use
  replacement-adjusted Expected Roster Gain while the nominee receives the
  exact price search.
- Do not interpret average-five salary prices as calibrated uncertainty or switch
  directly to one draw. Preserve the current default until roster-level price risk
  is retested with the current residual method; one draw improved distributional
  calibration but reduced realized-cap feasibility from 19.5% to 5.8% in the
  frozen-origin replay.
- The paired nominal-buffer replay found that `cap + $5` improved affordability
  more than `cap + $10`, but subsequent feasibility-first work supersedes any
  default selection from its unconditional point totals. Keep acquired-player
  salaries deterministic in every experimental price row.
- Do not choose between `+$5` and `+$10` from the sequential fixed-price replay.
  Its completion, discordance, order-sign, and precision requirements all fail;
  use it as a stress test and move the decision to the current-method
  walk-forward/coherent-market study with explicit personal keeper states.
- Do not choose `+$5` versus `+$10` from unconditional historical points on
  unaffordable rosters. Preserve both as unselected experimental guardrails
  until a roster-level calibrated salary-risk or feasible-recourse study can
  compare policies on an explicit ex-ante risk target.
- A sampled chance constraint is a valid way to express the ex-ante salary-risk
  target, but normalized marginal residual markets are not yet calibrated for
  optimizer-selected rosters. The roughly $29 selected-roster gap combines a
  positive actual-minus-point residual with an additional scenario discount
  relative to the point row; correct these separately before selecting a live
  threshold, and consider CVaR or a robust salary row if exact chance binaries
  are too expensive for live loops.
- Salary residual calibration must retain optimizer selection frequency. An
  ever-selected unique-player comparison hides the bias; frequent/core selections
  and the strongest within-position value-over-price ranks concentrate positive
  residuals even when the all-player mean residual is slightly negative. Use a
  strictly prior-origin, shrinkage selection-propensity/value correction and keep
  the point salary row as an anchor in the next replay.
- Optimizer-aligned projection strength is a valid preseason salary feature.
  Historical salary rows should use the existing OOS validation ensemble and
  current rows the final projection ensemble; derive value ranks within year,
  position, and keeper availability, preserve the full salary universe with a
  consensus fallback, and version the resulting method separately from the v1
  salary baseline.
- Projection interval calibration must use realized outcomes strictly before
  the forecast origin. For next-season rows, donor origin `d` realizes in
  `d + 1`, so it requires an additional origin embargo. Never treat terminal
  next-row placeholder `y_act` as observed when `resid_target_available = 0`.
- Residual and team/position share salary features should be tested together as
  a versioned specification. Collapse redundant source-specific role shares to
  row medians and disagreement, preserve deep-player coverage with causal
  projection-tier fallbacks, and require both player-error and selected-roster
  affordability gates before promotion.
- Optimizer selection frequency can support a targeted decision-price reserve
  without changing the coherent v5 market salary. Fit the reserve strictly on
  completed prior origins, keep the 2022 no-history seed at zero, and shrink
  positive predicted residuals. Half shrinkage currently gives the cleanest
  low-cost tradeoff, while full shrinkage is the more aggressive affordability
  option; neither is ready to become the default until the remaining low
  absolute feasibility and current-season two-pass workflow are addressed.

## Key Links

- Module tracker: `MODULE_TRACKER.md`
- Decision log: `DECISION_LOG.md`
- Cross-repo context: `CROSS_REPO_CONTEXT.md`
- Best-ball table contract: `../docs/data_contracts/best_ball_weekly_tables.md`
- Best-ball build runbook: `../docs/runbooks/best_ball_weekly_build.md`
- Research index: `../research/README.md`
- Latest chronological log: `Session_Notes/2026-07.md`

## Working Defaults

- Keep numbered scripts notebook-friendly and import-safe where practical.
- Favor surgical changes over broad refactors.
- Treat name cleaning, projection joins, ADP joins, and table schemas as
  first-class risk areas.
- Update app-facing contracts when Simulation tables change.
