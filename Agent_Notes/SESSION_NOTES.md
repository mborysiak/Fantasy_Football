# Session Notes Landing

Last updated: 2026-09-05

## Project Objective

Maintain and improve the fantasy football modeling pipeline that builds player
projections, residual distributions, simulation inputs, and best-ball weekly
template tables consumed by downstream draft apps.

## Current Focus

- Retired the display-only paired-breakout surface from production. Source,
  Auction, and Snake Simulation databases now contain 26 tables at about
  80.64/80.64/80.62 MiB; every retained table matches its pre-change snapshot.
  Manifest schema 8 removes the build/validation requirement and clears inherited
  breakout tables during staged compaction/app preparation. The Auction explorer
  is removed; optional research generation requires a separate output database.
  Expert/stat inputs remain tracked, Model_Inputs/Validations remain untracked
  with local files retained, and recoverable snapshots are under
  `Data/Production_Refresh_Backups/20260906_breakout_retirement/`. See
  `docs/runbooks/database_storage.md` and `Session_Notes/2026-09.md`.
- The completed 2026 Beta auction is published for hindsight review. The raw
  result has 180 unique players totaling `$3,581`, including 15 realized
  keepers at `$608`; the canonical `beta_actual` app pool has all 156 drafted
  offensive players totaling `$3,541`. Actual mode now takes keeper membership
  from the result slice, preserving result-only Jayden Daniels `$11`, and skips
  the inapplicable predicted selection-premium join. Josh Jacobs' exact `$26`
  price remains in hindsight data but his 2026 player-season is excluded from
  future salary-model fitting because suspension risk created an approximately
  `$31` non-market discount. Source/app slices match exactly, 42 focused model
  tests and all 120 Auction tests pass, and the rendered Actual view has 15
  keepers with zero exceptions.
- A staged 2026 Beta keeper/salary refresh is live. Bhayshul Tuten `$11`
  replaces Nico Collins, Quinshon Judkins is `$30`, and Omarion Hampton is
  `$67`; the 14 contracts total `$597`. The locked v6 salary model publishes
  323 keyed rows and reconciles 142 open slots exactly to `$2,979`. A fresh
  keeper-aware reserve completed 1,000/1,000 trials, publishes 309 non-keeper
  premiums, and has an expected roster reserve of `$8.8068`. The paired
  breakout map was republished so Tuten is a keeper and Collins is available;
  salary, reserve, source/app/Snake parity, SQLite, focused model tests, the
  119-test Auction suite, and live app smokes all pass. Rollbacks and receipts
  are under
  `Data/Production_Refresh_Backups/20260829_beta_tuten_hampton_judkins_salary_refresh/`.
- A read-only DK/beta rolling study rejects generic provider
  `max - median` PPG as a production point, tail, or template feature. The
  normalized primary worsens +3/+5 residual calibration in both leagues and
  misses point/template replication. Raw absolute gap improves pooled RMSE by
  roughly `0.0055` in both but is unstable in recent seasons and concentrated
  in a post-slice QB result; high-gap quartiles actually have lower +3/+5 hit
  rates. Production is unchanged. Durable study:
  `research/studies/2026-08-29_v2_asymmetric_expert_projection/`.
- A prespecified follow-up also rejects a large-denominator stabilized bullish
  fraction, `(max - median) / sqrt(median^2 + 5^2)`. It improves pooled
  controlled point RMSE by `0.00317/0.00409` DK/beta but worsens recent RMSE,
  is nonworse at only two of four positions, and has crossing clustered
  intervals. +3/+5 Brier and AUC worsen in both leagues, weekly-template gates
  fail, and the highest stabilized-gap quartile still has lower +3/+5 rates
  than the lowest. Production is unchanged. Durable study:
  `research/studies/2026-08-29_v2_stabilized_bull_gap/`.
- Governed schema-7 refresh `20260828T215922Z_c9cd883e` is live from stage
  `Data/Production_Refresh_Stages/20260828_beta_keeper_refresh_v4/`. All 31
  build/validation/app steps completed before explicit promotion. The release
  publishes 341 DK, 383 NFFC, 323 beta, and 323 NV projections; Beta has 14
  keepers spending `$597`, leaving 142 slots and `$2,979`, and its latest
  reserve seed completed 1,000/1,000 trials with 309 premiums. Paired-breakout generation is
  now a governed refresh step, both oversized app artifacts are verified as Git
  LFS paths, and Auction/Snake smokes passed with zero errors or exceptions.
  Durable pre-refresh copies are under
  `Data/Production_Refresh_Backups/20260828T215922Z_c9cd883e/`.
- A review-only paired breakout surface now joins each historical player's
  season-N managed weekly path to the same player's N+1 appearance and
  conditional production. Matching uses causal preseason projection, canonical
  ADP, experience, role/room context, signed uncapped N+1 growth, and a small
  separate appearance term; salary is excluded. All 1,167 current four-league
  RB/WR/TE rows receive 80 donors, including diagnostic keeper rows that the
  Auction UI hides by default. The source and Auction copies have exact parity
  at 11,867 templates, 93,360 pool rows, 1,167 player rows, and four audits.
  The Streamlit explorer ranks late-market players and exposes paired donors;
  it does not change optimizer scoring. Durable study:
  `research/studies/2026-08-27_paired_breakout_templates/`.
- A frozen 2022-2024 Beta multi-origin replay now tests the same exact-mean,
  pure expected-excess, and standardized 50/50 mean-plus-excess policies from
  the 2025 study. All three isolated artifacts use prior-year projection,
  salary, and donor cutoffs; actual auction prices define the hindsight cost
  surface, and actual weeks load only after every annual roster is selected.
  Pure excess loses actual score by `5.88/48.84/64.33` in 2022/23/24; 50/50
  gains `25.99` in 2022 but loses `59.56/83.23`. Both also lose independent
  holdout mean and P90 in all three origins. Combined with the post-hoc positive
  2025 result, keep expected excess diagnostic and leave production unchanged.
  Durable study:
  `research/studies/2026-08-27_auction_excess_multi_origin/`.
- A follow-up 2025 Beta power-win replay fixes the promising best-available
  waiver proxy and scores 130-131 candidates per block on win probability,
  expected winning margin, and blended power-win utility. The waiver control is
  also the exact mean frontier in all eight blocks, so paired-LCB arms at 0.5%
  and 1.0% correctly retain it. Exploratory direct optimization changes three
  blocks but reverses independently: win/power loses `3.08` mean and `2.25`
  P90, while direct excess loses `4.23` mean. Unguarded pure and standardized
  50/50 objectives make the reversal larger: pure win loses `25.78` mean and
  `3.185` win-probability points; 50/50 win loses `7.67` mean and `1.087`
  points. Production is unchanged; keep the tail metrics diagnostic pending
  dense local swaps and larger/cross-fitted, multi-origin validation. Durable study:
  `research/studies/2026-08-27_auction_power_win_objective/`.
- An isolated 2025 Beta actual-salary replay compares current managed-score
  construction with a best-available RB/WR waiver proxy, a 0.25%-mean-guarded
  championship tie-break, and both changes. The combined arm raises paired
  churn-scored EV `32.32` points (LCB80 `+28.92`) and actual 2025 score `69.81`,
  but raises dead-zone RB count from `0.75` to `0.88` and matches waiver-only in
  seven of eight blocks. The championship tie-break alone is directionally safe
  but modest. Production is unchanged; next test roster-marginal needle-mover
  value over accessible waiver replacement rather than youth or absolute q90
  residual production. Durable study:
  `research/studies/2026-08-27_auction_championship_waiver_objective/`.
- A one-command isolated 2025 Beta Auction replay now builds a staged
  `Simulation.sqlite3` without touching or syncing the live 2026 database. It
  publishes 309 rolling-origin projection/salary players, all 156 drafted
  offensive actual prices, the exact 15-keeper/`$407` historical context, 238
  keyed preseason ETR ranks, and 4,993 weekly donor profiles capped at 2024.
  Predicted- and actual-salary app smokes and 69 focused tests pass. This is a
  current-method replay with model specification selected as of 2026, not a
  pristine historical method holdout. Durable study:
  `research/studies/2026-08-26_auction_2025_historical_replay/`.
- A matched beta second-keeper roster tournament fixes Chase Brown at `$34`
  and separately fixes Tuten, Burden, or Loveland at `$11`, while leaving the
  other two draftable at governed counterfactual market prices. All three arms
  complete 192/192 shared hidden-auction paths and use 24,576 paired holdout
  score cells. Brown/Tuten leads Brown/Loveland `+3.08` and Brown/Burden
  `+8.68` managed-season points, with positive relative results in 7/8 and 6/8
  construction blocks. Tuten's `$19.31` modeled discount lets the policy spend
  about `$6` more at WR and `$3` more at TE than Loveland on average despite
  spending about `$6.5` less at RB. Prefer Tuten for current-season roster EV,
  but treat Loveland as close rather than dominated; no separate next-year
  option bonus is included. Durable study:
  `research/studies/2026-08-26_beta_keeper_roster_tournament/`.
- An isolated 2026 beta salary rebuild treats Chase Brown, Bhayshul Tuten,
  Luther Burden III, and Colston Loveland as non-keepers while leaving the
  other 12 active keepers fixed. Brown models at `$72.17` versus his `$34`
  contract; Tuten models at `$30.31` versus `$11`; Burden and Loveland remain
  near `$25.25/$27.03`. The counterfactual top-144 non-keeper market reconciles
  exactly to `$3,180`, all 324 salary keys match production beta projections,
  and production/app inputs are unchanged. Salary-only surplus ranks the `$11`
  choices Tuten (`+$19.31`), Loveland (`+$16.03`), then Burden (`+$14.25`);
  paired full-roster keeper scenarios remain the decision test for positional
  opportunity cost. Durable study:
  `research/studies/2026-08-26_beta_nonkeeper_salary_counterfactual/`.
- Tiered Sequential targeting is now App v21 on local Auction `main`. It retains
  the completed market-price Buy rosters already scored for the four highest
  positive confirmed LCB anchors, groups every uncertain auction outcome into
  semantic position-spend/shape families, and presents each leading family as
  an expensive-to-cheap target table. Family headers report managed-season EV,
  roster-EV P10-P90, and delta versus all captured high-LCB completions without
  another solve or scoring pass. V21 replaces raw position-budget emphasis with
  open-slot median and P10-P90 budgets while retaining the full min/max envelope.
  It also audits cap-scaled `<$40` and `<$50` total-roster WR spend, including
  fixed/acquired players, and reports share, within-family EV delta versus the
  higher-spend paths, and average total RB count. The audit is diagnostic only:
  no WR floor or joint solve was added. Commit `0ae0c02` contains the v21 update;
  all 106 tests, exact one-versus-four-worker parity, and two-batch accumulation
  pass. A budget-120 four-worker run took `7.922s`, with structure aggregation
  at `0.120s`. The main app is visually verified and healthy on port 8502; all
  modified/generated SQLite files remain untouched and no push was performed.
  The marginal tier cards are removed; one central conditional completion per
  anchor remains available as supporting detail. Add Evidence concatenates
  fresh Buy outcomes and rebuilds the families. The earlier eight-block
  Brown/Tuten beta replay surfaces Josh Allen `$37` (`+16.95` LCB80), Bijan
  `$106` (`+13.17`), Gibbs `$110` (`+6.41`), and Jonathan Taylor `$96` (`+3.39`)
  across 384 Buy outcomes; all 96 Allen outcomes occupy the Premium-QB family.
  Durable study:
  `research/studies/2026-08-25_lcb_aligned_structures/`.
- A Brown/Tuten/Tyson/Coleman beta peer audit finds no Shaheed-specific data
  bug. The comp explorer's signed residual averages omit side probabilities and
  production centers every donor pool: Shaheed has a 43.2% positive-residual
  share versus Coker's 38.1% and produces slightly more 10+/15+ weeks and
  points over waiver despite lower PPG and fewer games. Across nine matched
  confirmation roots, Shaheed/Coker/Doubs average LCB80 is
  `+10.68/+5.88/+3.35`, but Coker wins one root and Doubs two. Holding the other
  12 roster spots fixed, Doubs beats Shaheed by only `+2.17` mean while Coker
  trails `-1.70`. The larger board signal is a whole-roster Buy-versus-Pass
  recourse effect: Shaheed Pass plans never directly substitute Coker or Doubs
  in the four default blocks. No production change; consider equal-confirmation
  peer or fixed-roster substitution diagnostics. Durable study:
  `research/studies/2026-08-25_shaheed_peer_audit/`.
- A keeper-correct beta construction sensitivity starts from Chase Brown `$34`
  and Bhayshul Tuten `$11`, excludes the other 12 active keepers, and reuses
  four paired construction/holdout blocks. Relative to the current additive
  solver, one exact shared-opportunity swap is accepted in three blocks and
  changes average QB/RB/WR/TE shape from `1/6.00/4.25/1.75` to
  `1/5.75/4.75/1.50`; common-holdout mean is effectively flat (`-0.97`) while
  p10 rises `+12.33`. A hard QB1/TE1 plus RB/WR 5/6 or 6/5 shape loses
  `4.94` mean and `1.10` p10, while constructing with every waiver baseline
  raised `1.5` loses `10.34` mean and `0.49` p10 under the unchanged scoring
  authority. App v15 therefore disables joint refinement in both screen and
  confirmation while retaining current beta settings; treat the joint result
  as p10/diversification research rather than a live mean promotion. Durable
  study: `research/studies/2026-08-25_beta_joint_shape_waiver/`.
- Sequential App v15 keeps both the 64-player discovery screen and 18-player
  confirmation stage additive; the exact full-bank swap remains callable only
  for research. The keeper-correct Maye/Achane NV replay excludes
  all 14 opposing keepers; predicted organic mean/p10 improves
  `+7.38`/`+7.84` and actual hindsight improves `+5.96`/`+8.21`, with no
  completion loss. Exact confirmation-only fresh boards run in `9.24s`
  predicted and `7.58s` actual versus `6.72s`/`4.84s` additive and
  `20.08s`/`17.58s` exact in both stages. A utilization/add-one shortlist
  matched plan quality and protected locked fliers, but was not materially
  faster than exact confirmation and changed four actual-board calls, including
  Gibbs `$111` from PASS to TARGET; it remains research-only. Turning joint
  refinement off restores the additive timing baseline, avoids retaining the
  large construction banks, and advances the calculation/cache version so v14
  results cannot be reused. The 56 focused Sequential tests pass. Durable study:
  `research/studies/2026-08-24_sequential_shared_opportunity/`.
- Auction League Settings now defaults to predicted salaries and offers a
  governed `Use Actual Salaries` hindsight toggle when completed results have
  been published. The active 2026 NV actual slice contains exactly 156 drafted
  offensive players totaling `$3,435`; prices and keeper costs are exact,
  salary rescaling/selection reserve are disabled, and variation 0 is the
  baseline weekly-outcome view. Source/app slice parity, 92 App tests, a
  one-trial managed optimizer run, and the rendered UI smoke pass.
- Full governed refresh `20260826T212331Z_dbd5cac4` is live from the August 26
  source pulls and revised beta keeper file. All 30 build/validation/app steps
  completed before explicit promotion. The release publishes 348 DK, 382
  NFFC, 324 beta, and 324 NV projections with exact weekly-map parity. Beta
  now has 13 keepers spending `$383`, leaving 143 market slots and `$3,193`;
  NV remains at 16 keepers spending `$453`, 140 slots, and `$3,123`. The beta
  reserve seed completed 1,000/1,000 trials and published 311 premiums. Staged
  and live model/Auction/Snake artifacts match exactly; SQLite integrity,
  handoff idempotence, source/app parity, and all app smokes pass. The release
  report is under
  `Data/Production_Refresh_Stages/20260826_keeper_projection_refresh_v4/`.
- Fresh projection providers no longer supply a current expert center for
  Jayden Higgins, although DK/NFFC market feeds still rank him inside protected
  depth. The handoff therefore retains his market/audit evidence but does not
  fabricate a center from July or legacy data. DK and NFFC carry the explicit
  reviewed reason `market_only_without_current_projection_center`; the fresh
  production release omits him from all four league surfaces. The exclusion
  map compiles and all 37 handoff tests pass.
- Bounded same-position salary reinvestment remains the
  Fantasy_Football_App Sequential default (introduced in cache version 13;
  current cache version 14). Four paired
  evidence variations across prespecified early, middle, and late-cap states
  pass legality, completion, budget-use, and accumulated anchor-stability
  gates; unused salary falls by `$11.57` on Buy and `$24.81` on Pass paths.
  Gibbs remains an accumulated TARGET and late-cap Pitts remains PASS. Bowers
  at `$51` changes from baseline TARGET to bounded PASS because the Pass roster
  can now deploy its budget, not because his projection or weekly outcomes
  changed. The rollout uses no new solver and no direct spend reward. Durable
  study: `research/studies/2026-08-21_bounded_app_shadow/`. Its execution path
  now scans a pre-ranked salary order for dynamic top-N checks and applies
  incremental cap/top-N legality to same-position reinvestment swaps. Top-10
  low/high max-bid anchors are deferred to an explicit UI action; repeated Add
  Evidence runs remain market-only and invalidate any previously derived max
  bids until the final replay across all retained seeds. Fixed-seed checks are
  frame-exact; the focused mechanisms reduced direct reinvestment time 20.4%
  and initial-board time 12.3% at budget 120. Runtime study:
  `research/studies/2026-08-21_sequential_runtime_optimization/`. A follow-up
  incremental ordinary-refresh challenger preserved exact paths but was
  rejected and reverted: fresh early-state processes were 84% slower, while
  middle/late states were flat-to-slower, so that challenger remains rejected.
- Reconstructed the current beta Bijan decision after Gibbs `$110`, Chase
  Brown `$34`, and Bhayshul Tuten `$11`. At Bijan `$105`, eight production
  evidence variations all return TARGET with mean Buy-minus-Pass `+22.54`
  managed-season points; Top-N and the four-RB minimum do not force the result.
  Ordinary Pass rollouts leave about `$45` more unspent, though. A deliberate
  full-cap Tee Higgins plus Emeka Egbuka completion ties Bijan's mean within
  `0.18` season points and improves p10 by `15.28`. The engine is functioning
  as designed, but the action is policy-contingent rather than proof of strict
  roster dominance. No production change. Durable study:
  `research/studies/2026-08-20_bijan_fourth_rb_audit/`.
- Tested fixed-scale overall `log1p(ADP)` in the weekly-template matcher as a
  same-weight replacement for positional ADP rank and as an added 0.50-weight
  field. The 2,647-target-per-league role-tiered replay has complete coverage,
  but neither arm replicates across DK/beta and development/temporal cells.
  Adding it raises Brock Bowers' beta ADP-35-or-earlier donor weight from
  38.26% to 41.33% without improving centered q90 or P(+5), so production is
  unchanged and the field remains diagnostic-only. Durable study:
  `research/studies/2026-08-20_template_overall_log_adp/`.
- A staged salary-only refresh from the August 20 Beta/NV ESPN copies is live.
  Both active exports now use the variable-length terminal-`$0` contract: Beta
  parsed 180/180 records and NV parsed the user-confirmed 240/240. Beta replaces
  Puka Nacua at `$75` with Tucker Kraft at `$11` in the 14-player keeper set,
  leaving `$3,135` across 142 non-keeper slots; NV initially retained zero
  keepers and a `$3,576` top-156 budget before the August 22 keeper refresh
  above. Both leagues publish 327 keyed salary rows.
  The fresh Beta reserve completed 1,000/1,000 trials and publishes 313 rows
  with an expected roster reserve of `$9.0908`. Source/Snake share SHA-256
  `bae74108f2729194703fb2c7d2b5dfdbf2e7766ae636382b66ca2ad9a1a55a72`;
  Auction is `c4fcccd33e298774b1dccffd67f931b5c84ea6b385accb3b348286b44a814cd6`.
  All SQLite, parity, and live app-smoke gates pass. Rollbacks and the input
  receipt are under `Data/Production_Refresh_Backups/20260820T2132Z_salary_only/`.
  GitHub branches `codex/salary-refresh-20260820` are synchronized at Auction
  commit `e52af13` and Snake commit `a6ce708`; `main` is unchanged pending
  separate merge authorization.
- App/Snake publication for the FTN-adjusted release is complete. The first
  App push exposed corrupt loose database objects in both sibling repositories:
  App `fe8bc519142f7efafd20feb1ae0f61fcb7a27ae0` and Snake
  `ed6641a1b8d05589b818ccd886375ae2828a682b`. Each promoted working database
  matched its manifest SHA-256 and independently hashed to the exact missing
  Git blob ID. The corrupt compressed files were preserved under
  `Data/Production_Refresh_Backups/20260820T025602Z_7b2e9926/git_object_recovery_20260820/`,
  both objects were reconstructed, and strict full `git fsck` passed in both
  repositories. App commit `f306114` and Snake commit `ce3c4ff` are now live on
  GitHub `main`, with clean synchronized local branches.
- FTN-adjusted schema-6 stage `20260820T025601Z_a672217a` completed all 30
  governed steps from the refreshed 2026 inputs. The live release publishes 347
  DK, 382 NFFC, 327 beta, and 327 NV projections with exact weekly-map parity,
  80 donors/player, 1,000/1,000 auction trials, exact Auction/Snake database
  parity, and clean app smokes. The change report records 20 added and 14
  dropped league-player rows versus live. Haynes King and J'Mari Taylor now
  have complete NFFC current/next handoff evidence, so their stale
  `market_only_without_current_projection_center` exclusions were removed;
  focused handoff tests pass 37/37. The stage was explicitly promoted at
  `2026-08-20T15:29:51Z`; all 11 installed model/app artifacts match their
  staged hashes, and durable pre-refresh backups plus the release report are
  under `Data/Production_Refresh_Backups/20260820T025602Z_7b2e9926/`.
- Repaired the local Git object database after four large loose database blobs
  failed zlib validation. The one reachable blob—commit `4b44a678`'s
  `Data/Databases/Simulation.sqlite3`—was reconstructed byte-for-byte from the
  exact-hash managed-template pre-fix backup. Three objects absent from all
  branch and reflog reachability were quarantined outside `.git`; two also have
  exact governed database backups. The damaged originals and SHA-256-preserving
  copies are under
  `Data/Production_Refresh_Backups/20260815T0038Z_managed_template_contract/git_object_recovery_20260814/`.
  Full strict `git fsck`, connectivity, `git diff --check`, committed-blob reads,
  and normal status/diff operations now pass. The live working databases and
  unrelated changes were not altered by the repair.
- The weekly-template builder now owns a separate managed-auction center contract:
  `managed_profile_ppg`, `managed_residual_center_ppg`,
  `managed_active_ppg_resid`, and `managed_center_policy`. Zero-active donors use
  a positive conditional V2 center when available, otherwise a governed
  historical fallback, so a near-zero legacy prediction can no longer inflate
  a partial-season QB path. Build, export, and production-release gates reject
  inconsistent centers/residuals or any per-week/season multiplier above the
  league horizon. Rebuilt beta/DK/NFFC/NV templates were promoted to the source,
  Auction, and Snake databases with exact generated-table parity; the source and
  Snake files share SHA-256 `27ca20d41296f544932861040c2a4e10cdf7075ef583b1dc249c6f4479493be5`.
  Snake deliberately continues to consume its validated legacy/V2 fields and
  ignores the auction-only columns. Durable pre-fix copies are under
  `Data/Production_Refresh_Backups/20260815T0038Z_managed_template_contract/`.
- Schema-6 stage `20260814T153232Z_d5feb5a5` completed all 30
  governed steps and both app smokes. Final populations are 344 DK, 381 NFFC,
  326 beta, and 326 NV with exact weekly-map parity, 80 donors/player, and no
  unresolved ADP audit rows. NV publishes 326 unique `nvpred` salary rows with
  exact projection-key parity, zero keepers, and a top-156 total of `$3,576`
  (12 teams times the `$298` offensive cap). Auction smoke rendered beta and NV
  with zero errors/exceptions; Snake smoke rendered DK/NFFC cleanly. The saved
  release report treats NV as 326 first-time additions. The release was
  explicitly promoted at `2026-08-14T17:31:42Z`; all 11 installed model/app
  artifacts match their staged hashes, every SQLite quick/foreign-key check
  passes, and durable rollback copies plus the release report are under
  `Data/Production_Refresh_Backups/20260814T153233Z_2a4365dd/`. The final
  app-parity resume needed explicit SQLite digest-connection closure to avoid
  transient Windows read corruption; harden that connection lifetime before
  the next fresh refresh.
- Complete refreshes now produce a hash-bound pre-promotion change report. It
  prints every added/dropped league-player row, the ten largest published PPG
  increases/decreases, and the ten most positive/negative probability-weighted
  template residuals with prior values/deltas. JSON and Markdown copies are
  saved in staging, verified again before installation, and archived beside the
  durable rollback databases. A retrospective report for the August 8 release
  is under `Data/Production_Refresh_Backups/20260808T171956Z_710ed569/`.
- Refreshed stage `20260808T171956Z_9604d0b4` completed all 25
  governed steps with 345 DK, 382 NFFC, and 326 beta players, 80 weekly donors
  per player, 1,000/1,000 auction trials, exact Auction/Snake parity, and clean
  app smokes and was explicitly promoted at `2026-08-08T19:20:34Z`. All ten
  installed hashes and SQLite checks pass, and all ten durable pre-refresh
  backups exist. Ricky Pearsall is explicitly excluded from
  DK/NFFC because his user-confirmed season-ending PCL injury has no current or
  next projection center while recent ADP is still decaying; the canonical ADP
  rows remain for audit. NFFC reviews the first 363 offensive ADP candidates so
  three reviewed incomplete rows can be omitted while retaining the full
  360-pick market surface.
- Notebook 1 now handles FantasyPoints' grouped-header
  `projections.season.csv`, prints post-commit FantasyPros/NFFC/DK ADP save
  confirmations, and hardens FFToday against partial page pulls with retries,
  structural table selection, per-position depth floors, atomic replacement,
  and exact post-write confirmation. The repaired 2026 FFToday slice is
  QB/RB/WR/TE 50/95/131/50.
- Canonical ADP governance v2 is implemented and live. MFL remains modeled only
  through 2024. From 2025 onward the model
  median gives one family vote each to FantasyPros redraft, FantasyPros
  best-ball `AVG`, direct DraftKings ADP, and the NFFC Overall/$25-$50
  aggregate. NFFC literal downloads are `ADP.tsv` (Overall) and `ADP (1).tsv`
  ($25/$50); raw Rotowire/Cutline rows remain audit history. The NFFC center
  and bounds average the two available feeds, pooled SD includes within-feed
  range and between-feed disagreement, and one-feed fringe rows disclose
  `source_count=1` with null `feed_gap`. Beta template matching now prefers the
  same V2 canonical family ADP; ETR still controls beta eligibility and is only
  the last-resort match fallback. Fresh stage
  `20260804T193416Z_4136d153` completed all governed build/validation/app steps
  with 348 DK, 382 NFFC, and 326 beta rows, 80 donors/player, 1,000/1,000
  reserve trials, exact app parity, and zero smoke errors. It was promoted at
  `2026-08-04T20:27:28Z`; all ten installed digests matched the manifest at
  promotion, every durable pre-refresh backup exists under
  `Data/Production_Refresh_Backups/20260804T193416Z_b6992d72/`, and promotion
  reported no cleanup warnings. The repository Simulation digest changed only
  through the separately receipted post-promotion vacuum described below.
- Every governed refresh now runs `compact_simulation` after the final
  Simulation writer and before validation, app copies, and promotion. The step
  records before/after file and page receipts and requires integrity/foreign
  keys `ok` plus a zero freelist. The just-promoted live source was compacted
  once under the same rule from 114,450,432 to 62,353,408 bytes, with all 26
  table digests unchanged; its pre-vacuum copy and JSON receipt are retained in
  the current release backup folder.
- The frozen Ridge `alpha=10` replacement completed full corrected-lineage
  downstream validation. Annual 2017-2025 refits improve pooled RMSE by only
  0.001401 DK and 0.001264 beta with both player-cluster intervals crossing
  zero; 2025 loses in both, leaving 4/6 recent season wins. Strict-prior player
  distributions and all core/depth template cells pass, but fixed-roster score
  CRPS improves only DK development and worsens DK temporal by 0.311%, beta
  development by 0.378%, and beta temporal by 0.527% versus the 0.5% gate.
  Reject the Ridge swap for 2026 and retain equal-third Lasso/RF/LightGBM.
  HistGBM remains rejected; production is unchanged.
- Logged cross-provider expert-rank disagreement is rejected for both point
  prediction and residual uncertainty. After within-source season-position
  normalization, common-depth mapping, and `log1p` MAD, adding disagreement,
  source count, and coverage to normalized rank level worsens controlled RMSE
  by 0.00803 DK and 0.00326 beta and improves only RB. Strictly-prior scale
  CRPS also worsens by 0.07%/0.03%, and causal excess disagreement does not
  rescue the result. Keep normalized rank level as the unpromoted challenger;
  the RB-only movement is post-hoc. See
  `research/studies/2026-08-04_v2_logged_rank_disagreement/`.
- A direct current-lineage raw-log confirmation also leaves normalized expert
  rank preferred. `log1p(median raw overall rank)` worsens normalized-rank
  controlled/production RMSE by 0.00076/0.00174 in DK but improves by
  0.00162/0.00249 in beta. Both win only 5/9 controlled seasons and the
  relevant intervals cross zero. The scoring disagreement plus normalized
  rank's better depth/QB-placement semantics rejects raw log as a replacement.
  See `research/studies/2026-08-04_v2_log_expert_rank_confirmation/`.
- The final normalized-rank promotion test reselected the complete locked
  model grids at every forecast origin. DK improves pooled/recent RMSE by
  0.00181/0.00370 and player CRPS by 0.00112, but its interval crosses zero
  and only 2025 improves recently. Beta is flat-worse, worsens player CRPS by
  0.00042, and worsens QB RMSE by 0.01280. Only 3/6 recent league-season cells
  win. Stage A therefore rejects promotion and intentionally does not run
  post-failure template or roster transport. Keep normalized rank in
  research/audit only. See
  `research/studies/2026-08-04_v2_normalized_rank_promotion/`.
- A follow-up expanded both LightGBM and CatBoost from eight to 16 pre-2023
  candidates. DK retained the exact LightGBM 0.05/100 incumbent. Beta selected
  a new 0.01/500 schedule pre-2023 but worsened pooled 2023-2025 blend RMSE by
  0.000583 and slightly weakened the Extra Trees blend. Both CatBoost searches
  retained the original 0.03/300 candidate and its beta loss. Retain current
  LightGBM parameters and CatBoost's rejection; no production change.
- The beta weekly-template scoring-context correction is active. Governed run
  `20260803T040708Z_2075ac47` rebuilt and promoted the full pipeline with exact
  beta V2 preseason matcher context and `beta_scored_expert_fallback`; validated
  legacy OOS centers remain where available. The 39 sack-context-unavailable
  2018 QBs remain auditable and donor-ineligible. This was an explicit
  data-correctness override, not a predictive promotion: the full arm passed all
  player gates but worsened development roster CRPS by `+0.9061%` versus the
  `0.5%` gate and 2023-2025 by `+0.3790%`. All 24 refresh steps, 1,000/1,000
  reserve trials, both app smokes, installed hashes, SQLite integrity, and app
  content-parity checks passed. See
  `research/studies/2026-08-02_beta_scoring_context/`.
- A locked model-family screen selected eight Extra Trees and eight CatBoost
  configurations only on rolling 2013-2022 origins, then tested fixed
  equal-four blends on the reused 2023-2025 confirmation block. Extra Trees
  improved RMSE by 0.003461 DK and 0.006270 beta, won every league-season and
  non-QB position cell, and survived five estimator seeds per scoring system.
  It is a research shadow candidate only: DK uncertainty crosses zero and the
  scoring systems share player outcomes. CatBoost is rejected after its small
  DK gain reversed in beta; production remains the equal-third
  Lasso/RF/LightGBM blend.
- `ADP_Ranks` provider replacement is now source-scoped, depth-validated, and
  atomic. The live `Season_Stats_New` FantasyPros slices contain 313 valid 2024
  offense rows, 302 valid 2025 offense rows including 42 recovered QBs, and
  466 valid 2026 rows including one 32-team DST snapshot. Duplicate provider
  keys and legacy `DS`/`K1`-`K9` rows are gone. Governed run
  `20260802T174943Z_5b3cff69` rebuilt and promoted all three V2 lineages from
  the corrected source, so the refreshed market consensus and fingerprints are
  now active.
- Historical V2 team context now discards only the mutable/backfilled team
  labels from historical `FFA_RawStats`, `FFA_Projections`, and
  `FantasyPros_Best_Ball_ADP`; source rows and projection/market values remain.
  Team aliases are canonicalized before consensus, true trusted-source ties
  remain null, and the policy plus alias map is hashed into every foundation.
  Live DK/NFFC/beta each contain 6,665 identities, 56,162 aliases, and 13,824
  feature keys. Audited 2019 movers resolve to Hopkins=HOU, Cooks=LAR,
  Diggs=MIN, and Winston=TB; Christian Kirk and Trevor Lawrence share exact JAC
  QB context in 2022-2023.
- ESPN salary ingestion is structural rather than player-name-length based.
  The 2026 cycle atomically repairs only staged governed salary slices. The
  frozen historical counts remain exact at 200/160; the live preseason Beta
  and NV exports are variable-length, must end at an ESPN `$0` record, and
  preserve exact parsed/post-write parity. The current files parse 180/240
  records and both 327-key salary populations have exact projection parity.
  Team remains provenance-only for salary modeling;
  the only reviewed unresolved-team fallbacks are Stefon Diggs and Deebo Samuel
  Sr., both requiring `team_conflict=1` under
  `v2_nullable_team_conflict_v1`.
- Annual current/next model selection now uses a separately promoted
  `V2_Parameter_Cache.sqlite3`: 36 season/league/model entries are reusable only
  under an exact training-data and model-spec fingerprint. Cache hits skip the
  grids but still refit and predict every origin. Random forests use four
  workers. The schema-5 2026 production refresh promoted 36 validated entries:
  all 7 current and 5 next selections hit cache for each of DK, NFFC, and beta,
  while every selected-origin model was freshly refit and predicted.
- Raw provider exports remain a manual step in
  `Scripts/Data_Generation/1_Update_Projections.py`. After that boundary,
  `python -m Scripts.V2.refresh_production --year 2026` owns the complete
  downstream build in an isolated, resumable stage: canonical current/next
  Model_Inputs, separate DK/NFFC/beta V2 foundations and locked shadows,
  annual file-backed keepers, the idempotent ADP/projection handoff, three
  weekly surfaces/template audits, salaries, a fresh 1,000-trial selection
  reserve, release validation, and both app candidates. The approved-cycle
  registry binds exact annual runners, versions, floors, horizons, and template
  eras; its receipt/hash is immutable through resume and promotion. Only
  current season 2026 is approved, so current season 2027 fails closed until
  new annual inputs and evidence are registered. The NFFC candidate uses the
  core plus first-363 canonical offensive ADP union, NFFC scoring, and 17-week
  2021-forward templates. Snake labels it offense-only 3RR: canonical TK/TDSP
  remain audit rows, while K/DST and alternate contest formats are unsupported.
  The complete 24-step 2026 production run and atomic promotion passed: all
  four exact NFFC raw-feed labels/depth floors, the 383-player 17-week NFFC
  surface, fresh 1,000/1,000 auction reserve, and both app smokes passed. Live
  DK/NFFC/beta maps contain 350/383/328 players. The Windows host still retains
  bounded native-crash retries rather than being described as fully stable.
- The later Model_Inputs `0xC0000005` failure was reproduced independently of
  database completeness and traced to pandas' grouped rolling-window path,
  which also returned a corrupted internal window-bound type. The compiler now
  uses an equivalent within-player three-lag mean/max reduction; all 18 output
  tables match the prior successful build within `1e-12`, and repeated full
  compiles complete in about 22 seconds without warnings. Refresh manifest
  schema 5 retains immutable current/next Model_Inputs bases and restores both
  before every attempt or resume. Non-schema-5 stages must not be resumed.
- A subsequent `locked_dk` access violation was isolated to cumulative native
  LightGBM fitting, not its inputs, feature names, OpenMP inventory, or one bad
  fold. Annual current/next runners now execute each LightGBM grid origin in a
  fresh spawned worker with at most eight fits, use the same ceiling for
  selected replays, release each fitted pipeline, and retry one abruptly
  terminated batch. The unchanged locked-DK model replay then completed all
  families against the staged feature mart (660 current rows, 615 PPG rows;
  SQLite integrity `ok`), and the following-season runner published 660 2027
  shadows against the same disposable copy. Start a fresh governed stage
  because this runtime fix changes the code fingerprint; no live artifact was
  promoted.
- Production handoff exclusion policy v3 now treats incomplete market-only
  players in the final sixth of a league's draft surface as audited tail
  omissions rather than run-fatal rows. Core players, keepers, and new gaps in
  the protected first five-sixths still fail closed, and the remaining population
  must cover the entire draft. A disposable replay of the completed staged
  models published 350 DK, 383 NFFC, and 328 beta rows; J'Mari Taylor plus three
  NFFC ADP-316/317 players were audited out, while the now-complete Najee Harris
  row was restored by removing its stale annual exclusion. The prior stage's
  code fingerprint is stale, so a fresh governed refresh is required.
- FantasyPros season projections now require four manually exported QB/RB/WR/TE
  CSVs with the exact `FantasyPros_Fantasy_Football_Projections_<POS>.csv`
  filenames. The loader archives and schema-validates them; it no longer uses
  the login-limited HTML tables.
- The NFFC weekly matcher now takes all scoring-sensitive historical and
  current preseason context from the NFFC-scored V2 consensus; DK-scored
  `Model_Inputs` values are audit-only. Receiver environment uses the selected
  QB1's NFFC passing-point component rather than total QB fantasy PPG.
  Historical donors use
  `nffc_scored_expert_consensus`. A strict 540-target 2023-2025 replay rejects
  the locked OOF center: locked-minus-expert PPG CRPS is +0.002901, it loses
  3/3 seasons, its player-cluster interval is [-0.004914, +0.010748], and it
  passes 6/10 gates while failing all three promotion gates. The live surface
  has 1,509 2021-2025 templates with 17 populated weeks and a 383-player map.
  Annual rebuilds remove older active-league pool/map rows and
  Snake exposes only prediction slices backed by the current map, preventing
  regenerated template IDs from being paired to stale years.
- Weekly-template matcher validation now uses role-tiered objectives. Core
  players (main QB/RB/WR/TE cutoffs 18/36/48/18, with strict and broad
  sensitivities) optimize active-PPG CRPS first and managed contribution among
  one-SE PPG near-ties. Individual played-games CRPS is diagnostic for core
  players; aggregate played bias, extended-absence calibration, coverage,
  temporal/position slices, and replacement-aware roster CRPS remain gates.
  Depth players retain the equal-third PPG/contribution/played composite. A
  fresh 2,647-target per-league replay made the 0.25x all-distance matcher the
  only one-SE finalist, improving development core PPG CRPS by 0.007901 DK and
  0.005511 beta. It failed downstream 20-player roster CRPS non-inferiority:
  DK worsened 0.7096% in development and 0.5696% in 2023-2025; beta was
  slightly worse. Aggregate missed-week bias stayed within margin, so this is
  not an injury-prediction rejection. Keep production matching unchanged and
  use the role-tiered validator for future matcher studies.
- Weekly-template validation now also has a causal rare-upside objective. A
  league-winner player-season requires at least +5 active PPG over the held-out
  projection and managed contribution above a position-specific q90 estimated
  from comparable preseason-ranked players in the five strictly prior seasons;
  q95 is the severity sensitivity. Ordinary PPG/contribution CRPS remains the
  calibration gate. The tighter WR PPG plus 0.25 projected YPR/TD-rate arm was
  the only saved finalist to improve q90 Brier, log loss, tail-utility CRPS,
  and contribution CRPS in all four DK/beta development/temporal cells, but
  most season intervals crossed zero. It also failed to transport to 12-team
  championship probability, worsening Brier/log loss in both DK periods and
  beta development while improving only recent beta. Production matching and
  both app objectives remain unchanged. Future Auction/Snake objective tests
  should keep calibrated draws and expected-score non-inferiority, then use a
  paired championship-probability lower bound only among near-tied actions.
  Raw DK tail probabilities need calibration work first: production predicts
  7.45%/6.02% versus 16.39%/14.17% realized core q90 events in development and
  2023-2025.
- A corrected 2,647-target per-league replay tested causal prior-season
  nflfastR receiver profiles: target share, air-yards share, aDOT, red-zone
  target share, and targeted-week usage dispersion. Profiles are
  position-season ranked, opportunity-shrunk, and neutral for rookies/missing
  histories. All four WR/TE bundles failed to improve development core PPG in
  both leagues; the closest arm worsened mean cross-league CRPS by 0.0095%.
  The features reduced Ladd McConkey's Terrelle Pryor donor weight and moved
  Pryor from beta rank 3 to as low as rank 10, but the more intuitive pools
  did not forecast better. Retain production matching. TE-only usage/depth is
  exploratory follow-up evidence, not a promoted specification.
- A follow-up causal nflfastR RB replay tested red-zone/goal-line carry room
  share, third/fourth-down target room share, and combined role bundles. Main
  core RB coverage was 86.1%. No arm improved development core PPG in both
  leagues; the strongest combined arm was effectively flat on average but
  worsened DK and improved beta, with clustered intervals crossing zero.
  Scoring role alone weakened temporal performance. Passing-down share improved
  the depth-player PPG/contribution/played composite in both league/period
  cells, but that depth-only slice is post-hoc. Keep global production matching
  unchanged; only a separately frozen depth/tapered passing-down follow-up is
  supported.
- A strict 1,620-target/league height/weight ablation joins the existing
  nflverse player master through exact V2 IDs and covers every rolling target
  plus 5,291/5,298 historical templates. The 0.25+0.25 primary size arm changes
  about 9% of donors overall and 12% for WR, but does not transport: beta
  modestly improves PPG, contribution, played-games, and impact while DK
  slightly worsens PPG, contribution, and impact discrimination. Height alone
  is essentially neutral in DK and weakly favorable in beta; removing QB and
  increasing size weight do not resolve the disagreement. Keep production
  unchanged and defer combine acquisition unless a separate prespecified
  athletic-testing hypothesis justifies its lower coverage and missingness.
- NFFC now contributes exactly one modeled ADP observation per player-season:
  the existing `ADP_Averages(league='nffc')` composite of its four contest
  feeds. The raw `NFFC_ADP` rows remain identity/candidate evidence but no
  longer receive four additional votes in the market consensus. The live
  DK/beta marts each contain 28,801 market/rank rows and a maximum of eight ADP
  sources. The locked replay improves modestly to 3.1076 DK and 2.8841 beta
  RMSE. ETR overall rank remains the beta top-180 eligibility ordering.
  A leakage-safe expert-rank challenger normalizes each source within
  season/position before taking a median. A full-column-forest control removes
  the locked forest's feature-subsampling confound. Rank level improves that
  blend by 0.0022 DK and 0.0019 beta RMSE with seven of nine season wins; DK's
  two intervals exclude zero, while beta's season interval excludes zero and
  its player interval ends essentially at zero. The locked-production-surface
  sensitivity is smaller and uncertain, and the expert-minus-projection gap is
  neutral. A scoring-matched raw-rank follow-up rejects
  percentile-after-median plus coverage: controlled gains are 0.00138 DK and
  0.00016 beta, both intervals cross zero, and it trails the normalized
  comparator by 0.00173/0.00213. `log1p(raw median)` is not distinguishable
  from the normalized comparator in this study, so raw rank stays audit-only
  and normalized rank stays unpromoted pending future-season confirmation.
  The source,
  Simulation, Auction, and Snake cutover passed all population,
  app-owned-table, and integrity gates.
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
  unavailable 2018 QB diagnostics retain an audited legacy center and exact
  quarantine-linked reason but are donor-ineligible rather than importing DK
  or zero-sack values.
  Corrected 1,620-target/league replays retain current match weights and keep
  next-year fields out of the matcher: their small PPG gains do not survive the
  joint contribution/played-games gates. The promoted V2/Simulation artifacts
  match staging byte-for-byte; all 20 Auction generated tables match source,
  all six app-owned tables are unchanged, and every Snake table matches.
- The V2 production handoff is active for 2026 DK (350 players:
  56 QB/100 RB/143 WR/51 TE) and beta (328:
  50 QB/95 RB/133 WR/50 TE). Beta is the core plus top-180 ETR overall-rank
  ordering and all keepers. The 1,980-row
  eligibility audit retains every inclusion and exclusion decision.
  Current `pred_fp_per_game` uses the locked league-specific V2 center; current
  residual quantiles are zero and the joint matched donor supplies the only
  PPG residual plus weekly/played path. Next-year `pred_fp_per_game_ny` is
  conditional on appearing and `pred_appear_ny` supplies the separate
  Bernoulli risk; the auction keeper path zeros future market value on no
  appearance. DK uses its legacy/preseason historical-center policy; beta uses
  validated legacy OOS centers where available and the beta-scored expert
  fallback otherwise; NFFC uses the scoring-matched expert consensus described
  above.
  Legacy current/next fields are audit-only, while V2 is the production
  authority. DK/beta V2 historical donor centers remain diagnostic with
  `v2_recenter_promoted = 0`: recentering worsened PPG CRPS by 0.0057 DK and
  0.0051 beta, with both player-cluster intervals above zero.
- Weekly template and current player-map tables now require canonical V2
  `player_key` plus match provenance. Governed source aliases resolve first,
  confirmed identities beat redundant provisional aliases, team disambiguates
  true same-name collisions, and pre-play rookies retain stable provisional
  keys. Coverage is 10,596/10,596 historical beta+DK templates and 678/678
  current rows. Every player receives 80 donors. Required current context joins
  are key-first: DK has 342 exact ADP matches and eight governed fallbacks;
  beta has 237 exact matches and 91 governed fallbacks, with zero generic
  default/review rows. `LA`/`LAR` and `ARZ`/`ARI` are canonicalized only for
  room features, outward labels remain unchanged, and `FA` room features are
  zero. Tetairoa McMillan and Amon-Ra St. Brown retain their canonical
  identities.
- The canonical current ADP handoff publishes 416 live DK rows, 497 NFFC rows
  (431 offense, 33 `TK`, and 33 `TDSP`), and 243 ETR rows. The latest local
  NFFC and ETR exports are dated 2026-07-27. All eight governed handoff hashes
  are unchanged on the second publish. Final gates pass 187 main, 69 strict
  release, 49 Auction, and 16 Snake tests plus Snake `AppTest` with zero
  exceptions. Backups live under
  `research/studies/2026-07-30_canonical_adp_handoff/results/pre_promotion/`.
- Beta has a separate rebuilt `Projection_V2_beta.sqlite3` lineage with exactly
  the same 6,655 identities and 13,909 player-season keys as DK but independent
  outcomes, provider scoring, consensus features, fits, and hyperparameters.
  `v2_conditional_ppg_2026_candidate_beta_v1` scores 2.8841 RMSE versus 2.9600
  for expert recalibration and wins 9/9 seasons. The no-history route remains
  an unpromoted secondary diagnostic; its pre-quarantine exact metric is
  superseded. Prior-only point calibration is rejected. The beta fit publishes
  745 shadow rows, 673 PPG centers, and 745 participation probabilities.
- Projection V2 now has a versioned DK shadow lock:
  `v2_conditional_ppg_2026_candidate_v1`. A complete-season, strictly-prior
  2017-2025 replay scores 3.1076 RMSE versus 3.1951 for expert recalibration
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
- The earlier isolated NFFC setup preview with cloned DK
  projections/templates is superseded. Do not promote or present that database
  as NFFC output; use only the independently scored, approved-cycle candidate
  after its release gates pass.
- The modeling repo owns the source `Simulation.sqlite3`. The weekly builder
  syncs generated best-ball tables to the auction app without replacing its
  keeper/salary scenarios, and copies the complete database to Snake.
- Weekly template matching now uses absolute PPG, projection/market
  disagreement, uncapped experience, and full-universe workload-room structure.
  Adaptive capped donor weights retain broad ESS, ordinary zero-active outcomes
  remain eligible, and the declared Bell 2018 holdout is audit-only.
- A strict 1,620-target/league receiver-rate ablation now tests preseason
  projected yards/reception and receiving TDs/reception. The primary 0.50
  WR/TE arm changes about nine of 80 donors and modestly improves PPG and
  contribution point estimates, but it worsens WR played-games or impact
  guardrails; RB contribution also worsens. Keep both rates outside the global
  matcher. TE yards/reception improves full-period contribution CRPS by 0.0649
  DK and 0.0459 beta with season/player intervals below zero, but remains a
  same-evidence future-origin challenger because DK recent played-games safety
  fails.
- A 648-WR/league follow-up jointly tested the requested tighter 2.25 PPG
  weight with the projected rate profiles. It tightens Ladd McConkey's current
  beta donor PPG gap by 20.8% for 0.135 fewer expected games, but worsens DK
  held-out PPG CRPS and does not transport in the combined arms. Terrelle Pryor
  remains top-three and usually gains probability because his preseason YPR
  and raw TD/rec are close to Ladd's. Keep production unchanged; a direct,
  historically complete route/usage variable is the appropriate next
  archetype test, not an ad hoc measurable added to remove one low-weight comp.
- A second 648-WR/league follow-up tested one-year and recency-weighted
  three-year preseason projection trajectory at 0.25/0.50 weights. It fixes
  the visible comp semantics—one-year 0.50 moves Pryor from beta rank 3/2.23%
  to rank 12/1.43%—but every arm worsens PPG CRPS in both leagues over both
  full and recent periods. Three-year 0.25 improves contribution/impact
  diagnostics but still fails the PPG-first requirement. Explicit history
  availability/depth is harmful. Keep trajectory audit-only and production
  unchanged.
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
- The live salary surface is
  `current_locked_spec_v6_v2_population_11f`. It keeps v5's compact
  keeper-market/additive-calibration structure but drops
  `ensemble_pred_resid_90`: the 11-field rolling MAE is `$4.2975` versus
  `$4.2991` for 12 fields, and the current weekly-donor QB p90 scale was 9.35
  times the historical projection-residual scale. Centered donor p90 remains
  diagnostic only. The beta salary slice has exactly 328 canonical keys:
  326 `ProjOnly` rows plus V2 fallbacks for Stefon Diggs and Deebo Samuel;
  all 14 keepers are keyed and the top 142 non-keepers total `$3,071`.
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
  `Salary_Selection_Premium` to both source and auction-app databases. The live
  v6-population refresh completed 1,000/1,000 rosters across 314 non-keepers
  and produces an expected 13-player reserve of `$8.5598`. The governed
  historical-v5/current-v6 transfer is
  `historical_v5_selection_surface_to_current_v6_v1`; current common-player
  salaries have correlation `0.99957` and MAE `$0.274`.
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
  Target Buy/Pass and Current Nomination remain current-year estimands. The
  keeper-aware reserve has been refreshed against the live v6 population.
- The fresh keeper-aware annual reserve completed 1,000/1,000 optimal Target
  seeds across 314 non-keepers, sums to 13 roster slots, trains only through
  2025, and matches exactly between source and app databases. Its expected
  roster reserve is `$8.5598`.
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
- The archived salary-v2 study used an experimental
  `current_locked_spec_v2_ensemble_features` feature path that joins the
  optimizer-aligned OOS validation point ensemble for historical seasons and
  `Final_Predictions_Resid` for the current season. It adds positional ensemble
  strength, ensemble-versus-consensus and ensemble-versus-ESPN-price rank gaps,
  ensemble PPG per source dollar, and selected position interactions. The join
  covered all 2,696 validation-era projection rows exactly; deep players
  outside the then-180-row final ensemble retained a consensus-PPG fallback.
  The v1 validation slice remains preserved for that historical comparison.
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
  optimizer-tail correction before another full replay. That decision is
  historical and does not describe the live v6 population.
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
