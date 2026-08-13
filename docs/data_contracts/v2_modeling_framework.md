# V2 Modeling Framework Contract

Last updated: 2026-07-31

## Scope

V2 Milestone 4A began as a shadow comparison of conditional-PPG and
participation models. Its deterministic out-of-fold evidence remains in
`Data/Databases/Projection_V2.sqlite3`, while the locked 2026 current and 2027
next-season DK/beta forecasts are published to production through
`Scripts/V2/production_handoff.py`. The approved 2026 refresh also builds a
separate offense-only NFFC candidate; it is not live evidence until its
league-specific acceptance and release gates pass and explicit promotion
completes.

Run the initial comparison with:

```powershell
python -m Scripts.V2.build_milestone_4
```

The command reuses the active Milestone 3 feature mart. Pass
`--rebuild-feature-mart` only when a full upstream rebuild is intended.

## Target Populations

### Conditional PPG

The conditional-PPG comparison requires:

- a completed historical outcome;
- at least four position-specific opportunity games;
- a non-null conditional-PPG target; and
- a non-null same-season preseason expert consensus.

Requiring team-game consensus creates one common comparison population for the
incumbent, direct models, and residual models. The incumbent uses
`expert_ppg_active_median` when a provider supplies projected games and falls
back to `expert_ppg_team_game_median`. The raw team-game consensus remains a
reported diagnostic baseline. This hybrid acknowledges that recent providers
have better availability inputs without dropping older OOF seasons.

### Participation

The participation target includes completed candidate-spine rows whose
identity is confirmed (or whose outcome was directly observed) and uses
`appeared` as its binary label. Confirmed zero-game rows remain in this
population. Unresolved identities are not assumed to be zeros: historical
source aliases contain duplicate and truncated names whose outcome joins are
not auditable. The incumbent is a smoothed position appearance rate constructed
only from seasons strictly earlier than the target season.

Conditional production and participation are deliberately separate. M4A does
not multiply them into an unconditional projection.

## Cross-Validation Contract

The default validation window is 2017-2025, matching the existing S1 cutoff.
Every validation season is divided into five deterministic folds with random
seed 1234.

For each held fold:

1. `SciKitModel.time_series_cv` receives all pre-2017 rows and the other four
   folds.
2. For a target season `t`, both validation and held-fold predictions are fit
   only on rows with `season < t`.
3. Hyperparameters are selected from rolling predictions for the other four
   folds.
4. The held fold is predicted with those hyperparameters and never participates
   in its selection.

This produces one OOF prediction per eligible 2017-2025 player-season while
retaining a large donor population for later template work. A whole-season
rolling-origin replay remains a later temporal robustness check; it is not the
primary donor-generation scheme.

`model_fold_assignments.training_through_season` must always equal
`season - 1`. Split-control column `game_date` remains available to
`SciKitModel`, but the first pipeline step removes it before model fitting.

## Compact Model Surface

M4A intentionally limits the model families:

- raw team-game consensus and the active-game-when-available expert hybrid;
- position-aware consensus recalibration with Ridge;
- compact and full-manifest residual Ridge;
- full-manifest residual Ridge with KBest, PCA, or feature agglomeration;
- shallow residual LightGBM;
- full-manifest direct Ridge and shallow direct LightGBM;
- prior-position participation rate;
- compact and full-manifest logistic participation models;
- full-manifest logistic models with KBest, PCA, or feature agglomeration; and
- shallow participation LightGBM.

ElasticNet, random forests, unconstrained depth, feature unions, chained
selectors, and broad default hyperparameter ranges are excluded from the first
comparison. Ridge represents regularized linear behavior; shallow LightGBM is
the nonlinear challenger.

KBest, PCA, and agglomeration are mutually exclusive pipeline challengers.
They are never stacked together. Imputation, scaling, selection, and
transformation occur inside each rolling fit.

The default search has four trials per fold over deliberately narrow ranges:

- Ridge `alpha`: 1, 10, or 100;
- logistic `C`: 0.1, 1, or 10;
- KBest: small fixed candidate counts;
- PCA/agglomeration: small fixed component counts; and
- LightGBM: 100-200 trees, depth 3-4, 7-15 leaves, and explicit
  regularization/subsampling.

## Feature Governance

Full models may use only the reviewed Milestone 3 residual or participation
manifest. Compact models are named subsets of those manifests. Four position
indicators are modeling controls and are recorded in each specification.
Position-specific consensus interactions are used only for the recalibration
model. Identity confirmation is audit metadata, not a participation feature,
because it also governs whether a historical zero label is trustworthy.

Position, experience/history depth, provider era, and provider depth are
evaluation slices. They do not create separate model families in M4A. This
avoids multiplying small samples before OOF evidence shows a need for distinct
models.

The completed projection-consensus ladder retains a projection-only
consensus-core LightGBM as a separate sparse-history finalist. It is not a new
feature manifest and does not replace the full model. Any future combination
must route between the two component forecasts using only prior-origin OOF
evidence by position and history depth. The current research candidate favors
projection-only estimates for limited-history QB/TE/WR groups and the full
model for limited-history RBs and all veterans, but remains unpromoted because
season-bootstrap intervals cross zero. The provider median remains the
consensus input; causal provider weights are diagnostic only.

A follow-up position-specific replay does not support four independent
QB/RB/WR/TE models. Separate projection-core and full fits worsen pooled RMSE
by 0.0425 and 0.0160, respectively. A three-role QB/RB/WR+TE full model ties
the pooled full model overall and remains an unpromoted temporal-robustness
finalist because its favorable 2023-2025 point estimate covers only three
seasons. Pooled models remain the primary contract unless the role-group
candidate clears the final whole-season and calibration guardrails.

A QB-versus-all-skill split also ties pooled full and is weaker in the recent
window. Position-specific additions of experience-relative projection,
teammate ADP, role opportunity share, and richer room clarity produce no
family that survives correction across 16 prespecified tests. These fields
remain research/template candidates, not conditional-PPG manifest additions.
In particular, young-player slice patterns must not be promoted from the same
OOF evidence used to discover them.

The default 18-model M4A surface remains unchanged after a research-only KNN
and random-forest check. Scaled KNN is rejected for conditional PPG. Full
random forest ties deterministic full LightGBM, and their prespecified 50/50
average is an unpromoted finalist with a 0.0086 RMSE point gain and a season
interval narrowly crossing zero. Final fitting must preserve RF and LightGBM
component forecasts so whole-season calibration and history routing can test
the blend without reconstructing or independently resampling another residual.

A subsequent current-lineage blend study refits governed full Lasso on the
same 3,696 OOF rows. Lasso is weaker standalone but adds complementary error:
the fixed full Lasso/RF/LightGBM equal-third average scores 3.1000 versus
3.1143 for the RF/LightGBM average, improves all nine validation seasons, and
has a season interval fully below zero. A strictly prior-season-weighted
Lasso/tree-average blend scores 3.1027 and confirms the direction. These are
finalists, not additions to the default M4A search surface. Final fitting must
preserve Lasso, RF, and LightGBM component predictions, blend provenance, and
any origin-specific causal weights for whole-season calibration and joint
template evaluation.

A projection-anchored history-gap follow-up addresses the linear imputation
concern without altering the 31-feature incumbent. Missing prior-year,
three-year, and career PPG becomes a zero adjustment to the player's current
expert baseline, with explicit availability and opportunity-game reliability.
The representation improves rookie/no-career slices directionally, but direct
Lasso is 0.0018-0.0027 RMSE worse overall. Its equal-third blend improves only
0.0026-0.0029, reverses in the 2023-2025 mean, and has a season interval
crossing zero. Keep the gap family for the final sparse-history routing and
template replay; do not globally replace the incumbent from this OOF evidence.

A preseason projection-trajectory follow-up compares the current consensus
team-game PPG with the same player's exact prior-year projection and a
recency-weighted prior-three-year projection baseline. Missing projection
history is a neutral zero change with explicit availability/count, so rookies
are retained without fabricated history. Exact one-year change alone improves
the equal-third Lasso/RF/LightGBM blend by only 0.0016 RMSE; the three-year
context improves it by 0.0041, and the combined five-field family improves it
by 0.0051 with a slight favorable 2023-2025 mean. Keep the full trajectory
family as an unpromoted whole-season/template finalist.

The same study replaces raw ADP with `log1p(ADP)` rather than adding both.
Logged ADP improves Lasso by 0.0319 RMSE with eight of nine season wins, but is
neutral for random forest and slightly worse for LightGBM. Use the transform
only as a linear-model candidate. Although the full trajectory-plus-log
equal-third blend has the best pooled score at 3.0930, it is 0.0025 worse in
the 2023-2025 mean and weakens missing-ADP and selected sparse-history slices.
It is not promoted from the same OOF evidence.

A team-environment follow-up tests 11 preseason-only fields on top of the
trajectory reference: QB1 passing/rushing yards and TDs, QB1 rushing
fantasy-point share, capped core-skill and self-excluded supporting-cast
strength, and non-duplicated team rushing/offensive-TD context. The complete
family is flat globally. QB1 rushing share is the best compact addition,
improving the equal-third blend from 3.0949 to 3.0928 and the 2023-2025 mean
by 0.0089, but it wins five of nine seasons and its corrected comparison is
not significant.

The effect is position-specific: QB rushing share improves WR/TE RMSE by
0.0055/0.0068, is neutral for RB, and worsens QB. A same-evidence WR/TE-only
route scores 3.0916, wins seven seasons, and has interval
`[-0.0070, +0.0002]`. Carry that route into prespecified whole-season and
joint-template validation, particularly for mobile-QB pass-catcher downside;
do not deploy it as a global feature or RB adjustment.

## Published Tables

| Table | Grain and purpose |
|---|---|
| `model_runs` | One row per M4A execution and its feature-run lineage |
| `model_fold_assignments` | One row per target/player-season holdout assignment |
| `model_specifications` | Exact model, feature list, pipeline, and search space |
| `model_hyperparameter_results` | Trial score and selected flag by model/fold |
| `model_oof_predictions` | Template-ready OOF actual, baseline, residual, and final prediction |
| `model_score_summary` | Pooled and season-mean model metrics |
| `model_slice_summary` | Position, season, history, provider-depth, and provider-era metrics |

The active OOF and summary tables are replaced atomically. `model_runs` and
`build_runs` retain history and mark older complete M4A runs superseded.

## Metrics and Promotion Boundary

Conditional PPG reports RMSE, MAE, bias, and Spearman correlation.
Participation reports Brier score, log loss, calibration bias, and ROC AUC.
For error metrics, a negative `delta` means improvement over the incumbent.
For Spearman and ROC AUC, a positive `delta` means improvement.

No challenger is promoted merely because it has the best pooled score. A later
decision must consider:

- season-mean and recent-provider-era performance;
- rookie, second-year, and other no-history performance;
- position stability;
- calibration rather than classification accuracy alone; and
- compatibility with joint residual/weekly-path templates.

M4A OOF residuals and participation probabilities are template-ready evidence,
not yet a new template-generation rule.

The companion research study runs fold-identical family dropouts for the full
Ridge residual and logistic participation models. Each dropout removes one
cataloged family while retaining the same folds, target rows, narrow search,
and position controls. Season-paired bootstrap intervals govern pruning;
pooled correlation alone does not.

## Required Invariants

1. Every OOF row belongs to 2017-2025 by default.
2. Every target/player-season appears in exactly one fold.
3. Every validation season is represented in all five folds.
4. `training_through_season < season` for every OOF row.
5. Conditional models share the same consensus-available population.
6. Participation retains confirmed completed zero-game candidates and excludes
   unresolved identity joins.
7. Participation predictions are bounded probabilities.
8. Split-control columns cannot enter fitted pipelines.
9. Compact feature sets remain subsets of reviewed manifests.
10. Model outputs retain exact Milestone 3 feature-run lineage.
11. A lock is valid only for the rebuilt identity, source-season, league
    scoring hash, and `core_offensive_season_components_v1` provider estimand
    recorded by its Milestone 3 run.
12. Beta lock evidence includes QB provider completeness by season and proves
    that no complete beta QB row has null sacks.
13. Any foundation identity, effective-season, source-row-quarantine, or
    provider-scoring correction invalidates downstream performance claims
    until locked, calibration, template, and next-year replays are rerun.
14. A production league must have an explicit scoring contract. Unknown or
    misspelled league names fail rather than inheriting NFFC defaults.

## Approved Production Cycles

`Scripts/V2/production_cycle.py` is the sole registry for annual production
contracts. A cycle binds:

- current and following seasons plus exact shadow table names;
- exact accepted model versions for DK, NFFC, and beta;
- the locked-current, following-season, and template-audit runners;
- source, model-input, and league/position production floors; and
- league weekly horizons and minimum template seasons.

`Scripts/V2/refresh_production.py --year 2026` selects the only approved cycle.
The runner stores a deterministic receipt and hash in its stage manifest and
rechecks both on resume and promotion. A requested 2027 current-season run
fails closed even though the 2026 cycle produces a following-season 2027
forecast. Approving 2027 requires new annual validation runners, exact lock
versions, current inputs, population/exclusion reviews, and a new registry
entry; changing an environment variable cannot reuse 2026 evidence.

## Locked 2026 Production Candidate

`v2_conditional_ppg_2026_candidate_v1` freezes the reviewed DK shadow
specification:

- conditional PPG: fixed equal-third pooled Lasso, random forest, and
  deterministic shallow LightGBM;
- inputs: the 31-field residual incumbent, five preseason projection-trajectory
  fields, and four position controls;
- participation: pooled deterministic shallow LightGBM on the 19 reviewed
  participation fields and four position controls; and
- validation: complete 2017-2025 forecast seasons with every fit,
  hyperparameter decision, route, calibration estimate, and interval using
  only earlier seasons.

The latest source-quarantine-corrected, one-vote-NFFC replay scores 3.1076 RMSE
versus 3.1951 for expert recalibration and wins all nine seasons. Do not add a
point-calibration overlay: every tested strictly-prior policy worsens pooled
RMSE. The projection-anchored gap route remains a locked secondary component
and is not combined with other routes or substituted for the published primary
center. Exact secondary-route metrics from the pre-quarantine lineage are
superseded; no secondary promotion is made from this data correction.

Annual tuning state is stored separately in
`Data/Databases/V2_Parameter_Cache.sqlite3` at
`(season, league, runner, model_name)`. The cache holds the complete selected
parameter row for every required forecast origin plus a SHA-256 over the exact
selection inputs and specification. Any training-row, target, feature, grid,
origin, embargo, metric, or seed change invalidates that model entry and reruns
its grid. A valid entry skips only hyperparameter optimization; every selected
model is still refit and predicts again. Random forests use four workers with
their locked random seed and produce the same shadows as the prior one-worker
execution. `locked_parameter_cache_receipts` and
`next_year_parameter_cache_receipts` record hit/miss status and fingerprints in
each league database, while the central cache is a governed refresh/promotion
artifact rather than an app input.

The locked template handoff supplies a strict-OOS V2 historical point center,
not a second residual distribution. Production retains it as
`v2_historical_pred_fp_per_game` for diagnostics. For DK and beta, it does not
replace the previously validated historical donor center: a 2017-2025 rolling
replay found that V2 recentering worsened PPG CRPS by 0.0057 in DK and 0.0051
in beta, with both player-cluster 95% intervals entirely above zero. DK/beta
therefore keep `v2_recenter_promoted = 0`. Rows with a real validation-ensemble
center record `historical_center_policy = legacy_validated_oos`; rows without
one use and disclose `preseason_projection_fallback`.

NFFC has a separate scoring-context and center contract because the legacy
`Model_Inputs` projection context is DK-scored. Its historical and current
scoring-sensitive matcher fields are authoritative from the NFFC-scored V2
`player_season_features` preseason consensus; the old context is audit-only.
Historical NFFC donors use the expert team-game PPG with
`historical_center_policy = nffc_scored_expert_consensus`. A strict 540-target
2023-2025 replay retained that center: locked-minus-expert PPG CRPS was
`+0.002901`, the locked center lost all three seasons, and its player-cluster
95% interval was `[-0.004914, +0.010748]`. The locked arm passed six of ten
gates but failed pooled PPG improvement, a nonpositive interval upper bound,
and two-season wins, so its OOF center remains diagnostic. This NFFC decision
does not alter DK/beta behavior.

The FFToday quarantine can make a beta 2018 QB V2 diagnostic center unavailable
when the leaked future-vintage row was the only apparent sack donor. Preserve
that absence explicitly. Do not fill it from DK, from the quarantined vintage,
or from a zero-sack assumption; the complete active historical center remains
the validated legacy OOS value. The corrected beta template population has
2,657/2,696 V2 diagnostic centers; the 39 unavailable rows are all governed
2018 QB fallbacks with an explicit reason and active quarantine proof.

`v2_conditional_ppg_2026_candidate_beta_v1` is the separately fitted
beta-scored counterpart. It uses the same primary feature hashes and
equal-third architecture, scores 2.8841 RMSE versus 2.9600 for beta expert
recalibration, and wins all nine seasons. The beta no-history route remains a
secondary diagnostic rather than changing the published primary during a
data-lineage correction. Exact pre-quarantine secondary metrics are
superseded. Every tested prior-only calibration policy worsens pooled RMSE.

`v2_conditional_ppg_2026_candidate_nffc_v1` is the separately fitted
NFFC-scored candidate registered for the 2026 production cycle. Its scoring
hash, feature lineage, current-model comparisons, and following-season
comparisons must pass the same fail-closed acceptance checks before release.
NFFC eligibility is the core population unioned with the first 363 canonical
NFFC ADP rows after filtering to QB/RB/WR/TE. The three extra 2026 candidates
replace reviewed protected-market exclusions for Ricky Pearsall, Khalil
Herbert, and Haynes King, so the market surface still covers all 360 draft
slots. Canonical `TK` and `TDSP` market
units remain audit data; they are not model targets. This offense-only lock
does not assert K/DST or complete contest support.

The staged NFFC weekly handoff contains 1,509 2021-2025 templates with 17
populated weeks and a 385-player current map. Neither these artifacts nor the
NFFC model database have been promoted live.

Raw provider rank numbers remain audit-only because provider list depths and
objectives are not commensurate. A strict-prior challenger normalizes each
expert source within season and position, takes the cross-source median, and
lets ETR contribute one rank vote. Controlled attribution uses a full-column
random forest on both sides because adding a feature changes the incumbent
forest's locked 50% feature subsample. The normalized rank level improves its
controlled blend by 0.0022 RMSE in DK and 0.0019 in beta, with seven of nine
season wins in each league. DK's season and player-cluster intervals exclude
zero; beta's season interval excludes zero and its player interval ends
essentially at zero. The production-surface sensitivity remains smaller:
0.0017 DK and 0.0014 beta, with both intervals crossing zero. The
expert-minus-projection rank gap is neutral and unstable. Neither feature is
promoted because the gain is tiny, the controlled forest is not the locked
production architecture, and rank-provider coverage changes sharply in
2024-2025. The normalized level remains a governed challenger for
future-season confirmation.

The DK and beta locks are active for the expanded 2026 production population:
351 DK rows (56 QB, 101 RB, 143 WR, and 51 TE) and 328 beta rows (50 QB,
95 RB, 133 WR, and 50 TE). DK is the 326-player core plus the top-280 DK
market union after eight governed market-only/no-center exclusions: Tyreek
Hill, Joe Mixon,
DeAndre Hopkins, Nick Chubb, Austin Ekeler, Kareem Hunt, Brandin Cooks, and
Taysom Hill. Beta is the core plus the top-180 ETR overall-rank union and all
keepers. `production_handoff.py` joins each league lock by canonical
`player_key`, fails closed on incomplete coverage or scoring-hash mismatches,
and publishes `Final_Predictions_Resid`,
`V2_Production_Projection_Handoff`, and
`V2_Production_Projection_Audit`. The 1,490-row
`V2_Production_Eligibility_Audit` records the complete reviewed inclusion and
exclusion population. The original rows are retained once in
`V2_Projection_Legacy_Backup`.

The NFFC population and its audit rows become production data only through a
complete approved-cycle stage and explicit promotion. They must not be
described as part of the earlier live DK/beta release merely because canonical
NFFC ADP already exists.

Beginning with exclusion policy
`v2_market_only_incomplete_buffer_exclusion_v3`, a refresh does not fail for
an incomplete market-only tail row merely because it appears in the requested
ADP surface. Core `ProjOnly` players and league keepers always fail closed.
New market-only gaps also fail closed through the first five-sixths of the
expected draft (`200` DK picks, `300` NFFC picks, and `150` beta/ETR ranks)
unless separately accepted in the annual explicit-exclusion review. Beyond that
protected depth, an incomplete current or next-year handoff is omitted rather
than filled from legacy projections, is retained as a governed exclusion in
`V2_Production_Eligibility_Audit`, and is allowed only when the remaining
complete population still covers the full `240`/`360`/`180`-player draft.
`Avg_ADPs` continues to retain the full canonical market surface.
The audit records the required depth, protected depth, effective market draft
position, and whether the exclusion was automatic.

The canonical current market snapshot contains 416 live DK rows, 497 NFFC
rows (431 offense plus 33 `TK` and 33 `TDSP` draft units), and 243 ETR rows.
The latest local NFFC and ETR exports are dated 2026-07-27. ETR's exact source
ranks remain the beta population ordering, while NFFC contributes one
composite market vote. DK resolves 343 production rows to exact canonical ADP
and uses eight governed fallbacks; beta resolves 238 exactly and uses 90
governed fallbacks. Neither league permits a generic default or review route.

The corrected live DK/beta fit-through-2025 shadows contain 745 candidates in
each league, 715 DK point centers, 673 beta point centers, and 745
participation probabilities. Production coverage is 100% for the 351/328 app
populations. Tetairoa McMillan and Amon-Ra St. Brown retain their governed
canonical identities.
`production_handoff.py` is refresh-safe: a republish replaces its governed key
and metadata columns from the canonical weekly map rather than creating merge
suffixes. A second identical publish must produce zero point deltas.

The production current residual quantiles are exactly zero,
`independent_current_residual_draw_allowed = 0`, and
`current_uncertainty_source = joint_weekly_template_only`. A consumer must add
the sampled donor's centered `active_ppg_resid` directly to the V2 point center
and use that same donor's weekly path. It must not scale that residual to the
zeroed legacy model spread. DK, NFFC, and beta retain separate V2 databases and
scoring hashes, and every join is by league plus canonical key rather than
display name.

For current and following-season production, the locked V2 shadow is the point
and appearance authority. Legacy current/next fields are audit-only and must
not silently fill a missing locked center. This current/next rule is separate
from the historical donor-center contract: validated legacy OOS for DK/beta
and scoring-matched preseason expert consensus for NFFC.
The second production handoff reproduced unchanged hashes for all eight
governed handoff/audit tables. The canonical release passed 187
main-repository, 69 strict-release, 49 Auction, and 16 Snake tests; Snake
`AppTest` reported zero exceptions. Durable evidence and recoverable
pre-promotion copies live under
`research/studies/2026-07-30_canonical_adp_handoff/`.

The original replay lives under
`research/studies/2026-07-29_v2_locked_final_validation/`. Its data-lineage
claims are superseded by the corrected replay summarized under
`research/studies/2026-07-29_v2_weekly_fftoday_correction/`.

## Following-Season Production Contract

`v2_next_year_expert_residual_v1` forecasts the season after the preseason
origin using two separate quantities:

1. conditional PPG if the player appears in the following season; and
2. the probability of any following-season appearance.

The conditional target is:

```text
conditional PPG in t+1 - expert_ppg_team_game_median in t
```

No `t+1` preseason projection, ADP, roster, team, or realized statistic enters
the origin-`t` features. A forecast for origin `t` trains only on origins
through `t-2`, whose following-season outcomes end in `t-1`. Hyperparameter
selection uses the same embargo. Confirmed identities with no following-season
game evidence receive an appearance label of zero; conditional PPG stays null.
Unresolved identities are not converted to negatives.

The primary conditional forecast is the fixed equal-third average of Lasso,
random forest, and deterministic shallow LightGBM residuals. The primary
appearance forecast is deterministic shallow LightGBM. Whole-origin validation
uses 2017-2024 because 2025 is the latest completed target season.

| League | Conditional blend RMSE | Expert carry RMSE | Origin wins |
|---|---:|---:|---:|
| DK | 3.9137 | 5.2351 | 8/8 |
| beta | 3.5538 | 4.3653 | 8/8 |

These conditional metrics come from the source-quarantine-corrected lineage.
Pre-quarantine appearance-Brier figures are not carried forward as current
evidence; the separate appearance architecture and promotion status are
unchanged.

The output databases publish:

| Table | Grain and purpose |
|---|---|
| `next_year_targets` | Origin player-season with auditable `t+1` labels |
| `next_year_target_audit` | Origin/position label and identity counts |
| `next_year_selected_hyperparameters` | Strict-prior selection provenance |
| `next_year_parameter_cache_receipts` | Season/league/model cache fingerprint and hit/miss provenance |
| `next_year_whole_season_predictions` | Long-form validation predictions |
| `next_year_model_scores` | Pooled, origin, position, and history slices |
| `next_year_model_comparisons` | Origin-paired causal comparisons |
| `next_year_template_handoff` | Historical and shadow matching context by canonical key |
| `next_year_2027_shadow_predictions` | Current 2026-origin forecast for 2027 |

The complete corrected 2027 shadow contains 745 canonical candidates per
league, 715 DK and 673 beta conditional PPG centers, and 745 appearance
probabilities per league. The production handoff publishes the intersection
with the current app population: 351 DK and 328 beta rows. DK and beta use
independent scoring-specific features and fits. The approved NFFC refresh runs
the same embargoed following-season methodology under NFFC scoring, but its
output is not covered by the historical DK/beta metrics in the table above and
must clear its own acceptance gate.

The rolling weekly-template replay does not promote the two next-year fields
as donor-matching features. Residual rank improves weekly-PPG CRPS by only
about 0.001-0.002, while season-contribution CRPS is usually worse. They are
published for explicit keeper/next-horizon valuation, not template matching.

Consumer guardrail: a conditional PPG center is a survivor-only forecast. A
keeper or optimizer must retain the separate appearance probability and define
an explicit no-appearance/availability mixture. Production draws the
conditional residual distribution first, then a Bernoulli appearance outcome;
no appearance sets future market value to zero. It must not consume the
conditional center alone or add another independent current-season residual
draw to the joint weekly donor path.

The reproducible model and template replays live under
`research/studies/2026-07-29_v2_next_year_residual/`.
