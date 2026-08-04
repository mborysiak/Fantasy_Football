# PFF TE tackle-breaking confirmation

This study confirms the prior broad PFF screen in two independent tracks:

1. a selected-grid replay of the full locked Lasso, random-forest, and
   LightGBM conditional-PPG blend; and
2. a TE-only weekly-template distance replay.

Production is unchanged unless a later explicit promotion is approved.

## Frozen feature construction

- Season `t` uses only PFF receiving data from `t-1`.
- PFF rows join to `player_key` through numeric `player_identity.pff_id`.
- Only prior rows classified by PFF as TE are eligible.
- Avoided tackles per reception uses `receptions / (receptions + 20)`
  reliability. YAC per route uses `routes / (routes + 100)` reliability.
- Projection features are winsorized and shrunk to their source-season TE
  weighted means. Missing histories and non-TE rows are neutral zero after
  centering.
- Template features are source-season TE percentile ranks shrunk toward 0.5.
  Missing histories are 0.5, so they add no systematic high/low style label.

## Projection arms

- `production`: persisted locked blend.
- `te_pff_opportunity_control`: prior PFF availability, log routes, and log
  receptions.
- `te_pff_mtf`: opportunity control plus shrunk avoided tackles/reception.
- `te_pff_yac`: opportunity control plus shrunk YAC/route sensitivity.

Every challenger replays all three locked component models with the exact
per-origin hyperparameters already selected for production. No challenger is
retuned on the confirmation outcomes.

## Template arms

- `production`
- `te_pff_mtf_w025` (primary)
- `te_pff_mtf_w050` (weight sensitivity)
- `te_pff_yac_w025` (feature sensitivity)

All production weights, donor pools, recency, kernels, target cohorts, and
weekly outcome transport are otherwise unchanged. Template validation uses the
role-tiered policy plus the causal q90/q95 rare-upside diagnostics.

## Result

- Projection: the TE-routed avoided-tackles-per-reception arm improved TE RMSE
  and q90 Brier versus both production and the prior-PFF-opportunity control in
  DK and beta, in both 2017-2022 and 2023-2025. Treat it as a projection-only
  implementation candidate; do not interpret the replay as a new-origin
  confirmation because the broad screen used the same historical origins.
- Templates: reject the primary avoided-tackles arm. The YAC/route sensitivity
  cleared the player-level PPG/upside screen, but its roster transport worsened
  DK score CRPS and championship Brier/log loss in both periods and was mixed
  in beta.
- No production model, feature manifest, template weight, database, or app
  objective changed.

The compact decision tables and bootstrap diagnostics are in `results/`, with
the main readout in `results/findings.md`.

## Reproduction

Run the projection and template scripts separately for `dk` and `beta`, then
run `summarize_template_validation.py`. If the YAC arm remains the mechanical
template finalist, run `run_roster_validation.py` for each league. Finally run
`summarize_study.py` and `verify_results.py`. The runners expose `--help` for
their exact league and output-directory arguments. Large row-level projection
and template intermediates were compacted after the durable summaries were
verified; rerunning the upstream scripts recreates them before re-summarizing.
