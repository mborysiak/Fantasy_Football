# nflfastR RB role-opportunity template replay

## Question

Can causal prior-season high-value and passing-down RB opportunity distinguish
weekly-outcome archetypes beyond the production preseason projection, market,
experience, and backfield-room features?

## Frozen design

- Source: weekly `RB_Stats` in
  `Daily_Fantasy_Data/Databases/FastR_Beta.sqlite3`.
- A target/template season `t` receives only realized statistics from `t-1`.
- Every share is calculated against the RB room over weeks in which the player
  registered at least one carry or target. This estimates role conditional on
  opportunity without adding player-specific missed-game history.
- Raw shares are ranked within source season and position, then shrunk toward
  neutral (`0.5`) based on overall and situation-specific opportunity.
- Rookies and missing histories are neutral, not zero.
- nflfastR's stored situational fields combine third and fourth down. Outputs
  therefore use the exact `third_fourth` label rather than calling them pure
  third-down measures.
- Touchdowns, conversions, EPA, and yards per touch are excluded. The test is
  about opportunity role rather than realized success or efficiency.

### Profiles

| Profile | Definition | Reliability |
|---|---|---|
| Red-zone carry room share | Player red-zone carries / RB-room red-zone carries in opportunity weeks | overall opportunity reliability x room RZ sample reliability |
| Goal-line carry room share | Player goal-line carries / RB-room goal-line carries in opportunity weeks | overall opportunity reliability x room goal-line sample reliability |
| Third/fourth-down target room share | Player third/fourth-down targets / RB-room third/fourth-down targets in opportunity weeks | overall opportunity reliability x room situational-target sample reliability |

Overall opportunity reliability is `opportunities / (opportunities + 40)`.
Situation reliabilities use room denominators of 10 red-zone carries, 5
goal-line carries, and 8 third/fourth-down targets.

### Matching arms

| Method | Added RB weights |
|---|---|
| `production` | none |
| `rb_scoring_role_w050` | red-zone carry share 0.25; goal-line carry share 0.25 |
| `rb_passing_down_w050` | third/fourth-down target share 0.50 |
| `rb_dual_role_w050` | goal-line carry share 0.25; third/fourth-down target share 0.25 |
| `rb_dual_role_w100` | goal-line carry share 0.50; third/fourth-down target share 0.50 |

## Validation

- Corrected current-lineage DK and beta weekly data.
- Expanded rolling cohorts: QB 48, RB 90, WR 120, TE 48 per season.
- Frozen role-tiered policy from
  `research/studies/2026-07-31_template_role_tiered_validation/README.md`.
- Development seasons 2017-2022; untouched temporal check 2023-2025.
- Any weekly finalist must still pass replacement-aware roster CRPS before
  production promotion.

## Reproduction

```powershell
.venv_ff_312\Scripts\python.exe research/studies/2026-07-31_template_fastr_rb_roles/run_validation.py --league dk
.venv_ff_312\Scripts\python.exe research/studies/2026-07-31_template_fastr_rb_roles/run_validation.py --league beta
python research/studies/2026-07-31_template_fastr_rb_roles/run_role_tier_rescore.py
```

Production remains unchanged unless the full validation sequence passes.

## Results

The replay produced 2,647 rolling targets and 13,235 method-target rows per
league. Prior-role coverage was 81.0% across the expanded RB cohort and 86.1%
among main-core RBs. Missing profiles were confirmed neutral, non-RB candidate
rows were identical to production, and the production baseline reproduced the
prior corrected replay exactly.

No candidate improved development core PPG CRPS in both leagues, so none
passed the frozen weekly screen and roster replay was skipped.

- The stronger dual-role bundle was effectively flat in cross-league mean
  development core PPG (`-0.0007%`), but the league signs disagreed: DK worsened
  `0.0147%` and beta improved `0.0161%`. Both season- and player-cluster
  intervals crossed zero.
- The 0.50 dual-role arm likewise improved beta and worsened DK. Contribution
  improved in both leagues, but the core PPG gate correctly prevented it from
  advancing.
- Scoring role alone was the weakest temporal arm. Its beta temporal core PPG
  deterioration had a season-cluster interval entirely above zero.
- Passing-down share was the most stable descriptive signal. It slightly
  worsened DK and improved beta in development, then improved temporal core PPG
  in both leagues. Among depth RBs, its equal-weight
  PPG/contribution/played-games composite improved in both development and
  temporal periods for both leagues. This depth-only pattern was observed after
  the global arms were scored and is therefore exploratory, not promotion
  evidence.

**Decision: retain production matching.** If RB role receives one follow-up,
the justified hypothesis is a prespecified depth-only or smoothly rank-tapered
third/fourth-down target-share feature. Goal-line opportunity should not be
carried forward from this result, and no current evidence supports a global RB
role weight.
