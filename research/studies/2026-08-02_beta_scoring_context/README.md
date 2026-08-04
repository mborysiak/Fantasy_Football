# Beta scoring-matched weekly-template context

## Question

Does replacing the hybrid DK/beta weekly-template matching context with the
beta-scored V2 preseason context preserve or improve strict rolling
calibration?

The production beta path already uses beta-scored current V2 point forecasts
and beta-scored realized weekly donor outcomes. The defect under test is the
similarity context between those endpoints: historical component/room fields
come from DK-scored `Model_Inputs`, and historical rows without a validation
center also fall back to DK preseason PPG.

## Frozen arms

1. `production_hybrid`: current DK `Model_Inputs` context and current center
   policy.
2. `beta_context_only`: beta V2 component, uncertainty, room, and team-QB
   context while retaining the current donor centers, including the DK
   preseason fallback. This arm isolates the context change.
3. `beta_scored_full`: the same beta V2 context plus beta expert team-game PPG
   only where the legacy validation center is unavailable. Validated beta OOS
   centers remain unchanged.

All arms use identical beta weekly outcomes, target forecasts, target cohorts,
match weights, recency, seeds, donor eligibility, and probability rules.
Donors are always strictly earlier than the target season.

Beta sack scoring can make a fringe QB's passing component and total negative.
The V2 feature mart therefore preserves signed component shares whenever the
component sum is nonzero. This is a source-contract correction, not a local
template fill.

The governed beta 2018 QB source quarantine leaves that season-position
without a complete sack-aware preseason consensus. Those rows remain in the
template table for audit but are ineligible as donors, and 2018 QB targets are
excluded from every validation arm. The challenger does not substitute DK
context or invent a sack imputation for that unavailable stratum.

## Player-level promotion gates

The full arm must pass all established role-tier safety limits:

- development main-core PPG and contribution CRPS no worse than 0.25%;
- absolute played-games bias degradation no more than 0.15 games;
- absolute extended-absence calibration degradation no more than 0.01;
- P10-P90 PPG coverage decline no more than 0.01;
- no development core position PPG CRPS deterioration above 1%;
- 2023-2025 main-core PPG deterioration no more than 0.5%;
- depth equal-third PPG/contribution/played composite no worse than 0.5%,
  with no component worse than 1%; and
- strict/main/broad core sensitivity PPG and contribution each no worse than
  0.5% in development and 2023-2025.

Passing these gates advances `beta_scored_full` to the existing replacement-
aware roster replay. Roster-score CRPS must then remain within 0.5%. Failure of
the full arm is not permission to promote the context-only arm.

## Run

Use a completed staged refresh through `locked_beta`, so the corrected V2
feature run and its locked handoff have exact lineage:

```powershell
.venv_ff_312\Scripts\python.exe `
  research\studies\2026-08-02_beta_scoring_context\run_validation.py `
  --v2-db <stage>\databases\Projection_V2_beta.sqlite3 `
  --simulation-db <stage>\databases\Simulation.sqlite3
```

The runner writes only to this study's `results/` directory. It does not alter
production databases or app artifacts.

## Results and decision

The strict player replay covered 2,608 held-out player-seasons. The original
`beta_scored_full` arm passed every player gate, including development core
PPG/contribution deltas of `+0.0105%`/`+0.0379%`, temporal core PPG of
`+0.2434%`, and worst tier PPG/contribution deltas of `+0.4462%`/`+0.4144%`.
It did not pass the paired roster replay: development roster-score CRPS
worsened `+0.9061%` against the `0.5%` limit, while 2023-2025 worsened
`+0.3790%` and passed.

The roster investigation kept the target cohort, outcomes, weights, seeds,
and strictly earlier donor rule paired. Because the governed beta 2018 QB
quarantine leaves no complete sack-aware QB context, the roster replay omits
2018 from every arm; otherwise it could not construct a legal roster without
restoring DK QB units. It scores 1,152 paired rosters from the remaining eight
origins with 384 common scenarios per roster.

Targeted source/representation checks found:

| Challenger | Development roster CRPS delta | 2023-2025 delta | Decision |
| --- | ---: | ---: | --- |
| beta context with legacy fallback PPG anchor | `-0.4918%` | `+0.2647%` | passes, but retains the mixed old fallback anchor |
| beta expert fallback center plus full beta context | `+0.9061%` | `+0.3790%` | fails development |
| beta-only V2-era donors | `+2.8658%` | `+0.5113%` | fails both |
| beta match PPG decoupled from the donor residual center | `+1.0173%` | `+0.4062%` | fails development |
| decoupled beta PPG with beta rank and market gap removed | `+0.3708%` | `-0.2572%` | passes roster |
| decoupled beta PPG plus a small `0.50` rank weight | `+0.8456%` | `+0.3526%` | fails development |

The PPG-only arm's separate player confirmation failed one safety gate:
development depth-tier played-games CRPS worsened `+1.3506%` against the `1%`
component limit, although its depth PPG and contribution CRPS improved. Adding
the small rank term cleared every player gate but caused the roster failure in
the final row above. Thus no fully beta-scored arm clears both player and
roster protocols. The passing context-only arm is not promoted as a partial
fix because it retains the pre-2017 DK preseason PPG anchor.

The original statistical decision was no promotion because no fully
beta-scored arm cleared both protocols. That conclusion remains the correct
interpretation of the validation: this study does not show a predictive win.

On 2026-08-03, the user explicitly approved a data-correctness override. The
full beta context and `beta_scored_expert_fallback` were promoted because
retaining DK-derived matcher/fallback units in a beta-scored pipeline is a
known unit defect. The accepted tradeoff is the measured development roster
CRPS regression (`+0.9061%` versus the `0.5%` gate), with the smaller
2023-2025 regression (`+0.3790%`). Run `20260803T040708Z_2075ac47` rebuilt the
full pipeline, passed every release gate and both app smokes, and atomically
promoted the corrected databases. The 39 beta 2018 QB rows remain explicitly
unavailable and donor-ineligible; no DK or zero-sack fill was introduced.

Future work should treat the corrected beta unit boundary as the baseline and
seek new evidence or a prespecified matcher architecture. Do not restore the
hybrid context, relax the gates retrospectively, or retune weights by
validation era.
