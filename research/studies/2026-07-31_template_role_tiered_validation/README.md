# Weekly-template role-tiered validation

## Question

Would any previously tested weekly-template matcher change survive when the
selection policy emphasizes conditional PPG for secure fantasy roles, while
preserving contribution and aggregate missed-game risk rather than optimizing
the exact identity of injured players?

Production is unchanged unless a challenger passes the full sequence below.

## Frozen policy

The policy was frozen before Phase A outputs were generated.

### Role tiers

Preseason position rank is reconstructed within season and position by the
same ordering used to select validation targets: projected PPG descending,
ADP ascending, then player name.

| Tier | QB | RB | WR | TE |
|---|---:|---:|---:|---:|
| Strict core | 12 | 24 | 36 | 12 |
| Main core | 18 | 36 | 48 | 18 |
| Broad core | 24 | 48 | 60 | 24 |

`depth_main` is the portion of each saved validation cohort below the main
core threshold. Phase A cannot enforce team-QB1 status because the saved
prediction artifacts do not carry team. The fresh replay must add that field
and report the QB18 result both with and without the team-QB1 restriction.

### Core-player selection order

1. Active-PPG CRPS is the primary loss.
2. Managed-contribution CRPS breaks PPG near-ties. A near-tie is a candidate
   whose development-period PPG delta is within one season-cluster standard
   error of the best eligible candidate in the same experiment.
3. PPG bias, P10-P90 coverage, recent seasons, position slices, and league
   replication are guardrails.
4. Individual played-games CRPS is reported but is not a core-player gate.

Core availability is protected with aggregate quantities that the simulator
actually needs:

- absolute played-games bias may not deteriorate by more than 0.15 games;
- absolute extended-absence calibration error may not deteriorate by more
  than 0.01;
- PPG P10-P90 coverage may not decline by more than 0.01;
- contribution CRPS may not deteriorate by more than 0.25%; and
- no position's PPG CRPS may deteriorate by more than 1% in development; and
- temporal 2023-2025 core PPG CRPS may not deteriorate by more than 0.5%.

These are non-inferiority guardrails, not claims that player-specific injury
risk is predictable.

### Depth-player selection

Depth rows retain the incumbent equal-weight normalized composite of PPG,
managed contribution, and played-games CRPS. The composite may not deteriorate
by more than 0.5%, and no component may deteriorate by more than 1%.

### Time and replication

- Development: 2017-2022.
- Recent diagnostic: 2020-2025.
- Untouched temporal check: 2023-2025.
- Full diagnostic: 2017-2025.
- DK and beta are scored separately wherever both artifacts exist.
- Season-cluster bootstrap intervals are primary; player-cluster intervals are
  a sensitivity check for finalists.

### Promotion sequence

1. **Phase A:** rescore compatible saved strict-rolling predictions. Historical
   or superseded-lineage studies may nominate a fresh-replay challenger but
   cannot promote one.
2. **Phase B:** freeze a small challenger set and rerun the current corrected
   DK and beta pipeline with expanded cohorts (QB 48, RB 90, WR 120, TE 48).
3. **Phase C:** only Phase-B finalists receive roster-level replacement and
   lineup scoring. A finalist must be non-inferior on roster-score CRPS within
   0.5% while preserving the aggregate missed-game distribution.

If no challenger clears a phase, the next phase is skipped and production
remains unchanged.

## Phase A source policy

Current-production-baseline studies are treated as direct evidence. Earlier
context, pruning, and weight-sensitivity studies are clearly marked as
historical/superseded evidence because they predate the corrected explicit
league weekly foundation. The NFFC center replay is excluded: it compares
point-center sources rather than matcher criteria and its artifact contains
two records per target.

## Reproduction

```powershell
python research/studies/2026-07-31_template_role_tiered_validation/run_phase_a_rescore.py

.venv_ff_312\Scripts\python.exe research/studies/2026-07-31_template_role_tiered_validation/run_phase_b_replay.py --league dk
.venv_ff_312\Scripts\python.exe research/studies/2026-07-31_template_role_tiered_validation/run_phase_b_replay.py --league beta
python research/studies/2026-07-31_template_role_tiered_validation/run_phase_b_rescore.py

.venv_ff_312\Scripts\python.exe research/studies/2026-07-31_template_role_tiered_validation/run_phase_c_roster_replay.py --league dk
.venv_ff_312\Scripts\python.exe research/studies/2026-07-31_template_role_tiered_validation/run_phase_c_roster_replay.py --league beta
python research/studies/2026-07-31_template_role_tiered_validation/run_phase_c_decision.py
```

## Results

Phase A rescored 214,488 saved prediction rows from 13 compatible result sets.
It confirmed that the old equal-third PPG/contribution/played objective had
hidden several core-player PPG improvements, but only as nomination evidence.

Phase B reran six frozen methods on the corrected current lineage with 2,647
held-out targets per league. The 0.25x all-distance candidate was the only
one-SE finalist. Its core development PPG CRPS improved by 0.007901 DK and
0.005511 beta; both player-cluster intervals excluded zero. Temporal PPG was
slightly better DK and slightly worse beta, with both intervals crossing zero.
WR PPG/rate and TE YPR arms improved both-league point estimates but were not
within one standard error of the stronger flatter-distance candidate.

Phase C scored 1,296 paired 20-player rosters per league with 384 full weekly
scenarios per roster. The flatter candidate failed the frozen +0.5% roster
CRPS non-inferiority margin:

- DK development: +0.7096%; temporal: +0.5696%.
- beta development: +0.0269%; temporal: +0.1169%.

The DK season-cluster intervals were entirely adverse. Aggregate missed-week
bias remained within the 0.15-game-per-player margin in every cell and
missed-week CRPS was generally neutral to better. The rejection therefore
comes from replacement-aware roster scoring, not an individual injury-
prediction gate.

**Decision: retain production matching and adopt the role-tiered validation
contract for future matcher studies.** No database, app, or production matcher
was changed.
