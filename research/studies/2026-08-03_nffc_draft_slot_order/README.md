# NFFC Draft-Slot Order

## Question

For the current Snake app's NFFC offense-only Preview, how should a user rank
the 12 possible draft slots before the room assigns positions?

## Frozen design

- League: `nffc`
- Prediction year/dataset: `2026` / `final_ensemble`
- Teams/rounds: 12 / 20
- Draft order: the app's NFFC Third Round Reversal schedule
- Roster bounds: the app's default best-ball position ranges
- Policy: current Sequential Preview with stack preference enabled
- Ex-ante draft rooms per slot: 256
- Opponents before the user's first selection: exactly `slot - 1`, drawn from
  the same noisy-ADP room used for the rest of that draft
- User decisions: the Sequential policy's legal-roster, marginal best-ball,
  scarcity, urgency, survival, and stack-aware rollout rule at every turn
- Held-out audit bank: 512 seasons, disjoint from construction, pilot, and
  decision banks
- Seeds and score-bank columns: common across all 12 slots
- Execution: one fresh Python process per slot

Each room is a complete ex-ante draft. Players selected before the user's first
turn are removed before the policy chooses, and the policy adapts to actual
availability at all 20 user turns. Slot quality is the raw expected 17-week
best-ball score of those completed rosters on the held-out audit bank. Stack
utility influences the policy's choices but is not added to the primary
slot-quality outcome.

This uses the same rollout policy that the production D128 stage uses to
complete candidates. It does not incorrectly force a single first-pick player
to remain available in every slot, and it does not give the policy knowledge
of future opponent selections. Because an ex-ante room branches before the
user's first turn, it is a structural slot-value study rather than a literal
replay of the app's D128 recommendation screen at one already-observed state.

Uncertainty uses the app's conservative two-way approximation: variation in
draft-room means divided by the number of rooms plus variation in season-bank
means divided by the number of held-out seasons. Pairwise intervals use common
rooms and common season scenarios. A multivariate-normal draw from that paired
covariance estimates first-place probability and expected rank; these are
diagnostics, not historical guarantees.

## Run

From the modeling repository, using the Snake app's pinned Python 3.12
environment:

```powershell
..\Fantasy_Football_Snake\.venv_snake_312\Scripts\python.exe `
  research\studies\2026-08-03_nffc_draft_slot_order\run_study.py
```

Outputs are written under `results/`. Existing per-slot results make the run
resumable unless `--force` is supplied.

## Scope

This compares the app's governed offense-only 20-player NFFC adapter. It is
not a historical contest ROI study and does not represent the official
30-round K/DST roster or alternate NFFC formats. The result should be treated
as a model-based slot preference and monitored as ADP/projections change.
