# Upside objective audit

## Question

Can weekly-template matching and the downstream Auction/Snake decision layers
place more useful emphasis on rare league-winning outcomes without obtaining
that emphasis by inflating or degrading the calibrated player distribution?

## Frozen player outcome

The primary rare event is a player-season that satisfies both conditions:

1. realized active PPG is at least 5.0 points above the held-out preseason
   projection; and
2. realized managed contribution is at least the position-specific 90th
   percentile for comparable preseason-ranked players in the five seasons
   strictly prior to the prediction origin.

The 95th-percentile contribution threshold is a prespecified severity
sensitivity. Threshold cohorts use the expanded rolling target counts
`QB/RB/WR/TE = 48/90/120/48`. No target-season outcomes enter a threshold.

The continuous tail utility is zero unless the +5 PPG condition is met, then
equals contribution above the causal threshold. Its CRPS is a proper score for
the transformed outcome; it does not change or reweight the underlying player
forecast.

## Frozen evaluation hierarchy

1. Keep active-PPG and contribution CRPS as calibration/non-inferiority gates.
2. Evaluate rare-event Brier score and log loss for calibration.
3. Evaluate average precision, top-decile lift, and top-decile recall for
   league-winner identification. These are diagnostics, not substitutes for
   calibration.
4. Evaluate continuous tail-utility CRPS so the objective distinguishes a
   marginal hit from an extreme hit.
5. Require development and 2023-2025 temporal replication across both DK and
   beta before a matcher can advance.

No production matcher or app objective changes in this study.

## Roster objective prototype

The paired 12-team Phase-C replay is rerun on the same scenario bank. For each
room and scenario, the roster with the highest best-ball score wins. The
fraction of scenario wins is the forecast championship probability. It is
scored against the realized room winner using Brier score and room-level log
loss while ordinary roster-score CRPS remains a guardrail.

This is the intended objective shape for both applications:

- Auction: compare Buy versus Pass on paired championship probability in
  addition to paired expected season points.
- Snake: rank forced candidates on expected championship probability in
  addition to expected final roster score.

The historical Phase-C rooms are a validation proxy, not a claim that the
current simulations yet model every contest-level dependency.

## Commands

```powershell
$env:PYTHONPATH = (Resolve-Path Scripts)
python research/studies/2026-08-01_upside_objective_audit/run_player_tail_replay.py --league dk
python research/studies/2026-08-01_upside_objective_audit/run_player_tail_replay.py --league beta
python research/studies/2026-08-01_upside_objective_audit/run_roster_championship_replay.py --league dk
python research/studies/2026-08-01_upside_objective_audit/run_roster_championship_replay.py --league beta
python research/studies/2026-08-01_upside_objective_audit/summarize_results.py
```

The roster replay includes production, the prior `flatter_w025_all` Phase-B
finalist, and `wr_ppg225_both025`, which was the only saved Phase-B arm to
improve the primary q90 event Brier/log-loss and continuous tail score in all
four league-by-period cells while remaining inside the frozen PPG guardrail.
