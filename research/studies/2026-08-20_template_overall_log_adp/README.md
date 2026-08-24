# Overall Log-ADP Weekly-Template Matching

## Question

Does the weekly-template matcher lose useful market-strength information by
ranking ADP only within season and position? In particular, should an elite TE
drafted near pick 20 be distinguished from a season's TE1 drafted much later?

## Frozen challengers

The study compares exactly three methods:

- `production`: the current matcher;
- `replace_adp_rank_with_log_adp`: remove the direct `adp_rank_pct` distance
  and replace it at the same `0.50` weight with fixed-scale overall log ADP;
- `add_log_adp`: retain all production fields and add overall log ADP at
  weight `0.50`.

The added field is calculated from the actual overall pick before any
position-specific donor filtering:

```text
match_log_adp_scaled = log1p(clip(avg_pick, 1, 300)) / log1p(301)
```

Missing or nonpositive ADP remains missing and receives the existing neutral
distance fill. Donors remain same-position only. Experience, projection,
market-gap, role, room, kernel, recency, eligibility, pool-size, and donor-cap
rules are unchanged.

## Validation policy

The replay follows the active role-tiered contract from
`2026-07-31_template_role_tiered_validation`:

- 2,647 held-out targets per league from expanded annual cohorts;
- strictly earlier donors at every origin;
- separate DK and beta scoring;
- development selection on 2017-2022 and temporal checks on 2023-2025;
- core-player PPG-first selection with contribution, calibration, coverage,
  position, and availability guardrails;
- Phase C roster transport only if a challenger clears the joint Phase-B
  screen.

The live production matcher and databases are read-only inputs.

## Result

Both challengers fail Phase B, so Phase C roster transport is intentionally
skipped and production remains unchanged.

- DK and beta each replayed 2,647 held-out targets and 7,941 paired method
  rows. Log-ADP coverage is 100% for all 5,298 historical templates and all
  rolling targets.
- Replacement is effectively flat-worse in development core PPG: relative
  deltas are `+0.000014%` DK and `+0.001204%` beta. It improves temporal DK by
  `-0.030360%` but worsens temporal beta by `+0.064543%`.
- Addition worsens development DK by `+0.016723%` and improves beta by only
  `-0.005675%`; the temporal signs reverse (`-0.023722%` DK,
  `+0.037778%` beta). Player-cluster intervals cross zero.
- TE-only results do not transport across scoring systems. Adding log ADP
  slightly improves DK PPG CRPS in development and 2023-2025, while beta is
  slightly worse in both periods.
- In Brock Bowers' current beta pool, replacement/addition raise weight on
  donors with ADP 35 or earlier from `38.26%` to `39.59%`/`41.33%`.
  Addition raises combined Kelce/Gronk/Graham weight from `13.78%` to
  `15.16%`, but centered residual q90 falls from `+3.889` to `+3.836` PPG and
  `P(+5)` falls from `3.31%` to `3.14%`. The intended comp-composition change
  therefore does not produce a supported upside-calibration gain.

See `results_phase_b/phase_b_findings.md` and
`results_phase_b/finalist_decisions.csv`.

## Reproduction

```powershell
.venv_ff_312\Scripts\python.exe research\studies\2026-08-20_template_overall_log_adp\run_phase_b_replay.py --league dk
.venv_ff_312\Scripts\python.exe research\studies\2026-08-20_template_overall_log_adp\run_phase_b_replay.py --league beta
python research\studies\2026-08-20_template_overall_log_adp\run_phase_b_rescore.py
```
