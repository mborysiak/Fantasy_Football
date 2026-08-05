# Logged Expert-Rank Level Confirmation

## Frozen question

On the current promoted DK and beta V2 lineages, is
`log1p(median raw overall expert rank)` materially better than the existing
normalized expert-rank challenger?

The normalized challenger converts each source to a within-source,
season-position percentile before taking the cross-source median. The raw-log
challenger takes the median of the same scoring-specific sources' published
overall ranks and then applies `log1p`.

DK substitutes the pinned full-PPR ETR ranks for 2025 and 2026. Beta retains
the half-PPR ETR source. Both representations therefore use the same
league-specific rank-source rows and player availability.

## Frozen model comparison

- `incumbent`: locked production features.
- `normalized_rank`: incumbent plus the normalized within-position rank level.
- `raw_log`: incumbent plus `log1p(median raw overall expert rank)`.

Every forecast origin reuses the incumbent's strictly-prior selected
hyperparameters. The primary attribution surface gives the random forest every
column so feature subsampling cannot create an artificial difference. The
locked equal-thirds production surface is a sensitivity.

The direct comparison is `raw_log` versus `normalized_rank`. Raw log replaces
normalized rank; the two are not bundled.

## Frozen advancement gates

Raw log is preferred only if, in both DK and beta:

- controlled pooled RMSE improves by at least 0.001;
- controlled 2023-2025 RMSE is nonworse;
- at least 6 of 9 seasons improve;
- season- and player-cluster 95% interval upper bounds are at most zero;
- production-surface pooled and recent RMSE are nonworse; and
- at least three of four positions are nonworse on the controlled surface.

If the representations are tied, retain normalized rank because its
within-source position normalization is more robust to provider pool depth and
overall QB-placement differences. Passing only advances a representation to
the same future nested/prospective validation as normalized rank; it does not
promote expert rank into production.

## Governance

- The current V2 databases are opened read-only.
- SHA-256 hashes are checked before and after each run.
- Results are written only below this study.
- No production table, feature manifest, or model lock is changed.

Run one league or both:

```powershell
.\.venv_ff_312\Scripts\python.exe research\studies\2026-08-04_v2_log_expert_rank_confirmation\run_study.py --league dk
.\.venv_ff_312\Scripts\python.exe research\studies\2026-08-04_v2_log_expert_rank_confirmation\run_study.py --league beta
.\.venv_ff_312\Scripts\python.exe research\studies\2026-08-04_v2_log_expert_rank_confirmation\run_study.py --league all --combine-existing
```
