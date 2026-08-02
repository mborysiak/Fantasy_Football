# V2 Market and Expert-Rank Challengers

## Questions

1. What changes when the four correlated NFFC contest feeds plus their
   composite are replaced by one modeled NFFC family observation?
2. Does normalized expert-rank information improve the locked conditional-PPG
   model beyond projection consensus, ADP, history, and room context?

## NFFC policy

`ADP_Averages(league='nffc')` already stores one player-level arithmetic
average of the available Rotowire Online, Best Ball Overall, Best Ball $25/$50,
and Cutline feeds. V2 now admits only that `adp_average_nffc` row to
`player_season_market_values`. The four raw `NFFC_ADP` rows remain
candidate/identity evidence.

The isolated feature marts are:

- `artifacts/local/Projection_V2_single_nffc.sqlite3`
- `artifacts/local/Projection_V2_beta_single_nffc.sqlite3`

They are local staging artifacts and are intentionally not review evidence.
Durable findings and compact tabular outputs belong under `results/`.

## Expert-rank construction

Raw provider ranks are not commensurate because list depths and ranking
objectives differ. The study therefore:

1. orders each source within season and position;
2. converts that order to a percentile, with 1.0 best and 0.0 worst;
3. takes the median percentile across available sources; and
4. tests either the normalized level or its difference from the projection
   position percentile.

ETR contributes one normalized vote. A dedicated ETR coefficient is not
tested because half-PPR ETR history begins in 2024, leaving only one learnable
out-of-fold season before the 2026 fit.

The primary attribution reuses the incumbent model's strictly-prior selected
hyperparameters. Because the locked random forest samples 50% of columns at
each split, the primary attribution sets `max_features=1.0` for both the
incumbent and challenger forests; otherwise merely adding a column changes
which incumbent fields are sampled. Every origin is still fit only on earlier
seasons, and imputation remains inside each training fit. The original locked
forest is retained as a production-surface sensitivity.

## Prespecified raw-rank follow-up

The raw-rank follow-up tests the proposal to take the median of every rank
that actually exists for a player, regardless of provider list depth. Provider
positions are first mapped to the staged player-season position. The runner
fails if the number of observed source votes then differs from the raw-median
source count.

DK removes half-PPR ETR rows and uses `ETR_Ranks_PPR` for 2025-2026 only.
Beta retains half-PPR `ETR_Ranks` for 2024-2026. The normalized comparator is
rebuilt in-process from these same scoring-specific rows; it is not read from
the earlier normalized study.

The fixed ladder is:

1. incumbent;
2. scoring-specific normalized rank (comparison benchmark);
3. raw available-rank median;
4. `log1p` of that median;
5. season-wide percentile calculated after taking the median; and
6. that percentile plus publication coverage.

Publication coverage is the number of sources ranking the player divided by
the number publishing any rank for the staged season-position. It measures
partial-list availability; it is not a depth-adjusted rank. Raw provider count,
eligible count, source depths, and shallow-list omissions remain audit fields.

Only the percentile-plus-coverage variant is eligible to advance. A
single-league gate requires favorable pooled and recent deltas, at least six
of nine season wins, nonpositive season- and player-cluster interval upper
bounds, the same robustness on the locked production-surface blend, at least
0.001 RMSE improvement over the in-run normalized comparator, position and
history guardrails, non-worse early and expanded provider eras, and non-worse
incomplete-coverage behavior. Both DK and beta must pass. Passing advances the
feature only to a strict nested-retune study; it does not alter production.

## Run

First build the isolated marts and run the locked NFFC-only validations so the
selected incumbent hyperparameters and reproduction tables are present. Then:

```powershell
python research/studies/2026-07-30_v2_market_rank_challengers/run_expert_rank_challenger.py --league dk
python research/studies/2026-07-30_v2_market_rank_challengers/run_expert_rank_challenger.py --league beta
```

The runner writes OOF predictions, slice scores, season/player-cluster
intervals, 2026 shadows, source coverage, and an ETR leave-out diagnostic.
It never writes to the V2 database.

Run the raw follow-up and then create its cross-league receipt:

```powershell
python research/studies/2026-07-30_v2_market_rank_challengers/run_raw_rank_challenger.py --league dk
python research/studies/2026-07-30_v2_market_rank_challengers/run_raw_rank_challenger.py --league beta
python research/studies/2026-07-30_v2_market_rank_challengers/run_raw_rank_challenger.py --league all
```

The raw runner opens staged databases read-only, verifies exact locked feature
and hyperparameter lineage, and pins scoring-specific rank inputs and raw ETR
file hashes in `input_manifest.json`.

## Decision

The one-vote NFFC contract is promoted. Locked pooled RMSE changes from
3.10783 to 3.10756 in DK and from 2.88446 to 2.88411 in beta.

The normalized expert-rank level remains a challenger. In the controlled
full-column-forest blend it improves RMSE by 0.00221 DK and 0.00186 beta and
wins seven of nine seasons in both leagues. DK's season and player-cluster
intervals exclude zero; beta's season interval excludes zero and its player
interval ends essentially at zero. The locked-production-surface sensitivity
is smaller and uncertain, and the expert-minus-projection gap is neutral and
unstable. The effect is too small, and provider-era drift too large, to alter
the production lock from this study. ETR overall rank continues to order beta
production eligibility; ETR also contributes one normalized vote inside the
unpromoted rank challenger.

The prespecified raw-rank follow-up does not advance. Percentile-after-median
plus publication coverage improves the controlled blend by only 0.00138 DK
and 0.00016 beta, wins 6/9 and 5/9 seasons, and has season- and player-cluster
intervals crossing zero in both leagues. It is worse than the matched
scoring-specific normalized comparator by 0.00173 DK and 0.00213 beta, and its
early-provider-era delta is unfavorable. Among rank-available rows its
controlled delta is only -0.00024 DK and +0.00027 beta.

The `log1p(raw median)` scale diagnostic improves versus the incumbent by
0.00329 DK and 0.00231 beta in the controlled comparison, but is not
distinguishable from the matched normalized comparator in this study: direct
raw-log-minus-normalized deltas are -0.00018 and -0.00002 with both season-
and player-cluster intervals crossing zero. It remains research-only;
selecting it from this ladder would require an independently confirmed
decision rule.

See `results/findings.md` for the release and validation receipt.
See `results/raw_rank_combined/findings.md` for the raw-rank cross-league
receipt.
