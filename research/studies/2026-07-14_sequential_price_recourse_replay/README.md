# Sequential Price-Recourse Replay

This study tests the `$5` and `$10` nominal predicted-salary guardrails in a
sequential auction setting. It is deliberately narrower than a historical
auction replay because the source exports retain winning prices and final owner
rosters, but not nomination order, losing bids, or opponent budget responses.

The estimand is the paired `$5 - $10` result for a non-anticipating,
receding-horizon policy conditional on:

- the historical nonkeeper skill-player clearing-price tape;
- a prespecified synthetic nomination-order stress family; and
- either first-refusal at the recorded clearing price (`p`) or a `$1` outbid
  stress (`p + 1`).

Only the current nominee, current price, and prior aggregate sales are revealed.
Future tape membership, future order, future prices, and target-season outcomes
are not available to the policy. Every acquired player counts at the paid price
in both the stochastic salary cap and nominal salary guardrail. Target-season
weekly scores are accessed only after a legal 13-player roster is complete.

The raw `Data/OtherData/Salaries/beta_YYYY_results.csv` exports are parsed as 12
anonymous roster blocks. K and D/ST are removed to reconcile the historical 15-player,
`$300` league with the app's 13-skill-player, `$298` contract. Forecast-unmatched
historical skill players are passive opponent transactions and can never be
selected. The one unidentified modeled skill slot in each of 2023 and 2024 is
an opaque `$1` event with unknown position. No runtime fallback invents a player:
if an opaque event becomes mandatory, the path fails and the failure is reported.

Both policy modes preserve the real `$298` cap, roster size, and position limits:

- `strict` keeps the requested `$5` or `$10` nominal guardrail and Top-N rule;
- `operational` tries `$5 -> $10 -> no nominal row` or `$10 -> no nominal row`,
  then drops Top-N only if neither Buy nor Pass has a legal continuation.

Strict mode is the primary buffer comparison. Operational mode is a separately
labeled fallback sensitivity, not a pure `$5` versus `$10` estimand.

Order families are reported separately because the historical nomination order
cannot be recovered:

- `tier_early`: frozen preseason projection tiers, high to low;
- `uniform`: a uniform random permutation;
- `position_run`: short randomized position runs, high projections first; and
- `star_late`: projection tiers reversed as an adversarial extreme.

The salary uncertainty row is one coherent average of five frozen replay draws
per path and is dynamically renormalized after each observed sale. The nominal
point-salary row is independently renormalized to the same live market state.
The study therefore inherits the frozen replay's mostly legacy 2023-2025 salary
laws; it does not claim to validate the current 2026 residual-quantile model.
Every path begins with an empty personal roster after all league keepers have
been removed from the market. A close result must be checked in representative
fixed-personal-keeper states before adopting a universal live setting.

Run a smoke check:

```powershell
python research/studies/2026-07-14_sequential_price_recourse_replay/run_recourse_replay.py `
  --years 2025 --trials 1 --order-regimes uniform `
  --contexts 8 --evaluation-contexts 12 --context-draws 2 `
  --projection-draws 100 --salary-draws 100 `
  --output-dir research/studies/2026-07-14_sequential_price_recourse_replay/artifacts/local/smoke
```

Run the predeclared initial replay:

```powershell
python research/studies/2026-07-14_sequential_price_recourse_replay/run_recourse_replay.py
```

The initial run uses eight common random paths per year and order family. The
primary slice is strict `p + 1` over 2022-2024 for tier-early, uniform, and
position-run orders. Recorded price `p`, star-late, operational mode, and 2025
are sensitivities. Extend the trial count if paired-clean order/context
randomization error remains above two season points or split-half signs are
unstable. This error is not future-season uncertainty. Results are never pooled
across order families using invented probability weights.

No buffer is selected unless the primary families have effectively complete
paired-clean coverage, no meaningful discordant completion, stable signs, and
adequate randomization precision. The zero-point value recorded for an
incomplete path is only a harsh policy-invalid sensitivity, not an observed
season score.

## Completed Result

The initial replay completed 1,024 policy paths and 821 legal rosters. It does
not select either buffer. In the primary strict `p + 1` slice over 2022-2024 and
the three primary order families:

- `$5` completed 44/72 paths and `$10` completed 42/72;
- both buffers completed in only 38/72 pairs, with 10/72 discordant pairs; and
- only 15/72 pairs were clean enough for the prespecified point comparison.

No order family had clean observations in every development origin, so the
equal-origin point effect and its randomization error are undefined. Available
signs conflict by year and order. Assigning failed drafts zero points reverses
some completed-roster comparisons, confirming that feasible-only scoring is
selection-biased rather than a valid basis for choosing a buffer.

At the recorded winning price `p`, development completion improved to 70/72 for
`$5` and 69/72 for `$10`; 67/72 pairs both completed. That sensitivity still
had five discordant pairs and only 33/72 clean pairs. Conditional completed-pair
scores leaned toward `$10`, but the clean order/year signs were unresolved. The
2025 sensitivity also reversed completion direction relative to development and
had mixed point signs. Operational mode was identical to strict in every
development path because none of its nominal or Top-N fallbacks was accepted.

The earlier static replay's `+$5` preference therefore remains provisional only;
this sequential fixed-tape study does not validate it as a universal live
setting or justify switching to `+$10`. The next decision-quality test should
rebuild the current residual-quantile salary method walk-forward and evaluate
representative fixed-personal-keeper states with coherent market scenarios or a
direct chance-risk/recourse rule.

Full results and the explicit decision gate are in
`results/decision_readout.md`.

## Checkpoint and Native-Solver Reproduction

The full run should use validated year checkpoints:

```powershell
python research/studies/2026-07-14_sequential_price_recourse_replay/run_recourse_replay.py `
  --checkpoint-dir research/studies/2026-07-14_sequential_price_recourse_replay/artifacts/local/full_checkpoints_pickle `
  --resume
```

CVXOPT/GLPK exited natively and intermittently during the long 2025 process,
without a Python exception. The exact 2025 grid was therefore executed as 32
fresh order-regime/trial shards using `--trials 8 --trial-indices N`, preserving
the full original salary/context plans and order seeds. The assembler verifies
all 256 unique cells, ledgers, source-audit hashes, roster/cap rules, and a
separate prefix-invariance artifact before committing the 2025 checkpoint:

```powershell
python research/studies/2026-07-14_sequential_price_recourse_replay/assemble_2025_trial_shards.py
```

GLPK's hypothetical continuation rows are audited with a `$0.001` numerical
tolerance; the largest observed excess was `$0.0004223`. Completed rosters still
use the exact recorded paid prices and are independently required to stay at or
below `$298` within `1e-8`.
