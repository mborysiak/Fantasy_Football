# Decision Readout: One-Year, One-Hit Keeper Portfolio

## Decision

The revised keeper formulation worked well enough to advance to a
current-production-method replay.

Use only first-year keeper surplus, value the best outcome across all five
bench slots, fix the eight current starters, and allow at most two bench swaps
that do not reduce the causal construction-bank mean. Do not require two
keepers to hit and do not penalize a missed option beyond its measured
current-season bench opportunity cost.

Prefer `best1_lex0`. The two-point tolerance bought little additional keeper
value and is unnecessary when the zero-loss policy already finds the intended
players. Keep this out of the live app until the same post-process is replayed
with current v5 salaries, the selection reserve, and the converged organic
construction path.

## Revised Contract

For each next-year projection draw, convert PPG to position-specific Market `$`
and score each bench player as:

```text
max(future_market_value - (acquisition_price + 10), 0)
```

The roster utility is the mean best surplus across the five bench players, not
the sum of two players' expected values:

```text
E[max(bench_player_surplus_1, ..., bench_player_surplus_5)]
```

This matches the intended strategy: draft several cheap options, keep the one
that hits, and recycle misses through waivers. The second keeper slot remains a
supplementary top-two metric rather than a success requirement.

Historical next means and residual quantiles come from current-method OOS
`Model_Validations_Resid` rows with `current_or_next_year='next'`. The maximum
residual donor season is `origin - 1` in every replay. Players without a
dedicated next row use the current origin projection plus position-level next
residuals and are explicitly flagged as proxies.

## Primary Paired Results

Changes below are `best1_lex0` versus the same current-only roster. The eight
nominal starters are identical and at most two bench slots change.

| Origin | Forecast mean | Forecast p10 | Actual season | Weeks 13-16 | Best keeper surplus | Any `$20+` hit | Bench spend |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | +4.7 | +4.2 | +8.5 | +3.3 | +$17.9 | +16.4 pp | -$0.9 |
| 2023 | +0.9 | -0.7 | +4.9 | +2.9 | +$12.5 | +3.6 pp | -$1.9 |
| 2024 | -2.7 | -3.0 | -19.5 | -0.2 | +$6.4 | +3.6 pp | -$5.1 |
| 2025 | +1.5 | +1.8 | -3.7 | -5.5 | unavailable | unavailable | -$4.9 |

Across the three observable keeper origins, the policy improved best realized
first-year surplus by `$12.3` per roster and raised the probability of at least
one `$20+` surplus keeper by 7.9 percentage points. Both trial halves had
positive best-surplus effects in every observable origin.

The chance of any `$10+` hit was already high in the control—80.4%, 84.4%, and
76.0%—so it rose only 12.8, 0.4, and 0.0 points. The more informative result is
that the best hit became substantially more valuable. The policy also raised
maximum future PPG by 1.10 and 0.92 in 2022-2023. In 2024 maximum future PPG was
flat, but keeper surplus still increased because the successful players were
cheaper to acquire.

Observed-price surplus, despite incomplete historical salary coverage, moved
in the same direction: `+$9.9`, `+$13.8`, and `+$6.6` by origin.

## Current-Season Opportunity Cost

The average four-origin forecast effect was `+1.1` season points for the mean
and `+0.6` for p10. Average actual season points changed by `-2.5`, and average
weeks 13-16 points were effectively flat (`+0.1`). Thus the revised policy did
not buy keeper value by broadly degrading the modeled current roster.

The 2024 forecast and actual-season declines remain the cautionary slice.
Although the construction-bank mean improved by 5.5 points and the starters
were fixed, the independent evaluation mean fell 2.7 and that historical
season fell 19.5. This is the residual opportunity cost of replacing useful
bench depth. It is small in forecast terms but should remain a promotion gate
in the current-method replay.

## Player-Level Check

Relative to control, `best1_lex0` added the desired successful options to these
numbers of 250 trial rosters:

- Kenneth Walker `+44` and Rachaad White `+49` in 2022;
- Devon Achane `+48` and Zay Flowers `+5` in 2023; and
- Chase Brown `+31` and Bucky Irving `+24` in 2024.

This supports the user's one-hit framing. The method does not need every bench
bet to work; it needs to increase exposure to at least one cheap player who
becomes materially more valuable.

Cam Skattebo appeared on 39/250 control benches and 39/250 `best1_lex0` benches
in 2025. He was retained rather than rejected, but the policy did not increase
his selection because he has no dedicated 2025 next-validation row and uses the
explicit current-projection proxy. His 2026 keeper outcome is also not yet
observable. Prospective rookie/first-year next-row coverage is therefore an
important implementation gap.

## Why This Version Improved On The First Study

1. It removed the unsupported assumption that one next-year draw persists for
   three keeper seasons.
2. It evaluates the best realized option across the full bench instead of
   requiring the two players ranked highest in August to be the hits.
3. It uses causal historical next-model residual quantiles rather than the
   heterogeneous legacy truncated-normal surfaces.
4. It evaluates counterfactual acquisition cost for every player and retains
   observed salary only as an audit.
5. It fixes all starters and limits changes to two bench slots under an exact
   current-value constraint.

Selection concentration across simulation trials is not itself a failure. If
the same cheap player is the best option in many market draws, repeatedly
selecting him is correct. The relevant concentration is the five-player bench
portfolio within one roster, whose payoff here requires only one success.

## Remaining Boundaries And Next Step

- There are only three realized keeper origins. The 2025-to-2026 outcome is not
  available.
- Historical points use the current 2026 model specification on OOS origin
  data; this is a current-method walk-forward replay, not a nested holdout for
  model-spec selection.
- Twenty-two to 72 candidates per origin use explicit point proxies rather
  than dedicated next-model rows.
- The option portfolio uses a 0.25 common residual correlation. Dependence
  sensitivity should be checked before treating the expected maximum as
  calibrated dollars.
- Waiver churn remains implicit: misses have zero future keeper payoff while
  their current bench contribution is fully scored. Opponent claims and exact
  transaction timing are not modeled.

Next, reproduce `best1_lex0` as a post-processing candidate in the current app
method: start from the converged current roster, protect its starters, use the
v5-plus-reserve decision price as the strike, and accept at most two bench swaps
only when current mean/p10 and affordability remain inside explicit gates.

No production app or database behavior changed in this study.
