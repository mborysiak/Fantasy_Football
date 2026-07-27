# Decision Readout

> **Superseded on 2026-07-14:** The unconditional historical point comparison
> below rewards some rosters that could not be purchased at realized prices and
> cannot select a production buffer. The affordability differences remain valid.
> See `../../2026-07-14_salary_chance_frontier/results/decision_readout.md` for
> the feasibility-first follow-up and current decision.

## Outcome

The current-method replay confirms a real, stable tradeoff rather than an actual
points improvement from the tighter guardrail.

| Development 2022-2024 | No buffer | +$10 | +$5 |
| --- | ---: | ---: | ---: |
| Actual season points | 1,594.67 | 1,591.72 | 1,586.15 |
| Historical-price feasible | 10.53% | 16.67% | 23.20% |
| Mean cap overage | $31.86 | $24.43 | $20.88 |

Direct `+$5 minus +$10` effects were:

| Period | Season points | Team PPG (16 weeks) | Feasibility | Mean overage |
| --- | ---: | ---: | ---: | ---: |
| 2022-2024 | -5.57 | -0.35 | +6.53 pp | -$3.55 |
| 2025 check | -3.10 | -0.19 | +6.40 pp | -$3.32 |

The point effect was negative, feasibility positive, and overage negative in
all four seasons. Both trial halves retained the same directions. The tighter
constraint changed 83.7% of development rosters and 87.2% of 2025 rosters, so
this is a material policy difference rather than a cosmetic cap.

## Recommendation

Do not select a default from this replay. In development, moving from no buffer
to `+$10` changed unconditional season points by -2.95 while gaining 6.13
feasibility points and cutting mean overage by $7.43. Tightening from `+$10` to
`+$5` changed unconditional points by -5.57, gained another 6.53 feasibility
points, and reduced overage by another $3.55. Only the affordability effects are
valid policy evidence; the point effects include historically unaffordable
rosters.

Do not interpret this as solving affordability. Even `+$5` produced only 23.2%
development and 16.4% 2025 historical-price feasibility. Coherent salary-market
scenarios or direct chance/recourse constraints remain the larger opportunity.

## Verification and sensitivity

- All 3,000 paired cells solved optimally with 13 unique players, current
  position maxima (`QB <= 1`, `TE <= 2`), Top-N, sampled `$298`, and nominal
  constraints satisfied.
- Salary centers and residual histories respected their rolling-origin cutoffs;
  weekly construction donors preceded the target season.
- An independent reconstruction from `roster_trials.csv` exactly reproduced all
  direct effects and cap-feasibility flags.
- Development first/second-half `+$5 minus +$10` effects were respectively
  `-6.14/-4.99` points, `+6.4/+6.7` feasibility points, and
  `-$3.91/-$3.19` mean overage.
- Minimum salary fallback selection differed by at most 0.004 players per roster
  between the two buffers in any year; salary-model fallback differences were
  also negligible. The comparison is not being driven by differential fallback
  use.
- Jointly feasible point effects are too sparse and mixed to override the main
  result: only 105/750 development pairs and 20/250 2025 pairs fit historical
  prices under both buffers.

## Limits

- Historical final prices are exogenous; this is not a replay of opponent bids.
- The salary data roll by origin, but the model specification was selected as of
  2026 and is not a fresh historical method holdout.
- Missing historical actual prices use the intentional `$1` fallback, making
  absolute feasibility optimistic. Direct fallback-count differences between
  buffers are small, but the level remains a limitation.
- Four seasons are four independent outcome units. The 250 trials per year
  characterize Monte Carlo policy behavior, not 1,000 independent seasons.
