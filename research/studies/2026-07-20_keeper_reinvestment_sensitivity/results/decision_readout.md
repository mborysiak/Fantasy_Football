# Decision Readout: Full-Roster Keeper Reinvestment

## Decision

The budget-transfer mechanism is real, but a broad full-roster keeper
bonus is not ready for promotion. The k1/k2/k3 labels count incremental
forced additions, not mutually exclusive bench roles or total lottery
tickets. K1 is the safest tested forced-addition policy, but the next model
should use soft portfolio tradeoffs rather than a hard option count.

The one-option policy moved `$5.7` from the five-player bench to the
starting core while total spend increased only `$0.1`. Starter-only
forecast mean/p10 improved `4.6`/`3.6`, proving that cheap options can
finance stronger starters. Full-roster mean/p10 improved only `0.6`/`1.2`,
because the sacrificed bench depth offsets most of the starter gain.

## Across-Origin Tradeoff

| policy | mean_forced_options | cap_filled_rate | forced_young_share | starter_changes | starter_forecast_spend_effect | bench_forecast_spend_effect | starter_forecast_ev_effect | starter_forecast_p10_effect | forecast_ev_effect | forecast_p10_effect | actual_points_effect | actual_playoff_points_effect | actual_waiver_starts_effect | actual_best_keeper_surplus_effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reinvest_k1 | 0.828 | 82.8 | 97.464 | 1.381 | 5.718 | -5.61 | 4.567 | 3.623 | 0.576 | 1.241 | 8.09 | -0.544 | 0.344 | 2.066 |
| reinvest_k2 | 1.495 | 66.7 | 96.789 | 1.635 | 7.938 | -7.85 | 6.956 | 5.322 | 0.548 | 0.146 | 7.688 | -0.706 | 0.724 | 3.091 |
| reinvest_k3 | 1.931 | 43.6 | 95.443 | 1.735 | 8.511 | -8.294 | 7.974 | 5.466 | 0.294 | -0.649 | -0.472 | -4.483 | 1.377 | -0.022 |

The second forced option bought only about `$1.0` of additional realized
best keeper surplus versus k1 while reducing full-roster p10 by about `1.1`
points and adding about `0.4` waiver starts. The third option was dominated:
it raised waiver use, reduced p10 and playoff scoring, and erased the
incremental realized keeper surplus because the 2024 portfolio failed.

## Total Bench Composition

| policy | young_le2 | young_le3 | rookie | veteran_gt3 |
| --- | --- | --- | --- | --- |
| control | 3.336 | 3.768 | 1.994 | 1.232 |
| reinvest_k1 | 3.708 | 3.948 | 2.355 | 1.052 |
| reinvest_k2 | 3.955 | 4.123 | 2.507 | 0.877 |
| reinvest_k3 | 4.1 | 4.24 | 2.629 | 0.76 |

The forced-option count is incremental to the current optimizer's bench.
Using at most two years of experience as a descriptive upside proxy, the
control already averaged `3.34` young players among five bench slots. K1
raised that to `3.71`, k2 to `3.96`, and k3 to `4.10`. Thus k3 generally
tested a much more youth-heavy bench, not a clean test of a flexible
two-fill-in/two-to-three-ticket preference. Youth is not a role: a young
player can provide both startable current depth and keeper upside.

## Why Incremental Keeper Value Was Small

The rebuilt current-only control already put Kenneth Walker on 249/250
2022 benches, Achane on 211/250 and Flowers on 152/250 2023 benches, and
Chase Brown on 226/250 2024 benches. The option policy therefore had little
room to improve hit probability; it mostly added further bets. K1 did add
Rachaad White to 15 rosters and Bucky Irving to 13, but its most common
forced names were Dameon Pierce, Zach Charbonnet, Trey Benson, and Ray Davis.

Across all forced selections, 97.5% of k1 options and 96.8% of k2 options
were players with at most two years of experience. The signal is selecting
the intended young profile; the problem is incremental calibration and
bench opportunity cost, not a failure to target youth.

## Most Common Forced Options

| year | policy | player | forced_rosters | forced_roster_rate |
| --- | --- | --- | --- | --- |
| 2022 | reinvest_k1 | Dameon Pierce | 128 | 51.2 |
| 2022 | reinvest_k1 | Elijah Moore | 20 | 8.0 |
| 2022 | reinvest_k1 | Rachaad White | 15 | 6.0 |
| 2022 | reinvest_k1 | Cordarrelle Patterson | 7 | 2.8 |
| 2022 | reinvest_k1 | Treylon Burks | 7 | 2.8 |
| 2022 | reinvest_k2 | Dameon Pierce | 156 | 62.4 |
| 2022 | reinvest_k2 | Elijah Moore | 48 | 19.2 |
| 2022 | reinvest_k2 | Michael Carter | 32 | 12.8 |
| 2022 | reinvest_k2 | Rachaad White | 31 | 12.4 |
| 2022 | reinvest_k2 | Rhamondre Stevenson | 22 | 8.8 |
| 2022 | reinvest_k3 | Dameon Pierce | 166 | 66.4 |
| 2022 | reinvest_k3 | Elijah Moore | 61 | 24.4 |
| 2022 | reinvest_k3 | Michael Carter | 48 | 19.2 |
| 2022 | reinvest_k3 | Rachaad White | 45 | 18.0 |
| 2022 | reinvest_k3 | Rhamondre Stevenson | 41 | 16.4 |
| 2023 | reinvest_k1 | Zach Charbonnet | 64 | 25.6 |
| 2023 | reinvest_k1 | Tyjae Spears | 54 | 21.6 |
| 2023 | reinvest_k1 | Kendre Miller | 32 | 12.8 |
| 2023 | reinvest_k1 | Brian Robinson | 22 | 8.8 |
| 2023 | reinvest_k1 | Rashaad Penny | 18 | 7.2 |
| 2023 | reinvest_k2 | Zach Charbonnet | 96 | 38.4 |
| 2023 | reinvest_k2 | Tyjae Spears | 72 | 28.8 |
| 2023 | reinvest_k2 | Kendre Miller | 69 | 27.6 |
| 2023 | reinvest_k2 | Brian Robinson | 52 | 20.8 |
| 2023 | reinvest_k2 | Rashaad Penny | 39 | 15.6 |
| 2023 | reinvest_k3 | Zach Charbonnet | 109 | 43.6 |
| 2023 | reinvest_k3 | Kendre Miller | 92 | 36.8 |
| 2023 | reinvest_k3 | Tyjae Spears | 87 | 34.8 |
| 2023 | reinvest_k3 | Brian Robinson | 67 | 26.8 |
| 2023 | reinvest_k3 | Rashaad Penny | 54 | 21.6 |
| 2024 | reinvest_k1 | Zach Charbonnet | 146 | 58.4 |
| 2024 | reinvest_k1 | Trey Benson | 44 | 17.6 |
| 2024 | reinvest_k1 | Blake Corum | 28 | 11.2 |
| 2024 | reinvest_k1 | Bucky Irving | 13 | 5.2 |
| 2024 | reinvest_k1 | Ray Davis | 3 | 1.2 |
| 2024 | reinvest_k2 | Zach Charbonnet | 200 | 80.0 |
| 2024 | reinvest_k2 | Trey Benson | 121 | 48.4 |
| 2024 | reinvest_k2 | Blake Corum | 72 | 28.8 |
| 2024 | reinvest_k2 | Bucky Irving | 28 | 11.2 |
| 2024 | reinvest_k2 | Ray Davis | 27 | 10.8 |
| 2024 | reinvest_k3 | Zach Charbonnet | 216 | 86.4 |
| 2024 | reinvest_k3 | Trey Benson | 156 | 62.4 |
| 2024 | reinvest_k3 | Blake Corum | 90 | 36.0 |
| 2024 | reinvest_k3 | Bucky Irving | 59 | 23.6 |
| 2024 | reinvest_k3 | Ray Davis | 54 | 21.6 |
| 2025 | reinvest_k1 | Ray Davis | 108 | 43.2 |
| 2025 | reinvest_k1 | Trey Benson | 18 | 7.2 |
| 2025 | reinvest_k1 | Jacory Croskey Merritt | 15 | 6.0 |
| 2025 | reinvest_k1 | Braelon Allen | 11 | 4.4 |
| 2025 | reinvest_k1 | Tyrone Tracy | 11 | 4.4 |
| 2025 | reinvest_k2 | Ray Davis | 130 | 52.0 |
| 2025 | reinvest_k2 | Trey Benson | 43 | 17.2 |
| 2025 | reinvest_k2 | Jacory Croskey Merritt | 32 | 12.8 |
| 2025 | reinvest_k2 | Tyrone Tracy | 28 | 11.2 |
| 2025 | reinvest_k2 | Braelon Allen | 17 | 6.8 |
| 2025 | reinvest_k3 | Ray Davis | 134 | 53.6 |
| 2025 | reinvest_k3 | Trey Benson | 57 | 22.8 |
| 2025 | reinvest_k3 | Jacory Croskey Merritt | 49 | 19.6 |
| 2025 | reinvest_k3 | Tyrone Tracy | 33 | 13.2 |
| 2025 | reinvest_k3 | Braelon Allen | 20 | 8.0 |

## Selection Frequency

K1 accepted one new option in 82.8% of rosters. K2 selected two in 66.7%,
one in 16.1%, and none in 17.2%. K3 reached all three in only 43.6%; its
mean was 1.93 forced options. The gates therefore stop some additions, but
the third slot still activates often enough to cause material depth damage.

## Boundaries

- There are only three realized next-season keeper origins.
- The greedy eight-candidate shortlist is an explicit search approximation.
- The selection gate uses the full cached-bank expected profile; independent
  mean/p10 are evaluation outcomes and were not reused for selection.
- The rebuilt control differs from the preceding five-context control and is
  intentionally the fair same-engine comparator for full-roster reoptimization.
- This remains a frozen historical salary replay, not the final current
  v5-plus-selection-reserve production test.

## Recommended Next Step

Do not add a generic multi-slot keeper bonus to production. Next give every
bench player both a current fill-in value and a keeper-option value; the same
player may contribute to both. Optimize expected-best keeper surplus across
the whole bench only after roster mean/p10 and aggregate bench coverage are
protected, without age quotas or designated slot counts. Allow the resulting
number of lottery profiles to emerge. The waiver baseline can cap lineup
downside while causal drop/claim timing remains a separate enhancement.
