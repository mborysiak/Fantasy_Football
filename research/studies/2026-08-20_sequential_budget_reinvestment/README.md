# Sequential Budget Reinvestment

## Question

Does the production Sequential Auction policy overvalue buying Bijan Robinson
at `$105` because its Pass recourse fills the roster before reinvesting auction
savings? Can a bounded, deterministic target-upgrade policy capture the benefit
of an exact re-solve without bringing repeated-ILP instability into rollout?

## Arms

- `baseline`: unmodified production branch rollout.
- `slack_replan`: after an owned purchase, re-solve the remaining history-only
  roster when projected final spend leaves more than
  `max($5, $1 * open slots)` unused. A lost target also receives a full re-solve.
- `purchase_replan`: slower benchmark that fully re-solves after every owned
  purchase or lost target.
- `bounded_replan`: retain the compiled production plan, but on material slack
  search deterministically for up to three positive-value, same-position target
  upgrades among the top 24 remaining candidates and eight weakest targets.
  If a target is lost, rebuild a legal minimum-first plan before applying local
  upgrades.
- `bounded_guard`: `bounded_replan` plus a final-two-slot bargain rejection.

All arms use the same construction values, nomination tapes, revealed auction
prices, and weekly validation contexts within an evidence variation. The
experimental policies see only completed sales and the currently unresolved
player pool. Production App code and SQLite data remain unchanged.

## Exact replan result

The policy hypothesis is confirmed, but the exact repeated-ILP implementation
is not production-ready.

- Across variations 0-6 and 14, baseline returns `TARGET` eight of eight times
  with mean Buy-minus-Pass `+22.54` managed-season points. Slack-triggered
  replanning returns one `TARGET`, three `WATCH`, and four `PASS`, with mean
  Buy-minus-Pass `-2.63`.
- Average Pass EV rises from `1633.04` to `1661.00`, Pass p10 rises from
  `1445.75` to `1474.26`, and average unused salary falls from `$44.69` to
  `$22.53`.
- In variation 14, Bijan moves from `+27.44` / `+21.74` LCB80 (`TARGET`) to
  `+3.22` / `-5.19` (`WATCH`).
- Successful exact arms average about `1.38` seconds, but repeated fresh-process
  runs exposed individual GLPK solve times above 60 seconds and access
  violations. This arm is a policy benchmark, not a production candidate.

## Bounded follow-up result

The same-position bounded-upgrade arm passes this draft-state test.

- It returns two `TARGET`, two `WATCH`, and four `PASS` calls, with mean
  Buy-minus-Pass `-1.95`. Seven of eight classifications match the full replan;
  variation 3 moves from a marginal exact-arm `WATCH` to a marginal bounded-arm
  `TARGET` (`+0.23` LCB80).
- Average Pass EV reaches `1659.19`, recovering 93.5% of the full replan's
  `27.96`-point improvement over baseline. Average Pass p10 reaches `1472.47`,
  versus `1474.26` for the exact arm.
- Average Pass unused salary falls to `$22.81`, essentially matching the exact
  arm's `$22.53`. Completion is 99.5%, comparable with baseline's 99.2% and
  close to the exact arm's 100%.
- Fresh one-variation replay processes average about `0.67` seconds, roughly
  half the successful exact-arm runtime, with no solve required after the
  compiled plan is loaded.
- Variation 14 moves to `+1.47` / `-5.49` LCB80 (`WATCH`), closely matching the
  exact arm's direction.

An initial cross-position version was rejected because completion fell to
roughly 88-90%. The corrected same-position version preserves aggregate
position feasibility and uses a deterministic minimum-first rebuild when a
planned target becomes unavailable. The `bounded_guard` sensitivity was also
rejected: it was slightly favorable in variation 14 but developed a greater
than 90-second rejection chain in variation 4. A direct reward for salary spent
remains rejected.

### Alec Pierce exclusion sensitivity

Variation 14 was replayed with Alec Pierce removed from the draftable pool,
using a separately keyed compiled-plan bank and the same baseline/bounded
mechanics. All 96 Buy/Pass branches completed and no roster contained Pierce.

- Baseline remains `TARGET`, but Bijan's mean/LCB80 edge falls from
  `+27.44` / `+21.74` to `+24.33` / `+18.74`.
- Bounded reinvestment remains `WATCH`. Bijan's mean edge rises from `+1.47`
  to `+4.90`, while LCB80 remains below zero at `-3.26`.
- Bounded Pass EV falls modestly from `1653.40` to `1650.33`; Pass p10 is
  effectively flat-to-up (`1459.81` to `1460.42`). Average unused salary falls
  from `$20.96` to `$17.96`, and completion remains 100%.
- Pierce's roster share is redistributed across a broad receiver set rather
  than one replacement. The largest increases are Brian Thomas Jr., Ladd
  McConkey, and Rome Odunze (`+8.3` percentage points each), followed by
  Quentin Johnston (`+6.2`).

Pierce therefore contributes some of the Pass branch's mean value, but the
bounded conclusion is not dependent on him: passing remains competitive enough
that Bijan is a `WATCH`, not a robust `TARGET`.

### Gibbs after Chase Brown and Bhayshul Tuten

An earlier-state replay starts with Chase Brown at `$34` and Bhayshul Tuten at
`$11`, then evaluates Jahmyr Gibbs at the user's actual `$110` price. Across
variations 0-6 and 14:

- Baseline returns eight `TARGET` calls with mean Buy-minus-Pass `+40.27` and
  mean LCB80 `+32.35`.
- Bounded reinvestment returns six `TARGET`, one `WATCH`, and one `PASS`, with
  mean Buy-minus-Pass `+12.39` and mean LCB80 `+6.25`.
- The bounded Pass branch improves from baseline `1594.40` EV to `1649.37`,
  while bounded Buy improves from `1634.67` to `1661.76`. Gibbs retains a
  smaller but positive average edge after improving both branches.
- Bounded average unused salary is `$21.75` on Buy and `$25.09` on Pass,
  versus `$43.59` and `$80.24` under baseline. Completion remains 99%+.
- The primary variation 14 is `WATCH` (`+2.31`, `-3.36` LCB80), and variation 2
  is `PASS` (`-0.16`, `-5.73`). The other six variations are `TARGET`.

The evidence therefore supports Gibbs at `$110` on average, but not as an
unconditional recommendation in every evidence bank. This early-state replay
also exposed unnecessary repeated sorting in the research bounded search; a
semantics-preserving precomputed value order restores successful fresh-process
runtime to roughly `0.64-0.74` seconds per variation. Production remains
unchanged.

### Brown/Tuten full bounded target board

A full variation-14 Target Board uses the production 320-budget discovery and
confirmation surface, current modeled prices, one worker, and bounded rollout
recourse. It screens 64 candidates, exactly confirms 18, and ranks the confirmed
cohort by LCB80 then mean gain.

1. Bijan Robinson, RB, `$105`: `TARGET`, `+20.85`, LCB80 `+15.00`.
2. Kyren Williams, RB, `$53`: `TARGET`, `+15.17`, LCB80 `+11.05`.
3. Josh Allen, QB, `$36`: `TARGET`, `+11.96`, LCB80 `+7.04`.
4. Derrick Henry, RB, `$72`: `TARGET`, `+11.17`, LCB80 `+6.16`.
5. KC Concepcion, WR, `$4`: `TARGET`, `+5.80`, LCB80 `+1.85`.
6. Luther Burden III, WR, `$26`: `TARGET`, `+7.44`, LCB80 `+0.35`.
7. Dylan Sampson, RB, `$1`: `TARGET`, `+5.16`, LCB80 `+0.26`.
8. Khalil Shakir, WR, `$4`: `WATCH`, `+3.21`, LCB80 `-0.85`.
9. Jahmyr Gibbs, RB, `$108`: `WATCH`, `+4.69`, LCB80 `-0.88`.
10. Jordyn Tyson, WR, `$10`: `WATCH`, `+2.94`, LCB80 `-1.14`.

The top four form the substantive target tier. Ranks 5-7 only narrowly clear
zero and should be treated as variation-sensitive; ranks 8-10 are positive-mean
WATCH candidates. Gibbs' ninth-place result is the same variation-14 caution
seen in his direct nomination replay, while the eight-variation direct test is
positive on average. The board is research-only and does not change production.

Five additional Add Evidence-style batches freeze the same 18-player confirmed
cohort and all 37 confirmed price anchors, append independent variations 15-19,
and pool 24 evidence blocks per player/price. The separate 46-player discovery
watchlist is not refreshed because those rows cannot affect the confirmed
ranking. The final confirmed top ten is:

1. Bijan Robinson, RB, `$105`: `TARGET`, `+18.63`, LCB80 `+16.12`.
2. Jahmyr Gibbs, RB, `$108`: `TARGET`, `+15.77`, LCB80 `+13.35`.
3. Josh Allen, QB, `$36`: `TARGET`, `+13.43`, LCB80 `+11.31`.
4. Derrick Henry, RB, `$72`: `TARGET`, `+8.50`, LCB80 `+6.38`.
5. Kyren Williams, RB, `$53`: `TARGET`, `+7.96`, LCB80 `+5.05`.
6. RJ Harvey, RB, `$13`: `TARGET`, `+2.78`, LCB80 `+1.11`.
7. Jonathan Taylor, RB, `$96`: `WATCH`, `+2.22`, LCB80 `-0.10`.
8. Luther Burden III, WR, `$26`: `PASS`, `-0.08`, LCB80 `-2.33`.
9. Lamar Jackson, QB, `$17`: `PASS`, `-0.28`, LCB80 `-2.69`.
10. KC Concepcion, WR, `$4`: `PASS`, `-1.02`, LCB80 `-2.91`.

The pooled evidence preserves Bijan as the top target and changes Gibbs from a
single-batch WATCH to the second-strongest TARGET. The robust tier is Bijan,
Gibbs, Allen, Henry, and Kyren. Harvey clears the action threshold narrowly;
Taylor is essentially on the boundary. The single-batch cheap-WR calls do not
survive accumulation: Burden and Concepcion become PASS.

Windows process corruption prevented one long in-process accumulation. The
equivalent checkpointed runner evaluates each fixed confirmed player/price
cohort in a fresh, seed-preserving process, retries failed children without
counting them, and then uses the app's native evidence accumulator to rerank the
whole cohort. A direct HiGHS formulation matched the existing required-roster
MILP objective on 12 controlled cases. Durable final output is
`results/brown_tuten_bounded_add5_batch6_results.csv`.

This is strong evidence that bounded upgrades are the right implementation
direction across the reconstructed Gibbs and Bijan states. Production promotion
still requires additional prespecified middle- and late-draft states plus
focused App tests.

## Reproduction

The Windows-native exact solver is unstable across repeated solves. The durable
bounded runs therefore compile each variation's baseline plans in a fresh
process, persist them under `results/compiled_plans_variation*.json`, and replay
the solver-free bounded arm in a second fresh process.

```powershell
C:\Users\borys\GitHub\Fantasy_Football_App\streamlitvenv\Scripts\python.exe research\studies\2026-08-20_sequential_budget_reinvestment\run_experiment.py --variations 14 --arms baseline --stable-solver --summary-prefix plan_v14
C:\Users\borys\GitHub\Fantasy_Football_App\streamlitvenv\Scripts\python.exe research\studies\2026-08-20_sequential_budget_reinvestment\run_experiment.py --variations 14 --arms bounded_replan --summary-prefix bounded_v14
python research\studies\2026-08-20_sequential_budget_reinvestment\run_checkpointed_evidence.py --prior-prefix brown_tuten_bounded_add5_batch5 --output-prefix brown_tuten_bounded_add5_batch6 --variation 19 --compute-budget 320 --timeout-seconds 45
```

Durable aggregate evidence is in `results/bounded_aggregate_summary.csv` and
`results/bounded_variation_summary.csv`. The injury sensitivity is summarized
in `results/no_pierce_sensitivity_summary.csv`, and the Gibbs replay is in
`results/gibbs110_brown_tuten_variation_summary.csv`. The full bounded board is
in `results/brown_tuten_bounded_board_v14_results.csv`; its six-batch accumulated
confirmed board is in `results/brown_tuten_bounded_add5_batch6_results.csv`.
