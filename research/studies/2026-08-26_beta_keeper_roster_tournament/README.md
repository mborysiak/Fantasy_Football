# Beta Keeper Roster Tournament

## Purpose

Compare the current-season roster opportunity cost of three possible `$11`
second keepers while Chase Brown remains fixed at `$34`:

- Bhayshul Tuten;
- Luther Burden III; and
- Colston Loveland.

The comparison uses the governed four-player non-keeper salary rebuild from
`../2026-08-26_beta_nonkeeper_salary_counterfactual/`. The chosen candidate is
fixed at `$11`; the other two remain draftable at their counterfactual market
prices. The other 12 active beta keepers remain unavailable in every arm. Each
arm therefore has 14 league keepers spending `$441`, 142 remaining market
slots, and `$3,135` in remaining market budget.

## Method

This is not a perfect-knowledge roster ILP. It uses the current additive
history-only Sequential policy with bounded same-position reinvestment. The
three arms share:

- current beta weekly templates and waiver baselines;
- the same construction banks;
- the same hidden nomination orders and underlying salary draws;
- the same held-out managed-season scoring banks; and
- the current QB1/RB4-6/WR4-6/TE1-2 roster constraints.

Salary draws use the counterfactual centers. The existing residual shapes are
retained for unchanged market players. Tuten's formerly deterministic keeper
row receives an explicit monotone residual approximation anchored to the
counterfactual model's exact min, P10, P90, and max. The current selection
premium is disabled because it was calibrated on the Brown/Tuten keeper state
and is not valid for the Burden or Loveland counterfactuals.

## Run

```powershell
.\.venv_ff_312\Scripts\python.exe `
  research/studies/2026-08-26_beta_keeper_roster_tournament/run_tournament.py
```

Defaults use eight construction blocks, 24 hidden auction paths per block, and
128 independent managed-season contexts per completed roster.

## Result

Brown plus Tuten is the current-season winner under this policy. Across 192
completed hidden-auction paths and 24,576 paired scoring cells per arm, its
managed-season EV is `1,655.84`, versus `1,652.75` for Brown/Loveland and
`1,647.16` for Brown/Burden. Tuten beats Loveland in seven of eight construction
blocks and Burden in six of eight; the mean advantages are `+3.08` and `+8.68`
points, respectively. The differences are useful but small relative to the
roughly 145-point season-level outcome standard deviation, so this is a Tuten
preference rather than evidence that Loveland is categorically wrong.

The roster mechanism is consistent with the modeled keeper discounts. Tuten's
counterfactual market center is `$30.31`, creating `$19.31` of salary surplus at
the `$11` contract, versus `$16.03` for Loveland and `$14.25` for Burden. The
Tuten arm consequently spends less at RB on average (`$157`) and reallocates to
WR (`$82`) and TE (`$15`) while retaining a complete legal roster. Loveland
provides positional scarcity but can be reproduced from the market cheaply
enough in the scenarios where the policy wants another TE; Burden is also
drafted selectively when he clears below his modeled center.

See `results/summary.md` for aggregate scores, paired uncertainty, representative
rosters, and recurring non-fixed targets.
