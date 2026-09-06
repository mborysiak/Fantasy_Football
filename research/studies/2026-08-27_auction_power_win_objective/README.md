# Auction Power-Win Objective Test

This study holds the promising 2025 Beta best-available waiver proxy fixed and
compares roster-selection objectives over a broader candidate frontier:

- `waiver_control`: the additive mean-EV plan used by the preceding waiver arm;
- `mean_frontier`: the highest exact full-roster mean among every candidate;
- `win_g005`: highest paired win-probability LCB within 0.5% of the best mean;
- `excess_g005`: highest paired expected-winning-margin LCB within 0.5%;
- `power_g005`: highest paired power-win LCB within 0.5%;
- matching `g010` arms use a 1.0% mean guardrail as an aggressive sensitivity.

The frozen primary arms rank paired construction-bank LCB80. After that test
collapsed to the exact-mean roster, a labeled exploratory sensitivity also
maximizes the direct point estimate for win probability, expected excess, and
power utility inside the 1.0% guardrail. Those direct arms answer whether the
tail signal transports to independent contexts when the construction-stage
uncertainty penalty is removed; they are not promoted to primary evidence.

A second exploratory follow-up removes the EV guardrail entirely and selects
pure win, expected-excess, and power-win objectives. Matching 50/50 arms blend
block-standardized expected score with the corresponding standardized tail
metric. Standardization is required because season points, probabilities, and
winning-margin points have different native units.

For candidate score `S` and a scenario-specific best-of-11 opponent score `M`:

```text
win probability = P(S > M)
expected excess = E[max(S - M, 0)]
power win = P(S > M) + 0.25 * E[log1p(max(S - M, 0) / 25)]
dominant win = P(S > M + 50)
```

The opponent maximum distribution is evaluated exactly from the common
feasible-roster reference bank with replacement; it is not a binary realized
league tournament. Candidate selection uses paired lower confidence bounds on
the construction bank, and all reported primary comparisons use independent
simulation contexts.

The candidate frontier includes the ordinary full-bank mean solution, every
single construction context, randomized 2/4/8/16-context subsets, and player-
level P75/P90 marginal-value vectors. This lets the tail objectives choose among
meaningfully different rosters rather than merely rerank mean-like solutions.

The 2025 actual season was already inspected in the preceding study. It is
therefore descriptive here and cannot be treated as a fresh holdout. Production
files and app databases are never modified.

The decision readout is in [`results/findings.md`](results/findings.md). The
LCB-selected tail arms retain the exact-mean waiver roster in every block, and
the exploratory direct, pure, and standardized 50/50 objectives all fail to
transport to independent contexts. The aggressive arms mainly transfer spend
from WR to RB; they do not reliably suppress dead-zone veterans.

Run:

```powershell
.\.venv_ff_312\Scripts\python.exe research\studies\2026-08-27_auction_power_win_objective\run_power_win_test.py
```
