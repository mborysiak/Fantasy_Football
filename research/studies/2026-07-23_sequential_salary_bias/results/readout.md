# Blind sequential salary-bias replay

This rolling replay keeps the construction draw and nomination order paired within each origin. The static arm sees its entire sampled salary surface; the Sequential Target arm sees replay prices only as nominations occur.

## Headline

**The oracle/scenario-shopping component largely disappears, but the player-level residual bias does not.**

- Static full-surface selection spent **$291.8** on the sampled surface, but those players cost **$306.2** at point prices and **$323.0** historically. That is a **$14.4** scenario-shopping discount plus a **$16.8** actual-minus-point residual.
- Blind Sequential Target's initial plan cost **$283.3** at point prices and **$300.6** historically: a **$17.3** residual. That residual is essentially the same size as the static arm, but the plan is not built around an unusually cheap full salary draw.
- Consequently, initial-plan historical-cap feasibility improved from **14.1%** static to **43.0%** blind.
- With the half reserve, the blind initial plan fell to **$294.4** historical spend and **52.3%** feasibility.
- After live recourse, blind no-reserve completion was **93.8%** and every completed roster was legal, but it paid only **$252.6** and left **$45.4** unused on average. Half-reserve completion was **95.3%**.

The initial-plan comparison is the clean test of selection concentration. The acquired-roster comparison includes the benefit of observing prices and pivoting during the auction. A legal completed sequential roster is also audited against its actual paid p+1 spend. The acquired-roster gap of **$-8.0** therefore reflects genuine recourse, but the large unused budget says the current replay policy overcorrects and should not yet be read as an optimal spending policy.

## Design limits

- Historical nomination order and losing bids are unavailable; orders are current production-style noisy salary orderings.
- Historical clearing prices are treated as exogenous. The policy does not model opponents reacting to our purchases.
- Missing historical salaries retain the established $1 fallback and are reported through recorded-salary coverage.
- Four season origins are the independent time units; trial variation is paired Monte Carlo precision, not four dozen independent seasons.

Rows evaluated: 512. See `arm_summary.csv`, `paired_effects.csv`, and `player_selection_rates.csv` for the full decomposition.
