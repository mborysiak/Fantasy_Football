# Current-Season Veteran Value Results

## Coverage and estimand

- Long ADP history: 4,206 player-seasons from 2008-2025.
- Current-method auction history: 1,049 player-seasons from 2022-2025.
- Outcomes are current-season only. Managed points sum positive weekly points above the position waiver baseline.
- Bracketed ranges below are 95% player-cluster bootstrap intervals. They describe historical association, not a causal effect of age.

## Long-history matched results

Each veteran is compared with up to five same-position, same-season younger peers matched jointly on preseason PPG and market ADP.

### Top-100 ADP

- RB: 39 veteran-seasons/26 players; managed points +0.3 [-16.6, +15.6], miss rate -7.2 [-27.1, +15.6] pp, upside-hit rate -5.6 [-18.3, +7.0] pp, boom weeks -0.0 [-0.8, +0.6].
- WR: 47 veteran-seasons/26 players; managed points -6.2 [-19.5, +6.0], miss rate +10.2 [-4.4, +25.8] pp, upside-hit rate -0.0 [-14.7, +14.3] pp, boom weeks -0.1 [-0.7, +0.5].
- TE: only 6 adequately matched veteran-seasons/5 players; insufficient for interpretation.

### Top-200 ADP

- RB: 88 veteran-seasons/44 players; managed points -1.9 [-12.2, +8.2], miss rate -5.2 [-16.4, +7.4] pp, upside-hit rate -1.1 [-10.2, +7.2] pp, boom weeks -0.1 [-0.8, +0.4].
- WR: 88 veteran-seasons/43 players; managed points -2.7 [-10.0, +4.2], miss rate +3.6 [-5.2, +13.6] pp, upside-hit rate +4.3 [-4.1, +12.3] pp, boom weeks -0.0 [-0.4, +0.4].
- TE: 43 veteran-seasons/19 players; managed points +8.0 [-5.2, +18.5], miss rate -9.8 [-24.9, +8.6] pp, upside-hit rate +2.3 [-12.2, +16.4] pp, boom weeks -0.2 [-0.8, +0.3].

## Exact recent auction evidence

These matches use rolling v5 predicted auction salary rather than ADP. The full pool is usable as a direction check; the `$5+` veteran cells are too small for a stable production penalty.

### All modeled salaries

- RB: 19 veteran-seasons/16 players; managed points -1.4 [-18.3, +12.1], miss rate -3.2 [-25.7, +22.4] pp, upside-hit rate +2.1 [-10.0, +15.7] pp.
- WR: 25 veteran-seasons/15 players; managed points +0.5 [-12.1, +14.1], miss rate +4.8 [-15.2, +27.8] pp, upside-hit rate +17.6 [+6.7, +28.6] pp.
- TE: only 10 adequately matched veteran-seasons/7 players; insufficient for interpretation.

### Predicted salary at least $5

- RB: only 8 adequately matched veteran-seasons/7 players; insufficient for interpretation.
- WR: only 9 adequately matched veteran-seasons/5 players; insufficient for interpretation.
- TE: only 1 adequately matched veteran-seasons/1 players; insufficient for interpretation.

## Current 2026 named-player market context

- Alvin Kamara: raw experience 9, modeled experience 8, 7.15 PPG, market $1.3; salary is -0.9 versus the mean of five younger same-position projection peers.
- Derrick Henry: raw experience 10, modeled experience 8, 15.67 PPG, market $73.3; salary is +4.9 versus the mean of five younger same-position projection peers.
- George Kittle: raw experience 9, modeled experience 9, 8.86 PPG, market $5.1; salary is -3.0 versus the mean of five younger same-position projection peers.
- Travis Kelce: raw experience 13, modeled experience 10, 8.57 PPG, market $5.0; salary is -1.6 versus the mean of five younger same-position projection peers.

## Interpretation

- A blanket RB/WR/TE current-season age tax is not supported. Long-run managed value is close to younger matched peers for RB, modestly lower for WR, and not reliably estimable for premium TE.
- Premium veteran RBs look more compressed than worse: mean value is neutral, with both miss and upside-hit rates modestly lower than matched peers. That can justify an explicit ceiling preference, not an expected-value haircut.
- Premium veteran WRs provide the only recurring warning: fewer managed points and a higher miss rate in every leave-one-season-out Top-100 match. Player-cluster intervals remain wide, so this is a candidate for prospective template testing rather than a calibrated penalty.
- The market does not apply one uniform veteran discount. Projection-matched historical RB/WR veterans were drafted at nearly the same Top-100 ADP as younger peers; individual 2026 TE prices are discounted, while Derrick Henry is not cheaper than his projection peers.
- The recent `$5+` auction sample is too small and directionally unstable to estimate a dollar penalty: it contains only a handful of above-threshold RB/TE seasons.
- Do not alter current point forecasts from this evidence. If production changes, first validate a premium-WR current-outcome mixture. Express any broader veteran fade as an explicit ceiling/risk utility preference using uncapped experience, not as a claimed forecast correction.
