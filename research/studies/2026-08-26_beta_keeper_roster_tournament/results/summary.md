# Beta Keeper Roster Tournament Results

All three scenarios use 8 shared construction blocks, 24 hidden auction paths per block, and 128 held-out managed seasons per completed roster.

## Aggregate managed-season result

| Keeper start | EV | Delta vs Tuten | P10 | P10 delta | Completion | Avg spend | Avg QB/RB/WR/TE |
|---|---:|---:|---:|---:|---:|---:|---|
| Brown + Tuten | 1655.84 | +0.00 | 1469.36 | +0.00 | 100.0% | $271.78 | 1.00/5.68/4.46/1.86 |
| Brown + Loveland | 1652.75 | -3.08 | 1467.75 | -1.60 | 100.0% | $269.03 | 1.00/5.94/4.84/1.22 |
| Brown + Burden | 1647.16 | -8.68 | 1463.02 | -6.33 | 100.0% | $267.49 | 1.00/5.83/4.42/1.74 |

## Paired differences versus Brown + Tuten

| Scenario | Mean delta | P10 of paired delta | Win rate | Positive blocks | LCB80 |
|---|---:|---:|---:|---:|---:|
| Brown + Burden | -8.68 | -154.09 | 46.9% | 2/8 | -12.35 |
| Brown + Loveland | -3.08 | -146.84 | 49.3% | 1/8 | -5.80 |

## Representative completed rosters

### Brown + Burden — 1646.7 EV, 1471.8 P10, $268 spend

- QB: Lamar Jackson
- RB: Chase Brown, Chuba Hubbard, D'Andre Swift, Derrick Henry, Jordan Mason, TreVeyon Henderson
- WR: Christian Watson, Emeka Egbuka, Luther Burden III, Marvin Harrison Jr.
- TE: Harold Fannin Jr., Travis Kelce

### Brown + Loveland — 1652.8 EV, 1464.6 P10, $269 spend

- QB: Jalen Hurts
- RB: Aaron Jones, Chase Brown, David Montgomery, Jaylen Warren, Kyren Williams, Tony Pollard
- WR: DJ Moore, DK Metcalf, Emeka Egbuka, Rome Odunze
- TE: Colston Loveland, George Kittle

### Brown + Tuten — 1655.4 EV, 1478.1 P10, $266 spend

- QB: Josh Allen
- RB: Aaron Jones, Bhayshul Tuten, Chase Brown, Jaylen Warren, Omarion Hampton, Tony Pollard
- WR: Christian Watson, Emeka Egbuka, Jameson Williams, Makai Lemon, Michael Pittman
- TE: George Kittle

## Frequent non-fixed targets

- **Brown + Tuten:** Jameson Williams (53%, $21.3); Harold Fannin Jr. (42%, $8.6); Derrick Henry (39%, $68.1); Christian Watson (38%, $15.8); Kyle Pitts (35%, $8.8); George Kittle (32%, $3.9); Josh Allen (31%, $34.6); Emeka Egbuka (30%, $32.5); Luther Burden III (30%, $21.8); Tony Pollard (27%, $16.7)
- **Brown + Burden:** Jameson Williams (51%, $21.5); Harold Fannin Jr. (42%, $8.7); Derrick Henry (37%, $68.3); Christian Watson (35%, $15.9); Jaylen Warren (33%, $15.1); Kyle Pitts (33%, $8.8); Tony Pollard (32%, $17.2); TreVeyon Henderson (32%, $23.0); George Kittle (30%, $4.1); Rhamondre Stevenson (29%, $19.1)
- **Brown + Loveland:** Jameson Williams (57%, $21.4); Christian Watson (45%, $15.9); Derrick Henry (40%, $68.1); Jaylen Warren (40%, $15.4); Tony Pollard (34%, $17.1); Alec Pierce (32%, $8.4); Rhamondre Stevenson (29%, $18.5); Josh Allen (29%, $35.1); Rico Dowdle (29%, $12.8); TreVeyon Henderson (28%, $23.0)

## Interpretation boundary

This isolates current-season roster value under the current Sequential policy. It does not add a separate next-year keeper-option bonus. The selection-premium reserve is disabled because the stored calibration belongs to the production Brown/Tuten keeper state.
