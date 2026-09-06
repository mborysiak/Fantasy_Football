# Beta Joint, Shape, and Waiver Results

## State and comparison

The paired replay starts with Chase Brown at `$34` and Bhayshul Tuten at `$11`
under the 2026 beta lineup (QB1/RB2/WR2/TE1/FLEX2). The other 12 active beta
keepers are unavailable. All four arms reuse four independent 32-context
construction banks and four paired 64-context holdouts.

Cross-arm quality is always scored with the current waiver baselines:
QB `13.5`, RB `6.2`, WR `6.9`, and TE `6.3` PPG. The waiver sensitivity
constructs with `15.0`/`7.7`/`8.4`/`7.8`, but does not receive those extra
points in the common comparison.

## Aggregate result

| Arm | Mean | Delta | P10 | Delta | Avg QB/RB/WR/TE | Forecast spend |
|---|---:|---:|---:|---:|---|---:|
| Current additive | 1641.68 | — | 1457.93 | — | 1 / 6.00 / 4.25 / 1.75 | 297.66 |
| Joint one swap | 1640.71 | -0.97 | 1470.26 | +12.33 | 1 / 5.75 / 4.75 / 1.50 | 295.12 |
| Fixed shape additive | 1636.74 | -4.94 | 1456.83 | -1.10 | 1 / 5.75 / 5.25 / 1 | 297.80 |
| Waiver +1.5 additive | 1631.33 | -10.34 | 1457.44 | -0.49 | 1 / 6.00 / 4.50 / 1.50 | 297.68 |

One exact joint swap was accepted in three of four blocks and improved its own
construction-bank objective by 3.23 season points on average. Independent mean
quality is essentially flat-to-slightly lower, while average p10 improves 12.33
points. The effect is heterogeneous rather than a clean promotion result.

## Current versus joint rosters

### Block 0

Current roster (QB1/RB6/WR4/TE2):

- QB: Jayden Daniels
- RB: De'Von Achane, D'Andre Swift, Chase Brown, Jacory Croskey-Merritt,
  Jordan Mason, Bhayshul Tuten
- WR: Jameson Williams, Christian Watson, Marvin Harrison Jr., Jordan Addison
- TE: Harold Fannin Jr., Dallas Goedert

Joint change: **Dallas Goedert -> Rashid Shaheed**, producing QB1/RB6/WR5/TE1.
Holdout mean changes `1650.07 -> 1651.31`; p10 changes
`1449.58 -> 1484.10`.

### Block 1

Current roster (QB1/RB6/WR4/TE2):

- QB: Dak Prescott
- RB: Jahmyr Gibbs, D'Andre Swift, Chase Brown, J.K. Dobbins, Bhayshul Tuten,
  Aaron Jones
- WR: Christian Watson, DK Metcalf, Alec Pierce, Quentin Johnston
- TE: Harold Fannin Jr., Dalton Kincaid

Joint change: **none**. The additive roster was already a one-swap local
optimum, so its holdout mean/p10 remain `1659.72`/`1473.58`.

### Block 2

Current roster (QB1/RB6/WR4/TE2):

- QB: Jaxson Dart
- RB: Derrick Henry, Chase Brown, Jaylen Warren, J.K. Dobbins, Chuba Hubbard,
  Bhayshul Tuten
- WR: Amon-Ra St. Brown, Brian Thomas Jr., Alec Pierce, Rashid Shaheed
- TE: Dalton Kincaid, Juwan Johnson

Joint change: **Chuba Hubbard -> Makai Lemon**, producing QB1/RB5/WR5/TE2.
Holdout mean changes `1628.70 -> 1624.44`; p10 changes
`1431.88 -> 1429.73`.

### Block 3

Current roster (QB1/RB6/WR5/TE1):

- QB: Jaxson Dart
- RB: Bijan Robinson, Chase Brown, Jaylen Warren, J.K. Dobbins, Jordan Mason,
  Bhayshul Tuten
- WR: Tee Higgins, Michael Wilson, Alec Pierce, Makai Lemon, Jayden Reed
- TE: Harold Fannin Jr.

Joint change: **Alec Pierce -> Josh Downs**, leaving the same position counts.
Holdout mean changes `1628.22 -> 1627.38`; p10 changes
`1476.69 -> 1493.62`.

Unlike NV, beta cannot add a second or third QB because its current maximum is
QB1. The shared-opportunity correction therefore acts mainly on the second TE
or sixth RB and redirects the slot toward WR depth.

## Fixed-shape result

The requested QB1/TE1/RB5-6/WR5-6 constraint is already satisfied in block 3.
It changes the other three additive rosters, but averages `-4.94` mean and
`-1.10` p10 versus current. Block 1 improves materially; blocks 0 and 2 worsen.
The hard shape gets the intended composition but is too rigid to improve the
paired aggregate.

| Block | Removed from current | Added to current | Mean delta | P10 delta |
|---:|---|---|---:|---:|
| 0 | Dallas Goedert; Jameson Williams; Jayden Daniels | Brian Thomas Jr.; Luther Burden III; Patrick Mahomes | -15.83 | -15.29 |
| 1 | D'Andre Swift; DK Metcalf; Dak Prescott; Dalton Kincaid | Josh Allen; KC Concepcion; Marvin Harrison Jr.; Tyler Allgeier | +5.82 | +19.61 |
| 2 | Chuba Hubbard; Juwan Johnson | Chris Godwin Jr.; Matthew Golden | -9.76 | -8.73 |
| 3 | none | none | 0.00 | 0.00 |

## Waiver +1.5 result

Raising every waiver baseline does not reproduce the joint correction. The
roster still averages six RBs, shifts only 0.25 slot from TE to WR, and loses
10.34 mean points under the common current-waiver holdout. Its p10 is nearly
flat (`-0.49`).

If scored under its own raised-waiver assumption, the arm appears to gain
roughly 9.44 mean and 33.53 p10 versus the current arm. That comparison is not
valid evidence of a better roster: it mostly reflects awarding every lineup a
higher replacement score. The common-authority result is the relevant test.

| Block | Removed from current | Added to current | Mean delta | P10 delta |
|---:|---|---|---:|---:|
| 0 | Jacory Croskey-Merritt; Jameson Williams; Jordan Addison; Marvin Harrison Jr. | Antonio Williams; Denzel Boston; Josh Jacobs; Rashid Shaheed | -29.46 | -1.38 |
| 1 | Aaron Jones; Christian Watson; DK Metcalf; Dak Prescott; Dalton Kincaid | Josh Allen; Omar Cooper Jr.; Rashid Shaheed; Sam LaPorta; Tyler Allgeier | +4.49 | -9.36 |
| 2 | Alec Pierce; Dalton Kincaid; Juwan Johnson | Brenton Strange; Chris Godwin Jr.; Matthew Golden | -16.41 | +8.78 |
| 3 | none | none | 0.00 | 0.00 |

## Static compile timing

This stripped construction harness averages `0.0036s` for the current additive
solve, `0.0043s` for the fixed-shape additive solve, `0.0040s` for the raised-
waiver additive solve, and `0.1814s` for the exact one-swap solve. Thus the
shortcuts preserve additive-solve speed, while the exact joint correction is
about `51x` slower at this isolated compile step. These are not end-to-end
Target Board timings: production applies exact refinement only to confirmation
candidates, where the prior production-shaped NV measurement was roughly
`1.4-1.6x` additive rather than `51x` for a complete board.

## Decision

- Keep current beta position constraints and waiver baselines.
- Do not replace joint opportunity accounting with a blanket +1.5 waiver bump.
- Do not hard-code the requested 5/5 plus RB-or-WR shape from this evidence.
- Treat the beta one-swap result as a promising p10/diversification sensitivity,
  not a mean-quality promotion. The live App v14 confirmation-only exact swap
  can remain active, but this four-block beta slice does not justify extending
  joint refinement into broad screening or multiple swaps.
