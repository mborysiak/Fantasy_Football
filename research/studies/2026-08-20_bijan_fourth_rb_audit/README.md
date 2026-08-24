# Bijan Fourth-RB Auction Audit

## Question

Does the production Sequential Auction policy correctly value buying Bijan
Robinson after the user already owns Jahmyr Gibbs, Chase Brown, and Bhayshul
Tuten, or is the recommendation an artifact of raw PPG, the Top-N constraint,
or incomplete accounting for the cheaper WR completion forced by the purchase?

## Reconstructed state

The primary state is the same three-player state previously frozen in
`2026-07-20_sequential_seed_stability`, updated to the current screenshot and
production database:

- Jahmyr Gibbs at `$110`;
- Chase Brown at `$34`;
- Bhayshul Tuten at `$11`;
- Bijan Robinson evaluated at the current rounded model price;
- Chase Brown and Bhayshul Tuten retained as the user's persisted keepers;
- all other active 2026 beta league keepers unavailable at their persisted
  prices;
- no other off-screen non-keeper sale.

The screenshot shows random variation 14, so that is the primary production
evidence bank. Variation 0, the Top-N constraint, and the selection reserve are
separate sensitivities. If off-screen sales existed before the Bijan decision,
the exact auction-path estimates can change and should be rerun with those
sales entered.

## Logic being tested

- Buy and Pass share auction prices, nomination paths, and managed weekly
  outcomes.
- Completed rosters must contain at least QB1/RB4/WR4/TE1 within 13 total
  players, while weekly scoring starts QB1/RB2/WR2/TE1/FLEX2.
- Bijan is therefore the fourth required roster RB, not a fifth redundant RB.
- Weekly scoring uses lineup decisions that learn from prior weeks, explicit
  player availability, and waiver floors rather than summing roster PPG.
- The Top-N constraint must already be satisfied by Gibbs if he belongs to the
  candidate branch's Top 12; it must not force a second premium player.

## Result

The production recommendation is internally correct, but it should be read as
an auction-policy result rather than proof that Bijan strictly dominates a
deliberate mid-tier-WR plan.

- Current variation 14 at `$105` returns `+27.44` expected managed-season
  points with `6.78` SE, `+21.74` LCB80, 82.98% paired wins, and all four
  evidence blocks positive. The fast price curve gives a `$112` policy max bid.
- Eight standard production variations (0-6 and 14) all return `TARGET` at
  `$105`. Mean lift is `+22.54`, the range is `+9.84` to `+31.64`, and the
  weakest LCB80 remains barely positive at `+0.06`.
- Gibbs is the highest-priced player in the branch Top 12, so he already
  satisfies the Top-N constraint. Turning Top-N off leaves variation 14 nearly
  unchanged (`+27.51`, LCB80 `+19.37`). Relaxing the four-RB roster minimum on
  the exact same evidence bank is also unchanged; neither rule forces Bijan.
- Enabling the optional selection reserve weakens the result to `+12.86` with
  `+4.52` LCB80 and three of four blocks positive, but still returns `TARGET`.
- The primary Buy paths average `$293.64` of roster spend and `107.63` nominal
  starting-lineup PPG. Pass paths average only `$253.35` and `103.81` PPG.
  Common Pass alternatives are Tony Pollard, Christian Watson, and DK Metcalf;
  common Buy completions include Jordan Addison, Makai Lemon, and George
  Kittle. The stochastic comparison is therefore not mechanically two `$35`
  WRs versus two `$5` WRs: Pass often cannot deploy roughly `$45` of the cap
  through the non-anticipating auction policy.
- A separate central-completion counterfactual forces Tee Higgins at `$34` and
  Emeka Egbuka at `$37` after passing on Bijan, then spends essentially the full
  cap in both branches and scores them on the same 256 weekly contexts. Bijan
  wins mean score by only `0.18` season points (`1655.14` versus `1654.96`), an
  effective tie, while the two-WR roster has the higher pooled p10 (`1489.08`
  versus `1473.80`).

The scorer is not summing roster PPG. It uses player availability, learned
weekly lineup decisions, QB1/RB2/WR2/TE1/FLEX2 starts, and waiver floors. No
code, database, or production-policy change follows from this audit. The
remaining modeling question is the already-known late-draft reinvestment and
stranded-salary problem: if the user can deliberately execute the two-WR plan,
Bijan is mean-competitive rather than clearly superior.

## Reproduction

```powershell
C:\Users\borys\GitHub\Fantasy_Football_App\streamlitvenv\Scripts\python.exe research\studies\2026-08-20_bijan_fourth_rb_audit\run_audit.py
```

Results are written under `results/`.
