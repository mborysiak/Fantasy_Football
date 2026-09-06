# LCB-Aligned Structure Validation

The two-batch beta replay starts with Chase Brown at `$34` and Bhayshul Tuten
at `$11`. It accumulated eight evidence blocks and 96 completed confirmation
Buy rollouts per surfaced anchor (`384` conditional outcomes total).

The four highest positive confirmed LCB anchors were:

| Anchor | Price | LCB80 | Mean Buy-minus-Pass gain |
|---|---:|---:|---:|
| Josh Allen | $37 | +16.95 | +20.88 |
| Bijan Robinson | $106 | +13.17 | +17.09 |
| Jahmyr Gibbs | $110 | +6.41 | +12.18 |
| Jonathan Taylor | $96 | +3.39 | +7.78 |

Every conditional example contained its anchor. The most-supported outcome
families were:

- RB-heavy: `252/384` rollouts, led by Gibbs (`89/96`), Bijan (`83/96`), and
  Taylor (`80/96`) Buy branches;
- Premium-QB: `100/384`, including all `96/96` Josh Allen Buy branches; and
- Double-premium-RB: `27/384`.

This resolves the previous display contradiction: Allen, Bijan, and Gibbs now
anchor the structure output because those structures are drawn from the same
completed Buy rosters scored by their LCB80 calculations. No extra roster
optimization is performed. The initial replay exposed that applying the legacy
complete-link player-tier guide to all 384 outcomes made the second batch
unnecessarily expensive (`30.98s`). V19 now computes family support across all
outcomes but builds the detailed player-tier guide from only the four central
anchor examples. A 384-outcome post-change aggregation benchmark completes in
`0.0807s`, so the structure presentation no longer adds material draft-day
latency. A later full timing rerun encountered the repository's known repeated
Windows native-process slowdown and was terminated rather than reported as a
clean comparison.
