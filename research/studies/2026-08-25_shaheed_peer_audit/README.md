# Shaheed Peer Audit

Reconstruct the displayed 2026 beta draft state with Chase Brown `$34`,
Bhayshul Tuten `$11`, Jordyn Tyson `$7`, and Jonah Coleman `$1` owned. Compare
Rashid Shaheed, Jalen Coker, and Romeo Doubs at their displayed market prices
on the same calculation-v15 construction, auction, and validation evidence.

The audit uses the current beta lineup, roster bounds, waiver baselines, Top-12
salary constraint, selection reserve off, compute budget 320, and random
variation 1 (the default zero-indexed evidence seed).

Run one player per fresh process because long repeated native runs can become
unstable on Windows:

Run:

```powershell
python research\studies\2026-08-25_shaheed_peer_audit\run_audit.py --candidate "Rashid Shaheed" --variation 0
```

Use variations `0` through `8` for the paired sensitivity summarized in
`results/summary.md`.

