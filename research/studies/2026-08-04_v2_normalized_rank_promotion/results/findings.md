# Normalized Expert-Rank Promotion Decision

- Stage A nested point/distribution gate: `FAIL`.
- Recent season wins: `3/6`.
- Decision: retain normalized expert rank outside production.
- Stage B template/roster transport: not run because Stage A did not authorize
  it.

| League | Pooled RMSE delta | Recent RMSE delta | Player 95% | Player CRPS delta | Max position delta |
|---|---:|---:|---:|---:|---:|
| DK | -0.00181 | -0.00370 | [-0.00452, +0.00087] | -0.00112 | +0.00101 |
| Beta | +0.00008 | +0.00004 | [-0.00302, +0.00313] | +0.00042 | +0.01280 QB |

The DK recent aggregate gain comes entirely from 2025: 2023 and 2024 worsen.
Beta improves in 2024 and 2025 but worsens in 2023. Nested reselection changes
8/30 DK and 13/30 beta component-origin selections, demonstrating that the
earlier locked-hyperparameter attribution result did not survive the final
model-selection surface consistently.

Residual 50%/80% coverage remains acceptable in both leagues. That does not
override the failed point, uncertainty, beta-position, and 5-of-6 replication
gates.

Both production database hashes matched before and after. No production model,
parameter cache, template, SQLite table, or app artifact changed.
