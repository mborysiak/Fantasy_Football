# V2 Regularized Linear Sparsity Results

Negative comparison deltas favor the challenger. Confidence intervals bootstrap the nine validation-season RMSE differences.

## Pooled OOF

| Model | Features | RMSE |
|---|---:|---:|
| `direct_lasso_expanded` | 47 | 3.1615 |
| `direct_elastic_net_expanded` | 47 | 3.1651 |
| `direct_lasso_incumbent` | 35 | 3.1656 |
| `direct_elastic_net_incumbent` | 35 | 3.1699 |
| `direct_ridge_incumbent` | 35 | 3.1747 |
| `direct_ridge_expanded` | 47 | 3.1807 |

## Paired season comparisons

| Comparison | Pooled delta | Mean season delta | 95% interval | Wins |
|---|---:|---:|---:|---:|
| `lasso_vs_ridge_incumbent` | -0.0091 | -0.0092 | [-0.0238, +0.0027] | 5/9 |
| `elastic_net_vs_ridge_incumbent` | -0.0049 | -0.0050 | [-0.0174, +0.0057] | 5/9 |
| `lasso_expanded_vs_incumbent` | -0.0041 | -0.0038 | [-0.0233, +0.0159] | 5/9 |
| `elastic_net_expanded_vs_incumbent` | -0.0048 | -0.0044 | [-0.0256, +0.0177] | 6/9 |
| `ridge_expanded_vs_incumbent` | +0.0060 | +0.0062 | [-0.0159, +0.0349] | 5/9 |

## Sparsity

| Model | Mean raw selected | Raw range | Raw fraction | Mean indicators selected |
|---|---:|---:|---:|---:|
| `direct_elastic_net_expanded` | 32.4 | 24-39 | 68.8% | 19.4 |
| `direct_elastic_net_incumbent` | 26.5 | 18-33 | 75.7% | 15.8 |
| `direct_lasso_expanded` | 27.3 | 22-36 | 58.0% | 6.9 |
| `direct_lasso_incumbent` | 23.6 | 17-30 | 67.6% | 8.6 |
| `direct_ridge_expanded` | 43.4 | 43-47 | 92.4% | 27.4 |
| `direct_ridge_incumbent` | 32.3 | 32-35 | 92.4% | 19.3 |
