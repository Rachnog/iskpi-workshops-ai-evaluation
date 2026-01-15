---
name: portfolio-comparison
description: Compare different portfolio optimization approaches and present trade-offs. Use when the investor wants to see multiple options or understand alternatives.
---

# Portfolio Comparison Skill

## Comparison Framework

### What to Compare

1. **Optimization methods**: MVO vs HRP
2. **Optimization targets**: min_volatility vs max_sharpe vs max_return
3. **Universes**: conservative vs global_diversified vs aggressive
4. **Constraints**: different max_position limits

## Code Example: Full Comparison

```python
from portfolio_optimizer import (
    PortfolioConfig, optimize_portfolio, optimize_hrp,
    backtest_portfolio, download_market_data, get_universe
)

universe = "global_diversified"
tickers = get_universe(universe)
prices = download_market_data(tickers, "2019-01-01", "2024-01-01")

results = {}

# MVO - Min Volatility
config_mv = PortfolioConfig(
    tickers=list(prices.columns),
    start_date="2019-01-01",
    end_date="2024-01-01",
    optimization_target="min_volatility"
)
opt_mv = optimize_portfolio(config_mv, prices)
bt_mv = backtest_portfolio(opt_mv["weights"], prices)
results["MVO Min Vol"] = {**opt_mv, "backtest": bt_mv}

# MVO - Max Sharpe
config_ms = PortfolioConfig(
    tickers=list(prices.columns),
    start_date="2019-01-01",
    end_date="2024-01-01",
    optimization_target="max_sharpe"
)
opt_ms = optimize_portfolio(config_ms, prices)
bt_ms = backtest_portfolio(opt_ms["weights"], prices)
results["MVO Max Sharpe"] = {**opt_ms, "backtest": bt_ms}

# HRP
opt_hrp = optimize_hrp(prices)
bt_hrp = backtest_portfolio(opt_hrp["weights"], prices)
results["HRP"] = {**opt_hrp, "backtest": bt_hrp}

# Print comparison table
print(f"{'Method':<20} {'Exp Ret':>10} {'Vol':>10} {'Sharpe':>10} {'Max DD':>10}")
print("-" * 60)
for name, r in results.items():
    print(f"{name:<20} {r['expected_return']:>9.2%} {r['volatility']:>9.2%} "
          f"{r['sharpe_ratio']:>9.3f} {r['backtest']['max_drawdown']:>9.2%}")
```

## Presenting Trade-offs

### Risk vs Return Trade-off

| Approach | Expected Return | Risk | Best For |
|----------|----------------|------|----------|
| Min Volatility | Lower | Lowest | Capital preservation |
| Max Sharpe | Balanced | Medium | Risk-adjusted returns |
| Max Return | Highest | Highest | Growth seekers |
| HRP | Balanced | Medium | Diversification |

### When to Recommend Each

- **Min Volatility**: Short horizon, low risk tolerance, near retirement
- **Max Sharpe**: Medium horizon, balanced risk tolerance
- **Max Return**: Long horizon, high risk tolerance, young investors
- **HRP**: Uncertainty about correlations, want robust diversification

## Recommendation Template

```
Based on your profile:
- Time horizon: X years
- Risk tolerance: [Low/Medium/High]
- Investment goal: [Preservation/Growth/Income]

I recommend [Method] because:
1. [Reason aligned with time horizon]
2. [Reason aligned with risk tolerance]
3. [Reason aligned with goal]

Alternative: [Other method] would give you [trade-off explanation].
```
