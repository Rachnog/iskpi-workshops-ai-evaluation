---
name: backtesting
description: Backtest portfolio performance using historical data. Use when validating portfolio strategy with historical returns, calculating realized metrics, and stress testing.
---

# Backtesting Skill

## Backtest Process

### Step 1: Run Backtest

```python
from portfolio_optimizer import backtest_portfolio

# weights: dict of {ticker: weight}
# prices: DataFrame of historical prices
backtest_result = backtest_portfolio(weights, prices)

print(f"Cumulative Return: {backtest_result['cumulative_return']:.2%}")
print(f"Annualized Return: {backtest_result['annualized_return']:.2%}")
print(f"Volatility: {backtest_result['volatility']:.2%}")
print(f"Sharpe Ratio: {backtest_result['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {backtest_result['max_drawdown']:.2%}")
```

## Backtest Metrics Explained

| Metric | What It Tells You |
|--------|-------------------|
| `cumulative_return` | Total return over the period |
| `annualized_return` | Yearly average return |
| `volatility` | Realized risk (annualized std dev) |
| `sharpe_ratio` | Realized risk-adjusted return |
| `max_drawdown` | Worst historical decline |

## Interpreting Results

### Expected vs Realized Comparison

| Comparison | Interpretation |
|------------|----------------|
| Realized Sharpe > Expected | Strategy performed better than predicted |
| Realized Sharpe < Expected | Estimation error or regime change |
| Realized Vol > Expected | Higher risk than anticipated |
| Realized Vol < Expected | Lower risk (good if returns maintained) |

### Warning Signs

- **Large gap between expected and realized**: Model assumptions may be wrong
- **Max drawdown > -30%**: May be too risky for conservative investors
- **Negative Sharpe ratio**: Strategy lost money risk-adjusted

## Best Practices

1. **Use sufficient history**: At least 3-5 years of data
2. **Include market stress periods**: 2008, 2020, 2022
3. **Compare to benchmark**: SPY for equity, AGG for bonds
4. **Check for survivorship bias**: Include delisted securities if possible

## Code Example: Full Backtest Analysis

```python
from portfolio_optimizer import (
    PortfolioConfig, optimize_portfolio,
    backtest_portfolio, download_market_data, get_universe
)

# Setup
tickers = get_universe("global_diversified")
prices = download_market_data(tickers, "2019-01-01", "2024-01-01")

# Optimize
config = PortfolioConfig(
    tickers=list(prices.columns),
    start_date="2019-01-01",
    end_date="2024-01-01",
    optimization_target="max_sharpe"
)
opt_result = optimize_portfolio(config, prices)

# Backtest
bt_result = backtest_portfolio(opt_result["weights"], prices)

# Compare
print(f"Expected Return: {opt_result['expected_return']:.2%}")
print(f"Realized Return: {bt_result['annualized_return']:.2%}")
print(f"Expected Sharpe: {opt_result['sharpe_ratio']:.3f}")
print(f"Realized Sharpe: {bt_result['sharpe_ratio']:.3f}")
```
