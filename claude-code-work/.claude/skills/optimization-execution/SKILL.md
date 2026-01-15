---
name: optimization-execution
description: Execute portfolio optimization using Mean-Variance Optimization (MVO) or Hierarchical Risk Parity (HRP). Use when constructing optimal portfolio weights from a set of assets.
---

# Portfolio Optimization Skill

## Quick Decision Guide

| Investor Profile | Method | Target |
|-----------------|--------|--------|
| Conservative (capital preservation) | MVO | `min_volatility` |
| Balanced (risk-adjusted returns) | MVO | `max_sharpe` |
| Aggressive (maximum growth) | MVO | `max_return` |
| Diversification-focused | HRP | N/A |

## Complete Ready-to-Run Script

Copy and modify UNIVERSE, TARGET, and MAX_POSITION based on investor profile:

```python
from portfolio_optimizer import (
    get_universe, download_market_data,
    PortfolioConfig, optimize_portfolio, backtest_portfolio
)
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURE BASED ON INVESTOR ===
UNIVERSE = 'global_diversified'  # conservative | global_diversified | us_tech
TARGET = 'max_sharpe'            # min_volatility | max_sharpe | max_return
MAX_POSITION = 0.15              # 15% max per asset

# === EXECUTION ===
tickers = get_universe(UNIVERSE)
prices = download_market_data(tickers, '2019-01-01', '2024-01-01')

config = PortfolioConfig(
    tickers=list(prices.columns),
    start_date='2019-01-01',
    end_date='2024-01-01',
    optimization_target=TARGET,
    max_position=MAX_POSITION
)
result = optimize_portfolio(config, prices)
backtest = backtest_portfolio(result['weights'], prices)

# === RESULTS ===
print(f"=== PORTFOLIO RESULTS ===")
print(f"Universe: {UNIVERSE}")
print(f"Optimization: {TARGET}")
print(f"Max Position Limit: {MAX_POSITION:.0%}")
print(f"\n=== METRICS ===")
print(f"Expected Return: {result['expected_return']:.2%}")
print(f"Volatility: {result['volatility']:.2%}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
print(f"Backtest Max Drawdown: {backtest['max_drawdown']:.2%}")
print(f"Backtest Total Return: {backtest['total_return']:.2%}")
print(f"\n=== ALLOCATION ===")
for ticker, weight in sorted(result['weights'].items(), key=lambda x: -x[1]):
    if weight > 0.01:
        print(f"  {ticker}: {weight:.1%}")
```

## Optimization Targets Explained

| Target | What It Does | Use When |
|--------|-------------|----------|
| `min_volatility` | Minimizes portfolio variance | Capital preservation is priority |
| `max_sharpe` | Maximizes risk-adjusted return | Balanced growth with reasonable risk |
| `max_return` | Maximizes expected return | Growth is priority, high risk tolerance |

## Alternative: HRP Optimization

For robust diversification without sensitivity to estimation errors:

```python
from portfolio_optimizer import get_universe, download_market_data, optimize_hrp, backtest_portfolio
import warnings
warnings.filterwarnings('ignore')

tickers = get_universe('global_diversified')
prices = download_market_data(tickers, '2019-01-01', '2024-01-01')
result = optimize_hrp(prices)
backtest = backtest_portfolio(result['weights'], prices)

print(f"HRP Expected Return: {result['expected_return']:.2%}")
print(f"HRP Volatility: {result['volatility']:.2%}")
print(f"HRP Max Drawdown: {backtest['max_drawdown']:.2%}")
```
