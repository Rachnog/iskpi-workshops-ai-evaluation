---
name: risk-assessment
description: Assess portfolio risk using volatility, drawdown, and other risk metrics. Use when evaluating the risk characteristics of a portfolio or comparing risk profiles.
---

# Risk Assessment Skill

## Key Risk Metrics

| Metric | Description | Calculation | Good Value |
|--------|-------------|-------------|------------|
| **Volatility** | Annualized standard deviation | σ × √252 | < 15% conservative, < 25% aggressive |
| **Max Drawdown** | Worst peak-to-trough decline | min(cumulative returns) | > -20% conservative |
| **Sharpe Ratio** | Risk-adjusted return | (R - Rf) / σ | > 1.0 good, > 2.0 excellent |
| **Sortino Ratio** | Downside risk-adjusted return | (R - Rf) / σ_down | > 1.5 good |
| **Value at Risk (VaR)** | Maximum expected loss at confidence | Percentile of returns | 95% VaR < 2% daily |

## Risk Assessment Process

### Step 1: Analyze Volatility
```python
import numpy as np

returns = prices.pct_change().dropna()
volatility = returns.std() * np.sqrt(252)  # Annualized
print(f"Annualized Volatility: {volatility:.2%}")
```

### Step 2: Calculate Maximum Drawdown
```python
def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

max_dd = calculate_max_drawdown(portfolio_returns)
print(f"Maximum Drawdown: {max_dd:.2%}")
```

### Step 3: Compute Sharpe Ratio
```python
risk_free_rate = 0.02  # 2% annual
excess_return = portfolio_return - risk_free_rate
sharpe = excess_return / volatility
print(f"Sharpe Ratio: {sharpe:.3f}")
```

## Risk Tolerance Mapping

| Risk Tolerance | Target Volatility | Max Drawdown Tolerance | Suitable Universe |
|---------------|-------------------|------------------------|-------------------|
| Low | < 10% | > -15% | conservative |
| Medium | 10-18% | > -25% | global_diversified |
| High | 18-30% | > -40% | us_tech, aggressive |

## Risk Flags

Warn the investor if:
- Volatility exceeds their stated tolerance by > 20%
- Max drawdown exceeds tolerance threshold
- Sharpe ratio < 0.5 (poor risk-adjusted returns)
- Single position exceeds max_position constraint
