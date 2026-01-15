---
name: universe-selection
description: Select appropriate asset universes for portfolio construction based on investor profile, risk tolerance, and investment goals. Use when determining which assets to include in a portfolio.
---

# Universe Selection Skill

## Quick Reference

| Investor Type | Universe | Code |
|---------------|----------|------|
| Conservative (low risk, near retirement, preservation) | `conservative` | `get_universe('conservative')` |
| Balanced (moderate risk, long horizon, diversified) | `global_diversified` | `get_universe('global_diversified')` |
| Aggressive (high risk, growth focus) | `us_tech` | `get_universe('us_tech')` |

## Decision Matrix

| Time Horizon | Risk Tolerance | Recommended Universe |
|-------------|----------------|---------------------|
| < 5 years | Low | `conservative` |
| < 5 years | Medium | `conservative` |
| 5-15 years | Low | `conservative` |
| 5-15 years | Medium | `global_diversified` |
| 5-15 years | High | `us_tech` |
| > 15 years | Low | `global_diversified` |
| > 15 years | Medium | `global_diversified` |
| > 15 years | High | `us_tech` |

## Available Universes

| Universe | Assets | Risk Level |
|----------|--------|------------|
| `conservative` | BND, AGG, TLT, IEF, GOVT, LQD, MBB, VMBS | Low |
| `global_diversified` | SPY, EFA, EEM, VWO, TLT, GLD, VNQ, LQD, HYG, DBC, IEF, GOVT, AGG, BND, VTI | Medium |
| `us_tech` | AAPL, MSFT, GOOG, AMZN, META, NVDA, TSLA, CRM, ADBE, INTC, CSCO, ORCL, IBM, QCOM, AMD | High |

## Ready-to-Run Code

```python
from portfolio_optimizer import get_universe

# For conservative investor (capital preservation, low risk, near retirement)
tickers = get_universe('conservative')
# Returns: ['BND', 'AGG', 'TLT', 'IEF', 'GOVT', 'LQD', 'MBB', 'VMBS']

# For balanced investor (moderate risk, diversification)
tickers = get_universe('global_diversified')
# Returns: ['SPY', 'EFA', 'EEM', 'VWO', 'TLT', 'GLD', 'VNQ', 'LQD', 'HYG', 'DBC', 'IEF', 'GOVT', 'AGG', 'BND', 'VTI']

# For aggressive investor (high risk, growth)
tickers = get_universe('us_tech')
# Returns: ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', ...]
```
