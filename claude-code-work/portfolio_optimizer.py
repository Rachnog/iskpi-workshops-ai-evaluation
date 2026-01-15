"""
Portfolio Optimization Module

Shared utility functions for portfolio optimization used across all notebooks.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, HRPOpt, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimization."""
    tickers: List[str]
    start_date: str
    end_date: str
    optimization_target: str  # 'min_volatility', 'max_sharpe', 'max_return'
    weight_bounds: Tuple[float, float] = (0, 1)
    risk_free_rate: float = 0.02
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    max_position: Optional[float] = None
    min_constituents: Optional[int] = None


def download_market_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical price data for given tickers."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # Handle different yfinance column formats
    if isinstance(data.columns, pd.MultiIndex):
        # Check which column names are available
        if 'Adj Close' in data.columns.get_level_values(0):
            prices = data['Adj Close']
        elif 'Close' in data.columns.get_level_values(0):
            prices = data['Close']
        else:
            # Fallback: use first level of MultiIndex
            prices = data.iloc[:, data.columns.get_level_values(0) == data.columns.get_level_values(0)[0]]
            if isinstance(prices.columns, pd.MultiIndex):
                prices.columns = prices.columns.droplevel(0)
    else:
        # Single ticker case
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            prices = data

    # Ensure we have a DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0] if len(tickers) == 1 else 'Price')

    return prices.dropna()


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from prices."""
    return prices.pct_change().dropna()


def estimate_expected_returns(prices: pd.DataFrame, method: str = 'mean_historical') -> pd.Series:
    """Estimate expected returns using various methods."""
    if method == 'mean_historical':
        return expected_returns.mean_historical_return(prices)
    elif method == 'ema_historical':
        return expected_returns.ema_historical_return(prices)
    elif method == 'capm':
        return expected_returns.capm_return(prices)
    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_covariance(prices: pd.DataFrame, method: str = 'sample') -> pd.DataFrame:
    """Estimate covariance matrix using various methods."""
    if method == 'sample':
        return risk_models.sample_cov(prices)
    elif method == 'semicovariance':
        return risk_models.semicovariance(prices)
    elif method == 'exp_cov':
        return risk_models.exp_cov(prices)
    elif method == 'ledoit_wolf':
        return risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    else:
        raise ValueError(f"Unknown method: {method}")


def optimize_portfolio(
    config: PortfolioConfig,
    prices: pd.DataFrame,
    returns_method: str = 'mean_historical',
    cov_method: str = 'sample'
) -> Dict[str, Any]:
    """
    Optimize portfolio based on configuration.

    Returns dict with weights, performance metrics, and optimization details.
    """
    mu = estimate_expected_returns(prices, returns_method)
    S = estimate_covariance(prices, cov_method)

    # Apply position constraints if specified
    weight_bounds = config.weight_bounds
    if config.max_position is not None:
        weight_bounds = (0, config.max_position)

    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

    # Apply optimization target
    if config.optimization_target == 'min_volatility':
        weights = ef.min_volatility()
    elif config.optimization_target == 'max_sharpe':
        weights = ef.max_sharpe(risk_free_rate=config.risk_free_rate)
    elif config.optimization_target == 'max_return':
        if config.target_volatility is None:
            raise ValueError("target_volatility required for max_return optimization")
        weights = ef.efficient_risk(config.target_volatility)
    elif config.optimization_target == 'efficient_return':
        if config.target_return is None:
            raise ValueError("target_return required for efficient_return optimization")
        weights = ef.efficient_return(config.target_return)
    else:
        raise ValueError(f"Unknown optimization target: {config.optimization_target}")

    cleaned_weights = ef.clean_weights()
    performance = ef.portfolio_performance(
        verbose=False,
        risk_free_rate=config.risk_free_rate
    )

    return {
        'weights': dict(cleaned_weights),
        'expected_return': performance[0],
        'volatility': performance[1],
        'sharpe_ratio': performance[2],
        'optimization_target': config.optimization_target,
        'returns_method': returns_method,
        'cov_method': cov_method
    }


def optimize_hrp(prices: pd.DataFrame, linkage_method: str = 'single') -> Dict[str, Any]:
    """
    Optimize portfolio using Hierarchical Risk Parity.

    Returns dict with weights, clusters, and performance metrics.
    """
    returns = calculate_returns(prices)
    hrp = HRPOpt(returns=returns)
    weights = hrp.optimize(linkage_method=linkage_method)
    cleaned_weights = hrp.clean_weights()
    performance = hrp.portfolio_performance(verbose=False)

    return {
        'weights': dict(cleaned_weights),
        'expected_return': performance[0],
        'volatility': performance[1],
        'sharpe_ratio': performance[2],
        'clusters': hrp.clusters,
        'linkage_method': linkage_method
    }


def plot_efficient_frontier(
    prices: pd.DataFrame,
    optimized_portfolios: Dict[str, Dict],
    n_points: int = 100,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot efficient frontier with optimized portfolios marked."""
    mu = estimate_expected_returns(prices)
    S = estimate_covariance(prices)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot efficient frontier
    ef = EfficientFrontier(mu, S)
    ef_returns = []
    ef_risks = []

    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    for tr in target_returns:
        try:
            ef_temp = EfficientFrontier(mu, S)
            ef_temp.efficient_return(tr)
            perf = ef_temp.portfolio_performance(verbose=False)
            ef_risks.append(perf[1])
            ef_returns.append(perf[0])
        except Exception:
            continue

    ax.plot(ef_risks, ef_returns, 'b-', linewidth=2, label='Efficient Frontier')

    # Plot optimized portfolios
    colors = ['red', 'green', 'orange', 'purple']
    markers = ['*', 's', '^', 'D']

    for idx, (name, portfolio) in enumerate(optimized_portfolios.items()):
        ax.scatter(
            portfolio['volatility'],
            portfolio['expected_return'],
            marker=markers[idx % len(markers)],
            s=200,
            c=colors[idx % len(colors)],
            label=f"{name} (SR: {portfolio['sharpe_ratio']:.2f})",
            zorder=5
        )

    ax.set_xlabel('Volatility (Std Dev)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Efficient Frontier with Optimized Portfolios')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_hrp_dendrogram(
    prices: pd.DataFrame,
    linkage_method: str = 'single',
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """Plot dendrogram for Hierarchical Risk Parity clustering."""
    returns = calculate_returns(prices)
    corr = returns.corr()

    # Compute distance matrix
    dist = np.sqrt((1 - corr) / 2)
    dist_condensed = dist.values[np.triu_indices(len(dist), k=1)]

    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(dist_condensed, method=linkage_method)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Dendrogram
    sch.dendrogram(
        linkage_matrix,
        labels=corr.index.tolist(),
        ax=axes[0],
        leaf_rotation=90
    )
    axes[0].set_title('Hierarchical Clustering Dendrogram')
    axes[0].set_xlabel('Assets')
    axes[0].set_ylabel('Distance')

    # Correlation heatmap
    sns.heatmap(
        corr,
        ax=axes[1],
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt='.2f',
        square=True
    )
    axes[1].set_title('Asset Correlation Matrix')

    plt.tight_layout()
    return fig


def plot_weights_comparison(
    portfolios: Dict[str, Dict],
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """Plot weight comparison across different portfolios."""
    weights_df = pd.DataFrame({
        name: pd.Series(p['weights'])
        for name, p in portfolios.items()
    }).fillna(0)

    fig, ax = plt.subplots(figsize=figsize)
    weights_df.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Assets')
    ax.set_ylabel('Weight')
    ax.set_title('Portfolio Weight Comparison')
    ax.legend(title='Portfolio')
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def backtest_portfolio(
    weights: Dict[str, float],
    prices: pd.DataFrame,
    initial_value: float = 100000
) -> Dict[str, Any]:
    """
    Backtest a portfolio with given weights.

    Returns performance metrics and portfolio value series.
    """
    returns = calculate_returns(prices)

    # Filter weights to match available tickers
    available_tickers = [t for t in weights.keys() if t in returns.columns]
    filtered_weights = {t: weights[t] for t in available_tickers}
    total_weight = sum(filtered_weights.values())
    if total_weight > 0:
        filtered_weights = {t: w/total_weight for t, w in filtered_weights.items()}

    weights_series = pd.Series(filtered_weights)

    # Calculate portfolio returns
    portfolio_returns = (returns[weights_series.index] * weights_series).sum(axis=1)

    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    portfolio_value = initial_value * cumulative_returns

    # Calculate metrics
    total_return = (portfolio_value.iloc[-1] / initial_value - 1) * 100
    annualized_return = portfolio_returns.mean() * 252
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0

    # Maximum drawdown
    rolling_max = portfolio_value.cummax()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Value at Risk (95%)
    var_95 = portfolio_returns.quantile(0.05)

    return {
        'portfolio_value': portfolio_value,
        'portfolio_returns': portfolio_returns,
        'total_return_pct': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95
    }


def plot_backtest(
    backtest_results: Dict[str, Dict],
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """Plot backtest results for multiple portfolios."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Portfolio value over time
    for name, result in backtest_results.items():
        axes[0, 0].plot(result['portfolio_value'], label=name)
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Value ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Drawdown
    for name, result in backtest_results.items():
        rolling_max = result['portfolio_value'].cummax()
        drawdown = (result['portfolio_value'] - rolling_max) / rolling_max * 100
        axes[0, 1].fill_between(drawdown.index, drawdown.values, alpha=0.5, label=name)
    axes[0, 1].set_title('Drawdown (%)')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Returns distribution
    for name, result in backtest_results.items():
        axes[1, 0].hist(
            result['portfolio_returns'] * 100,
            bins=50,
            alpha=0.5,
            label=name
        )
    axes[1, 0].set_title('Daily Returns Distribution')
    axes[1, 0].set_xlabel('Return (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Performance metrics comparison
    metrics = ['annualized_return', 'annualized_volatility', 'sharpe_ratio', 'max_drawdown']
    metric_labels = ['Ann. Return', 'Ann. Volatility', 'Sharpe Ratio', 'Max Drawdown']

    metrics_df = pd.DataFrame({
        name: [result[m] for m in metrics]
        for name, result in backtest_results.items()
    }, index=metric_labels)

    metrics_df.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Performance Metrics Comparison')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(title='Portfolio')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Universe definitions for different markets/sectors
UNIVERSE_DEFINITIONS = {
    'us_large_cap': [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B',
        'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'MA'
    ],
    'us_tech': [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'CRM',
        'ADBE', 'INTC', 'CSCO', 'ORCL', 'IBM', 'QCOM', 'AMD'
    ],
    'global_diversified': [
        'SPY', 'EFA', 'EEM', 'VWO', 'TLT', 'GLD', 'VNQ', 'LQD',
        'HYG', 'DBC', 'IEF', 'GOVT', 'AGG', 'BND', 'VTI'
    ],
    'european': [
        'EWG', 'EWU', 'EWQ', 'EWI', 'EWP', 'EWN', 'EWK', 'EIRL'
    ],
    'asian': [
        'EWJ', 'FXI', 'EWY', 'EWH', 'EWT', 'EWS', 'INDA', 'THD'
    ],
    'conservative': [
        'BND', 'AGG', 'TLT', 'IEF', 'GOVT', 'LQD', 'MBB', 'VMBS'
    ],
    'aggressive': [
        'QQQ', 'ARKK', 'SMH', 'XLK', 'SOXX', 'IGV', 'SKYY', 'WCLD'
    ]
}


def get_universe(category: str) -> List[str]:
    """Get ticker universe by category name."""
    if category in UNIVERSE_DEFINITIONS:
        return UNIVERSE_DEFINITIONS[category]
    else:
        raise ValueError(f"Unknown universe category: {category}. Available: {list(UNIVERSE_DEFINITIONS.keys())}")
