"""
analysis - Performance metrics for backtest results.

Module:
    performance  Win rate, profit factor, Sharpe, Sortino, expectancy,
                 max drawdown, max consecutive losses, etc. - everything
                 the backtest panel displays + extended metrics from the spec.
                 Pure functions; consumes a list of utils.helpers.Trade.

Import directly:

    from analysis.performance import (
        PerformanceMetrics,
        compute_metrics,
        equity_curve,
    )
"""
