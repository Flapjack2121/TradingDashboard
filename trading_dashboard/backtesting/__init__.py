"""
backtesting - Historical strategy evaluation engine.

Module:
    backtester  Bar-by-bar simulation with commissions, slippage, and
                ATR/percent stops. Runs a single strategy or many; supports
                walk-forward validation; yields List[Trade] (frozen) for
                analysis.performance and CSV export.

Import directly:

    from backtesting.backtester import (
        Backtester,
        BacktestResult,
        run_backtest,
        run_walk_forward,
    )
"""
