"""
utils - Shared helpers used by every other module.

Modules:
    indicators  Technical indicators (EMA, RSI, MACD, ATR, S/R, Fib, candles).
                Single source of truth - strategies never reimplement these.
    helpers     Scoring, formatting, validation, timezone, dataclasses for
                signals and trades.

Import directly from the submodule, e.g.:

    from utils.indicators import ema, rsi, atr
    from utils.helpers import Signal, compute_confluence_score
"""
