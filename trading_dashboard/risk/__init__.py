"""
risk - Risk management engine.

Module:
    risk_manager  Per-trade position sizing (from stop-loss distance) and
                  the gate that rejects setups violating R:R, max-open-trades,
                  or daily-loss limits. Same rules used in both live execution
                  and the backtester so simulated and real PnL match.

Import directly:

    from risk.risk_manager import (
        RiskManager,
        RiskDecision,
        position_size_from_stop,
    )
"""
