"""
execution - Live brokerage integration (Alpaca paper trading).

Module:
    alpaca_client  Thin wrapper over alpaca-py's TradingClient. Exposes
                   account snapshot, open positions, market / bracket
                   order submission, and emergency cancel-all / flatten-all
                   helpers. Used by the dashboard's Portfolio panel and by
                   the (optional) auto-execute button on Signal Cards.

US equities only — Alpaca does not cover EU / Asia tickers. Calls degrade
gracefully (logged warning + empty result) when ALPACA_API_KEY /
ALPACA_SECRET_KEY are not configured.

Import directly:

    from execution.alpaca_client import (
        AlpacaClient,
        AccountSnapshot,
        PositionSnapshot,
        OrderResult,
    )
"""
