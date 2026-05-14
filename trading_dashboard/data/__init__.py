"""
data - Market-data layer.

Modules:
    data_fetcher  Historical OHLCV via yfinance (cached on disk + in memory),
                  plus Alpaca historical bars where applicable. Handles region
                  routing, 4H resampling, NaN cleanup, and timezone alignment.

Import directly from the submodule, e.g.:

    from data.data_fetcher import fetch_history, fetch_many
"""
