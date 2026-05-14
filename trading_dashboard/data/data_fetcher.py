"""
Market-data layer.

Primary source: yfinance (historical OHLCV).
Alpaca is used elsewhere for live account / positions / orders (execution/).

Responsibilities:
    - Fetch single / batch / whole-region OHLCV
    - Resample 1H -> 4H where yfinance has no native interval
    - Drop NaN price rows, fill missing volume with 0
    - Cache results on disk (diskcache) with timeframe-aware TTLs
    - Never raise on network/data failures - log and return empty frames
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import diskcache
import pandas as pd
import yfinance as yf

from config import (
    CACHE,
    Region,
    Timeframe,
    get_index_for_region,
    get_tickers_for_region,
    resample_rule_for_timeframe,
    yf_interval_for_timeframe,
    yf_period_for_timeframe,
)
from utils.helpers import REQUIRED_OHLCV, drop_nan_safe, get_logger

logger = get_logger("data_fetcher")


# =============================================================================
# Cache singleton
# =============================================================================

_cache = diskcache.Cache(str(CACHE.disk_cache_dir))


def _cache_key(ticker: str, timeframe_value: str, period: str) -> str:
    """Build a stable disk-cache key for a (ticker, timeframe, period) triple."""
    return f"yf::{ticker}::{timeframe_value}::{period}"


def _ttl_for(timeframe: Timeframe) -> int:
    """TTL in seconds for the given timeframe (daily gets a longer TTL)."""
    return (
        CACHE.daily_ttl_seconds
        if timeframe == Timeframe.D1
        else CACHE.intraday_ttl_seconds
    )


# =============================================================================
# Public API: single-ticker fetch
# =============================================================================

def fetch_history(
    ticker: str,
    timeframe: Timeframe | str = Timeframe.D1,
    period: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch post-processed historical OHLCV for a single ticker.

    Returns a DataFrame indexed by Timestamp with columns
    [Open, High, Low, Close, Volume]. Returns an empty DataFrame on any
    failure (network, delisted ticker, etc.) - never raises.

    Args:
        ticker: yfinance symbol (e.g. "AAPL", "SAP.DE", "0700.HK", "^GSPC").
        timeframe: Timeframe enum or its string value ("1D", "4H", "1H").
        period: Optional yfinance period override (defaults from config).
        use_cache: When True, serves from / writes to the disk cache.
    """
    if isinstance(timeframe, str):
        timeframe = Timeframe(timeframe)

    period = period or yf_period_for_timeframe(timeframe)
    key = _cache_key(ticker, timeframe.value, period)

    if use_cache and CACHE.enabled:
        cached = _cache.get(key)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached.copy()

    raw = _yf_download(ticker, timeframe, period)
    if raw.empty:
        return _empty_ohlcv()

    df = _post_process(raw, timeframe)

    if use_cache and CACHE.enabled and not df.empty:
        _cache.set(key, df, expire=_ttl_for(timeframe))

    return df


# =============================================================================
# Public API: batch & region fetch
# =============================================================================

def fetch_many(
    tickers: List[str],
    timeframe: Timeframe | str = Timeframe.D1,
    period: Optional[str] = None,
    use_cache: bool = True,
    max_workers: int = 4,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple tickers concurrently.

    Returns a dict mapping ticker -> DataFrame. Tickers that fail are still
    present in the dict with an empty DataFrame (so callers can iterate
    uniformly).
    """
    results: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return results

    if len(tickers) == 1:
        t = tickers[0]
        results[t] = fetch_history(t, timeframe, period, use_cache)
        return results

    workers = max(1, min(max_workers, len(tickers)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_ticker = {
            executor.submit(fetch_history, t, timeframe, period, use_cache): t
            for t in tickers
        }
        for fut in as_completed(future_to_ticker):
            t = future_to_ticker[fut]
            try:
                results[t] = fut.result()
            except Exception as exc:
                logger.warning("Fetch failed for %s: %s", t, exc)
                results[t] = _empty_ohlcv()
    return results


def fetch_region(
    region: Region | str,
    timeframe: Timeframe | str = Timeframe.D1,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Fetch all tickers in the region's default universe."""
    tickers = get_tickers_for_region(region)
    return fetch_many(tickers, timeframe=timeframe, use_cache=use_cache)


def fetch_index(
    region: Region | str,
    timeframe: Timeframe | str = Timeframe.D1,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch the reference index for the region (S&P 500 / DAX / Nikkei)."""
    ticker = get_index_for_region(region)
    return fetch_history(ticker, timeframe=timeframe, use_cache=use_cache)


# =============================================================================
# Cache management
# =============================================================================

def clear_cache() -> int:
    """Empty the disk cache. Returns the number of entries removed."""
    n = len(_cache)
    _cache.clear()
    logger.info("Cleared %d cache entries from %s", n, CACHE.disk_cache_dir)
    return n


def cache_info() -> Dict[str, object]:
    """Return cache stats for the UI / debugging."""
    return {
        "enabled": CACHE.enabled,
        "entries": len(_cache),
        "directory": str(CACHE.disk_cache_dir),
        "intraday_ttl_seconds": CACHE.intraday_ttl_seconds,
        "daily_ttl_seconds": CACHE.daily_ttl_seconds,
    }


# =============================================================================
# Internal: yfinance call & post-processing
# =============================================================================

def _yf_download(
    ticker: str,
    timeframe: Timeframe,
    period: str,
) -> pd.DataFrame:
    """
    Wrap yf.download with hardening:
        - threads=False (we control concurrency above)
        - auto_adjust=True (split/dividend adjusted)
        - log + return empty on any exception
        - flatten MultiIndex columns (a known yfinance quirk for some tickers)
    """
    interval = yf_interval_for_timeframe(timeframe)
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        logger.warning("yfinance error for %s (%s/%s): %s",
                       ticker, interval, period, exc)
        return _empty_ohlcv()

    if df is None or df.empty:
        logger.info("yfinance returned no data for %s (%s/%s)",
                    ticker, interval, period)
        return _empty_ohlcv()

    # Recent yfinance versions sometimes return MultiIndex columns even for
    # a single ticker. Flatten so downstream code sees plain columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only the standard OHLCV columns we care about.
    keep = [c for c in REQUIRED_OHLCV if c in df.columns]
    if not keep:
        logger.warning("yfinance frame for %s missing all OHLCV columns; "
                       "got %s", ticker, list(df.columns))
        return _empty_ohlcv()

    return df[keep].copy()


def _post_process(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """
    Clean + (optionally) resample a freshly-downloaded yfinance frame.

    Steps:
        1. Sort by index (yfinance is usually sorted but we don't trust it)
        2. Drop rows with NaN in any price column
        3. Fill NaN volume with 0
        4. Resample to the timeframe's rule if needed (e.g. 1H -> 4H)
        5. Normalize column order
    """
    if df.empty:
        return df

    df = df.sort_index()
    df = drop_nan_safe(df, subset=[c for c in ["Open", "High", "Low", "Close"]
                                   if c in df.columns])
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    rule = resample_rule_for_timeframe(timeframe)
    if rule is not None:
        df = _resample_ohlcv(df, rule)

    cols = [c for c in REQUIRED_OHLCV if c in df.columns]
    return df[cols]


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV to a coarser bar size.

    Aggregation:
        Open   -> first
        High   -> max
        Low    -> min
        Close  -> last
        Volume -> sum

    Drops empty/incomplete bars (NaN in OHLC) produced by gaps.
    """
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    available = {k: v for k, v in agg.items() if k in df.columns}
    out = df.resample(rule).agg(available)
    return out.dropna(subset=[c for c in ["Open", "High", "Low", "Close"]
                              if c in out.columns])


def _empty_ohlcv() -> pd.DataFrame:
    """Empty DataFrame with the canonical OHLCV column layout."""
    return pd.DataFrame(columns=REQUIRED_OHLCV)
