"""
Feature engineering for the ML trade ranker.

Single source of truth for the model's input vector. The same function is
used at training time (over closed backtest trades) and at inference time
(over the current live signal), guaranteeing parity.

Feature design (all numeric):
    Trend / momentum
        ema_fast_over_slow      EMA50 / EMA200 - 1  (positive = uptrend)
        price_over_ema_fast     Close / EMA50 - 1   (extension from MA)
        rsi                     0-100, neutral 50
        macd_hist               raw histogram value at the entry bar

    Volatility / volume
        atr_over_price          ATR / Close (normalized)
        volume_ratio            Volume / Volume_avg

    Setup geometry
        rr_ratio                |TP - entry| / |entry - SL|
        distance_to_sr_pct      % distance to nearest S/R level
        fib_proximity           1 - min(|price - level|/price) over fib levels;
                                0 when no fib structure exists

    Direction & context (one-hot)
        direction_is_buy        1 if BUY, 0 if SELL
        region_us / region_europe / region_asia
        strat_trend_following / strat_breakout_retest / strat_price_action /
        strat_smc / strat_swing_trading / strat_fibonacci

The order of `FEATURE_COLUMNS` is locked: the model is serialized with the
column list alongside it (see ml.model) so reloads validate consistency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import INDICATORS, Region, StrategyName
from utils.helpers import Signal, Trade, get_logger
from utils.indicators import (
    fibonacci_levels,
    find_support_resistance,
    with_indicators,
)

logger = get_logger("ml.features")


# =============================================================================
# Canonical feature order (DO NOT REORDER — model serialization depends on it)
# =============================================================================

FEATURE_COLUMNS: List[str] = [
    # Trend / momentum
    "ema_fast_over_slow",
    "price_over_ema_fast",
    "rsi",
    "macd_hist",
    # Volatility / volume
    "atr_over_price",
    "volume_ratio",
    # Setup geometry
    "rr_ratio",
    "distance_to_sr_pct",
    "fib_proximity",
    # Direction
    "direction_is_buy",
    # Region one-hot
    "region_us",
    "region_europe",
    "region_asia",
    # Strategy one-hot
    "strat_trend_following",
    "strat_breakout_retest",
    "strat_price_action",
    "strat_smc",
    "strat_swing_trading",
    "strat_fibonacci",
]


# =============================================================================
# Mapping tables for one-hot encoding
# =============================================================================

_REGION_COLUMNS: Dict[str, str] = {
    Region.US.value: "region_us",
    Region.EUROPE.value: "region_europe",
    Region.ASIA.value: "region_asia",
}

_STRATEGY_COLUMNS: Dict[str, str] = {
    StrategyName.TREND_FOLLOWING.value: "strat_trend_following",
    StrategyName.BREAKOUT_RETEST.value: "strat_breakout_retest",
    StrategyName.PRICE_ACTION.value: "strat_price_action",
    StrategyName.SMC.value: "strat_smc",
    StrategyName.SWING_TRADING.value: "strat_swing_trading",
    StrategyName.FIBONACCI.value: "strat_fibonacci",
}


# =============================================================================
# Public builders
# =============================================================================

def build_features_from_window(
    df: pd.DataFrame,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    region: str,
    strategy_name: str,
) -> Dict[str, float]:
    """
    Build the feature dict from an OHLCV window ending at the entry bar.

    The function tolerates a window that does not yet have indicator columns
    (it will call `with_indicators` itself) and gracefully zero-fills any
    feature that can't be computed (insufficient history, NaN values, etc.).
    """
    feat: Dict[str, float] = {col: 0.0 for col in FEATURE_COLUMNS}

    if df is None or df.empty:
        return feat

    # Ensure indicators are present without mutating the caller's frame.
    needed = {"ema_fast", "ema_slow", "rsi", "atr", "vol_avg", "macd_hist"}
    if not needed.issubset(df.columns):
        df = with_indicators(df)

    last = df.iloc[-1]
    close = _safe_float(last.get("Close"), default=entry_price)
    if close <= 0:
        close = max(entry_price, 1e-9)

    # ---- Trend / momentum --------------------------------------------------
    ema_fast = _safe_float(last.get("ema_fast"))
    ema_slow = _safe_float(last.get("ema_slow"))
    feat["ema_fast_over_slow"] = _safe_ratio(ema_fast, ema_slow) - 1.0 if ema_slow > 0 else 0.0
    feat["price_over_ema_fast"] = _safe_ratio(close, ema_fast) - 1.0 if ema_fast > 0 else 0.0
    feat["rsi"] = _safe_float(last.get("rsi"), default=50.0)
    feat["macd_hist"] = _safe_float(last.get("macd_hist"))

    # ---- Volatility / volume ----------------------------------------------
    atr_val = _safe_float(last.get("atr"))
    feat["atr_over_price"] = (atr_val / close) if close > 0 else 0.0

    volume = _safe_float(last.get("Volume"))
    vol_avg = _safe_float(last.get("vol_avg"))
    feat["volume_ratio"] = (volume / vol_avg) if vol_avg > 0 else 0.0

    # ---- Setup geometry ----------------------------------------------------
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    feat["rr_ratio"] = (reward / risk) if risk > 0 else 0.0

    feat["distance_to_sr_pct"] = _nearest_level_distance_pct(df, entry_price)
    feat["fib_proximity"] = _fib_proximity_score(df, entry_price)

    # ---- Direction ---------------------------------------------------------
    feat["direction_is_buy"] = 1.0 if str(direction).upper() == "BUY" else 0.0

    # ---- Region one-hot ----------------------------------------------------
    region_col = _REGION_COLUMNS.get(str(region))
    if region_col:
        feat[region_col] = 1.0

    # ---- Strategy one-hot --------------------------------------------------
    strat_col = _STRATEGY_COLUMNS.get(str(strategy_name))
    if strat_col:
        feat[strat_col] = 1.0

    # Final sanity sweep: replace any non-finite with 0.0
    for k, v in feat.items():
        if not np.isfinite(v):
            feat[k] = 0.0

    return feat


def build_features_from_signal(
    signal: Signal,
    df: pd.DataFrame,
) -> Dict[str, float]:
    """Convenience wrapper for the live inference path."""
    return build_features_from_window(
        df=df,
        direction=signal.direction,
        entry_price=signal.entry,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        region=signal.region,
        strategy_name=signal.strategy,
    )


def build_features_from_trade(
    trade: Trade,
    df_full: pd.DataFrame,
) -> Optional[Dict[str, float]]:
    """
    Training path: rebuild features from a closed Trade.

    Uses the OHLCV slice that ENDS at the trade's entry bar so the feature
    vector reflects what the strategy saw at decision time (no peeking at
    the trade's outcome).
    """
    if df_full is None or df_full.empty:
        return None

    try:
        entry_idx = df_full.index.get_indexer([trade.entry_time], method="pad")[0]
    except Exception:
        logger.debug("Could not locate entry_time %s in df", trade.entry_time)
        return None

    if entry_idx < 0:
        return None

    window = df_full.iloc[: entry_idx + 1]
    if len(window) < INDICATORS.ema_slow:
        return None

    return build_features_from_window(
        df=window,
        direction=trade.side,
        entry_price=trade.entry_price,
        stop_loss=trade.stop_loss,
        take_profit=trade.take_profit,
        region="",                # filled by caller via merge if needed
        strategy_name=trade.strategy,
    )


# =============================================================================
# Vector / matrix helpers
# =============================================================================

def features_dict_to_row(feat: Dict[str, float]) -> List[float]:
    """Project a feature dict into the locked column order."""
    return [float(feat.get(col, 0.0)) for col in FEATURE_COLUMNS]


def features_to_dataframe(rows: List[Dict[str, float]]) -> pd.DataFrame:
    """
    Build a DataFrame from a list of feature dicts.

    Columns are forced to `FEATURE_COLUMNS` order so downstream slicing
    by position is safe (sklearn doesn't track column names internally).
    """
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    df = pd.DataFrame(rows)
    # Add any missing columns (zero-filled), then reorder.
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_COLUMNS].astype(float)


# =============================================================================
# Internal helpers
# =============================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce to float, returning `default` for NaN / None / unparseable."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(f):
        return float(default)
    return f


def _safe_ratio(num: float, denom: float) -> float:
    """num / denom, returning 1.0 (== "no effect" after the -1 shift) on bad input."""
    if denom is None or not np.isfinite(denom) or denom == 0:
        return 1.0
    return num / denom


def _nearest_level_distance_pct(df: pd.DataFrame, price: float) -> float:
    """
    Fractional distance from `price` to the closest S/R level detected
    in the recent window. Returns 1.0 (i.e. "very far") when no levels
    are found, so the model can distinguish "no structure" from "at level".
    """
    if price <= 0:
        return 1.0
    try:
        levels = find_support_resistance(df)
    except Exception:
        return 1.0

    all_levels = list(levels.support) + list(levels.resistance)
    if not all_levels:
        return 1.0

    nearest = min(abs(price - lvl) for lvl in all_levels)
    return float(nearest / price)


def _fib_proximity_score(df: pd.DataFrame, price: float) -> float:
    """
    Score in [0, 1]: 1.0 when price sits exactly on a fib level, decaying
    with distance. 0.0 when no fib structure is identifiable.
    """
    if price <= 0:
        return 0.0
    try:
        fib = fibonacci_levels(df)
    except Exception:
        return 0.0
    if fib is None or not fib.levels:
        return 0.0

    nearest_pct = min(
        abs(price - level_price) / price
        for level_price in fib.levels.values()
    )
    # Convert distance into a 0-1 proximity score with a soft cap at 5%.
    return float(max(0.0, 1.0 - min(nearest_pct, 0.05) / 0.05))
