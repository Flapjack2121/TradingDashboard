"""
Technical indicators - single source of truth.

Every strategy and the ML feature pipeline imports from this module.
All functions are PURE:
    - input:  pandas Series / DataFrame with OHLCV columns
    - output: pandas Series / DataFrame indexed identically to the input
    - no in-place mutation
    - no lookahead bias: the value at time t depends only on data <= t

OHLCV convention: DataFrame columns ["Open", "High", "Low", "Close", "Volume"].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from config import INDICATORS


# =============================================================================
# Trend & momentum
# =============================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average with `period` warm-up bars before output."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def rsi(close: pd.Series, period: Optional[int] = None) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing.

    Returns 0-100 with neutral 50 substituted where the formula is undefined
    (e.g. flat warm-up periods with no losses).
    """
    period = period or INDICATORS.rsi_period
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing == EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # Fill the early warm-up and divide-by-zero rows with the neutral value.
    return out.fillna(50.0)


def macd(
    close: pd.Series,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    signal: Optional[int] = None,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence.

    Returns a DataFrame with columns ["macd", "signal", "histogram"].
    """
    fast = fast or INDICATORS.macd_fast
    slow = slow or INDICATORS.macd_slow
    signal = signal or INDICATORS.macd_signal

    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


# =============================================================================
# Volatility
# =============================================================================

def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range = max(H-L, |H - prev_close|, |L - prev_close|)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
    """Average True Range with Wilder's smoothing."""
    period = period or INDICATORS.atr_period
    return true_range(df).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()


def atr_average(
    df: pd.DataFrame,
    atr_period: Optional[int] = None,
    avg_period: Optional[int] = None,
) -> pd.Series:
    """Rolling SMA of ATR - the baseline used by the global volatility filter."""
    avg_period = avg_period or INDICATORS.atr_avg_period
    return atr(df, period=atr_period).rolling(
        window=avg_period, min_periods=avg_period
    ).mean()


# =============================================================================
# Volume
# =============================================================================

def volume_average(volume: pd.Series, period: Optional[int] = None) -> pd.Series:
    """Rolling SMA of volume."""
    period = period or INDICATORS.volume_avg_period
    return volume.rolling(window=period, min_periods=period).mean()


def volume_spike(
    volume: pd.Series,
    period: Optional[int] = None,
    threshold: float = 1.0,
) -> pd.Series:
    """Boolean: volume strictly greater than `threshold * rolling-avg volume`."""
    avg = volume_average(volume, period=period)
    return volume > (avg * threshold)


# =============================================================================
# Support / Resistance
# =============================================================================

@dataclass(frozen=True)
class Levels:
    """Detected support and resistance prices, nearest-to-price first."""
    support: List[float]
    resistance: List[float]


def find_pivots(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect pivot highs and pivot lows using a left/right confirmation window.

    A pivot high at bar i is the unique maximum High in bars [i-left, i+right].
    Pivot lows are dual. The last `right` bars cannot be confirmed yet, which
    is the correct behavior - it prevents lookahead bias.

    Returns two Series aligned to df.index: NaN where no pivot, else the price.
    """
    high, low = df["High"], df["Low"]
    n = len(df)
    pivot_high = pd.Series(np.nan, index=df.index, dtype="float64")
    pivot_low = pd.Series(np.nan, index=df.index, dtype="float64")

    if n < left + right + 1:
        return pivot_high, pivot_low

    for i in range(left, n - right):
        win_h = high.iloc[i - left : i + right + 1]
        win_l = low.iloc[i - left : i + right + 1]
        center_h = high.iloc[i]
        center_l = low.iloc[i]

        if center_h == win_h.max() and (win_h == center_h).sum() == 1:
            pivot_high.iloc[i] = center_h
        if center_l == win_l.min() and (win_l == center_l).sum() == 1:
            pivot_low.iloc[i] = center_l

    return pivot_high, pivot_low


def find_support_resistance(
    df: pd.DataFrame,
    lookback: Optional[int] = None,
    tolerance_pct: Optional[float] = None,
    max_levels: int = 3,
) -> Levels:
    """
    Return up to `max_levels` support and resistance prices.

    Detects pivots in the last `lookback` bars, then clusters pivots within
    `tolerance_pct` of each other so we don't return duplicate levels.
    Supports lie below the current close, resistances lie above.
    """
    lookback = lookback or INDICATORS.sr_lookback
    tolerance_pct = tolerance_pct or INDICATORS.sr_tolerance_pct

    recent = df.tail(lookback)
    if len(recent) < 7:  # need at least left+right+1 bars for pivots
        return Levels(support=[], resistance=[])

    pivot_high, pivot_low = find_pivots(recent)
    last_close = float(df["Close"].iloc[-1])
    tolerance = last_close * (tolerance_pct / 100.0)

    highs = pivot_high.dropna().tolist()
    lows = pivot_low.dropna().tolist()

    def cluster(values: List[float]) -> List[float]:
        if not values:
            return []
        sorted_vals = sorted(values, reverse=True)
        clusters: List[List[float]] = []
        for v in sorted_vals:
            placed = False
            for c in clusters:
                if abs(v - float(np.mean(c))) <= tolerance:
                    c.append(v)
                    placed = True
                    break
            if not placed:
                clusters.append([v])
        return [float(np.mean(c)) for c in clusters]

    resistance = [v for v in cluster(highs) if v > last_close]
    support = [v for v in cluster(lows) if v < last_close]

    # Nearest to current price first.
    resistance.sort(key=lambda v: v - last_close)
    support.sort(key=lambda v: last_close - v)

    return Levels(support=support[:max_levels], resistance=resistance[:max_levels])


def near_level(
    price: float,
    levels: List[float],
    tolerance_pct: Optional[float] = None,
) -> bool:
    """True if `price` is within `tolerance_pct` of any value in `levels`."""
    if not levels:
        return False
    tolerance_pct = tolerance_pct or INDICATORS.sr_tolerance_pct
    tol = price * (tolerance_pct / 100.0)
    return any(abs(price - lvl) <= tol for lvl in levels)


def distance_to_nearest_level(price: float, levels: List[float]) -> float:
    """Absolute distance to the nearest level. Returns +inf if `levels` is empty."""
    if not levels:
        return float("inf")
    return min(abs(price - lvl) for lvl in levels)


# =============================================================================
# Fibonacci
# =============================================================================

FIB_LEVELS: List[float] = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]


@dataclass(frozen=True)
class FibLevels:
    """Auto-detected Fibonacci retracement levels for the most recent swing."""
    swing_high: float
    swing_low: float
    direction: Literal["up", "down"]
    levels: Dict[float, float]   # ratio -> price


def fibonacci_levels(
    df: pd.DataFrame,
    lookback: Optional[int] = None,
) -> Optional[FibLevels]:
    """
    Detect the most recent swing high / swing low within `lookback` bars and
    return Fibonacci retracement levels.

    direction == "up"   -> swing low came first; retracement measures pullback
                           down from the swing high.
    direction == "down" -> swing high came first; retracement measures bounce
                           up from the swing low.

    Returns None if a clean swing can't be detected.
    """
    lookback = lookback or INDICATORS.fib_lookback
    recent = df.tail(lookback)
    if len(recent) < 10:
        return None

    high_idx = recent["High"].idxmax()
    low_idx = recent["Low"].idxmin()
    swing_high = float(recent["High"].loc[high_idx])
    swing_low = float(recent["Low"].loc[low_idx])

    if swing_high <= swing_low:
        return None

    direction: Literal["up", "down"] = "up" if low_idx < high_idx else "down"
    span = swing_high - swing_low

    if direction == "up":
        # Levels descend from the swing high
        levels = {ratio: swing_high - span * ratio for ratio in FIB_LEVELS}
    else:
        # Levels ascend from the swing low
        levels = {ratio: swing_low + span * ratio for ratio in FIB_LEVELS}

    return FibLevels(
        swing_high=swing_high,
        swing_low=swing_low,
        direction=direction,
        levels=levels,
    )


# =============================================================================
# Trend direction & pullback detection
# =============================================================================

def trend_direction(df: pd.DataFrame) -> pd.Series:
    """
    Per-bar trend label using EMA_fast vs EMA_slow:

        "up"   -> EMA_fast > EMA_slow
        "down" -> EMA_fast < EMA_slow
        "flat" -> equal or warm-up (NaN) periods
    """
    fast = ema(df["Close"], INDICATORS.ema_fast)
    slow = ema(df["Close"], INDICATORS.ema_slow)
    out = pd.Series("flat", index=df.index, dtype="object")
    valid = ~(fast.isna() | slow.isna())
    out[valid & (fast > slow)] = "up"
    out[valid & (fast < slow)] = "down"
    return out


def is_pullback_to_ema(
    df: pd.DataFrame,
    direction: Literal["up", "down"] = "up",
    tolerance_pct: Optional[float] = None,
) -> pd.Series:
    """
    Boolean Series: price has pulled back to within tolerance of the fast EMA
    while still respecting the prevailing trend.

    "up":   Low <= EMA_fast * (1 + tol)  AND  Close > EMA_fast
    "down": High >= EMA_fast * (1 - tol)  AND  Close < EMA_fast
    """
    tolerance_pct = tolerance_pct or INDICATORS.pullback_tolerance_pct
    tol = tolerance_pct / 100.0
    fast = ema(df["Close"], INDICATORS.ema_fast)

    if direction == "up":
        touched = df["Low"] <= fast * (1.0 + tol)
        respected = df["Close"] > fast
    else:
        touched = df["High"] >= fast * (1.0 - tol)
        respected = df["Close"] < fast

    return (touched & respected).fillna(False)


# =============================================================================
# Candlestick patterns
# =============================================================================

def _candle_geometry(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Compute body/range/upper-wick/lower-wick as Series. Internal helper."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    rng = (h - l).replace(0.0, np.nan)
    body = (c - o).abs()
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    return {
        "open": o, "high": h, "low": l, "close": c,
        "range": rng, "body": body,
        "upper_wick": upper_wick, "lower_wick": lower_wick,
    }


def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    True at bar i when:
        - bar i-1 is bearish
        - bar i   is bullish
        - bar i's body fully contains bar i-1's body
    """
    o, c = df["Open"], df["Close"]
    prev_o, prev_c = o.shift(1), c.shift(1)

    prev_bearish = prev_c < prev_o
    curr_bullish = c > o
    engulfs = (o <= prev_c) & (c >= prev_o)

    return (prev_bearish & curr_bullish & engulfs).fillna(False)


def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    True at bar i when:
        - bar i-1 is bullish
        - bar i   is bearish
        - bar i's body fully contains bar i-1's body
    """
    o, c = df["Open"], df["Close"]
    prev_o, prev_c = o.shift(1), c.shift(1)

    prev_bullish = prev_c > prev_o
    curr_bearish = c < o
    engulfs = (o >= prev_c) & (c <= prev_o)

    return (prev_bullish & curr_bearish & engulfs).fillna(False)


def detect_hammer(
    df: pd.DataFrame,
    body_max_pct: float = 0.35,
    lower_min_ratio: float = 2.0,
) -> pd.Series:
    """
    Hammer: small body in the upper third, long lower wick (>= 2x body),
    tiny upper wick. Reversal signal at the bottom of a downswing.
    """
    g = _candle_geometry(df)
    small_body = g["body"] / g["range"] <= body_max_pct
    long_lower = g["lower_wick"] >= lower_min_ratio * g["body"]
    small_upper = g["upper_wick"] <= g["body"]
    return (small_body & long_lower & small_upper).fillna(False)


def detect_shooting_star(
    df: pd.DataFrame,
    body_max_pct: float = 0.35,
    upper_min_ratio: float = 2.0,
) -> pd.Series:
    """
    Shooting Star: small body in the lower third, long upper wick (>= 2x body),
    tiny lower wick. Reversal signal at the top of an upswing.
    """
    g = _candle_geometry(df)
    small_body = g["body"] / g["range"] <= body_max_pct
    long_upper = g["upper_wick"] >= upper_min_ratio * g["body"]
    small_lower = g["lower_wick"] <= g["body"]
    return (small_body & long_upper & small_lower).fillna(False)


# =============================================================================
# SMC primitives: BOS + Order Block
# =============================================================================

def swing_highs_lows(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """Convenience wrapper around `find_pivots`."""
    return find_pivots(df, left=left, right=right)


def detect_break_of_structure(
    df: pd.DataFrame,
    lookback: int = 30,
    direction: Literal["up", "down"] = "up",
) -> pd.Series:
    """
    Boolean Series flagging Break of Structure bars.

    Bullish BOS: close strictly above the most recent confirmed swing high
                 within the last `lookback` bars.
    Bearish BOS: close strictly below the most recent confirmed swing low
                 within the last `lookback` bars.
    """
    close = df["Close"]
    pivot_high, pivot_low = find_pivots(df, left=3, right=3)
    n = len(df)
    out = pd.Series(False, index=df.index)

    if n < lookback:
        return out

    target = pivot_high if direction == "up" else pivot_low
    comparator = (lambda a, b: a > b) if direction == "up" else (lambda a, b: a < b)

    for i in range(lookback, n):
        window = target.iloc[max(0, i - lookback) : i].dropna()
        if window.empty:
            continue
        last_pivot = float(window.iloc[-1])
        if comparator(float(close.iloc[i]), last_pivot):
            out.iloc[i] = True
    return out


def detect_order_block(
    df: pd.DataFrame,
    direction: Literal["up", "down"] = "up",
    lookback: int = 30,
) -> Optional[Tuple[float, float]]:
    """
    Find the most recent order block in the given direction.

    Bullish OB: the last bearish candle that was followed by a strong bullish
                move (a later close above its high). Returns (low, high) of
                that bearish candle - i.e. the price zone to watch.
    Bearish OB: the last bullish candle that was followed by a strong bearish
                move (a later close below its low). Returns (low, high) of
                that bullish candle.

    Returns None if no order block can be detected within `lookback`.
    """
    recent = df.tail(lookback)
    if len(recent) < 10:
        return None

    o, h, l, c = recent["Open"], recent["High"], recent["Low"], recent["Close"]
    n = len(recent)

    if direction == "up":
        # Walk back from the second-to-last bar.
        for i in range(n - 2, 0, -1):
            if c.iloc[i] < o.iloc[i]:  # bearish candle
                forward_max = c.iloc[i + 1 :].max()
                if forward_max > h.iloc[i]:
                    return float(l.iloc[i]), float(h.iloc[i])
    else:
        for i in range(n - 2, 0, -1):
            if c.iloc[i] > o.iloc[i]:  # bullish candle
                forward_min = c.iloc[i + 1 :].min()
                if forward_min < l.iloc[i]:
                    return float(l.iloc[i]), float(h.iloc[i])
    return None


# =============================================================================
# Convenience: attach the standard indicator pack
# =============================================================================

def with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of `df` enriched with the canonical indicator columns:

        ema_fast, ema_slow, rsi, atr, atr_avg, vol_avg,
        macd, macd_signal, macd_hist

    Strategies and the ML feature pipeline call this once per ticker.
    """
    out = df.copy()
    out["ema_fast"] = ema(out["Close"], INDICATORS.ema_fast)
    out["ema_slow"] = ema(out["Close"], INDICATORS.ema_slow)
    out["rsi"] = rsi(out["Close"])
    out["atr"] = atr(out)
    out["atr_avg"] = atr_average(out)
    out["vol_avg"] = volume_average(out["Volume"])

    macd_df = macd(out["Close"])
    out["macd"] = macd_df["macd"]
    out["macd_signal"] = macd_df["signal"]
    out["macd_hist"] = macd_df["histogram"]

    return out
