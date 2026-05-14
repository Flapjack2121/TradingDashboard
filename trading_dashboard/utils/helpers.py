"""
Shared helpers: signal/trade types, scoring, formatting, timezone, validation.

This module sits above utils/indicators.py and below the strategies. It defines
the data contract (Signal, Trade, ConfluenceScore) that the rest of the system
consumes, plus the canonical scoring math from the spec.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from config import (
    ML_CONFIDENCE_BANDS,
    RISK,
    SCORING,
    SIGNAL_COLOR_MAP,
    UI_COLORS,
    Region,
    get_market_hours_for_region,
    get_timezone_for_region,
)


# =============================================================================
# Type aliases
# =============================================================================

Direction = Literal["BUY", "SELL", "HOLD"]
Side = Literal["BUY", "SELL"]
ExitReason = Literal["TP", "SL", "TIMEOUT", "SIGNAL_FLIP", "END_OF_DATA"]

REQUIRED_OHLCV: List[str] = ["Open", "High", "Low", "Close", "Volume"]


# =============================================================================
# Logging
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger. Single handler, single formatter, no
    propagation - safe to call repeatedly from any module.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                "%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level_name, logging.INFO))
        logger.propagate = False
    return logger


# =============================================================================
# Confluence scoring
# =============================================================================

@dataclass(frozen=True)
class ConfluenceScore:
    """
    Breakdown of confluence points awarded to a setup.

    Point allocation (spec-mandated):
        trend_alignment           -> +25
        volume_confirmation       -> +15
        support_resistance        -> +20
        candlestick_confirmation  -> +20
        ml_high_score             -> +20  (awarded when ML score >= threshold)
        ----------------------------------
        total                     ->  0-100
    """
    trend_alignment: int = 0
    volume_confirmation: int = 0
    support_resistance: int = 0
    candlestick_confirmation: int = 0
    ml_high_score: int = 0

    @property
    def total(self) -> int:
        return (
            self.trend_alignment
            + self.volume_confirmation
            + self.support_resistance
            + self.candlestick_confirmation
            + self.ml_high_score
        )

    @property
    def passes_threshold(self) -> bool:
        """True if total >= the strategy-score gate from config (default 75)."""
        return self.total >= SCORING.min_strategy_score

    def as_dict(self) -> Dict[str, int]:
        """Flat dict (incl. total) for serialization / UI display."""
        return {
            "trend_alignment": self.trend_alignment,
            "volume_confirmation": self.volume_confirmation,
            "support_resistance": self.support_resistance,
            "candlestick_confirmation": self.candlestick_confirmation,
            "ml_high_score": self.ml_high_score,
            "total": self.total,
        }


def compute_confluence_score(
    trend_aligned: bool,
    volume_confirmed: bool,
    near_support_resistance: bool,
    candle_confirmed: bool,
    ml_score: float = 0.0,
) -> ConfluenceScore:
    """Award the spec-defined points for each confluence factor present."""
    return ConfluenceScore(
        trend_alignment=(
            SCORING.trend_alignment_pts if trend_aligned else 0
        ),
        volume_confirmation=(
            SCORING.volume_confirmation_pts if volume_confirmed else 0
        ),
        support_resistance=(
            SCORING.support_resistance_pts if near_support_resistance else 0
        ),
        candlestick_confirmation=(
            SCORING.candlestick_confirmation_pts if candle_confirmed else 0
        ),
        ml_high_score=(
            SCORING.ml_high_score_pts
            if ml_score >= SCORING.ml_high_threshold
            else 0
        ),
    )


def combine_final_score(strategy_score: float, ml_score: float) -> float:
    """
    Blend the strategy confluence score (0-100) with the ML probability (0-1)
    using the spec weights. Returns a value on the 0-1 scale.

        FINAL = strategy_weight * (strategy_score / 100) + ml_weight * ml_score
    """
    s = _clamp(float(strategy_score) / 100.0, 0.0, 1.0)
    m = _clamp(float(ml_score), 0.0, 1.0)
    return SCORING.strategy_weight * s + SCORING.ml_weight * m


def ml_confidence_band(score: float) -> str:
    """
    Map a 0-1 ML probability to "High" / "Medium" / "Low"
    using the thresholds defined in config.ML_CONFIDENCE_BANDS.
    """
    if score is None or not np.isfinite(score):
        return "Low"
    for threshold, label in ML_CONFIDENCE_BANDS:
        if score >= threshold:
            return label
    return "Low"


# =============================================================================
# Signal & Trade dataclasses
# =============================================================================

@dataclass
class Signal:
    """
    A trade setup emitted by a strategy.

    Fields with defaults are filled in over the pipeline:
        - strategy emits with strategy_score and provisional ml_score=0
        - ml/model.py fills ml_score
        - helpers.combine_final_score fills final_score
        - confluence_breakdown is the per-factor point map for UI tooltips
    """
    ticker: str
    region: str
    strategy: str
    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float
    timeframe: str = "1D"
    timestamp: Optional[pd.Timestamp] = None
    strategy_score: float = 0.0   # 0-100
    ml_score: float = 0.0         # 0-1
    final_score: float = 0.0      # 0-1
    reason: str = ""
    confluence_breakdown: Dict[str, int] = field(default_factory=dict)

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop_loss)

    @property
    def reward_per_share(self) -> float:
        return abs(self.take_profit - self.entry)

    @property
    def rr_ratio(self) -> float:
        risk = self.risk_per_share
        return self.reward_per_share / risk if risk > 0 else 0.0

    @property
    def ml_band(self) -> str:
        return ml_confidence_band(self.ml_score)

    @property
    def is_actionable(self) -> bool:
        """True if direction is BUY/SELL and final_score passes the gate."""
        return (
            self.direction in ("BUY", "SELL")
            and self.final_score >= SCORING.min_final_score
        )

    def as_row(self) -> Dict[str, Any]:
        """Flat dict suitable for a DataFrame row in the trade table."""
        return {
            "Ticker": self.ticker,
            "Region": self.region,
            "Strategy": self.strategy,
            "Direction": self.direction,
            "Entry": self.entry,
            "Stop Loss": self.stop_loss,
            "Take Profit": self.take_profit,
            "R:R": round(self.rr_ratio, 2),
            "Strategy Score": round(self.strategy_score, 1),
            "ML Score": round(self.ml_score, 3),
            "ML Band": self.ml_band,
            "Final Score": round(self.final_score, 3),
            "Timeframe": self.timeframe,
            "Timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class Trade:
    """A closed trade produced by the backtester."""
    ticker: str
    strategy: str
    side: Side
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    size: float                 # number of shares (>= 0)
    stop_loss: float
    take_profit: float
    strategy_score: float       # 0-100
    ml_score: float             # 0-1
    final_score: float          # 0-1
    exit_reason: ExitReason
    commission: float = 0.0
    slippage: float = 0.0

    @property
    def gross_pnl(self) -> float:
        sign = 1.0 if self.side == "BUY" else -1.0
        return sign * (self.exit_price - self.entry_price) * self.size

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.commission - self.slippage

    @property
    def return_pct(self) -> float:
        """Net PnL as a fraction of the trade's notional entry value."""
        notional = self.entry_price * self.size
        return self.net_pnl / notional if notional > 0 else 0.0

    @property
    def holding_period(self) -> pd.Timedelta:
        return self.exit_time - self.entry_time

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    def as_row(self) -> Dict[str, Any]:
        return {
            "Ticker": self.ticker,
            "Strategy": self.strategy,
            "Side": self.side,
            "Entry Time": self.entry_time,
            "Entry": self.entry_price,
            "Exit Time": self.exit_time,
            "Exit": self.exit_price,
            "Size": self.size,
            "SL": self.stop_loss,
            "TP": self.take_profit,
            "Net PnL": self.net_pnl,
            "Return %": self.return_pct,
            "Exit Reason": self.exit_reason,
            "Strategy Score": self.strategy_score,
            "ML Score": self.ml_score,
            "Final Score": self.final_score,
        }


# =============================================================================
# Stop-Loss / Take-Profit construction
# =============================================================================

def atr_stop_loss_take_profit(
    entry: float,
    atr_value: float,
    direction: Side,
    sl_mult: Optional[float] = None,
    tp_mult: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute SL and TP using ATR multiples.

    BUY  -> SL = entry - sl_mult * ATR,   TP = entry + tp_mult * ATR
    SELL -> SL = entry + sl_mult * ATR,   TP = entry - tp_mult * ATR
    """
    sl_mult = sl_mult if sl_mult is not None else RISK.default_atr_sl_multiplier
    tp_mult = tp_mult if tp_mult is not None else RISK.default_atr_tp_multiplier

    if not np.isfinite(atr_value) or atr_value <= 0:
        raise ValueError(f"ATR must be positive and finite, got {atr_value!r}")

    if direction == "BUY":
        sl = entry - sl_mult * atr_value
        tp = entry + tp_mult * atr_value
    else:
        sl = entry + sl_mult * atr_value
        tp = entry - tp_mult * atr_value

    return float(sl), float(tp)


def calculate_rr_ratio(entry: float, stop_loss: float, take_profit: float) -> float:
    """Reward-to-Risk ratio. Returns 0.0 if risk is zero."""
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    return reward / risk if risk > 0 else 0.0


# =============================================================================
# Formatting
# =============================================================================

def format_price(value: Optional[float], decimals: int = 2) -> str:
    """Locale-friendly price string. Returns an em-dash for invalid values."""
    if value is None or not np.isfinite(value):
        return "—"  # em dash
    return f"{value:,.{decimals}f}"


def format_percent(value: Optional[float], decimals: int = 2) -> str:
    """Value is a fraction (0.12 -> '12.00%'). Em-dash for invalid input."""
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.{decimals}f}%"


def format_score_100(value: Optional[float], decimals: int = 1) -> str:
    """0-100 score formatter."""
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}"


def signal_color(direction: str) -> str:
    """Hex color for a signal direction. Falls back to the neutral UI color."""
    return SIGNAL_COLOR_MAP.get(direction.upper(), UI_COLORS["neutral"])


# =============================================================================
# Timezone & market hours
# =============================================================================

def now_in_region(region: Region | str) -> datetime:
    """Current wall-clock datetime in the region's IANA timezone."""
    tz = pytz.timezone(get_timezone_for_region(region))
    return datetime.now(tz)


def is_market_open(region: Region | str, now: Optional[datetime] = None) -> bool:
    """
    True if it is currently a weekday and within the region's trading hours.

    This is a regular-hours heuristic - it does not account for half-days or
    market holidays.
    """
    tz_name = get_timezone_for_region(region)
    if now is None:
        now = datetime.now(pytz.timezone(tz_name))
    elif now.tzinfo is None:
        now = pytz.timezone(tz_name).localize(now)
    else:
        now = now.astimezone(pytz.timezone(tz_name))

    if now.weekday() >= 5:   # Sat / Sun
        return False

    open_str, close_str = get_market_hours_for_region(region)
    open_t = dtime.fromisoformat(open_str)
    close_t = dtime.fromisoformat(close_str)
    return open_t <= now.time() <= close_t


def ensure_tz_aware(ts: pd.Timestamp, tz: str = "UTC") -> pd.Timestamp:
    """Localize a naive Timestamp to `tz`; pass through if already aware."""
    return ts.tz_localize(tz) if ts.tzinfo is None else ts


def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert a Timestamp to UTC (localizing if naive)."""
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# =============================================================================
# Validation
# =============================================================================

def ensure_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Raise if any required OHLCV column is missing. Returns df unchanged."""
    missing = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required OHLCV columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def drop_nan_safe(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Drop rows with NaN in any required column. Returns a copy."""
    return df.dropna(subset=subset or REQUIRED_OHLCV).copy()


# =============================================================================
# Private utilities
# =============================================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp `value` into the [lo, hi] interval."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
