"""
Abstract base class for all strategies + the mandatory global trade filter.

The global filter is the floor every actionable BUY/SELL must clear:
    1. Trend alignment (EMA fast vs slow in the trade's direction)
    2. ATR > 20-period ATR average  (volatility regime)
    3. Volume > 20-period volume average  (participation)

If any condition fails, the strategy must reject the trade or downgrade
the signal to HOLD. Concrete strategies inherit this class and implement
generate_signal(); they reuse passes_global_filter and build_signal to stay
consistent with the spec.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from config import INDICATORS, StrategyName
from utils.helpers import (
    Direction,
    Side,
    Signal,
    atr_stop_loss_take_profit,
    combine_final_score,
    compute_confluence_score,
    ensure_ohlcv_columns,
    get_logger,
)
from utils.indicators import (
    atr,
    atr_average,
    ema,
    find_support_resistance,
    near_level,
    volume_average,
    with_indicators,
)

logger = get_logger("strategy")

# Set to True from the UI (app.py) to bypass the global filter for exploration.
BYPASS_GLOBAL_FILTER: bool = False

# Global filter thresholds.
# ATR must be >= 90% of its 20-period average (prevents trades in extremely
# low-vol regimes without blocking near-average conditions).
ATR_FILTER_THRESHOLD: float = 0.90

# Volume: previous completed bar must be >= 80% of the 20-period average.
# We use the PREVIOUS bar (iloc[-2]) because today's intraday bar is
# incomplete until market close and always understates volume.
VOL_FILTER_THRESHOLD: float = 0.70


class Strategy(ABC):
    """
    Abstract base. Every concrete strategy must:

        - set a class-level `name: ClassVar[StrategyName]`
        - implement generate_signal(df, ticker, region, timeframe)

    The base class provides:
        - prepare(df)            -> validates OHLCV + attaches indicator pack
        - passes_global_filter   -> the mandatory trend/ATR/volume check
        - is_trend_aligned       -> confluence helper
        - is_volume_confirmed    -> confluence helper
        - is_near_sr             -> confluence helper
        - build_signal(...)      -> constructs Signal with ATR SL/TP + score

    Conventions:
        * generate_signal returns ONE Signal for the most recent bar, or
          None when data is insufficient (warm-up).
        * The returned Signal may carry direction "HOLD" when conditions
          don't match. The global filter is only checked for BUY/SELL.
        * Strategies NEVER set ml_score / final_score - those are filled
          later by the ML scorer / orchestrator.
    """

    name: ClassVar[StrategyName]
    min_bars_required: ClassVar[int] = INDICATORS.ema_slow + 20

    # ------------------------------------------------------------------
    # Abstract hook
    # ------------------------------------------------------------------
    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        ticker: str,
        region: str,
        timeframe: str = "1D",
    ) -> Optional[Signal]:
        """Return the latest Signal for the final bar of `df`, or None."""

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    @classmethod
    def prepare(cls, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Validate OHLCV columns and attach the canonical indicator pack.

        Returns:
            A copy of `df` with indicator columns appended, OR None if the
            frame is empty / too short for the EMA_slow warm-up.
        """
        if df is None or df.empty:
            return None
        if len(df) < cls.min_bars_required:
            return None
        ensure_ohlcv_columns(df)
        return with_indicators(df)

    # ------------------------------------------------------------------
    # Global trade filter (MANDATORY per spec)
    # ------------------------------------------------------------------
    @staticmethod
    def passes_global_filter(
        df: pd.DataFrame,
        direction: Side,
        bypass: bool = False,
    ) -> bool:
        """
        Verify ALL three global conditions at the latest bar:

            1. Trend:    EMA_fast vs EMA_slow in the trade's direction
            2. ATR:      latest ATR strictly greater than its 20-period MA
            3. Volume:   latest Volume strictly greater than its 20-period MA

        Pass bypass=True to skip the filter (exploration / debug mode).
        NaN values from warm-up periods are treated as failures (return False).
        """
        if bypass or BYPASS_GLOBAL_FILTER:
            return True

        if len(df) < INDICATORS.ema_slow + INDICATORS.atr_avg_period:
            return False

        fast = float(ema(df["Close"], INDICATORS.ema_fast).iloc[-1])
        slow = float(ema(df["Close"], INDICATORS.ema_slow).iloc[-1])
        atr_now = float(atr(df).iloc[-1])
        atr_ma = float(atr_average(df).iloc[-1])
        vol_now = float(df["Volume"].iloc[-1])
        vol_ma = float(volume_average(df["Volume"]).iloc[-1])

        if not np.isfinite([fast, slow, atr_now, atr_ma, vol_ma]).all():
            return False

        trend_ok = (fast > slow) if direction == "BUY" else (fast < slow)
        # Allow ATR within 10% below its average — prevents blocking
        # near-average volatility during consolidation phases.
        atr_ok = atr_now >= atr_ma * ATR_FILTER_THRESHOLD

        # Use the PREVIOUS completed bar for volume (iloc[-2]) because
        # today's intraday bar is incomplete and understates participation.
        # Allow volume down to 80% of average to pass near-average days.
        vol_prev = float(df["Volume"].iloc[-2]) if len(df) >= 2 else vol_now
        vol_ok = vol_prev >= vol_ma * VOL_FILTER_THRESHOLD

        return bool(trend_ok and atr_ok and vol_ok)

    @staticmethod
    def global_filter_detail(df: pd.DataFrame, direction: Side) -> dict:
        """Return per-condition booleans for diagnostics / the UI."""
        if len(df) < INDICATORS.ema_slow + INDICATORS.atr_avg_period:
            return {"trend": False, "atr": False, "volume": False,
                    "ema_fast": 0.0, "ema_slow": 0.0, "atr_now": 0.0,
                    "atr_ma": 0.0, "vol_ratio": 0.0,
                    "atr_threshold": ATR_FILTER_THRESHOLD,
                    "vol_threshold": VOL_FILTER_THRESHOLD}
        fast = float(ema(df["Close"], INDICATORS.ema_fast).iloc[-1])
        slow = float(ema(df["Close"], INDICATORS.ema_slow).iloc[-1])
        atr_now = float(atr(df).iloc[-1])
        atr_ma = float(atr_average(df).iloc[-1])
        vol_now = float(df["Volume"].iloc[-1])
        vol_prev = float(df["Volume"].iloc[-2]) if len(df) >= 2 else vol_now
        vol_ma = float(volume_average(df["Volume"]).iloc[-1])
        return {
            "trend": bool((fast > slow) if direction == "BUY" else (fast < slow)),
            "atr": bool(atr_now >= atr_ma * ATR_FILTER_THRESHOLD),
            "volume": bool(vol_prev >= vol_ma * VOL_FILTER_THRESHOLD),
            "ema_fast": round(fast, 2),
            "ema_slow": round(slow, 2),
            "atr_now": round(atr_now, 4),
            "atr_ma": round(atr_ma, 4),
            "vol_ratio": round(vol_prev / vol_ma, 2) if vol_ma > 0 else 0.0,
            "atr_threshold": ATR_FILTER_THRESHOLD,
            "vol_threshold": VOL_FILTER_THRESHOLD,
        }

    # ------------------------------------------------------------------
    # Confluence helpers (each maps to one factor in the 25/15/20/20/20 grid)
    # ------------------------------------------------------------------
    @staticmethod
    def is_trend_aligned(df: pd.DataFrame, direction: Side) -> bool:
        """Confluence: EMA_fast vs EMA_slow agrees with the trade direction."""
        if len(df) < INDICATORS.ema_slow + 1:
            return False
        fast = float(ema(df["Close"], INDICATORS.ema_fast).iloc[-1])
        slow = float(ema(df["Close"], INDICATORS.ema_slow).iloc[-1])
        if not np.isfinite(fast) or not np.isfinite(slow):
            return False
        return (fast > slow) if direction == "BUY" else (fast < slow)

    @staticmethod
    def is_volume_confirmed(df: pd.DataFrame) -> bool:
        """Confluence: latest volume strictly above its 20-period average."""
        if len(df) < INDICATORS.volume_avg_period + 1:
            return False
        vol_avg = float(volume_average(df["Volume"]).iloc[-1])
        vol_now = float(df["Volume"].iloc[-1])
        if not np.isfinite(vol_avg):
            return False
        return vol_now > vol_avg

    @staticmethod
    def is_near_sr(df: pd.DataFrame, direction: Side) -> bool:
        """
        Confluence: current close is within tolerance of a relevant S/R level.

        BUY  -> near a detected support
        SELL -> near a detected resistance
        """
        levels = find_support_resistance(df)
        relevant = levels.support if direction == "BUY" else levels.resistance
        last_close = float(df["Close"].iloc[-1])
        return near_level(last_close, relevant)

    # ------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------
    def build_signal(
        self,
        ticker: str,
        region: str,
        df: pd.DataFrame,
        direction: Direction,
        timeframe: str = "1D",
        candle_confirmed: bool = False,
        reason: str = "",
    ) -> Signal:
        """
        Construct the latest Signal.

        - HOLD: zeroed SL/TP/scores, no further computation.
        - BUY/SELL: ATR-based SL/TP, confluence score breakdown, and the
          provisional final_score (ml_score=0; orchestrator fills it later).

        Args:
            df: OHLCV with indicator columns attached (output of `prepare`).
            direction: "BUY" / "SELL" / "HOLD".
            candle_confirmed: True if the entry bar's candle supports the
                setup (strategy-specific; e.g. bullish candle on a pullback).
            reason: Short human-readable explanation for the UI tooltip.
        """
        last_close = float(df["Close"].iloc[-1])
        timestamp = df.index[-1]

        if direction == "HOLD":
            return Signal(
                ticker=ticker,
                region=region,
                strategy=self.name.value,
                direction="HOLD",
                entry=last_close,
                stop_loss=last_close,
                take_profit=last_close,
                timeframe=timeframe,
                timestamp=timestamp,
                strategy_score=0.0,
                ml_score=0.0,
                final_score=0.0,
                reason=reason,
                confluence_breakdown={},
            )

        # ATR-derived SL/TP -- canonical helper from utils.helpers.
        atr_now = float(atr(df).iloc[-1])
        if not np.isfinite(atr_now) or atr_now <= 0:
            # Degenerate-data fallback: tiny pct stop so callers never
            # divide by zero. Only triggers on flat/synthetic data.
            atr_now = max(last_close * 0.005, 1e-6)

        sl, tp = atr_stop_loss_take_profit(
            entry=last_close,
            atr_value=atr_now,
            direction=direction,  # type: ignore[arg-type]  # narrowed to Side
        )

        # Confluence breakdown (25/15/20/20/20). ML factor stays at 0 here;
        # the ML scorer can re-issue confluence with a non-zero ml_score later.
        breakdown = compute_confluence_score(
            trend_aligned=self.is_trend_aligned(df, direction),  # type: ignore[arg-type]
            volume_confirmed=self.is_volume_confirmed(df),
            near_support_resistance=self.is_near_sr(df, direction),  # type: ignore[arg-type]
            candle_confirmed=candle_confirmed,
            ml_score=0.0,
        )
        strategy_score = float(breakdown.total)

        return Signal(
            ticker=ticker,
            region=region,
            strategy=self.name.value,
            direction=direction,
            entry=last_close,
            stop_loss=sl,
            take_profit=tp,
            timeframe=timeframe,
            timestamp=timestamp,
            strategy_score=strategy_score,
            ml_score=0.0,
            final_score=combine_final_score(strategy_score, 0.0),
            reason=reason,
            confluence_breakdown=breakdown.as_dict(),
        )

    # ------------------------------------------------------------------
    # Convenience: emit a HOLD shortcut with optional reason
    # ------------------------------------------------------------------
    def hold(
        self,
        ticker: str,
        region: str,
        df: pd.DataFrame,
        timeframe: str = "1D",
        reason: str = "No setup",
    ) -> Signal:
        """Return a HOLD Signal with zeroed SL/TP and scores."""
        return self.build_signal(
            ticker=ticker,
            region=region,
            df=df,
            direction="HOLD",
            timeframe=timeframe,
            candle_confirmed=False,
            reason=reason,
        )
