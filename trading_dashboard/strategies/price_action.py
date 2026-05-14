"""
Price Action - high-probability candlestick patterns at the correct context.

Per the high-win-rate spec, only four pattern/context combinations are eligible:

    1. Bullish Engulfing  in an uptrend AND near a support level   -> BUY
    2. Hammer             near a support level                     -> BUY
    3. Bearish Engulfing  in a downtrend AND near a resistance level -> SELL
    4. Shooting Star      near a resistance level                  -> SELL

Pattern shape is already filtered by utils.indicators (body <= 35% of range,
wick >= 2x body, etc.), so "weak or mid-range" candles never reach this
strategy. The job here is to check CONTEXT (trend + S/R) and then defer to
the global trade filter.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from config import StrategyName
from strategies.base_strategy import Strategy
from utils.helpers import Signal, get_logger
from utils.indicators import (
    detect_bearish_engulfing,
    detect_bullish_engulfing,
    detect_hammer,
    detect_shooting_star,
    find_support_resistance,
    near_level,
)

logger = get_logger("strategy.price_action")


class PriceActionStrategy(Strategy):
    """Context-aware candlestick pattern strategy."""

    name: ClassVar[StrategyName] = StrategyName.PRICE_ACTION

    def generate_signal(
        self,
        df: pd.DataFrame,
        ticker: str,
        region: str,
        timeframe: str = "1D",
    ) -> Optional[Signal]:
        prepared = self.prepare(df)
        if prepared is None:
            return None

        last_close = float(prepared["Close"].iloc[-1])
        ema_fast = float(prepared["ema_fast"].iloc[-1])
        ema_slow = float(prepared["ema_slow"].iloc[-1])

        if not np.isfinite([ema_fast, ema_slow]).all():
            return None

        is_uptrend = ema_fast > ema_slow
        is_downtrend = ema_fast < ema_slow

        levels = find_support_resistance(prepared)
        at_support = near_level(last_close, levels.support)
        at_resistance = near_level(last_close, levels.resistance)

        # Detect patterns only at the latest bar.
        bull_engulf = bool(detect_bullish_engulfing(prepared).iloc[-1])
        bear_engulf = bool(detect_bearish_engulfing(prepared).iloc[-1])
        hammer = bool(detect_hammer(prepared).iloc[-1])
        shooting_star = bool(detect_shooting_star(prepared).iloc[-1])

        # ------------------------------------------------------------------
        # BUY signals
        # ------------------------------------------------------------------
        if bull_engulf and is_uptrend and at_support:
            return self._emit_or_hold(
                prepared, ticker, region, timeframe,
                direction="BUY",
                reason="Bullish engulfing in uptrend at support",
            )

        if hammer and at_support:
            return self._emit_or_hold(
                prepared, ticker, region, timeframe,
                direction="BUY",
                reason="Hammer at support",
            )

        # ------------------------------------------------------------------
        # SELL signals
        # ------------------------------------------------------------------
        if bear_engulf and is_downtrend and at_resistance:
            return self._emit_or_hold(
                prepared, ticker, region, timeframe,
                direction="SELL",
                reason="Bearish engulfing in downtrend at resistance",
            )

        if shooting_star and at_resistance:
            return self._emit_or_hold(
                prepared, ticker, region, timeframe,
                direction="SELL",
                reason="Shooting star at resistance",
            )

        # ------------------------------------------------------------------
        # No qualifying pattern at the right context
        # ------------------------------------------------------------------
        return self.hold(
            ticker, region, prepared, timeframe,
            reason="No high-probability pattern with correct trend/SR context",
        )

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _emit_or_hold(
        self,
        prepared: pd.DataFrame,
        ticker: str,
        region: str,
        timeframe: str,
        direction: str,
        reason: str,
    ) -> Signal:
        """Run the global filter; emit the signal on pass, HOLD on fail."""
        if self.passes_global_filter(prepared, direction):  # type: ignore[arg-type]
            return self.build_signal(
                ticker=ticker,
                region=region,
                df=prepared,
                direction=direction,  # type: ignore[arg-type]
                timeframe=timeframe,
                candle_confirmed=True,
                reason=reason,
            )
        return self.hold(
            ticker, region, prepared, timeframe,
            reason=f"{reason} (global filter rejected)",
        )
