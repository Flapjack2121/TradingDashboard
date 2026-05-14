"""
Trend Following - EMA50 pullback entries (refined per the high-win-rate spec).

Raw EMA50/EMA200 crossovers are explicitly rejected by the spec because they
chase late and produce too many false signals in chop. Instead, this strategy
waits for a pullback to EMA50 within the prevailing trend, then requires a
bullish/bearish candle that closes back on the right side of EMA50.

Entry rules
-----------
BUY:
    1. EMA_fast > EMA_slow                      (uptrend)
    2. bar.Low <= EMA_fast * (1 + pullback_tol)  (price retested EMA_fast)
    3. bar.Close > bar.Open                      (bullish candle confirmation)
    4. bar.Close > EMA_fast                      (closed back above EMA_fast)
    5. Strategy.passes_global_filter(BUY)        (trend + ATR + volume floor)

SELL:
    mirror conditions (with the EMA inequalities and candle direction flipped).
"""

from __future__ import annotations

from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from config import INDICATORS, StrategyName
from strategies.base_strategy import Strategy
from utils.helpers import Signal, get_logger

logger = get_logger("strategy.trend_following")


class TrendFollowingStrategy(Strategy):
    """Pullback-to-EMA50 trend follower."""

    name: ClassVar[StrategyName] = StrategyName.TREND_FOLLOWING

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

        last_open = float(prepared["Open"].iloc[-1])
        last_close = float(prepared["Close"].iloc[-1])
        last_high = float(prepared["High"].iloc[-1])
        last_low = float(prepared["Low"].iloc[-1])
        ema_fast = float(prepared["ema_fast"].iloc[-1])
        ema_slow = float(prepared["ema_slow"].iloc[-1])

        if not np.isfinite([ema_fast, ema_slow, last_close, last_open]).all():
            return None

        tol = INDICATORS.pullback_tolerance_pct / 100.0

        # ------------------------------------------------------------------
        # Uptrend pullback -> BUY
        # ------------------------------------------------------------------
        if ema_fast > ema_slow:
            pulled_back = last_low <= ema_fast * (1.0 + tol)
            bullish_candle = last_close > last_open
            respected = last_close > ema_fast

            if pulled_back and bullish_candle and respected:
                if self.passes_global_filter(prepared, "BUY"):
                    return self.build_signal(
                        ticker=ticker,
                        region=region,
                        df=prepared,
                        direction="BUY",
                        timeframe=timeframe,
                        candle_confirmed=True,
                        reason=(
                            "Uptrend pullback to EMA50 with bullish candle "
                            "and close back above EMA50"
                        ),
                    )
                return self.hold(
                    ticker, region, prepared, timeframe,
                    reason="Pullback detected but global filter rejected",
                )

            return self.hold(
                ticker, region, prepared, timeframe,
                reason="Uptrend but no clean pullback / candle confirmation",
            )

        # ------------------------------------------------------------------
        # Downtrend retest -> SELL
        # ------------------------------------------------------------------
        if ema_fast < ema_slow:
            pulled_back = last_high >= ema_fast * (1.0 - tol)
            bearish_candle = last_close < last_open
            respected = last_close < ema_fast

            if pulled_back and bearish_candle and respected:
                if self.passes_global_filter(prepared, "SELL"):
                    return self.build_signal(
                        ticker=ticker,
                        region=region,
                        df=prepared,
                        direction="SELL",
                        timeframe=timeframe,
                        candle_confirmed=True,
                        reason=(
                            "Downtrend retest of EMA50 with bearish candle "
                            "and close back below EMA50"
                        ),
                    )
                return self.hold(
                    ticker, region, prepared, timeframe,
                    reason="Retest detected but global filter rejected",
                )

            return self.hold(
                ticker, region, prepared, timeframe,
                reason="Downtrend but no clean retest / candle confirmation",
            )

        # ------------------------------------------------------------------
        # Flat / undefined trend -> HOLD
        # ------------------------------------------------------------------
        return self.hold(
            ticker, region, prepared, timeframe,
            reason="EMA fast and slow are equal - no defined trend",
        )
