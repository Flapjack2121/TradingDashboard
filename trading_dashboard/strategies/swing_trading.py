"""
Swing Trading - RSI pullback zone strategy (per the high-win-rate spec).

Crucially, this is NOT a classic mean-reversion strategy. Oversold (RSI < 30)
and overbought (RSI > 70) are explicitly rejected as PRIMARY triggers because
they catch knife-edge reversals against momentum.

Instead:

    BUY  : EMA_fast > EMA_slow  AND  RSI in [40, 50]
           (shallow pullback in an uptrend - trend continuation)

    SELL : EMA_fast < EMA_slow  AND  RSI in [50, 60]
           (shallow rally in a downtrend - trend continuation)

A matching candlestick direction awards the +20 candle confluence point but
is not strictly required for the BUY/SELL signal to fire. The strategy gate
(>=75) and the global trade filter handle the rest.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from config import INDICATORS, StrategyName
from strategies.base_strategy import Strategy
from utils.helpers import Signal, get_logger

logger = get_logger("strategy.swing_trading")


class SwingTradingStrategy(Strategy):
    """RSI pullback / rally zone strategy with strict trend filter."""

    name: ClassVar[StrategyName] = StrategyName.SWING_TRADING

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

        last_rsi = float(prepared["rsi"].iloc[-1])
        ema_fast = float(prepared["ema_fast"].iloc[-1])
        ema_slow = float(prepared["ema_slow"].iloc[-1])
        last_open = float(prepared["Open"].iloc[-1])
        last_close = float(prepared["Close"].iloc[-1])

        if not np.isfinite([last_rsi, ema_fast, ema_slow]).all():
            return None

        is_uptrend = ema_fast > ema_slow
        is_downtrend = ema_fast < ema_slow

        in_buy_zone = (
            INDICATORS.swing_tf_rsi_buy_low
            <= last_rsi
            <= INDICATORS.swing_tf_rsi_buy_high
        )
        in_sell_zone = (
            INDICATORS.swing_tf_rsi_sell_low
            <= last_rsi
            <= INDICATORS.swing_tf_rsi_sell_high
        )

        # ------------------------------------------------------------------
        # BUY: uptrend + RSI in 40-50 pullback zone
        # ------------------------------------------------------------------
        if is_uptrend and in_buy_zone:
            bullish_candle = last_close > last_open
            if self.passes_global_filter(prepared, "BUY"):
                return self.build_signal(
                    ticker=ticker,
                    region=region,
                    df=prepared,
                    direction="BUY",
                    timeframe=timeframe,
                    candle_confirmed=bullish_candle,
                    reason=(
                        f"Uptrend pullback: RSI={last_rsi:.1f} in "
                        f"[{INDICATORS.swing_tf_rsi_buy_low:.0f}, "
                        f"{INDICATORS.swing_tf_rsi_buy_high:.0f}]"
                    ),
                )
            return self.hold(
                ticker, region, prepared, timeframe,
                reason="Swing BUY setup but global filter rejected",
            )

        # ------------------------------------------------------------------
        # SELL: downtrend + RSI in 50-60 rally zone
        # ------------------------------------------------------------------
        if is_downtrend and in_sell_zone:
            bearish_candle = last_close < last_open
            if self.passes_global_filter(prepared, "SELL"):
                return self.build_signal(
                    ticker=ticker,
                    region=region,
                    df=prepared,
                    direction="SELL",
                    timeframe=timeframe,
                    candle_confirmed=bearish_candle,
                    reason=(
                        f"Downtrend rally: RSI={last_rsi:.1f} in "
                        f"[{INDICATORS.swing_tf_rsi_sell_low:.0f}, "
                        f"{INDICATORS.swing_tf_rsi_sell_high:.0f}]"
                    ),
                )
            return self.hold(
                ticker, region, prepared, timeframe,
                reason="Swing SELL setup but global filter rejected",
            )

        # ------------------------------------------------------------------
        # No qualifying setup
        # ------------------------------------------------------------------
        trend_label = "up" if is_uptrend else "down" if is_downtrend else "flat"
        return self.hold(
            ticker, region, prepared, timeframe,
            reason=(
                f"No swing setup: trend={trend_label}, RSI={last_rsi:.1f} "
                "(not in pullback/rally zone)"
            ),
        )
