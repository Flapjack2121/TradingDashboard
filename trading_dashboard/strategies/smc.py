"""
Smart Money Concepts (SMC) - simplified, high-accuracy mode per the spec.

Only two SMC primitives are used:

    1. Break of Structure (BOS):
         bullish BOS = close strictly above the most recent confirmed swing high
         bearish BOS = close strictly below the most recent confirmed swing low

    2. Order Block (OB):
         bullish OB = the last bearish candle preceding the strong bullish move
                      that produced the BOS - the price zone is its (low, high)
         bearish OB = the last bullish candle preceding the strong bearish move

Entry rule:
    - A BOS must have occurred within the last `bos_recency_window` bars.
    - The current bar must overlap the matching Order Block (price has
      returned into the zone).
    - The current bar must be a confirmation candle in the trade direction.
    - The global trade filter must pass.

This intentionally excludes more elaborate SMC concepts (FVGs, liquidity
sweeps, mitigation blocks, etc.) - per spec, the goal is accuracy, not
coverage.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import pandas as pd

from config import StrategyName
from strategies.base_strategy import Strategy
from utils.helpers import Signal, get_logger
from utils.indicators import detect_break_of_structure, detect_order_block

logger = get_logger("strategy.smc")


class SMCStrategy(Strategy):
    """Break of Structure + Order Block pullback strategy."""

    name: ClassVar[StrategyName] = StrategyName.SMC

    # How far back to scan for swing highs/lows and OB candidates.
    bos_lookback: ClassVar[int] = 30
    # The BOS must have occurred within this many bars of "now".
    bos_recency_window: ClassVar[int] = 10

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
        if len(prepared) < self.bos_lookback + 5:
            return None

        # Did a BOS happen recently?
        bullish_bos_series = detect_break_of_structure(
            prepared, lookback=self.bos_lookback, direction="up"
        )
        bearish_bos_series = detect_break_of_structure(
            prepared, lookback=self.bos_lookback, direction="down"
        )
        had_bullish_bos = bool(
            bullish_bos_series.iloc[-self.bos_recency_window :].any()
        )
        had_bearish_bos = bool(
            bearish_bos_series.iloc[-self.bos_recency_window :].any()
        )

        last_open = float(prepared["Open"].iloc[-1])
        last_close = float(prepared["Close"].iloc[-1])
        last_high = float(prepared["High"].iloc[-1])
        last_low = float(prepared["Low"].iloc[-1])

        # ------------------------------------------------------------------
        # Bullish BOS + pullback into bullish Order Block -> BUY
        # ------------------------------------------------------------------
        if had_bullish_bos:
            ob = detect_order_block(
                prepared, direction="up", lookback=self.bos_lookback
            )
            if ob is not None:
                ob_low, ob_high = ob
                in_block = self._bars_overlap(last_low, last_high, ob_low, ob_high)
                bullish_candle = last_close > last_open
                respected_zone = last_close > ob_low

                if in_block and bullish_candle and respected_zone:
                    if self.passes_global_filter(prepared, "BUY"):
                        return self.build_signal(
                            ticker=ticker,
                            region=region,
                            df=prepared,
                            direction="BUY",
                            timeframe=timeframe,
                            candle_confirmed=True,
                            reason=(
                                f"Bullish BOS followed by pullback into order "
                                f"block [{ob_low:.2f} - {ob_high:.2f}] with "
                                "bullish confirmation"
                            ),
                        )
                    return self.hold(
                        ticker, region, prepared, timeframe,
                        reason="BOS + OB detected but global filter rejected",
                    )

        # ------------------------------------------------------------------
        # Bearish BOS + pullback into bearish Order Block -> SELL
        # ------------------------------------------------------------------
        if had_bearish_bos:
            ob = detect_order_block(
                prepared, direction="down", lookback=self.bos_lookback
            )
            if ob is not None:
                ob_low, ob_high = ob
                in_block = self._bars_overlap(last_low, last_high, ob_low, ob_high)
                bearish_candle = last_close < last_open
                respected_zone = last_close < ob_high

                if in_block and bearish_candle and respected_zone:
                    if self.passes_global_filter(prepared, "SELL"):
                        return self.build_signal(
                            ticker=ticker,
                            region=region,
                            df=prepared,
                            direction="SELL",
                            timeframe=timeframe,
                            candle_confirmed=True,
                            reason=(
                                f"Bearish BOS followed by pullback into order "
                                f"block [{ob_low:.2f} - {ob_high:.2f}] with "
                                "bearish confirmation"
                            ),
                        )
                    return self.hold(
                        ticker, region, prepared, timeframe,
                        reason="BOS + OB detected but global filter rejected",
                    )

        # ------------------------------------------------------------------
        # No BOS + OB setup
        # ------------------------------------------------------------------
        return self.hold(
            ticker, region, prepared, timeframe,
            reason="No recent BOS with order-block pullback",
        )

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _bars_overlap(
        bar_low: float,
        bar_high: float,
        zone_low: float,
        zone_high: float,
    ) -> bool:
        """Standard 1D interval overlap test."""
        return max(bar_low, zone_low) <= min(bar_high, zone_high)
