"""
Breakout & Retest (STRICT mode per the high-win-rate spec).

Raw breakouts are explicitly rejected because they bait fakeouts. The strategy
emits a signal only when all three sequential conditions are met:

    1. STRUCTURE: pivots in an older window establish a historical resistance
       (R) and support (S) line.
    2. BREAK:     a bar in the more recent window CLOSED beyond the level
       (close > R for bullish; close < S for bearish).
    3. RETEST:    the current bar returns to the level (low <= R or high >= S
       within tolerance) AND rejects with a directional candle that closes
       back on the correct side of the level.

Then the global trade filter (trend + ATR + volume) must pass.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from config import INDICATORS, StrategyName
from strategies.base_strategy import Strategy
from utils.helpers import Signal, get_logger
from utils.indicators import find_pivots

logger = get_logger("strategy.breakout_retest")


class BreakoutRetestStrategy(Strategy):
    """Resistance/support break + retest with rejection candle."""

    name: ClassVar[StrategyName] = StrategyName.BREAKOUT_RETEST

    # Structure detection window (older bars, used to identify R/S).
    structure_lookback: ClassVar[int] = 50
    # How many of the most recent bars are EXCLUDED from structure
    # (they may contain the breakout or the retest, not the old level).
    structure_buffer: ClassVar[int] = 10
    # Bars in which the breakout must have occurred.
    breakout_window: ClassVar[int] = 10
    # How close to the broken level the retest bar must come (% of price).
    retest_tolerance_pct: ClassVar[float] = 1.0
    # Tighter than the base class default: must cover EMA200 warm-up AND
    # the structure + buffer windows. Computed at class-definition time.
    min_bars_required: ClassVar[int] = max(
        INDICATORS.ema_slow + 20,
        structure_lookback + structure_buffer + 5,
    )

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
        if len(prepared) < self.structure_lookback + self.structure_buffer + 5:
            return None

        # Two non-overlapping windows:
        #   structure: [start, structure_end)         -> establishes R/S
        #   recent:    [structure_end, end-1]         -> must contain the break
        #   current bar (iloc[-1])                    -> the retest + rejection
        structure_end = -(self.structure_buffer + 1)
        structure_start = structure_end - self.structure_lookback
        structure = prepared.iloc[structure_start:structure_end]
        recent = prepared.iloc[structure_end:]  # includes current bar

        old_resistance, old_support = self._historical_levels(structure)
        if not np.isfinite(old_resistance) or not np.isfinite(old_support):
            return self.hold(
                ticker, region, prepared, timeframe,
                reason="Insufficient pivot structure",
            )

        last_open = float(prepared["Open"].iloc[-1])
        last_close = float(prepared["Close"].iloc[-1])
        last_high = float(prepared["High"].iloc[-1])
        last_low = float(prepared["Low"].iloc[-1])
        tol = self.retest_tolerance_pct / 100.0

        # All bars in `recent` EXCEPT the current one are candidates for the break.
        recent_pre_current = recent["Close"].iloc[:-1]

        # ------------------------------------------------------------------
        # Bullish breakout + retest -> BUY
        # ------------------------------------------------------------------
        broke_up = bool((recent_pre_current > old_resistance).any())
        retesting_from_above = (
            last_low <= old_resistance * (1.0 + tol)
            and last_close > old_resistance
        )
        bullish_rejection = last_close > last_open

        if broke_up and retesting_from_above and bullish_rejection:
            if self.passes_global_filter(prepared, "BUY"):
                return self.build_signal(
                    ticker=ticker,
                    region=region,
                    df=prepared,
                    direction="BUY",
                    timeframe=timeframe,
                    candle_confirmed=True,
                    reason=(
                        f"Bullish break of {old_resistance:.2f}, retest with "
                        "rejection candle and close back above level"
                    ),
                )
            return self.hold(
                ticker, region, prepared, timeframe,
                reason="Breakout-retest detected but global filter rejected",
            )

        # ------------------------------------------------------------------
        # Bearish breakdown + retest -> SELL
        # ------------------------------------------------------------------
        broke_down = bool((recent_pre_current < old_support).any())
        retesting_from_below = (
            last_high >= old_support * (1.0 - tol)
            and last_close < old_support
        )
        bearish_rejection = last_close < last_open

        if broke_down and retesting_from_below and bearish_rejection:
            if self.passes_global_filter(prepared, "SELL"):
                return self.build_signal(
                    ticker=ticker,
                    region=region,
                    df=prepared,
                    direction="SELL",
                    timeframe=timeframe,
                    candle_confirmed=True,
                    reason=(
                        f"Bearish break of {old_support:.2f}, retest with "
                        "rejection candle and close back below level"
                    ),
                )
            return self.hold(
                ticker, region, prepared, timeframe,
                reason="Breakdown-retest detected but global filter rejected",
            )

        return self.hold(
            ticker, region, prepared, timeframe,
            reason="No breakout/retest with rejection candle",
        )

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _historical_levels(structure: pd.DataFrame) -> tuple[float, float]:
        """
        Return (resistance, support) from the structure window.

        Uses pivot detection first (more robust) and falls back to the window's
        absolute max High / min Low if too few pivots were found.
        """
        pivots_h, pivots_l = find_pivots(structure, left=3, right=3)
        piv_h = pivots_h.dropna().tolist()
        piv_l = pivots_l.dropna().tolist()

        resistance = max(piv_h) if piv_h else float(structure["High"].max())
        support = min(piv_l) if piv_l else float(structure["Low"].min())
        return resistance, support
