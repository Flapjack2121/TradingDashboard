"""
Fibonacci - confluence-only retracement strategy (per the high-win-rate spec).

A standalone Fibonacci level is far too weak to trade. The spec mandates that
THREE independent factors line up at the SAME price zone:

    1. Price near the 61.8% retracement of the most recent swing.
    2. EMA_fast (50) sitting near that same level.
    3. A detected support (BUY) or resistance (SELL) zone near that level.

If any one of the three is missing, the strategy emits HOLD with a reason.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from config import INDICATORS, StrategyName
from strategies.base_strategy import Strategy
from utils.helpers import Signal, get_logger
from utils.indicators import (
    fibonacci_levels,
    find_support_resistance,
)

logger = get_logger("strategy.fibonacci")


class FibonacciStrategy(Strategy):
    """61.8% retracement strategy with mandatory EMA + S/R confluence."""

    name: ClassVar[StrategyName] = StrategyName.FIBONACCI

    # Tolerance for the EMA and S/R proximity to the 61.8% level.
    # Slightly looser than the price-to-fib tolerance because indicators and
    # pivots are unlikely to align to the cent.
    confluence_tolerance_pct: ClassVar[float] = 2.0

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

        fib = fibonacci_levels(prepared)
        if fib is None:
            return self.hold(
                ticker, region, prepared, timeframe,
                reason="No clean swing high/low for Fibonacci",
            )

        fib_618 = fib.levels[INDICATORS.fib_entry_level]
        last_close = float(prepared["Close"].iloc[-1])
        last_open = float(prepared["Open"].iloc[-1])
        ema_fast = float(prepared["ema_fast"].iloc[-1])

        if not np.isfinite([fib_618, ema_fast, last_close]).all() or fib_618 <= 0:
            return self.hold(
                ticker, region, prepared, timeframe,
                reason="Degenerate Fibonacci level",
            )

        price_tol = INDICATORS.fib_entry_tolerance_pct / 100.0
        conf_tol = self.confluence_tolerance_pct / 100.0

        price_near_fib = abs(last_close - fib_618) / fib_618 <= price_tol
        ema_near_fib = abs(ema_fast - fib_618) / fib_618 <= conf_tol

        if not price_near_fib:
            return self.hold(
                ticker, region, prepared, timeframe,
                reason=f"Price not at 61.8% level (fib={fib_618:.2f}, close={last_close:.2f})",
            )

        levels = find_support_resistance(prepared)

        # ------------------------------------------------------------------
        # BUY: 61.8% retracement in an UP swing -> needs support confluence
        # ------------------------------------------------------------------
        if fib.direction == "up":
            support_near_fib = self._any_level_near(levels.support, fib_618, conf_tol)

            if not ema_near_fib or not support_near_fib:
                missing = self._missing_factors(
                    ema_near_fib=ema_near_fib,
                    sr_near_fib=support_near_fib,
                    sr_label="support",
                )
                return self.hold(
                    ticker, region, prepared, timeframe,
                    reason=f"Price at 61.8% but missing confluence: {missing}",
                )

            if self.passes_global_filter(prepared, "BUY"):
                return self.build_signal(
                    ticker=ticker,
                    region=region,
                    df=prepared,
                    direction="BUY",
                    timeframe=timeframe,
                    candle_confirmed=(last_close > last_open),
                    reason=(
                        f"61.8% retracement at {fib_618:.2f} in confluence "
                        f"with EMA50 and support"
                    ),
                )
            return self.hold(
                ticker, region, prepared, timeframe,
                reason="Fib confluence detected but global filter rejected",
            )

        # ------------------------------------------------------------------
        # SELL: 61.8% retracement in a DOWN swing -> needs resistance confluence
        # ------------------------------------------------------------------
        if fib.direction == "down":
            resistance_near_fib = self._any_level_near(
                levels.resistance, fib_618, conf_tol
            )

            if not ema_near_fib or not resistance_near_fib:
                missing = self._missing_factors(
                    ema_near_fib=ema_near_fib,
                    sr_near_fib=resistance_near_fib,
                    sr_label="resistance",
                )
                return self.hold(
                    ticker, region, prepared, timeframe,
                    reason=f"Price at 61.8% but missing confluence: {missing}",
                )

            if self.passes_global_filter(prepared, "SELL"):
                return self.build_signal(
                    ticker=ticker,
                    region=region,
                    df=prepared,
                    direction="SELL",
                    timeframe=timeframe,
                    candle_confirmed=(last_close < last_open),
                    reason=(
                        f"61.8% retracement at {fib_618:.2f} in confluence "
                        f"with EMA50 and resistance"
                    ),
                )
            return self.hold(
                ticker, region, prepared, timeframe,
                reason="Fib confluence detected but global filter rejected",
            )

        return self.hold(
            ticker, region, prepared, timeframe,
            reason="Indeterminate Fibonacci swing direction",
        )

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _any_level_near(levels: list[float], target: float, tol: float) -> bool:
        """True if any level in `levels` is within `tol` (relative) of `target`."""
        if not levels or target <= 0:
            return False
        return any(abs(lvl - target) / target <= tol for lvl in levels)

    @staticmethod
    def _missing_factors(
        ema_near_fib: bool,
        sr_near_fib: bool,
        sr_label: str,
    ) -> str:
        """Build a short human-readable list of missing confluence factors."""
        missing = []
        if not ema_near_fib:
            missing.append("EMA50")
        if not sr_near_fib:
            missing.append(sr_label)
        return " + ".join(missing) if missing else "none"
