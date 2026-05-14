"""
strategies - Validated trading strategies.

Modules:
    base_strategy     Abstract Strategy class + the mandatory global trade
                      filter (trend direction + ATR > avg + volume > avg).
                      Every concrete strategy inherits from this.

    trend_following   EMA 50/200 pullback entries (refined - not a raw crossover).
    breakout_retest   Strict mode: breakout + retest + rejection candle.
    price_action      Context-filtered engulfing / hammer / shooting-star.
    smc               Break of Structure + Order Block pullback.
    swing_trading     RSI 40-50 / 50-60 pullback zones with trend filter.
    fibonacci         61.8% retracement only when EMA & S/R confluence is present.

Import strategies directly, e.g.:

    from strategies.trend_following import TrendFollowingStrategy

A registry of all enabled strategies is exposed via:

    from strategies import REGISTRY, build_strategy

so that the dashboard can instantiate any subset by name.
"""

from __future__ import annotations

from typing import Dict, Type

from config import StrategyName
from strategies.base_strategy import Strategy
from strategies.breakout_retest import BreakoutRetestStrategy
from strategies.fibonacci import FibonacciStrategy
from strategies.price_action import PriceActionStrategy
from strategies.smc import SMCStrategy
from strategies.swing_trading import SwingTradingStrategy
from strategies.trend_following import TrendFollowingStrategy


# Single source of truth: name -> Strategy class.
# UI selections and ensemble scoring iterate over this registry.
REGISTRY: Dict[StrategyName, Type[Strategy]] = {
    StrategyName.TREND_FOLLOWING: TrendFollowingStrategy,
    StrategyName.BREAKOUT_RETEST: BreakoutRetestStrategy,
    StrategyName.PRICE_ACTION: PriceActionStrategy,
    StrategyName.SMC: SMCStrategy,
    StrategyName.SWING_TRADING: SwingTradingStrategy,
    StrategyName.FIBONACCI: FibonacciStrategy,
}


def build_strategy(name: StrategyName | str) -> Strategy:
    """
    Instantiate a strategy by its enum value or string label.

    Raises KeyError if the name is unknown.
    """
    if isinstance(name, str):
        name = StrategyName(name)
    cls = REGISTRY[name]
    return cls()


__all__ = [
    "Strategy",
    "REGISTRY",
    "build_strategy",
    "TrendFollowingStrategy",
    "BreakoutRetestStrategy",
    "PriceActionStrategy",
    "SMCStrategy",
    "SwingTradingStrategy",
    "FibonacciStrategy",
]
