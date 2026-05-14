"""
Risk Management Engine.

Two things live here:

    1. position_size_from_stop(...)
       Pure function. Given equity, entry, stop-loss and a risk %, returns
       (quantity, risk_amount). No state, no side effects - safe to call
       from anywhere.

    2. RiskManager (stateful)
       Tracks the account's open-trade count, realized PnL for the current
       day, and the running equity. Method evaluate(signal) returns a
       RiskDecision that approves or rejects the trade with EVERY violated
       rule listed (so the UI can show all problems at once).

The same RiskManager class is used by both the live execution path and the
backtester so simulated and real PnL share an identical rule set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional, Tuple

from config import RISK
from utils.helpers import Signal, calculate_rr_ratio, get_logger

logger = get_logger("risk_manager")


# =============================================================================
# Pure functions
# =============================================================================

def position_size_from_stop(
    account_equity: float,
    entry: float,
    stop_loss: float,
    risk_pct: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute the position size (shares) and the dollar amount at risk.

    Args:
        account_equity: Current portfolio equity (cash + positions).
        entry: Intended entry price (>0).
        stop_loss: Stop-loss price.
        risk_pct: Fraction of equity to risk (defaults to RISK.max_risk_per_trade_pct).

    Returns:
        (quantity, risk_amount). Returns (0.0, 0.0) for degenerate inputs
        (zero stop distance, non-positive equity, etc.) - never raises.
    """
    if account_equity <= 0 or entry <= 0:
        return 0.0, 0.0

    risk_pct = (
        risk_pct
        if risk_pct is not None
        else RISK.max_risk_per_trade_pct
    )
    risk_amount = account_equity * risk_pct
    stop_distance = abs(entry - stop_loss)
    if stop_distance <= 0:
        return 0.0, 0.0

    quantity = risk_amount / stop_distance
    return float(quantity), float(risk_amount)


# =============================================================================
# Decision type
# =============================================================================

@dataclass(frozen=True)
class RiskDecision:
    """
    Outcome of risk evaluation.

    approved == True  : trade is sized and ready to execute
    approved == False : `reasons` lists every rule that failed
    """
    approved: bool
    quantity: float
    risk_amount: float
    reasons: List[str] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return not self.approved

    def __str__(self) -> str:
        if self.approved:
            return (
                f"APPROVED  qty={self.quantity:.4f}  "
                f"risk=${self.risk_amount:.2f}"
            )
        return f"REJECTED  {' ; '.join(self.reasons) or 'no reason'}"


# =============================================================================
# Stateful manager
# =============================================================================

class RiskManager:
    """
    Stateful risk gate used by both live execution and the backtester.

    State:
        current_equity        Account equity (updated on register_close)
        open_trades           Count of currently-open positions
        realized_pnl_today    PnL realized since the start of the current day
        current_day           The date used as "today" for the loss limit
    """

    def __init__(
        self,
        starting_equity: float,
        max_risk_per_trade_pct: Optional[float] = None,
        max_open_trades: Optional[int] = None,
        daily_loss_limit_pct: Optional[float] = None,
        min_rr_ratio: Optional[float] = None,
    ):
        if starting_equity <= 0:
            raise ValueError(f"starting_equity must be positive; got {starting_equity!r}")

        self.starting_equity: float = float(starting_equity)
        self.current_equity: float = float(starting_equity)
        self.open_trades: int = 0
        self.realized_pnl_today: float = 0.0
        self.current_day: Optional[date] = None

        self.max_risk_per_trade_pct: float = (
            max_risk_per_trade_pct
            if max_risk_per_trade_pct is not None
            else RISK.max_risk_per_trade_pct
        )
        self.max_open_trades: int = (
            max_open_trades
            if max_open_trades is not None
            else RISK.max_open_trades
        )
        self.daily_loss_limit_pct: float = (
            daily_loss_limit_pct
            if daily_loss_limit_pct is not None
            else RISK.daily_loss_limit_pct
        )
        self.min_rr_ratio: float = (
            min_rr_ratio
            if min_rr_ratio is not None
            else RISK.min_rr_ratio
        )

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def evaluate(
        self,
        signal: Signal,
        now: Optional[datetime] = None,
    ) -> RiskDecision:
        """
        Run a Signal through every risk gate.

        Collects ALL violations (not short-circuit) so the UI can show
        every reason at once. Returns APPROVED with a sized quantity only
        when every gate passes.
        """
        self._roll_day_if_needed(now)
        reasons: List[str] = []

        # Gate 1: Direction must be actionable
        if signal.direction not in ("BUY", "SELL"):
            return RiskDecision(
                approved=False,
                quantity=0.0,
                risk_amount=0.0,
                reasons=[f"Direction is {signal.direction!r}, not actionable"],
            )

        # Gate 2: Stop distance must be positive
        stop_distance = abs(signal.entry - signal.stop_loss)
        if stop_distance <= 0:
            reasons.append("Stop-loss distance is zero")

        # Gate 3: R:R minimum
        rr = calculate_rr_ratio(signal.entry, signal.stop_loss, signal.take_profit)
        if rr < self.min_rr_ratio:
            reasons.append(
                f"R:R {rr:.2f} below minimum {self.min_rr_ratio:.2f}"
            )

        # Gate 4: Max open trades
        if self.open_trades >= self.max_open_trades:
            reasons.append(
                f"Open trades {self.open_trades} at max ({self.max_open_trades})"
            )

        # Gate 5: Daily loss limit
        daily_limit_dollars = self.current_equity * self.daily_loss_limit_pct
        if self.realized_pnl_today <= -daily_limit_dollars:
            reasons.append(
                f"Daily loss limit hit: realized PnL "
                f"${self.realized_pnl_today:.2f} <= "
                f"-${daily_limit_dollars:.2f}"
            )

        if reasons:
            return RiskDecision(
                approved=False, quantity=0.0, risk_amount=0.0, reasons=reasons
            )

        # All gates passed -> size the position
        quantity, risk_amount = position_size_from_stop(
            account_equity=self.current_equity,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            risk_pct=self.max_risk_per_trade_pct,
        )

        if quantity <= 0:
            return RiskDecision(
                approved=False,
                quantity=0.0,
                risk_amount=0.0,
                reasons=["Position size rounds to zero shares"],
            )

        return RiskDecision(
            approved=True, quantity=quantity, risk_amount=risk_amount, reasons=[]
        )

    # ----------------------------------------------------------------------
    # State updates (called by backtester / executor)
    # ----------------------------------------------------------------------
    def register_open(self, when: Optional[datetime] = None) -> None:
        """Increment open-trade count when a new position is opened."""
        self._roll_day_if_needed(when)
        self.open_trades += 1

    def register_close(
        self,
        pnl: float,
        when: Optional[datetime] = None,
    ) -> None:
        """
        Record a closed trade.

        Updates equity, decrements open-trade count, adds PnL to today's
        running realized total.
        """
        self._roll_day_if_needed(when)
        self.open_trades = max(0, self.open_trades - 1)
        self.realized_pnl_today += float(pnl)
        self.current_equity += float(pnl)

    def reset(self, starting_equity: Optional[float] = None) -> None:
        """Reset the manager to a fresh state (used between backtest runs)."""
        if starting_equity is not None:
            if starting_equity <= 0:
                raise ValueError(
                    f"starting_equity must be positive; got {starting_equity!r}"
                )
            self.starting_equity = float(starting_equity)
        self.current_equity = self.starting_equity
        self.open_trades = 0
        self.realized_pnl_today = 0.0
        self.current_day = None

    # ----------------------------------------------------------------------
    # Diagnostics / UI
    # ----------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Return current state for display in the dashboard."""
        return {
            "starting_equity": self.starting_equity,
            "current_equity": self.current_equity,
            "open_trades": self.open_trades,
            "realized_pnl_today": self.realized_pnl_today,
            "daily_loss_limit": self.current_equity * self.daily_loss_limit_pct,
            "max_risk_per_trade_pct": self.max_risk_per_trade_pct,
            "max_open_trades": self.max_open_trades,
            "min_rr_ratio": self.min_rr_ratio,
            "current_day": self.current_day,
        }

    # ----------------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------------
    def _roll_day_if_needed(self, when: Optional[datetime]) -> None:
        """Reset the daily realized-PnL counter when the date changes."""
        # Prefer the supplied timestamp's date (backtester uses bar time);
        # fall back to wall clock for live use.
        ts = when if when is not None else datetime.now()
        today = ts.date() if isinstance(ts, datetime) else ts  # already a date
        if self.current_day != today:
            self.current_day = today
            self.realized_pnl_today = 0.0
