"""
Alpaca paper-trading client.

Thin wrapper over alpaca-py's `TradingClient`. All public methods are
exception-safe — broker errors are caught, logged, and returned as a
failure-shaped result rather than raised, so the Streamlit UI never
crashes on a transient network blip.

Public types:
    AccountSnapshot    Frozen view of the trading account.
    PositionSnapshot   Frozen view of one open position.
    OrderResult        Submission result (ok / failed + reason).
    AlpacaClient       The wrapper class.

Scope:
    - US equities only (Alpaca's coverage). Calls with non-US regions
      return a failure result without ever hitting the wire.
    - Paper trading by default (config-driven via ALPACA_PAPER env var).
    - Designed for low call volume from a single dashboard session;
      no rate limiting or batching is added here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from config import Region, get_alpaca_config
from utils.helpers import Signal, get_logger

logger = get_logger("alpaca_client")


# =============================================================================
# Snapshots (UI-friendly frozen views)
# =============================================================================

@dataclass(frozen=True)
class AccountSnapshot:
    """Read-only view of the trading account."""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    status: str
    paper: bool
    connected: bool      # False when credentials missing or API call failed
    error: str = ""


@dataclass(frozen=True)
class PositionSnapshot:
    """Read-only view of one open position."""
    ticker: str
    quantity: float
    side: str               # "long" or "short"
    market_value: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float   # fraction


@dataclass(frozen=True)
class OrderResult:
    """Outcome of an order submission."""
    submitted: bool
    ticker: str
    side: str
    quantity: float
    order_id: str = ""
    status: str = ""
    error: str = ""

    def __str__(self) -> str:
        if self.submitted:
            return (
                f"SUBMITTED  {self.side} {self.quantity} {self.ticker}  "
                f"id={self.order_id[:8]}…  status={self.status}"
            )
        return f"FAILED  {self.side} {self.quantity} {self.ticker}  err={self.error}"


# =============================================================================
# Client
# =============================================================================

class AlpacaClient:
    """
    Wrapper around `alpaca.trading.TradingClient`.

    The class is import-safe even when alpaca-py is not installed — the
    constructor disables the client and every method returns an empty /
    failure result. This means the dashboard can be developed and run
    locally without the broker dependency.
    """

    def __init__(self) -> None:
        cfg = get_alpaca_config()
        self._paper: bool = cfg.paper
        self._enabled: bool = False
        self._client = None       # alpaca.trading.TradingClient instance
        self._last_error: str = ""

        if not cfg.is_configured:
            self._last_error = "Alpaca credentials not configured"
            logger.info("AlpacaClient disabled: %s", self._last_error)
            return

        try:
            # Import lazily so the project runs without alpaca-py installed.
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(
                api_key=cfg.api_key,
                secret_key=cfg.secret_key,
                paper=cfg.paper,
            )
            self._enabled = True
            logger.info("AlpacaClient connected (paper=%s)", cfg.paper)
        except ImportError:
            self._last_error = "alpaca-py is not installed"
            logger.warning("%s — `pip install alpaca-py`", self._last_error)
        except Exception as exc:
            self._last_error = f"Init failed: {exc}"
            logger.exception("AlpacaClient init failed")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def is_enabled(self) -> bool:
        """True when credentials are valid and the SDK is importable."""
        return self._enabled

    @property
    def paper(self) -> bool:
        """True when the client targets the paper endpoint."""
        return self._paper

    @property
    def last_error(self) -> str:
        return self._last_error

    # ------------------------------------------------------------------
    # Account / positions
    # ------------------------------------------------------------------
    def get_account(self) -> AccountSnapshot:
        """Fetch the account snapshot (or a disconnected stub on failure)."""
        if not self._enabled:
            return self._zero_account()
        try:
            acc = self._client.get_account()
            return AccountSnapshot(
                equity=float(acc.equity),
                cash=float(acc.cash),
                buying_power=float(acc.buying_power),
                portfolio_value=float(acc.portfolio_value),
                status=str(acc.status),
                paper=self._paper,
                connected=True,
            )
        except Exception as exc:
            logger.exception("get_account failed")
            self._last_error = str(exc)
            return self._zero_account(error=str(exc))

    def get_positions(self) -> List[PositionSnapshot]:
        """Return all open positions (empty list on failure / disabled)."""
        if not self._enabled:
            return []
        try:
            positions = self._client.get_all_positions()
            out: List[PositionSnapshot] = []
            for p in positions:
                qty = float(p.qty)
                avg_entry = float(p.avg_entry_price)
                current = float(p.current_price) if p.current_price else avg_entry
                unrealized = float(p.unrealized_pl) if p.unrealized_pl else 0.0
                unrealized_pct = (
                    float(p.unrealized_plpc) if p.unrealized_plpc else 0.0
                )
                out.append(
                    PositionSnapshot(
                        ticker=str(p.symbol),
                        quantity=qty,
                        side=str(p.side).lower(),
                        market_value=float(p.market_value),
                        avg_entry_price=avg_entry,
                        current_price=current,
                        unrealized_pnl=unrealized,
                        unrealized_pnl_pct=unrealized_pct,
                    )
                )
            return out
        except Exception as exc:
            logger.exception("get_positions failed")
            self._last_error = str(exc)
            return []

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def submit_market_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
    ) -> OrderResult:
        """Submit a plain market order. Side must be 'BUY' or 'SELL'."""
        if not self._enabled:
            return self._disabled_order(ticker, side, quantity)

        side_upper = side.upper()
        if side_upper not in ("BUY", "SELL"):
            return OrderResult(
                submitted=False, ticker=ticker, side=side, quantity=quantity,
                error=f"Invalid side {side!r}",
            )
        qty = self._normalize_qty(quantity)
        if qty <= 0:
            return OrderResult(
                submitted=False, ticker=ticker, side=side_upper, quantity=qty,
                error="Quantity rounds to zero",
            )

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            req = MarketOrderRequest(
                symbol=ticker.upper(),
                qty=qty,
                side=OrderSide.BUY if side_upper == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(order_data=req)
            return OrderResult(
                submitted=True,
                ticker=ticker.upper(),
                side=side_upper,
                quantity=qty,
                order_id=str(order.id),
                status=str(order.status),
            )
        except Exception as exc:
            logger.exception("submit_market_order failed")
            self._last_error = str(exc)
            return OrderResult(
                submitted=False, ticker=ticker, side=side_upper,
                quantity=qty, error=str(exc),
            )

    def submit_bracket_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        take_profit: float,
        stop_loss: float,
    ) -> OrderResult:
        """
        Submit an OTO bracket: entry + take-profit + stop-loss as one ticket.

        Alpaca brackets only support BUY-then-sell or SELL-then-cover with
        valid TP > entry > SL (BUY) or SL > entry > TP (SELL); the broker
        will reject misordered legs. We don't enforce those here — the
        strategy layer already does.
        """
        if not self._enabled:
            return self._disabled_order(ticker, side, quantity)

        side_upper = side.upper()
        if side_upper not in ("BUY", "SELL"):
            return OrderResult(
                submitted=False, ticker=ticker, side=side, quantity=quantity,
                error=f"Invalid side {side!r}",
            )
        qty = self._normalize_qty(quantity)
        if qty <= 0:
            return OrderResult(
                submitted=False, ticker=ticker, side=side_upper, quantity=qty,
                error="Quantity rounds to zero",
            )
        if take_profit <= 0 or stop_loss <= 0:
            return OrderResult(
                submitted=False, ticker=ticker, side=side_upper, quantity=qty,
                error="TP and SL must be positive",
            )

        try:
            from alpaca.trading.requests import (
                MarketOrderRequest,
                TakeProfitRequest,
                StopLossRequest,
            )
            from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce

            req = MarketOrderRequest(
                symbol=ticker.upper(),
                qty=qty,
                side=OrderSide.BUY if side_upper == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
                stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
            )
            order = self._client.submit_order(order_data=req)
            return OrderResult(
                submitted=True,
                ticker=ticker.upper(),
                side=side_upper,
                quantity=qty,
                order_id=str(order.id),
                status=str(order.status),
            )
        except Exception as exc:
            logger.exception("submit_bracket_order failed")
            self._last_error = str(exc)
            return OrderResult(
                submitted=False, ticker=ticker, side=side_upper,
                quantity=qty, error=str(exc),
            )

    def submit_from_signal(
        self,
        signal: Signal,
        quantity: float,
    ) -> OrderResult:
        """
        Submit a bracket order from a strategy `Signal`.

        Guards:
            - Region must be US (Alpaca limitation).
            - Direction must be BUY or SELL.
            - Signal must be actionable (entry, SL, TP all set).

        The sized `quantity` should come from RiskManager.evaluate(), not
        from the signal itself (signals are unsized by design).
        """
        if signal.region != Region.US.value:
            return OrderResult(
                submitted=False, ticker=signal.ticker, side=signal.direction,
                quantity=quantity,
                error=f"Alpaca supports US only; signal region is {signal.region}",
            )
        if not signal.is_actionable:
            return OrderResult(
                submitted=False, ticker=signal.ticker, side=signal.direction,
                quantity=quantity, error="Signal is not actionable (HOLD or invalid)",
            )
        return self.submit_bracket_order(
            ticker=signal.ticker,
            side=signal.direction,
            quantity=quantity,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
        )

    # ------------------------------------------------------------------
    # Emergency controls
    # ------------------------------------------------------------------
    def cancel_all_orders(self) -> int:
        """Cancel every open order. Returns the count cancelled."""
        if not self._enabled:
            return 0
        try:
            responses = self._client.cancel_orders()
            return len(responses) if responses else 0
        except Exception as exc:
            logger.exception("cancel_all_orders failed")
            self._last_error = str(exc)
            return 0

    def close_all_positions(self, cancel_orders: bool = True) -> int:
        """
        Liquidate every open position at market. Returns count flattened.

        When `cancel_orders` is True (default), all open orders are
        cancelled first so they don't fight the close-out.
        """
        if not self._enabled:
            return 0
        try:
            responses = self._client.close_all_positions(
                cancel_orders=cancel_orders
            )
            return len(responses) if responses else 0
        except Exception as exc:
            logger.exception("close_all_positions failed")
            self._last_error = str(exc)
            return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _zero_account(self, error: str = "") -> AccountSnapshot:
        """Return an empty account snapshot used when the client is offline."""
        return AccountSnapshot(
            equity=0.0,
            cash=0.0,
            buying_power=0.0,
            portfolio_value=0.0,
            status="DISCONNECTED",
            paper=self._paper,
            connected=False,
            error=error or self._last_error,
        )

    def _disabled_order(
        self, ticker: str, side: str, quantity: float
    ) -> OrderResult:
        return OrderResult(
            submitted=False,
            ticker=ticker,
            side=side.upper(),
            quantity=quantity,
            error=self._last_error or "AlpacaClient disabled",
        )

    @staticmethod
    def _normalize_qty(quantity: float) -> float:
        """
        Alpaca accepts fractional qty for many US equities but not all.
        Round to 4 decimals as a safe middle ground; callers can pre-round
        to integer shares if they want.
        """
        return round(float(quantity), 4)
