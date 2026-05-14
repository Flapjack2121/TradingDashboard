"""
Multi-channel alert dispatcher.

Notifier classes encapsulate one transport each (console / Telegram /
Discord). The `AlertDispatcher` composes whichever notifiers are
enabled by the runtime AlertConfig (env-driven) and fans out events to
all of them. A failure in one channel never affects the others, and
network errors never propagate into the trading pipeline.

Typical usage from the orchestrator:

    dispatcher = get_dispatcher()
    dispatcher.notify_signal(signal)
    dispatcher.notify_trade_close(trade)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

from config import SCORING, get_alert_config
from utils.helpers import Signal, Trade, format_percent, format_price, get_logger

logger = get_logger("alerts.notifier")


# =============================================================================
# Event
# =============================================================================

@dataclass(frozen=True)
class AlertEvent:
    """A single notification payload."""
    kind: str                                # "signal" / "trade_open" / "trade_close" / "risk_rejected" / "error"
    message: str
    ticker: str = ""
    direction: str = ""
    strategy: str = ""
    final_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    extra: Dict[str, Any] = field(default_factory=dict)

    def format_short(self) -> str:
        """One-line plain-text representation used by simple channels."""
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        head = f"[{ts}] {self.kind.upper()}"
        if self.ticker:
            head += f" {self.ticker}"
        if self.direction:
            head += f" {self.direction}"
        return f"{head}  {self.message}"


# =============================================================================
# Notifier base + implementations
# =============================================================================

class Notifier(ABC):
    """Abstract base for a single notification channel."""

    name: str = "notifier"

    @abstractmethod
    def send(self, event: AlertEvent) -> bool:
        """Send the event. Return True on success, False on failure."""


class ConsoleNotifier(Notifier):
    """Logs every event to the application logger."""

    name = "console"

    def send(self, event: AlertEvent) -> bool:
        line = event.format_short()
        if event.kind == "error":
            logger.error(line)
        elif event.kind == "risk_rejected":
            logger.warning(line)
        else:
            logger.info(line)
        return True


class TelegramNotifier(Notifier):
    """Posts to the Telegram Bot API."""

    name = "telegram"

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._url = (
            f"https://api.telegram.org/bot{bot_token}/sendMessage"
            if bot_token else ""
        )

    def send(self, event: AlertEvent) -> bool:
        if not self._bot_token or not self._chat_id:
            return False
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed — Telegram alert dropped")
            return False
        try:
            resp = requests.post(
                self._url,
                json={
                    "chat_id": self._chat_id,
                    "text": event.format_short(),
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=5,
            )
            if resp.status_code >= 400:
                logger.warning(
                    "Telegram returned %s: %s", resp.status_code, resp.text[:200]
                )
                return False
            return True
        except Exception:
            logger.exception("Telegram send failed")
            return False


class DiscordNotifier(Notifier):
    """Posts to a Discord webhook URL."""

    name = "discord"

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url

    def send(self, event: AlertEvent) -> bool:
        if not self._url:
            return False
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed — Discord alert dropped")
            return False
        try:
            resp = requests.post(
                self._url,
                json={"content": event.format_short()},
                timeout=5,
            )
            if resp.status_code >= 400:
                logger.warning(
                    "Discord returned %s: %s", resp.status_code, resp.text[:200]
                )
                return False
            return True
        except Exception:
            logger.exception("Discord send failed")
            return False


# =============================================================================
# Dispatcher
# =============================================================================

class AlertDispatcher:
    """
    Fan-out coordinator for all enabled notifiers.

    Channels are wired at construction time from `get_alert_config()`.
    Trading-pipeline callers use the typed helpers (notify_signal,
    notify_trade_open, etc.) which build an `AlertEvent` and call
    `dispatch`.
    """

    def __init__(
        self,
        notifiers: Optional[List[Notifier]] = None,
        min_final_score_for_signal: Optional[float] = None,
    ) -> None:
        cfg = get_alert_config()
        if notifiers is None:
            notifiers = []
            if cfg.console_enabled:
                notifiers.append(ConsoleNotifier())
            if cfg.telegram_enabled:
                notifiers.append(
                    TelegramNotifier(
                        bot_token=cfg.telegram_bot_token,
                        chat_id=cfg.telegram_chat_id,
                    )
                )
            if cfg.discord_enabled:
                notifiers.append(DiscordNotifier(webhook_url=cfg.discord_webhook_url))

        self._notifiers: List[Notifier] = notifiers
        self._min_signal_score: float = (
            min_final_score_for_signal
            if min_final_score_for_signal is not None
            else cfg.min_final_score_for_alert
        )
        logger.info(
            "AlertDispatcher: channels=%s  signal_threshold=%.2f",
            [n.name for n in self._notifiers],
            self._min_signal_score,
        )

    # ------------------------------------------------------------------
    # Public typed helpers
    # ------------------------------------------------------------------
    def notify_signal(self, signal: Signal) -> None:
        """Only fires when final_score meets the alert threshold."""
        if signal.final_score < self._min_signal_score:
            return
        msg = (
            f"{signal.strategy}  entry={format_price(signal.entry)}  "
            f"SL={format_price(signal.stop_loss)}  "
            f"TP={format_price(signal.take_profit)}  "
            f"R:R={signal.rr_ratio:.2f}  "
            f"score={signal.final_score:.2f}  "
            f"ML={signal.ml_score:.2f} ({signal.ml_band})"
        )
        self.dispatch(AlertEvent(
            kind="signal",
            message=msg,
            ticker=signal.ticker,
            direction=signal.direction,
            strategy=signal.strategy,
            final_score=signal.final_score,
        ))

    def notify_trade_open(self, signal: Signal, quantity: float) -> None:
        msg = (
            f"OPENED  qty={quantity:g}  entry={format_price(signal.entry)}  "
            f"SL={format_price(signal.stop_loss)}  "
            f"TP={format_price(signal.take_profit)}  "
            f"strat={signal.strategy}"
        )
        self.dispatch(AlertEvent(
            kind="trade_open",
            message=msg,
            ticker=signal.ticker,
            direction=signal.direction,
            strategy=signal.strategy,
            final_score=signal.final_score,
        ))

    def notify_trade_close(self, trade: Trade) -> None:
        pnl_sign = "+" if trade.net_pnl >= 0 else ""
        msg = (
            f"CLOSED  exit={format_price(trade.exit_price)}  "
            f"PnL={pnl_sign}{format_price(trade.net_pnl)}  "
            f"return={format_percent(trade.return_pct)}  "
            f"reason={trade.exit_reason}  strat={trade.strategy}"
        )
        self.dispatch(AlertEvent(
            kind="trade_close",
            message=msg,
            ticker=trade.ticker,
            direction=trade.side,
            strategy=trade.strategy,
            extra={
                "net_pnl": trade.net_pnl,
                "return_pct": trade.return_pct,
                "exit_reason": trade.exit_reason,
            },
        ))

    def notify_risk_rejection(
        self,
        signal: Signal,
        reasons: List[str],
    ) -> None:
        joined = " ; ".join(reasons) or "no reason"
        msg = f"REJECTED  reasons=[{joined}]"
        self.dispatch(AlertEvent(
            kind="risk_rejected",
            message=msg,
            ticker=signal.ticker,
            direction=signal.direction,
            strategy=signal.strategy,
            final_score=signal.final_score,
            extra={"reasons": list(reasons)},
        ))

    def notify_error(self, msg: str, ticker: str = "") -> None:
        self.dispatch(AlertEvent(kind="error", message=msg, ticker=ticker))

    # ------------------------------------------------------------------
    # Generic
    # ------------------------------------------------------------------
    def dispatch(self, event: AlertEvent) -> Dict[str, bool]:
        """
        Fan the event out to every channel. Returns a per-channel
        success map (used by tests / the diagnostics panel).
        """
        results: Dict[str, bool] = {}
        for notifier in self._notifiers:
            try:
                results[notifier.name] = bool(notifier.send(event))
            except Exception:
                logger.exception("Notifier %s raised", notifier.name)
                results[notifier.name] = False
        return results

    @property
    def channels(self) -> List[str]:
        return [n.name for n in self._notifiers]


# =============================================================================
# Module-level singleton
# =============================================================================

@lru_cache(maxsize=1)
def get_dispatcher() -> AlertDispatcher:
    """Return a process-wide singleton dispatcher (cached)."""
    return AlertDispatcher()
