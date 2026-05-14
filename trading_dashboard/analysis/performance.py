"""
Performance metrics for a list of closed trades.

The spec requires both the basic four (win rate, profit factor, max drawdown,
total return) and the extended set:

    - Sharpe Ratio              (annualized; trades-per-year estimated)
    - Sortino Ratio             (downside-only volatility)
    - Expectancy                ($ per trade)
    - Average Win / Average Loss   (in $ and as % of entry notional)
    - Realized Risk-Reward      (avg_win / |avg_loss|)
    - Max Consecutive Losses

All public functions are pure. The data contract is List[utils.helpers.Trade].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from config import BACKTEST
from utils.helpers import Trade, format_percent, format_price, get_logger

logger = get_logger("performance")


# =============================================================================
# Container
# =============================================================================

@dataclass(frozen=True)
class PerformanceMetrics:
    """Full metric set for a closed backtest run."""

    # Counts
    total_trades: int
    winning_trades: int
    losing_trades: int

    # Basic
    win_rate: float              # fraction in [0, 1]
    total_pnl: float             # net $ across all trades
    total_return_pct: float      # fraction
    profit_factor: float         # gross wins / |gross losses|; inf if no losses
    max_drawdown_pct: float      # fraction; <= 0

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float

    # Trade quality
    expectancy: float            # $ expected per trade
    avg_win: float               # mean $ on winners
    avg_loss: float              # mean $ on losers (<= 0)
    avg_win_pct: float           # mean fractional return on winners
    avg_loss_pct: float          # mean fractional return on losers
    realized_rr: float           # avg_win / |avg_loss|
    max_consecutive_losses: int

    # Equity
    initial_capital: float
    ending_equity: float

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        """Flat dict (useful for JSON export / Streamlit data_editor)."""
        return {
            "Total Trades": self.total_trades,
            "Winning Trades": self.winning_trades,
            "Losing Trades": self.losing_trades,
            "Win Rate": self.win_rate,
            "Total PnL": self.total_pnl,
            "Total Return": self.total_return_pct,
            "Profit Factor": self.profit_factor,
            "Max Drawdown": self.max_drawdown_pct,
            "Sharpe Ratio": self.sharpe_ratio,
            "Sortino Ratio": self.sortino_ratio,
            "Expectancy": self.expectancy,
            "Avg Win": self.avg_win,
            "Avg Loss": self.avg_loss,
            "Avg Win %": self.avg_win_pct,
            "Avg Loss %": self.avg_loss_pct,
            "Realized R:R": self.realized_rr,
            "Max Consecutive Losses": self.max_consecutive_losses,
            "Initial Capital": self.initial_capital,
            "Ending Equity": self.ending_equity,
        }

    def format_for_ui(self) -> Dict[str, str]:
        """Same data, pre-formatted as strings for the metrics table."""
        return {
            "Total Trades": f"{self.total_trades}",
            "Winning Trades": f"{self.winning_trades}",
            "Losing Trades": f"{self.losing_trades}",
            "Win Rate": format_percent(self.win_rate),
            "Total PnL": f"${format_price(self.total_pnl)}",
            "Total Return": format_percent(self.total_return_pct),
            "Profit Factor": _format_finite(self.profit_factor),
            "Max Drawdown": format_percent(self.max_drawdown_pct),
            "Sharpe Ratio": _format_finite(self.sharpe_ratio),
            "Sortino Ratio": _format_finite(self.sortino_ratio),
            "Expectancy": f"${format_price(self.expectancy)}",
            "Avg Win": f"${format_price(self.avg_win)}",
            "Avg Loss": f"${format_price(self.avg_loss)}",
            "Avg Win %": format_percent(self.avg_win_pct),
            "Avg Loss %": format_percent(self.avg_loss_pct),
            "Realized R:R": _format_finite(self.realized_rr),
            "Max Consecutive Losses": f"{self.max_consecutive_losses}",
            "Initial Capital": f"${format_price(self.initial_capital)}",
            "Ending Equity": f"${format_price(self.ending_equity)}",
        }


# =============================================================================
# Public API
# =============================================================================

def compute_metrics(
    trades: List[Trade],
    initial_capital: float,
    risk_free_rate_annual: float = BACKTEST.risk_free_rate_annual,
) -> PerformanceMetrics:
    """
    Compute the full metric set from a list of closed Trades.

    Returns zeroed metrics when `trades` is empty (so the UI never crashes
    on a fresh run).
    """
    if not trades:
        return _zero_metrics(initial_capital)

    pnls = np.array([t.net_pnl for t in trades], dtype=float)
    returns = np.array([t.return_pct for t in trades], dtype=float)

    winning_mask = pnls > 0
    losing_mask = pnls < 0

    total_trades = len(trades)
    winning_trades = int(winning_mask.sum())
    losing_trades = int(losing_mask.sum())
    win_rate = winning_trades / total_trades

    total_pnl = float(pnls.sum())
    ending_equity = initial_capital + total_pnl
    total_return_pct = (
        total_pnl / initial_capital if initial_capital > 0 else 0.0
    )

    profit_factor = _profit_factor(pnls, winning_mask, losing_mask)
    max_drawdown_pct = _max_drawdown(pnls, initial_capital)

    sharpe_ratio, sortino_ratio = _sharpe_and_sortino(
        trades=trades,
        returns=returns,
        risk_free_rate_annual=risk_free_rate_annual,
    )

    avg_win = float(pnls[winning_mask].mean()) if winning_mask.any() else 0.0
    avg_loss = float(pnls[losing_mask].mean()) if losing_mask.any() else 0.0
    avg_win_pct = (
        float(returns[winning_mask].mean()) if winning_mask.any() else 0.0
    )
    avg_loss_pct = (
        float(returns[losing_mask].mean()) if losing_mask.any() else 0.0
    )

    loss_rate = losing_trades / total_trades
    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

    realized_rr = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0

    max_consec = _max_consecutive_losses(pnls)

    return PerformanceMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        total_pnl=total_pnl,
        total_return_pct=total_return_pct,
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        expectancy=expectancy,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        realized_rr=realized_rr,
        max_consecutive_losses=max_consec,
        initial_capital=initial_capital,
        ending_equity=ending_equity,
    )


def equity_curve(
    trades: List[Trade],
    initial_capital: float,
) -> pd.Series:
    """
    Build the equity curve as a time-indexed Series.

    Each trade's exit_time becomes a step in the curve; the value is the
    running equity after that trade closes. An anchor point is inserted at
    the first entry_time with `initial_capital` so the Plotly chart has a
    clean starting tick.
    """
    if not trades:
        return pd.Series([initial_capital], name="equity")

    sorted_trades = sorted(trades, key=lambda t: t.exit_time)
    times: List[pd.Timestamp] = [sorted_trades[0].entry_time]
    values: List[float] = [float(initial_capital)]

    equity = float(initial_capital)
    for t in sorted_trades:
        equity += float(t.net_pnl)
        times.append(t.exit_time)
        values.append(equity)

    return pd.Series(
        values,
        index=pd.DatetimeIndex(times, name="time"),
        name="equity",
    )


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    """Render a list of Trades as a DataFrame for display / CSV export."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([t.as_row() for t in trades])


# =============================================================================
# Internal helpers
# =============================================================================

def _profit_factor(
    pnls: np.ndarray,
    winning_mask: np.ndarray,
    losing_mask: np.ndarray,
) -> float:
    """Sum of wins / abs(sum of losses). Returns inf if no losses, 0 if no wins."""
    gross_wins = float(pnls[winning_mask].sum()) if winning_mask.any() else 0.0
    gross_losses = (
        float(abs(pnls[losing_mask].sum())) if losing_mask.any() else 0.0
    )
    if gross_losses > 0:
        return gross_wins / gross_losses
    return float("inf") if gross_wins > 0 else 0.0


def _max_drawdown(pnls: np.ndarray, initial_capital: float) -> float:
    """
    Maximum peak-to-trough drawdown of the cumulative-equity curve, expressed
    as a non-positive fraction (-0.18 = -18%).
    """
    equity = np.concatenate([[initial_capital], initial_capital + np.cumsum(pnls)])
    peaks = np.maximum.accumulate(equity)
    # Guard against divide-by-zero when initial_capital is 0.
    safe_peaks = np.where(peaks > 0, peaks, np.nan)
    drawdowns = (equity - peaks) / safe_peaks
    drawdowns = np.where(np.isfinite(drawdowns), drawdowns, 0.0)
    return float(drawdowns.min()) if len(drawdowns) > 0 else 0.0


def _sharpe_and_sortino(
    trades: List[Trade],
    returns: np.ndarray,
    risk_free_rate_annual: float,
) -> tuple[float, float]:
    """
    Compute annualized Sharpe and Sortino from per-trade fractional returns.

    Trades-per-year is estimated from the actual time span of the trade
    list (so a backtest covering 2 months annualizes correctly).
    """
    n = len(returns)
    if n < 2:
        return 0.0, 0.0

    first = min(t.entry_time for t in trades)
    last = max(t.exit_time for t in trades)
    span = last - first
    days = max(span.total_seconds() / 86400.0, 1.0)
    trades_per_year = n * 365.25 / days

    rf_per_trade = risk_free_rate_annual / trades_per_year if trades_per_year > 0 else 0.0
    excess = returns - rf_per_trade

    mean_excess = float(np.mean(excess))
    std_total = float(np.std(excess, ddof=1))

    if std_total > 0:
        sharpe = (mean_excess / std_total) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    downside = excess[excess < 0]
    if len(downside) > 1:
        downside_std = float(np.std(downside, ddof=1))
        if downside_std > 0:
            sortino = (mean_excess / downside_std) * np.sqrt(trades_per_year)
        else:
            sortino = 0.0
    elif len(downside) == 0:
        sortino = float("inf") if mean_excess > 0 else 0.0
    else:
        sortino = 0.0

    return float(sharpe), float(sortino)


def _max_consecutive_losses(pnls: np.ndarray) -> int:
    """Longest run of consecutive losing trades."""
    max_run = 0
    current = 0
    for pnl in pnls:
        if pnl < 0:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return int(max_run)


def _zero_metrics(initial_capital: float) -> PerformanceMetrics:
    """Empty-trade-list result so the UI renders cleanly on the first load."""
    return PerformanceMetrics(
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        total_pnl=0.0,
        total_return_pct=0.0,
        profit_factor=0.0,
        max_drawdown_pct=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        expectancy=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        realized_rr=0.0,
        max_consecutive_losses=0,
        initial_capital=float(initial_capital),
        ending_equity=float(initial_capital),
    )


def _format_finite(value: float, decimals: int = 2) -> str:
    """Format a possibly-infinite float for the UI."""
    if not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{decimals}f}"
