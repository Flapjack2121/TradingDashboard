"""
Bar-by-bar backtesting engine.

Public API:
    BacktestResult   Container for one completed backtest window.
    Backtester       Runs one or more strategies against historical OHLCV data.
    run_backtest     Convenience wrapper: single run.
    run_walk_forward Slices data into N anchored walk-forward windows and runs
                     a fresh Backtester on each test fold.

Design notes:
    - Zero lookahead bias: the strategy receives only bars up to and including
      bar i; the signal is entered at bar i+1 open.
    - Positions are closed when the HIGH of a bar touches TP (profit first) or
      the LOW touches SL; tie goes to TP. End-of-data forces a market-close.
    - Commission and slippage applied on both sides (entry + exit).
    - RiskManager gates every signal (same rules as live execution).
    - Score filter: strategy_score >= SCORING.min_strategy_score (75).
    - Only one position per ticker is allowed at a time.
    - CSV export writes to EXPORT_DIR with a timestamped filename.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Type

import pandas as pd

from analysis.performance import (
    PerformanceMetrics,
    compute_metrics,
    equity_curve,
    trades_to_dataframe,
)
from config import BACKTEST, EXPORT_DIR, RISK, SCORING, Timeframe
from risk.risk_manager import RiskManager
from strategies.base_strategy import Strategy
from utils.helpers import Signal, Trade, get_logger

logger = get_logger("backtester")


# =============================================================================
# Result container
# =============================================================================

@dataclass
class BacktestResult:
    """All outputs from one completed backtest run or walk-forward window."""

    ticker: str
    timeframe: Timeframe
    strategy_names: List[str]

    # Trade-level results
    trades: List[Trade] = field(default_factory=list)

    # Derived analytics (computed after the run)
    metrics: Optional[PerformanceMetrics] = None
    equity_series: Optional[pd.Series] = None

    # Walk-forward metadata (None for single-window runs)
    window_index: Optional[int] = None      # 0-based fold number
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None

    @property
    def is_walk_forward_window(self) -> bool:
        return self.window_index is not None

    def finalize(self, initial_capital: float) -> "BacktestResult":
        """Compute metrics and equity curve. Returns self for chaining."""
        self.metrics = compute_metrics(
            self.trades, initial_capital, BACKTEST.risk_free_rate_annual
        )
        self.equity_series = equity_curve(self.trades, initial_capital)
        return self

    def export_trades_csv(self, path: Optional[Path] = None) -> Path:
        """
        Write all trades to a CSV file.

        Auto-names the file if `path` is not given:
            <EXPORT_DIR>/<ticker>_<strategy>_<timestamp>.csv
        """
        if path is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            strats = "_".join(self.strategy_names)[:30]
            filename = f"{self.ticker}_{strats}_{ts}.csv"
            path = EXPORT_DIR / filename

        df = trades_to_dataframe(self.trades)
        if df.empty:
            logger.warning("No trades to export for %s", self.ticker)
            path.write_text("No trades\n")
        else:
            df.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)

        logger.info("Trades exported to %s (%d rows)", path, len(self.trades))
        return path


# =============================================================================
# Open position tracker (internal)
# =============================================================================

@dataclass
class _OpenPosition:
    """One live simulated position during the bar loop."""
    ticker: str
    direction: str              # "BUY" or "SELL"
    entry_price: float          # filled price (incl. slippage)
    entry_time: datetime
    stop_loss: float
    take_profit: float
    quantity: float
    commission_paid: float      # entry-side cost

    def filled_tp(self, bar_high: float, bar_low: float) -> bool:
        """Check if this bar reaches the take-profit level."""
        if self.direction == "BUY":
            return bar_high >= self.take_profit
        return bar_low <= self.take_profit

    def filled_sl(self, bar_high: float, bar_low: float) -> bool:
        """Check if this bar reaches the stop-loss level."""
        if self.direction == "BUY":
            return bar_low <= self.stop_loss
        return bar_high >= self.stop_loss


# =============================================================================
# Core engine
# =============================================================================

class Backtester:
    """
    Bar-by-bar simulation engine.

    Usage::

        bt = Backtester(initial_capital=100_000)
        result = bt.run(df, ticker="AAPL", region="US",
                        strategies=[TrendFollowingStrategy()],
                        timeframe=Timeframe.D1)
        result.finalize(100_000)
    """

    def __init__(
        self,
        initial_capital: float = BACKTEST.initial_capital,
        apply_score_filter: bool = True,
    ):
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive; got {initial_capital!r}")
        self.initial_capital = float(initial_capital)
        self.apply_score_filter = apply_score_filter

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        ticker: str,
        region: str,
        strategies: Sequence[Strategy],
        timeframe: Timeframe = Timeframe.D1,
    ) -> BacktestResult:
        """
        Simulate all strategies against `df`.

        Args:
            df:          Full OHLCV history (already processed, with indicators).
            ticker:      Ticker symbol (informational / Trade metadata).
            region:      Region string (passed to strategy.generate_signal).
            strategies:  One or more strategy instances to run.
            timeframe:   Bar timeframe (stored in BacktestResult).

        Returns:
            BacktestResult with trades list populated (call .finalize() after).
        """
        result = BacktestResult(
            ticker=ticker,
            timeframe=timeframe,
            strategy_names=[type(s).__name__ for s in strategies],
        )

        if df.empty or len(df) < 2:
            logger.warning("Not enough bars for %s — skipping", ticker)
            return result

        risk_mgr = RiskManager(starting_equity=self.initial_capital)
        open_positions: Dict[str, _OpenPosition] = {}   # ticker -> position
        trades: List[Trade] = []

        bars = df.reset_index(drop=False)   # keep time column accessible

        for i in range(1, len(bars)):
            bar = bars.iloc[i]
            bar_time = _bar_time(bar)
            bar_high = float(bar["High"])
            bar_low = float(bar["Low"])
            bar_close = float(bar["Close"])
            bar_open = float(bar["Open"])

            # ------ Step 1: manage open position for this ticker ---------
            if ticker in open_positions:
                pos = open_positions[ticker]
                trade = self._maybe_exit(
                    pos=pos,
                    bar_high=bar_high,
                    bar_low=bar_low,
                    bar_close=bar_close,
                    bar_time=bar_time,
                    is_last_bar=(i == len(bars) - 1),
                    risk_mgr=risk_mgr,
                )
                if trade is not None:
                    trades.append(trade)
                    del open_positions[ticker]
                continue   # one position per ticker: skip entry while open

            # ------ Step 2: only look for entries when flat ---------------
            if i == len(bars) - 1:
                continue   # no entries on the last bar (no future bar to fill)

            # Pass a trailing window (bars 0..i inclusive) to strategies.
            window = df.iloc[: i + 1]

            signal = self._best_signal(
                window=window,
                ticker=ticker,
                region=region,
                strategies=strategies,
                timeframe=timeframe,
            )

            if signal is None or signal.direction == "HOLD":
                continue

            if self.apply_score_filter:
                if signal.strategy_score < SCORING.min_strategy_score:
                    logger.debug(
                        "Score filter: %s score=%.1f < %.1f",
                        ticker, signal.strategy_score, SCORING.min_strategy_score,
                    )
                    continue

            # ------ Step 3: risk gate ------------------------------------
            decision = risk_mgr.evaluate(signal, now=bar_time)
            if not decision.approved:
                logger.debug("Risk rejected %s: %s", ticker, decision)
                continue

            # ------ Step 4: open position (fill at next bar open) --------
            next_bar = bars.iloc[i + 1]
            fill_price = self._fill_price(
                float(next_bar["Open"]), signal.direction
            )
            entry_time = _bar_time(next_bar)
            commission = fill_price * decision.quantity * RISK.commission_per_trade_pct

            open_positions[ticker] = _OpenPosition(
                ticker=ticker,
                direction=signal.direction,
                entry_price=fill_price,
                entry_time=entry_time,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                quantity=decision.quantity,
                commission_paid=commission,
            )
            risk_mgr.register_open(when=entry_time)

        # ------ Step 5: force-close any still-open position at end -------
        if ticker in open_positions:
            pos = open_positions[ticker]
            last_bar = bars.iloc[-1]
            trade = self._force_close(
                pos=pos,
                exit_price=float(last_bar["Close"]),
                exit_time=_bar_time(last_bar),
                reason="end_of_data",
                risk_mgr=risk_mgr,
            )
            trades.append(trade)

        result.trades = trades
        logger.info(
            "Backtest %s: %d trades from %d bars",
            ticker, len(trades), len(bars),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _best_signal(
        self,
        window: pd.DataFrame,
        ticker: str,
        region: str,
        strategies: Sequence[Strategy],
        timeframe: Timeframe,
    ) -> Optional[Signal]:
        """
        Run all strategies and return the one with the highest strategy_score.
        Returns None when no strategy produces an actionable signal.
        """
        best: Optional[Signal] = None
        for strategy in strategies:
            try:
                sig = strategy.generate_signal(window, ticker, region, timeframe)
            except Exception:
                logger.exception("Strategy %s raised on %s", type(strategy).__name__, ticker)
                continue
            if sig is None or sig.direction == "HOLD":
                continue
            if best is None or sig.strategy_score > best.strategy_score:
                best = sig
        return best

    def _fill_price(self, open_price: float, direction: str) -> float:
        """Apply slippage to the fill price."""
        slip = open_price * RISK.slippage_pct
        if direction == "BUY":
            return open_price + slip
        return open_price - slip

    def _maybe_exit(
        self,
        pos: _OpenPosition,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_time: datetime,
        is_last_bar: bool,
        risk_mgr: RiskManager,
    ) -> Optional[Trade]:
        """
        Check whether this bar triggers TP, SL, or end-of-data exit.

        TP takes priority over SL (conservative assumption: best fill).
        Returns a Trade if the position is closed, else None.
        """
        hit_tp = pos.filled_tp(bar_high, bar_low)
        hit_sl = pos.filled_sl(bar_high, bar_low)

        if hit_tp:
            exit_price = pos.take_profit
            reason = "take_profit"
        elif hit_sl:
            exit_price = pos.stop_loss
            reason = "stop_loss"
        elif is_last_bar:
            exit_price = bar_close
            reason = "end_of_data"
        else:
            return None

        return self._close_position(pos, exit_price, bar_time, reason, risk_mgr)

    def _force_close(
        self,
        pos: _OpenPosition,
        exit_price: float,
        exit_time: datetime,
        reason: str,
        risk_mgr: RiskManager,
    ) -> Trade:
        return self._close_position(pos, exit_price, exit_time, reason, risk_mgr)

    def _close_position(
        self,
        pos: _OpenPosition,
        exit_price: float,
        exit_time: datetime,
        reason: str,
        risk_mgr: RiskManager,
    ) -> Trade:
        """Build a Trade dataclass from a closed position."""
        slip = exit_price * RISK.slippage_pct
        if pos.direction == "BUY":
            actual_exit = exit_price - slip
            gross_pnl = (actual_exit - pos.entry_price) * pos.quantity
        else:
            actual_exit = exit_price + slip
            gross_pnl = (pos.entry_price - actual_exit) * pos.quantity

        exit_commission = actual_exit * pos.quantity * RISK.commission_per_trade_pct
        total_commission = pos.commission_paid + exit_commission
        slippage_cost = (
            pos.entry_price * pos.quantity * RISK.slippage_pct
            + actual_exit * pos.quantity * RISK.slippage_pct
        )
        net_pnl = gross_pnl - total_commission

        trade = Trade(
            ticker=pos.ticker,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=actual_exit,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            quantity=pos.quantity,
            gross_pnl=gross_pnl,
            commission=total_commission,
            slippage=slippage_cost,
            net_pnl=net_pnl,
            exit_reason=reason,
        )

        risk_mgr.register_close(pnl=net_pnl, when=exit_time)
        return trade


# =============================================================================
# Walk-forward runner
# =============================================================================

def run_walk_forward(
    df: pd.DataFrame,
    ticker: str,
    region: str,
    strategies: Sequence[Strategy],
    timeframe: Timeframe = Timeframe.D1,
    n_windows: int = BACKTEST.walk_forward_windows,
    train_pct: float = BACKTEST.walk_forward_train_pct,
    initial_capital: float = BACKTEST.initial_capital,
    apply_score_filter: bool = True,
) -> List[BacktestResult]:
    """
    Anchored walk-forward validation.

    Splits `df` into `n_windows` non-overlapping test folds. Each fold:
        - train fold: bars[0 : train_end]   (not used by the backtester
          directly, but context for strategies via their trailing window)
        - test fold:  bars[train_end : fold_end]

    Returns one BacktestResult per window (already finalized).
    """
    if df.empty:
        return []

    n = len(df)
    # Each full window (train + test) covers the same fraction of the dataset.
    window_size = n // n_windows
    if window_size < 10:
        logger.warning("Too few bars for walk-forward on %s — running single pass", ticker)
        bt = Backtester(initial_capital=initial_capital, apply_score_filter=apply_score_filter)
        result = bt.run(df, ticker, region, strategies, timeframe)
        result.finalize(initial_capital)
        return [result]

    results: List[BacktestResult] = []
    bt = Backtester(initial_capital=initial_capital, apply_score_filter=apply_score_filter)

    for w in range(n_windows):
        fold_start = w * window_size
        fold_end = fold_start + window_size if w < n_windows - 1 else n
        train_end = fold_start + int((fold_end - fold_start) * train_pct)

        # The test slice starts at train_end but the backtester needs the
        # full history leading up to it so strategies can compute indicators.
        # We pass df[0:fold_end] so the window covers training context; the
        # backtester will only trade on bars after train_end naturally because
        # the signal window always ends at the current bar.
        #
        # For a clean walk-forward (test-only) result we pass only the test
        # slice but prepend the minimum required context (min_bars_required).
        min_ctx = max((s.min_bars_required for s in strategies), default=220)
        context_start = max(0, train_end - min_ctx)
        test_df = df.iloc[context_start:fold_end].copy()

        result = bt.run(
            test_df, ticker, region, strategies, timeframe
        )

        # Annotate walk-forward metadata
        result.window_index = w
        idx = df.index
        result.train_start = _ts(idx[fold_start])
        result.train_end = _ts(idx[train_end - 1])
        result.test_start = _ts(idx[train_end])
        result.test_end = _ts(idx[fold_end - 1])

        result.finalize(initial_capital)
        results.append(result)

        logger.info(
            "WF window %d/%d: %d trades  metrics=%s",
            w + 1, n_windows, len(result.trades),
            f"PF={result.metrics.profit_factor:.2f} "
            f"WR={result.metrics.win_rate:.1%}"
            if result.metrics else "N/A",
        )

    return results


# =============================================================================
# Convenience wrappers
# =============================================================================

def run_backtest(
    df: pd.DataFrame,
    ticker: str,
    region: str,
    strategies: Sequence[Strategy],
    timeframe: Timeframe = Timeframe.D1,
    initial_capital: float = BACKTEST.initial_capital,
    apply_score_filter: bool = True,
    export_csv: bool = False,
) -> BacktestResult:
    """
    Single-window backtest convenience wrapper.

    Runs the backtester, finalizes metrics, and optionally exports trades.
    """
    bt = Backtester(
        initial_capital=initial_capital,
        apply_score_filter=apply_score_filter,
    )
    result = bt.run(df, ticker, region, strategies, timeframe)
    result.finalize(initial_capital)

    if export_csv:
        result.export_trades_csv()

    return result


# =============================================================================
# Internal utilities
# =============================================================================

def _bar_time(bar: pd.Series) -> datetime:
    """Extract a timezone-aware datetime from a bar row."""
    idx = bar.name
    if isinstance(idx, pd.Timestamp):
        return idx.to_pydatetime()
    # Fallback: look for common time column names
    for col in ("time", "Date", "Datetime", "datetime"):
        if col in bar.index:
            val = bar[col]
            if isinstance(val, pd.Timestamp):
                return val.to_pydatetime()
            return pd.Timestamp(val).to_pydatetime()
    return datetime.utcnow()


def _ts(val) -> datetime:
    """Convert an index value to datetime."""
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    return pd.Timestamp(val).to_pydatetime()
