"""
Global Trading Dashboard — Streamlit entry-point.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ── project root on path ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── config & helpers ─────────────────────────────────────────────────────────
from config import (
    CACHE,
    EXPORT_DIR,
    MODEL_PATH,
    FEATURE_PATH,
    SCORING,
    UI_COLORS,
    Region,
    StrategyName,
    Timeframe,
    get_tickers_for_region,
    yf_period_for_timeframe,
)
from utils.helpers import Signal, Trade, combine_final_score, compute_confluence_score

# ── modules ──────────────────────────────────────────────────────────────────
import strategies.base_strategy as _base_strategy

from alerts.notifier import get_dispatcher
from analysis.performance import compute_metrics, equity_curve, trades_to_dataframe
from backtesting.backtester import BacktestResult, run_backtest, run_walk_forward
from data.data_fetcher import fetch_history, fetch_many
from execution.alpaca_client import AlpacaClient
from ml.model import TradeRanker
from strategies import REGISTRY, build_strategy
from utils.indicators import with_indicators


# =============================================================================
# Page config (must be the FIRST Streamlit call)
# =============================================================================

st.set_page_config(
    page_title="Global Trading Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS overrides
# =============================================================================

st.markdown(
    """
    <style>
    [data-testid="stMetricLabel"]  { font-size: 0.8rem; color: #9ca3af; }
    [data-testid="stMetricValue"]  { font-size: 1.15rem; font-weight: 600; }
    .signal-card {
        border: 1px solid #2a2f3a;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        background: #161a24;
    }
    .signal-card h4 { margin: 0 0 6px; font-size: 1rem; }
    .badge {
        display: inline-block;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Singletons (cached in st.session_state)
# =============================================================================

@st.cache_resource
def _get_alpaca() -> AlpacaClient:
    return AlpacaClient()


@st.cache_resource
def _get_ranker() -> Optional[TradeRanker]:
    if MODEL_PATH.exists() and FEATURE_PATH.exists():
        try:
            return TradeRanker.load(MODEL_PATH, FEATURE_PATH)
        except Exception as exc:
            st.warning(f"ML model failed to load: {exc}")
    return None


# =============================================================================
# Data helpers (TTL-cached)
# =============================================================================

@st.cache_data(ttl=CACHE.daily_ttl_seconds, show_spinner=False)
def _cached_fetch(ticker: str, timeframe_val: str, period: str) -> pd.DataFrame:
    tf = Timeframe(timeframe_val)
    return fetch_history(ticker, timeframe=tf, period=period)


@st.cache_data(ttl=CACHE.daily_ttl_seconds, show_spinner=False)
def _cached_fetch_many(tickers_key: str, timeframe_val: str) -> Dict[str, pd.DataFrame]:
    tickers = tickers_key.split(",")
    tf = Timeframe(timeframe_val)
    return fetch_many(tickers, timeframe=tf)


# =============================================================================
# Signal pipeline
# =============================================================================

def _run_strategies_debug(
    df: pd.DataFrame,
    ticker: str,
    region: str,
    strategy_names: List[str],
    timeframe: Timeframe,
    ranker: Optional[TradeRanker],
) -> Dict[str, Any]:
    """
    Same as _run_strategies but returns a detailed diagnostics dict.
    Used by the debug expander in the Signals tab.
    """
    strategies = [build_strategy(StrategyName(n)) for n in strategy_names]
    results = []
    for strat in strategies:
        entry: Dict[str, Any] = {"strategy": type(strat).__name__, "direction": "HOLD",
                                  "score": 0, "error": None}
        try:
            sig = strat.generate_signal(df, ticker, region, timeframe)
            if sig is not None:
                entry["direction"] = sig.direction
                entry["score"] = sig.strategy_score
                entry["breakdown"] = sig.confluence_breakdown
        except Exception as exc:
            entry["error"] = str(exc)
        results.append(entry)
    return {
        "ticker": ticker,
        "bars": len(df),
        "strategies": results,
        "best_score": max((r["score"] for r in results), default=0),
    }


def _run_strategies(
    df: pd.DataFrame,
    ticker: str,
    region: str,
    strategy_names: List[str],
    timeframe: Timeframe,
    ranker: Optional[TradeRanker],
    min_strategy_score: float = 60.0,
    min_final_score: float = 0.60,
) -> Optional[Signal]:
    """
    Run selected strategies, pick the highest-scoring signal, apply ML scoring,
    recompute final confluence, and return the enriched signal (or None).
    """
    strategies = [build_strategy(StrategyName(n)) for n in strategy_names]
    best: Optional[Signal] = None
    for strat in strategies:
        try:
            sig = strat.generate_signal(df, ticker, region, timeframe)
        except Exception:
            continue
        if sig is None or sig.direction == "HOLD":
            continue
        if best is None or sig.strategy_score > best.strategy_score:
            best = sig

    if best is None or best.direction == "HOLD":
        return None

    # ML scoring
    ml_score = 0.0
    if ranker and ranker.is_trained:
        ml_score = ranker.score_signal(best, df)
    best.ml_score = ml_score

    # Recompute confluence with ML score, then final score
    confluence = compute_confluence_score(
        trend_aligned=(best.confluence_breakdown.get("trend_alignment", 0) > 0),
        volume_confirmed=(best.confluence_breakdown.get("volume_confirmation", 0) > 0),
        near_support_resistance=(best.confluence_breakdown.get("support_resistance", 0) > 0),
        candle_confirmed=(best.confluence_breakdown.get("candlestick_confirmation", 0) > 0),
        ml_score=ml_score,
    )
    best.strategy_score = float(confluence.total)
    best.confluence_breakdown = confluence.as_dict()
    best.final_score = combine_final_score(best.strategy_score, ml_score)

    # Gate: use the slider values passed in from the sidebar
    if best.strategy_score < min_strategy_score:
        return None
    if best.final_score < min_final_score:
        return None

    return best


def _scan_universe(
    tickers: List[str],
    region: Region,
    strategy_names: List[str],
    timeframe: Timeframe,
    ranker: Optional[TradeRanker],
    min_strategy_score: float = 60.0,
    min_final_score: float = 0.60,
) -> List[Signal]:
    """Fetch all tickers in parallel and generate signals."""
    tickers_key = ",".join(sorted(set(tickers)))
    dfs = _cached_fetch_many(tickers_key, timeframe.value)

    signals: List[Signal] = []
    for ticker, df in dfs.items():
        if df is None or df.empty:
            continue
        sig = _run_strategies(
            df, ticker, region.value, strategy_names, timeframe, ranker,
            min_strategy_score=min_strategy_score,
            min_final_score=min_final_score,
        )
        if sig is not None:
            signals.append(sig)

    signals.sort(key=lambda s: s.final_score, reverse=True)
    return signals


# =============================================================================
# Plotly helpers
# =============================================================================

def _candlestick_chart(df: pd.DataFrame, ticker: str, signal: Optional[Signal] = None):
    """Plotly candlestick + EMA50/EMA200 + optional signal levels."""
    import plotly.graph_objects as go

    prepared = with_indicators(df.copy())
    display = prepared.tail(120)
    x = display.index

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x, open=display["Open"], high=display["High"],
        low=display["Low"], close=display["Close"],
        name=ticker, increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
    ))
    if "ema_fast" in display.columns:
        fig.add_trace(go.Scatter(
            x=x, y=display["ema_fast"], name="EMA 50",
            line=dict(color="#3b82f6", width=1.2),
        ))
    if "ema_slow" in display.columns:
        fig.add_trace(go.Scatter(
            x=x, y=display["ema_slow"], name="EMA 200",
            line=dict(color="#f59e0b", width=1.2),
        ))

    if signal:
        color = "#22c55e" if signal.direction == "BUY" else "#ef4444"
        fig.add_hline(y=signal.entry, line=dict(color=color, width=1, dash="dash"),
                      annotation_text="Entry", annotation_position="right")
        fig.add_hline(y=signal.stop_loss, line=dict(color="#ef4444", width=1, dash="dot"),
                      annotation_text="SL", annotation_position="right")
        fig.add_hline(y=signal.take_profit, line=dict(color="#22c55e", width=1, dash="dot"),
                      annotation_text="TP", annotation_position="right")

    fig.update_layout(
        height=420, template="plotly_dark",
        margin=dict(l=0, r=0, t=28, b=0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    )
    return fig


def _equity_curve_chart(eq_series: pd.Series, title: str = "Equity Curve"):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq_series.index, y=eq_series.values,
        fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="#3b82f6", width=2), name="Equity",
    ))
    fig.update_layout(
        height=300, template="plotly_dark",
        margin=dict(l=0, r=0, t=28, b=0),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title=dict(text=title, x=0.01, font=dict(size=14)),
    )
    return fig


def _feature_importance_chart(importances: Dict[str, float]):
    import plotly.graph_objects as go

    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color="#3b82f6",
    ))
    fig.update_layout(
        height=480, template="plotly_dark",
        margin=dict(l=0, r=20, t=28, b=0),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        yaxis=dict(autorange="reversed"),
        title=dict(text="Feature Importances", x=0.01, font=dict(size=14)),
    )
    return fig


# =============================================================================
# Formatting / styling helpers
# =============================================================================

def _color_direction(val: str) -> str:
    if str(val).upper() == "BUY":
        return "color: #22c55e; font-weight:600"
    if str(val).upper() == "SELL":
        return "color: #ef4444; font-weight:600"
    return ""


def _color_score(val, threshold: float = SCORING.min_strategy_score) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v >= threshold:
        return "color: #22c55e"
    return "color: #f59e0b"


def _color_pnl(val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    return "color: #22c55e" if v > 0 else ("color: #ef4444" if v < 0 else "")


def _signal_table(signals: List[Signal]) -> pd.DataFrame:
    rows = [s.as_row() for s in signals]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df


# =============================================================================
# Tab implementations
# =============================================================================

def _tab_signals(
    region: Region,
    tickers: List[str],
    strategy_names: List[str],
    timeframe: Timeframe,
    ranker: Optional[TradeRanker],
) -> None:
    """Signals tab: scan → table → cards → price chart."""
    min_strat = float(st.session_state.get("sb_min_score", 40))
    min_final = float(st.session_state.get("sb_min_final", 0.40))

    # Sync the module-level bypass flag so all strategies pick it up.
    _base_strategy.BYPASS_GLOBAL_FILTER = bool(
        st.session_state.get("bypass_global_filter", False)
    )

    with st.spinner("Scanning universe…"):
        signals = _scan_universe(
            tickers, region, strategy_names, timeframe, ranker,
            min_strategy_score=min_strat,
            min_final_score=min_final,
        )

    # ── Debug expander ────────────────────────────────────────────────────────
    with st.expander("🔍 Diagnostics (open if no signals appear)"):
        tickers_key = ",".join(sorted(set(tickers)))
        dfs_debug = _cached_fetch_many(tickers_key, timeframe.value)
        nodata = [t for t, d in dfs_debug.items() if d is None or d.empty]
        hasdata = {t: d for t, d in dfs_debug.items() if d is not None and not d.empty}
        st.write(f"**Tickers with data:** {len(hasdata)} / {len(dfs_debug)}")
        if nodata:
            st.warning(f"No data returned for: {nodata}")

        if hasdata:
            from strategies.base_strategy import Strategy as _S, ATR_FILTER_THRESHOLD, VOL_FILTER_THRESHOLD
            from utils.indicators import with_indicators as _wi

            # ── Global filter summary table (one row per ticker) ──────────
            st.markdown("**Global filter per ticker** (first 10 tickers, BUY direction)")
            filter_rows = []
            for _t, _df in list(hasdata.items())[:10]:
                try:
                    _prep = _wi(_df)
                    _d = _S.global_filter_detail(_prep, "BUY")
                    filter_rows.append({
                        "Ticker": _t,
                        "Trend": "✅" if _d["trend"] else "❌",
                        "EMA50": _d["ema_fast"],
                        "EMA200": _d["ema_slow"],
                        f"ATR≥{ATR_FILTER_THRESHOLD:.0%}avg": "✅" if _d["atr"] else "❌",
                        "ATR now": _d["atr_now"],
                        "ATR avg": _d["atr_ma"],
                        f"Vol≥{VOL_FILTER_THRESHOLD:.0%}avg": "✅" if _d["volume"] else "❌",
                        "Vol ratio": _d["vol_ratio"],
                        "PASS": "✅ PASS" if all([_d["trend"], _d["atr"], _d["volume"]]) else "❌ FAIL",
                    })
                except Exception as _ex:
                    filter_rows.append({"Ticker": _t, "PASS": f"error: {_ex}"})
            if filter_rows:
                st.dataframe(pd.DataFrame(filter_rows), use_container_width=True, hide_index=True)

            # ── Strategy signal table (first 5 tickers) ───────────────────
            st.markdown("**Strategy signal breakdown** (first 5 tickers)")
            debug_rows = []
            for ticker_d, df_d in list(hasdata.items())[:5]:
                diag = _run_strategies_debug(
                    df_d, ticker_d, region.value, strategy_names, timeframe, ranker
                )
                for r in diag["strategies"]:
                    debug_rows.append({
                        "Ticker": ticker_d,
                        "Strategy": r["strategy"],
                        "Direction": r["direction"],
                        "Score": r["score"],
                        "Error": r.get("error") or "—",
                    })
            if debug_rows:
                st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, hide_index=True)
            st.info("💡 Enable **Bypass global filter** in the sidebar to explore signals in calm markets.")

    if not signals:
        st.info("No signals meeting the score threshold right now. Open the Diagnostics expander above to see raw strategy output.")
        return

    # ── Direction filter ──────────────────────────────────────────────────────
    dir_filter = st.radio(
        "Show signals",
        options=["All", "BUY only", "SELL only"],
        horizontal=True,
        key="sig_dir_filter",
    )
    if dir_filter == "BUY only":
        signals = [s for s in signals if s.direction == "BUY"]
    elif dir_filter == "SELL only":
        signals = [s for s in signals if s.direction == "SELL"]

    if not signals:
        st.info(f"No {dir_filter.replace(' only', '')} signals right now.")
        return

    st.success(f"Found **{len(signals)}** signal(s) · {sum(s.direction=='BUY' for s in signals)} BUY · {sum(s.direction=='SELL' for s in signals)} SELL")

    # ── Signal table ──────────────────────────────────────────────────────────
    st.subheader("Signal Table")
    rows = []
    for sig in signals:
        # Estimate hold time from timeframe + ATR-based TP distance.
        # TP = 3× ATR from entry → takes ~3-8 bars to reach on average.
        tf = sig.timeframe
        if tf == "1D":
            hold_est = "3–8 trading days"
        elif tf == "4H":
            hold_est = "12–32 hours (3–8 bars)"
        elif tf == "1H":
            hold_est = "3–8 hours"
        else:
            hold_est = "3–8 bars"
        row = sig.as_row()
        row["Est. Hold"] = hold_est
        row["Reason"] = sig.reason or "—"
        rows.append(row)

    df_table = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df_table.empty:
        # Reorder: trade details first, then scores
        cols_order = [
            "Ticker", "Direction", "Strategy", "Timeframe",
            "Entry", "Stop Loss", "Take Profit", "R:R",
            "Est. Hold", "Strategy Score", "Final Score",
            "Reason", "Timestamp",
        ]
        df_table = df_table[[c for c in cols_order if c in df_table.columns]]
        styled = df_table.style
        if "Direction" in df_table.columns:
            styled = styled.applymap(_color_direction, subset=["Direction"])
        if "Strategy Score" in df_table.columns:
            styled = styled.applymap(_color_score, subset=["Strategy Score"])
        styled = styled.format({
            "Entry": "{:.2f}", "Stop Loss": "{:.2f}", "Take Profit": "{:.2f}",
            "R:R": "{:.2f}", "Strategy Score": "{:.1f}", "Final Score": "{:.3f}",
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Signal cards (top 5) ──────────────────────────────────────────────────
    st.subheader("Trade Setup Details")
    top = signals[:5]
    for sig in top:
        dir_color = "#22c55e" if sig.direction == "BUY" else "#ef4444"
        score_color = "#22c55e" if sig.strategy_score >= SCORING.min_strategy_score else "#f59e0b"
        breakdown = sig.confluence_breakdown
        breakdown_str = " | ".join(
            f"{k.replace('_', ' ').title()}: {v}"
            for k, v in breakdown.items()
            if k != "total"
        )
        risk_pct = abs(sig.entry - sig.stop_loss) / sig.entry * 100 if sig.entry else 0
        reward_pct = abs(sig.take_profit - sig.entry) / sig.entry * 100 if sig.entry else 0
        tf = sig.timeframe
        if tf == "1D":
            hold_est = "3–8 trading days"
        elif tf == "4H":
            hold_est = "12–32 hours"
        elif tf == "1H":
            hold_est = "3–8 hours"
        else:
            hold_est = "3–8 bars"
        st.markdown(
            f"""
            <div class="signal-card">
                <h4>
                  <span style="color:{dir_color}">{sig.direction}</span>
                  &nbsp;{sig.ticker}
                  <span style="color:#9ca3af; font-size:0.8rem"> · {sig.strategy} · {sig.timeframe}</span>
                </h4>
                <span class="badge" style="background:{score_color}22;color:{score_color}">
                    Score {sig.strategy_score:.0f}/100
                </span>
                &nbsp;
                <span class="badge" style="background:#6366f122;color:#6366f1">
                    Final {sig.final_score:.2f}
                </span>
                &nbsp;
                <span class="badge" style="background:#8b5cf622;color:#8b5cf6">
                    Hold ~{hold_est}
                </span>
                <p style="margin:10px 0 6px; font-size:0.85rem; color:#e2e8f0;
                          background:#1e2535; padding:8px 12px; border-radius:6px;
                          border-left:3px solid {dir_color}">
                    {sig.reason or 'No reason provided'}
                </p>
                <table style="width:100%;border-collapse:collapse;font-size:0.85rem;margin-top:6px">
                  <tr>
                    <td style="padding:4px 12px 4px 0;color:#9ca3af">Entry</td>
                    <td style="padding:4px 12px 4px 0"><b>{sig.entry:.2f}</b></td>
                    <td style="padding:4px 12px 4px 0;color:#9ca3af">Stop Loss</td>
                    <td style="padding:4px 12px 4px 0"><b style="color:#ef4444">{sig.stop_loss:.2f}</b>
                      <span style="color:#9ca3af;font-size:0.75rem"> ({risk_pct:.1f}% risk)</span></td>
                    <td style="padding:4px 12px 4px 0;color:#9ca3af">Take Profit</td>
                    <td style="padding:4px 0"><b style="color:#22c55e">{sig.take_profit:.2f}</b>
                      <span style="color:#9ca3af;font-size:0.75rem"> ({reward_pct:.1f}% gain)</span></td>
                  </tr>
                  <tr>
                    <td style="padding:4px 12px 4px 0;color:#9ca3af">R:R Ratio</td>
                    <td style="padding:4px 12px 4px 0"><b>{sig.rr_ratio:.2f}</b></td>
                    <td colspan="4" style="padding:4px 0;color:#9ca3af;font-size:0.8rem">
                      {breakdown_str}
                    </td>
                  </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Price chart ───────────────────────────────────────────────────────────
    st.subheader("Price Chart")
    chart_ticker = st.selectbox(
        "Select ticker", [s.ticker for s in signals], key="chart_ticker_signals"
    )
    focused = next((s for s in signals if s.ticker == chart_ticker), None)
    period = yf_period_for_timeframe(timeframe)
    df_chart = _cached_fetch(chart_ticker, timeframe.value, period)
    if not df_chart.empty:
        st.plotly_chart(
            _candlestick_chart(df_chart, chart_ticker, focused),
            use_container_width=True,
        )

    # ── Alert dispatch ────────────────────────────────────────────────────────
    if st.session_state.get("alerts_enabled", True):
        dispatcher = get_dispatcher()
        for sig in signals:
            dispatcher.notify_signal(sig)


def _tab_backtest(
    region: Region,
    tickers: List[str],
    strategy_names: List[str],
    timeframe: Timeframe,
) -> None:
    """Backtest tab: run engine, show equity curve + metrics + trades."""
    st.subheader("Backtest Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        bt_ticker = st.selectbox("Ticker", tickers, key="bt_ticker")
    with col2:
        bt_strategy = st.selectbox(
            "Strategy", strategy_names, key="bt_strategy"
        )
    with col3:
        walk_forward = st.checkbox("Walk-forward validation", value=False, key="bt_wf")

    capital = st.number_input(
        "Initial Capital ($)", min_value=1000.0, value=100_000.0,
        step=1000.0, key="bt_capital"
    )
    run_btn = st.button("Run Backtest", type="primary", key="bt_run")

    if not run_btn:
        return

    period = yf_period_for_timeframe(timeframe)
    with st.spinner(f"Fetching {bt_ticker}…"):
        df_bt = _cached_fetch(bt_ticker, timeframe.value, period)

    if df_bt is None or df_bt.empty:
        st.error(f"No data for {bt_ticker}.")
        return

    strategy = build_strategy(StrategyName(bt_strategy))

    with st.spinner("Running backtest…"):
        if walk_forward:
            results = run_walk_forward(
                df=df_bt, ticker=bt_ticker, region=region.value,
                strategies=[strategy], timeframe=timeframe,
                initial_capital=float(capital),
            )
            # Aggregate across windows for a summary view
            all_trades = [t for r in results for t in r.trades]
            metrics = compute_metrics(all_trades, float(capital))
            eq = equity_curve(all_trades, float(capital))
        else:
            result = run_backtest(
                df=df_bt, ticker=bt_ticker, region=region.value,
                strategies=[strategy], timeframe=timeframe,
                initial_capital=float(capital),
            )
            all_trades = result.trades
            metrics = result.metrics
            eq = result.equity_series

    if metrics is None:
        st.warning("Backtest produced no metrics.")
        return

    # ── Equity curve ──────────────────────────────────────────────────────────
    if eq is not None and len(eq) > 1:
        st.plotly_chart(_equity_curve_chart(eq, f"{bt_ticker} Equity Curve"),
                        use_container_width=True)

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.subheader("Performance Metrics")
    formatted = metrics.format_for_ui()
    half = len(formatted) // 2
    items = list(formatted.items())
    col_a, col_b = st.columns(2)
    for k, v in items[:half]:
        col_a.metric(k, v)
    for k, v in items[half:]:
        col_b.metric(k, v)

    # ── Walk-forward window summary ────────────────────────────────────────────
    if walk_forward and results:
        st.subheader("Walk-Forward Windows")
        wf_rows = []
        for r in results:
            m = r.metrics
            if m:
                wf_rows.append({
                    "Window": r.window_index,
                    "Test Start": r.test_start,
                    "Test End": r.test_end,
                    "Trades": m.total_trades,
                    "Win Rate": f"{m.win_rate:.1%}",
                    "Profit Factor": f"{m.profit_factor:.2f}",
                    "Total Return": f"{m.total_return_pct:.2%}",
                    "Max DD": f"{m.max_drawdown_pct:.2%}",
                    "Sharpe": f"{m.sharpe_ratio:.2f}",
                })
        if wf_rows:
            st.dataframe(pd.DataFrame(wf_rows), use_container_width=True, hide_index=True)

    # ── Trades table ──────────────────────────────────────────────────────────
    if all_trades:
        st.subheader(f"Trades ({len(all_trades)})")
        df_trades = trades_to_dataframe(all_trades)
        if not df_trades.empty:
            styled = df_trades.style
            if "Net PnL" in df_trades.columns:
                styled = styled.applymap(_color_pnl, subset=["Net PnL"])
            styled = styled.format(
                {col: "{:.2f}" for col in df_trades.select_dtypes("number").columns},
                na_rep="—",
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

        # CSV export
        csv = df_trades.to_csv(index=False)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "Export trades CSV",
            data=csv,
            file_name=f"{bt_ticker}_{bt_strategy}_{ts}.csv",
            mime="text/csv",
        )


def _tab_portfolio(alpaca: AlpacaClient) -> None:
    """Portfolio tab: account snapshot + positions + emergency controls."""
    if not alpaca.is_enabled:
        st.info(
            "Alpaca is not connected. Set `ALPACA_API_KEY` and "
            "`ALPACA_SECRET_KEY` in `.env` to enable live trading."
        )
        return

    with st.spinner("Fetching account…"):
        account = alpaca.get_account()

    if not account.connected:
        st.error(f"Could not reach Alpaca: {account.error}")
        return

    # ── Account metrics ───────────────────────────────────────────────────────
    st.subheader("Account Snapshot")
    mode_badge = ":test_tube: Paper" if account.paper else ":zap: Live"
    st.caption(f"{mode_badge}  ·  Status: **{account.status}**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Equity", f"${account.equity:,.2f}")
    col2.metric("Cash", f"${account.cash:,.2f}")
    col3.metric("Buying Power", f"${account.buying_power:,.2f}")
    col4.metric("Portfolio Value", f"${account.portfolio_value:,.2f}")

    # ── Open positions ────────────────────────────────────────────────────────
    st.subheader("Open Positions")
    with st.spinner("Fetching positions…"):
        positions = alpaca.get_positions()

    if not positions:
        st.info("No open positions.")
    else:
        pos_rows = [
            {
                "Ticker": p.ticker,
                "Side": p.side.upper(),
                "Qty": p.quantity,
                "Avg Entry": p.avg_entry_price,
                "Current": p.current_price,
                "Market Value": p.market_value,
                "Unreal PnL": p.unrealized_pnl,
                "Unreal PnL %": f"{p.unrealized_pnl_pct:.2%}",
            }
            for p in positions
        ]
        df_pos = pd.DataFrame(pos_rows)
        styled = df_pos.style.applymap(_color_pnl, subset=["Unreal PnL"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Emergency controls ────────────────────────────────────────────────────
    st.subheader("Emergency Controls")
    st.warning(
        "These actions affect your **live paper account** immediately.",
        icon=":warning:",
    )
    col_cancel, col_close = st.columns(2)
    with col_cancel:
        if st.button("Cancel All Orders", type="secondary", key="cancel_all"):
            n = alpaca.cancel_all_orders()
            st.success(f"Cancelled {n} order(s).")
    with col_close:
        if st.button(
            "Close All Positions", type="secondary", key="close_all",
            help="Liquidates every open position at market price."
        ):
            n = alpaca.close_all_positions()
            st.success(f"Flattened {n} position(s).")


def _tab_ml(ranker: Optional[TradeRanker]) -> None:
    """ML diagnostics tab: model status + train button + metrics + feature importances."""
    st.subheader("ML Model Status")

    # ── Train model button ────────────────────────────────────────────────────
    st.markdown("### Train the Model")
    st.info(
        "Training scans historical data for all tickers, runs every strategy "
        "as a backtest, labels each trade win/loss, and fits a RandomForest "
        "classifier. Takes **2–5 minutes** depending on how many tickers you include. "
        "Run this once — then re-run whenever you want to refresh the model."
    )

    from config import METRICS_PATH, Region as _R, Timeframe as _TF

    col_r, col_n, col_tf = st.columns(3)
    with col_r:
        train_regions = st.multiselect(
            "Regions to train on",
            options=["us", "europe", "asia"],
            default=["us"],
            key="train_regions",
        )
    with col_n:
        max_tickers = st.slider(
            "Tickers per region",
            min_value=3, max_value=20, value=8, step=1,
            key="train_max_tickers",
            help="More tickers = better model but slower training. 5–10 is a good start.",
        )
    with col_tf:
        train_tf = st.selectbox(
            "Timeframe",
            options=[tf.value for tf in _TF],
            index=0,
            key="train_tf",
        )

    if st.button("🚀 Train Model Now", type="primary", key="train_btn"):
        from ml.train import _parse_regions, train_pipeline
        import json as _json

        prog = st.progress(0, text="Starting training…")
        status = st.empty()

        try:
            status.info("Fetching historical data and running backtests…")
            prog.progress(20, text="Fetching data & running backtests…")

            regions = _parse_regions(train_regions or ["us"])
            result = train_pipeline(
                regions=regions,
                timeframe=_TF(train_tf),
                max_tickers_per_region=max_tickers,
                save_artifacts=True,
            )
            prog.progress(100, text="Done!")

            if result.model is None:
                status.error(
                    "Training failed — not enough trades were collected. "
                    "Try adding more tickers or a longer timeframe."
                )
            else:
                m = result.metrics
                status.success(
                    f"Model trained on **{result.n_trades}** trades.  "
                    f"Accuracy: **{m.get('accuracy', 0):.1%}**  |  "
                    f"F1: **{m.get('f1', 0):.3f}**  |  "
                    f"ROC-AUC: **{m.get('roc_auc', 0):.3f}**"
                )
                st.balloons()
                # Force reload of the cached ranker on next run
                _get_ranker.clear()
                st.info("Model saved. Click **Refresh Data** in the sidebar to activate it.")
        except Exception as _exc:
            prog.progress(100, text="Error")
            status.error(f"Training error: {_exc}")

    st.divider()

    # ── Status & metrics ──────────────────────────────────────────────────────
    if ranker is None or not ranker.is_trained:
        st.warning("No trained model loaded yet. Use the button above to train one.")
        return

    st.success("Model loaded and ready.")
    st.caption(f"Artifact: `{MODEL_PATH.name}`")

    st.subheader("Training Metrics")
    if METRICS_PATH.exists():
        import json
        metrics = json.loads(METRICS_PATH.read_text())
        display_keys = [
            "accuracy", "precision", "recall", "f1", "roc_auc",
            "n_train", "n_test", "pos_rate_test", "trained_at",
            "regions", "timeframe",
        ]
        metric_pairs = [(k, metrics.get(k, "—")) for k in display_keys if k in metrics]
        m_col1, m_col2 = st.columns(2)
        for i, (k, v) in enumerate(metric_pairs):
            col = m_col1 if i % 2 == 0 else m_col2
            col.metric(k.replace("_", " ").title(), str(v) if not isinstance(v, float) else f"{v:.4f}")
    else:
        st.info("No metrics file found. Retrain to generate metrics.")

    # ── Feature importances ───────────────────────────────────────────────────
    st.subheader("Feature Importances")
    importances = ranker.feature_importances()
    if importances:
        st.plotly_chart(
            _feature_importance_chart(importances),
            use_container_width=True,
        )


# =============================================================================
# Sidebar
# =============================================================================

def _build_sidebar() -> Tuple[Region, List[str], List[str], Timeframe]:
    """Render the sidebar and return (region, tickers, strategy_names, timeframe)."""
    with st.sidebar:
        st.title("Controls")
        st.divider()

        # Region
        region_val = st.selectbox(
            "Region",
            options=[r.value for r in Region],
            key="sb_region",
        )
        region = Region(region_val)

        # Timeframe
        tf_val = st.selectbox(
            "Timeframe",
            options=[tf.value for tf in Timeframe],
            key="sb_timeframe",
        )
        timeframe = Timeframe(tf_val)

        # Tickers
        default_tickers = get_tickers_for_region(region)
        custom = st.text_input(
            "Custom tickers (comma-separated, blank = universe)",
            value="",
            key="sb_tickers",
        )
        if custom.strip():
            tickers = [t.strip().upper() for t in custom.split(",") if t.strip()]
            max_t = len(tickers)
        else:
            max_t = st.slider(
                "Max tickers to scan",
                min_value=5, max_value=len(default_tickers),
                value=min(25, len(default_tickers)),
                step=5,
                key="sb_max_tickers",
                help="Fewer tickers = faster initial load. Increase once the cache is warm.",
            )
            tickers = default_tickers[:max_t]

        st.caption(f"{len(tickers)} ticker(s) selected")

        # Strategies
        all_strat_names = [s.value for s in StrategyName]
        strategy_names = st.multiselect(
            "Strategies",
            options=all_strat_names,
            default=all_strat_names,
            key="sb_strategies",
        )
        if not strategy_names:
            strategy_names = all_strat_names

        # Score overrides
        st.divider()
        st.caption("Score Gates  (lower = more signals)")
        st.slider(
            "Min Strategy Score", 0, 100,
            40, key="sb_min_score",
            help="Without a trained ML model, max achievable score is 80. "
                 "Raise to 75 for strict filtering, lower to see more signals."
        )
        st.slider(
            "Min Final Score", 0.0, 1.0,
            0.20, step=0.05, key="sb_min_final",
            help="Final score = 50% strategy score + 50% ML score. "
                 "Without a trained ML model the max achievable is ~0.40. "
                 "Keep at 0.20 until ML is trained, then raise to 0.50+."
        )

        # Filter controls
        st.divider()
        st.checkbox(
            "Bypass global filter",
            value=False,
            key="bypass_global_filter",
            help="Skip the mandatory EMA-trend + ATR + Volume gate. "
                 "Use this to explore raw strategy signals in calm markets.",
        )

        # Alerts toggle
        st.divider()
        st.checkbox("Enable Alerts", value=True, key="alerts_enabled")

        # Status badges
        st.divider()
        alpaca = _get_alpaca()
        ranker = _get_ranker()
        st.caption("System Status")
        st.markdown(
            f"Alpaca: {'🟢 connected' if alpaca.is_enabled else '🔴 offline'}  \n"
            f"ML Model: {'🟢 loaded' if (ranker and ranker.is_trained) else '🟡 not trained'}"
        )

        # Refresh
        st.divider()
        if st.button("Refresh Data", use_container_width=True, key="refresh"):
            st.cache_data.clear()
            # Also clear disk cache so stale yfinance frames are re-downloaded.
            try:
                from data.data_fetcher import clear_cache as _clear_disk_cache
                _clear_disk_cache()
            except Exception:
                pass
            st.rerun()

    return region, tickers, strategy_names, timeframe


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # Header
    st.markdown(
        "## 📊 Global Trading Dashboard",
        unsafe_allow_html=False,
    )
    last_refresh = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    st.caption(f"Last refresh: {last_refresh}")
    st.divider()

    # Sidebar
    region, tickers, strategy_names, timeframe = _build_sidebar()

    # Singletons
    alpaca = _get_alpaca()
    ranker = _get_ranker()

    # Tabs
    tab_signals, tab_backtest, tab_portfolio, tab_ml = st.tabs(
        ["Signals", "Backtest", "Portfolio", "ML Diagnostics"]
    )

    with tab_signals:
        _tab_signals(region, tickers, strategy_names, timeframe, ranker)

    with tab_backtest:
        _tab_backtest(region, tickers, strategy_names, timeframe)

    with tab_portfolio:
        _tab_portfolio(alpaca)

    with tab_ml:
        _tab_ml(ranker)


if __name__ == "__main__":
    main()
