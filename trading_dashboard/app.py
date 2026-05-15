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
    min_strat = float(st.session_state.get("sb_min_score", 60))
    min_final = float(st.session_state.get("sb_min_final", 0.60))

    with st.spinner("Scanning universe…"):
        signals = _scan_universe(
            tickers, region, strategy_names, timeframe, ranker,
            min_strategy_score=min_strat,
            min_final_score=min_final,
        )

    if not signals:
        st.info("No signals meeting the score threshold right now.")
        return

    st.success(f"Found **{len(signals)}** actionable signal(s).")

    # ── Signal table ──────────────────────────────────────────────────────────
    st.subheader("Signal Table")
    df_table = _signal_table(signals)
    if not df_table.empty:
        styled = df_table.style
        if "Direction" in df_table.columns:
            styled = styled.applymap(_color_direction, subset=["Direction"])
        if "Strategy Score" in df_table.columns:
            styled = styled.applymap(_color_score, subset=["Strategy Score"])
        if "Final Score" in df_table.columns:
            styled = styled.applymap(
                lambda v: _color_score(float(v) * 100, SCORING.min_strategy_score),
                subset=["Final Score"],
            )
        styled = styled.format({
            "Entry": "{:.2f}", "Stop Loss": "{:.2f}", "Take Profit": "{:.2f}",
            "R:R": "{:.2f}", "Strategy Score": "{:.1f}",
            "ML Score": "{:.3f}", "Final Score": "{:.3f}",
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Signal cards (top 5) ──────────────────────────────────────────────────
    st.subheader("Top Signals")
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
        st.markdown(
            f"""
            <div class="signal-card">
                <h4>
                  <span style="color:{dir_color}">{sig.direction}</span>
                  {sig.ticker}
                  <span style="color:#9ca3af; font-size:0.8rem"> · {sig.strategy} · {sig.timeframe}</span>
                </h4>
                <span class="badge" style="background:{score_color}22;color:{score_color}">
                    Score {sig.strategy_score:.0f}/100
                </span>
                &nbsp;
                <span class="badge" style="background:#3b82f622;color:#3b82f6">
                    ML {sig.ml_score:.2f} ({sig.ml_band})
                </span>
                &nbsp;
                <span class="badge" style="background:#6366f122;color:#6366f1">
                    Final {sig.final_score:.2f}
                </span>
                <p style="margin:8px 0 4px; font-size:0.85rem; color:#9ca3af">{breakdown_str}</p>
                <p style="margin:0; font-size:0.85rem">
                    Entry <b>{sig.entry:.2f}</b>
                    &nbsp;|&nbsp; SL <b style="color:#ef4444">{sig.stop_loss:.2f}</b>
                    &nbsp;|&nbsp; TP <b style="color:#22c55e">{sig.take_profit:.2f}</b>
                    &nbsp;|&nbsp; R:R <b>{sig.rr_ratio:.2f}</b>
                </p>
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
    """ML diagnostics tab: model status + metrics + feature importances."""
    st.subheader("ML Model Status")

    if ranker is None or not ranker.is_trained:
        st.warning(
            "No trained model found. Run `python -m ml.train` from the "
            "`trading_dashboard/` directory to train the model."
        )
        st.code("cd trading_dashboard\npython -m ml.train --regions us --max-tickers 5")
        return

    from config import METRICS_PATH
    col1, col2 = st.columns(2)
    col1.success("Model loaded and ready.")
    col2.caption(f"Artifact: `{MODEL_PATH.name}`")

    # ── Training metrics ──────────────────────────────────────────────────────
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
        else:
            tickers = default_tickers

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
            0.40, step=0.05, key="sb_min_final",
            help="Final score = 50% strategy score + 50% ML score. "
                 "With no ML model, max is ~0.40. Raise once ML is trained."
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
