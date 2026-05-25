"""
=============================================================================
GLOBAL SWING DESK — Data Engine
=============================================================================
Fetches real market data from Yahoo Finance, computes technical indicators,
identifies trading setups, ranks them by Trade Priority Score (0-100),
and outputs data.json for the dashboard.

Run:    python3 fetch_signals.py
Output: data.json (loaded by dashboard.html)
=============================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# UNIVERSE
# =============================================================================

UNIVERSE = {
    'fx': [
        {'symbol': 'EURUSD=X', 'display': 'EUR/USD', 'region': 'Major'},
        {'symbol': 'GBPUSD=X', 'display': 'GBP/USD', 'region': 'Major'},
        {'symbol': 'USDJPY=X', 'display': 'USD/JPY', 'region': 'Major'},
        {'symbol': 'AUDUSD=X', 'display': 'AUD/USD', 'region': 'Major'},
        {'symbol': 'USDCHF=X', 'display': 'USD/CHF', 'region': 'Major'},
        {'symbol': 'USDCAD=X', 'display': 'USD/CAD', 'region': 'Major'},
        {'symbol': 'NZDUSD=X', 'display': 'NZD/USD', 'region': 'Major'},
        {'symbol': 'EURJPY=X', 'display': 'EUR/JPY', 'region': 'Cross'},
        {'symbol': 'GBPJPY=X', 'display': 'GBP/JPY', 'region': 'Cross'},
    ],
    'us': [
        {'symbol': 'NVDA', 'display': 'NVDA', 'name': 'NVIDIA', 'sector': 'Technology'},
        {'symbol': 'MSFT', 'display': 'MSFT', 'name': 'Microsoft', 'sector': 'Technology'},
        {'symbol': 'AAPL', 'display': 'AAPL', 'name': 'Apple', 'sector': 'Technology'},
        {'symbol': 'GOOGL', 'display': 'GOOGL', 'name': 'Alphabet', 'sector': 'Communication'},
        {'symbol': 'META', 'display': 'META', 'name': 'Meta Platforms', 'sector': 'Communication'},
        {'symbol': 'AMZN', 'display': 'AMZN', 'name': 'Amazon', 'sector': 'Consumer Disc.'},
        {'symbol': 'TSLA', 'display': 'TSLA', 'name': 'Tesla', 'sector': 'Consumer Disc.'},
        {'symbol': 'AVGO', 'display': 'AVGO', 'name': 'Broadcom', 'sector': 'Technology'},
        {'symbol': 'LLY', 'display': 'LLY', 'name': 'Eli Lilly', 'sector': 'Healthcare'},
        {'symbol': 'JPM', 'display': 'JPM', 'name': 'JPMorgan', 'sector': 'Financials'},
        {'symbol': 'V', 'display': 'V', 'name': 'Visa', 'sector': 'Financials'},
        {'symbol': 'COST', 'display': 'COST', 'name': 'Costco', 'sector': 'Consumer Stap.'},
        {'symbol': 'NFLX', 'display': 'NFLX', 'name': 'Netflix', 'sector': 'Communication'},
        {'symbol': 'XOM', 'display': 'XOM', 'name': 'Exxon Mobil', 'sector': 'Energy'},
        {'symbol': 'UNH', 'display': 'UNH', 'name': 'UnitedHealth', 'sector': 'Healthcare'},
        {'symbol': 'MA', 'display': 'MA', 'name': 'Mastercard', 'sector': 'Financials'},
    ],
    'eu': [
        {'symbol': 'ASML', 'display': 'ASML', 'name': 'ASML Holding', 'sector': 'Technology', 'region': 'Netherlands'},
        {'symbol': 'NVO', 'display': 'NVO', 'name': 'Novo Nordisk', 'sector': 'Healthcare', 'region': 'Denmark'},
        {'symbol': 'MC.PA', 'display': 'MC.PA', 'name': 'LVMH', 'sector': 'Consumer Disc.', 'region': 'France'},
        {'symbol': 'SAP', 'display': 'SAP', 'name': 'SAP', 'sector': 'Technology', 'region': 'Germany'},
        {'symbol': 'NESN.SW', 'display': 'NESN', 'name': 'Nestle', 'sector': 'Consumer Stap.', 'region': 'Switzerland'},
        {'symbol': 'ROG.SW', 'display': 'ROG', 'name': 'Roche', 'sector': 'Healthcare', 'region': 'Switzerland'},
        {'symbol': 'SIE.DE', 'display': 'SIE', 'name': 'Siemens', 'sector': 'Industrials', 'region': 'Germany'},
        {'symbol': 'HSBA.L', 'display': 'HSBA', 'name': 'HSBC', 'sector': 'Financials', 'region': 'UK'},
        {'symbol': 'SHEL', 'display': 'SHEL', 'name': 'Shell', 'sector': 'Energy', 'region': 'UK'},
        {'symbol': 'AZN', 'display': 'AZN', 'name': 'AstraZeneca', 'sector': 'Healthcare', 'region': 'UK'},
        {'symbol': 'OR.PA', 'display': 'OR.PA', 'name': "L'Oreal", 'sector': 'Consumer Stap.', 'region': 'France'},
        {'symbol': 'AIR.PA', 'display': 'AIR.PA', 'name': 'Airbus', 'sector': 'Industrials', 'region': 'France'},
    ],
    'asia': [
        {'symbol': '7203.T', 'display': '7203', 'name': 'Toyota', 'sector': 'Consumer Disc.', 'region': 'Japan'},
        {'symbol': '6758.T', 'display': '6758', 'name': 'Sony', 'sector': 'Technology', 'region': 'Japan'},
        {'symbol': '9984.T', 'display': '9984', 'name': 'SoftBank Group', 'sector': 'Communication', 'region': 'Japan'},
        {'symbol': '8035.T', 'display': '8035', 'name': 'Tokyo Electron', 'sector': 'Technology', 'region': 'Japan'},
        {'symbol': '7974.T', 'display': '7974', 'name': 'Nintendo', 'sector': 'Communication', 'region': 'Japan'},
        {'symbol': 'TSM', 'display': 'TSM', 'name': 'TSMC (ADR)', 'sector': 'Technology', 'region': 'Taiwan'},
        {'symbol': 'BABA', 'display': 'BABA', 'name': 'Alibaba (ADR)', 'sector': 'Consumer Disc.', 'region': 'China'},
        {'symbol': 'JD', 'display': 'JD', 'name': 'JD.com (ADR)', 'sector': 'Consumer Disc.', 'region': 'China'},
        {'symbol': 'INFY', 'display': 'INFY', 'name': 'Infosys (ADR)', 'sector': 'Technology', 'region': 'India'},
    ],
    'smallcap': [
        {'symbol': 'CELH', 'display': 'CELH', 'name': 'Celsius Holdings', 'sector': 'Consumer Stap.', 'region': 'US Small'},
        {'symbol': 'ELF',  'display': 'ELF',  'name': 'e.l.f. Beauty', 'sector': 'Consumer Disc.', 'region': 'US Small'},
        {'symbol': 'IBP',  'display': 'IBP',  'name': 'Installed Building Products', 'sector': 'Industrials', 'region': 'US Small'},
        {'symbol': 'LNTH', 'display': 'LNTH', 'name': 'Lantheus Holdings', 'sector': 'Healthcare', 'region': 'US Small'},
        {'symbol': 'ATKR', 'display': 'ATKR', 'name': 'Atkore', 'sector': 'Industrials', 'region': 'US Small'},
        {'symbol': 'CRS',  'display': 'CRS',  'name': 'Carpenter Technology', 'sector': 'Materials', 'region': 'US Small'},
        {'symbol': 'PI',   'display': 'PI',   'name': 'Impinj', 'sector': 'Technology', 'region': 'US Small'},
        {'symbol': 'BMI',  'display': 'BMI',  'name': 'Badger Meter', 'sector': 'Industrials', 'region': 'US Small'},
        {'symbol': 'RMBS', 'display': 'RMBS', 'name': 'Rambus', 'sector': 'Technology', 'region': 'US Small'},
        {'symbol': 'SPSC', 'display': 'SPSC', 'name': 'SPS Commerce', 'sector': 'Technology', 'region': 'US Small'},
    ],
    'em': [
        {'symbol': 'VALE', 'display': 'VALE', 'name': 'Vale', 'sector': 'Materials', 'region': 'Brazil'},
        {'symbol': 'ITUB', 'display': 'ITUB', 'name': 'Itau Unibanco', 'sector': 'Financials', 'region': 'Brazil'},
        {'symbol': 'PBR',  'display': 'PBR',  'name': 'Petrobras', 'sector': 'Energy', 'region': 'Brazil'},
        {'symbol': 'MELI', 'display': 'MELI', 'name': 'MercadoLibre', 'sector': 'Consumer Disc.', 'region': 'LatAm'},
        {'symbol': 'HDB',  'display': 'HDB',  'name': 'HDFC Bank (ADR)', 'sector': 'Financials', 'region': 'India'},
        {'symbol': 'IBN',  'display': 'IBN',  'name': 'ICICI Bank (ADR)', 'sector': 'Financials', 'region': 'India'},
        {'symbol': 'PDD',  'display': 'PDD',  'name': 'PDD Holdings', 'sector': 'Consumer Disc.', 'region': 'China'},
        {'symbol': 'NTES', 'display': 'NTES', 'name': 'NetEase', 'sector': 'Communication', 'region': 'China'},
        {'symbol': 'SE',   'display': 'SE',   'name': 'Sea Limited', 'sector': 'Technology', 'region': 'Singapore'},
        {'symbol': 'AMX',  'display': 'AMX',  'name': 'America Movil', 'sector': 'Communication', 'region': 'Mexico'},
    ],
}

BENCHMARK = 'SPY'   # for US RS computation
BENCH_EU  = '^STOXX50E'
BENCH_ASIA = '^N225'
BENCH_SMALL = 'IWM'   # Russell 2000 ETF
BENCH_EM    = 'EEM'   # iShares MSCI Emerging Markets

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(window=period).mean()

def atr(df, period=14):
    """Wilder's ATR."""
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def rsi(series, period=14):
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def adx(df, period=14):
    """Average Directional Index. Measures trend strength."""
    h, l, c = df['High'], df['Low'], df['Close']
    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_v = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_v)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_v)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()

# =============================================================================
# STAGE ANALYSIS (Weinstein)
# =============================================================================

def detect_stage(df):
    """
    Stage 1 = basing (sideways, below 30W MA flat)
    Stage 2 = advancing (above 30W MA, MA rising)
    Stage 3 = topping (above 30W MA but flattening, price stalling)
    Stage 4 = declining (below 30W MA, MA falling)
    """
    if len(df) < 200:
        return 1, 'Insufficient data'

    close = df['Close']
    ma30w = sma(close, 150)  # 30 weeks ≈ 150 trading days
    ma50  = sma(close, 50)

    last_close = close.iloc[-1]
    last_ma30w = ma30w.iloc[-1]
    ma30w_slope = (ma30w.iloc[-1] - ma30w.iloc[-22]) / ma30w.iloc[-22] if ma30w.iloc[-22] != 0 else 0
    ma50_slope  = (ma50.iloc[-1] - ma50.iloc[-10]) / ma50.iloc[-10] if ma50.iloc[-10] != 0 else 0

    if last_close > last_ma30w and ma30w_slope > 0.005 and ma50_slope > 0:
        return 2, 'Advancing — institutional accumulation'
    elif last_close < last_ma30w and ma30w_slope < -0.005 and ma50_slope < 0:
        return 4, 'Declining — institutional distribution'
    elif last_close > last_ma30w and ma30w_slope <= 0.005:
        return 3, 'Topping — momentum stalling, watch for breakdown'
    else:
        return 1, 'Basing — no trend, sit out or wait for break'

# =============================================================================
# RELATIVE STRENGTH RATING (O'Neil-style)
# =============================================================================

def compute_rs_rating(df, bench_df):
    """
    Compute IBD-style RS rating (1-99).
    Weighted blend of trailing performance vs benchmark across multiple windows.
    """
    if len(df) < 60 or len(bench_df) < 60:
        return 50.0

    def perf(s, n):
        if len(s) <= n: return 0
        return (s.iloc[-1] / s.iloc[-n-1] - 1) if s.iloc[-n-1] != 0 else 0

    q1 = perf(df['Close'], 63)
    q2 = perf(df['Close'], 126)
    q3 = perf(df['Close'], 189)
    q4 = perf(df['Close'], 252) if len(df) > 252 else q3

    b1 = perf(bench_df['Close'], 63)
    b2 = perf(bench_df['Close'], 126)
    b3 = perf(bench_df['Close'], 189)
    b4 = perf(bench_df['Close'], 252) if len(bench_df) > 252 else b3

    rel = 2*(q1 - b1) + (q2 - b2) + (q3 - b3) + (q4 - b4)
    score = 50 + 49 * np.tanh(rel * 2.5)
    return float(np.clip(score, 1, 99))

# =============================================================================
# SETUP DETECTION
# =============================================================================

def detect_vcp(df, lookback=60):
    """Volatility Contraction Pattern — Minervini's flagship signal."""
    if len(df) < lookback:
        return False, 0.0

    recent = df.tail(lookback)
    rolling_range = (recent['High'].rolling(10).max() - recent['Low'].rolling(10).min()) / recent['Close']

    if rolling_range.isna().sum() > 30:
        return False, 0.0

    rr = rolling_range.dropna()
    if len(rr) < 30:
        return False, 0.0
    early = rr.iloc[:len(rr)//3].mean()
    late  = rr.iloc[-len(rr)//3:].mean()

    contraction_ratio = late / early if early > 0 else 1.0

    vol_early = recent['Volume'].iloc[:lookback//3].mean()
    vol_late  = recent['Volume'].iloc[-lookback//3:].mean()
    vol_declining = vol_late < vol_early * 0.85 if vol_early > 0 else False

    near_high = recent['Close'].iloc[-1] >= recent['High'].max() * 0.95

    is_vcp = contraction_ratio < 0.6 and near_high
    quality = 0.0
    if is_vcp:
        quality = 0.5
        if vol_declining: quality += 0.3
        if contraction_ratio < 0.4: quality += 0.2

    return is_vcp, min(quality, 1.0)

def detect_pullback_to_ema(df, ema_period=21, tolerance=0.02):
    """Detects pullback to 21 EMA in an uptrend (or short equivalent)."""
    if len(df) < ema_period + 5:
        return False, None, 0.0
    ema_v = ema(df['Close'], ema_period)
    last_close = df['Close'].iloc[-1]
    last_ema = ema_v.iloc[-1]
    ma200 = sma(df['Close'], 200).iloc[-1] if len(df) >= 200 else None

    near_ema = abs(last_close - last_ema) / last_ema < tolerance
    if not near_ema:
        return False, None, 0.0

    if ma200 and last_close > ma200:
        direction = 'long'
        quality = 0.7
        for i in range(1, 4):
            if df['Low'].iloc[-i] <= ema_v.iloc[-i] * 1.005 and df['Close'].iloc[-i] > ema_v.iloc[-i]:
                quality += 0.1
                break
        return True, direction, min(quality, 1.0)
    elif ma200 and last_close < ma200:
        return True, 'short', 0.6
    return False, None, 0.0

def detect_breakout(df, lookback=20):
    """Detects breakout above 20-day high (or below 20-day low) on volume."""
    if len(df) < lookback + 5:
        return False, None, 0.0
    recent = df.tail(lookback + 1)
    prior_high = recent['High'].iloc[:-1].max()
    prior_low  = recent['Low'].iloc[:-1].min()
    last_close = df['Close'].iloc[-1]
    last_vol = df['Volume'].iloc[-1]
    avg_vol = df['Volume'].tail(50).mean()
    vol_confirmed = last_vol > avg_vol * 1.3 if avg_vol > 0 else False

    if last_close > prior_high:
        quality = 0.7 + (0.2 if vol_confirmed else 0)
        return True, 'long', quality
    elif last_close < prior_low:
        quality = 0.6 + (0.2 if vol_confirmed else 0)
        return True, 'short', quality
    return False, None, 0.0

def detect_connors(df):
    """Connors RSI(2) pullback in quality."""
    if len(df) < 200:
        return False, 0.0
    rsi2 = rsi(df['Close'], 2).iloc[-1]
    ma200 = sma(df['Close'], 200).iloc[-1]
    last = df['Close'].iloc[-1]
    if last > ma200 and rsi2 < 10:
        return True, 0.7
    return False, 0.0

# =============================================================================
# SIGNAL GENERATION & SCORING
# =============================================================================

def compute_signal(symbol_info, df, bench_df, asset_class):
    """Master function: identifies best setup, scores it 0-100."""
    if df is None or len(df) < 60:
        return None

    close = df['Close']
    high  = df['High']
    low   = df['Low']
    last_close = float(close.iloc[-1])

    ema8   = ema(close, 8).iloc[-1]
    ema21  = ema(close, 21).iloc[-1]
    ema50  = ema(close, 50).iloc[-1] if len(close) >= 50 else None
    sma50  = sma(close, 50).iloc[-1] if len(close) >= 50 else None
    sma150 = sma(close, 150).iloc[-1] if len(close) >= 150 else None
    sma200 = sma(close, 200).iloc[-1] if len(close) >= 200 else None
    atr_v  = atr(df).iloc[-1]
    rsi_v  = rsi(close).iloc[-1]
    adx_v  = adx(df).iloc[-1] if len(df) >= 28 else 0
    high_52w = high.tail(252).max() if len(high) >= 252 else high.max()
    low_52w  = low.tail(252).min() if len(low) >= 252 else low.min()
    pct_off_high = (last_close / high_52w - 1) * 100
    daily_change = (last_close / close.iloc[-2] - 1) * 100 if len(close) > 1 else 0

    avg_vol = df['Volume'].tail(50).mean() if 'Volume' in df.columns else 0
    last_vol = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1

    if asset_class == 'fx':
        stage, stage_desc = 0, 'N/A for FX'
        if sma200 and last_close > sma200 * 1.01:
            bias_default = 'long'
        elif sma200 and last_close < sma200 * 0.99:
            bias_default = 'short'
        else:
            bias_default = 'flat'
    else:
        stage, stage_desc = detect_stage(df)
        bias_default = 'long' if stage == 2 else 'short' if stage == 4 else 'flat'

    if bench_df is not None and asset_class != 'fx':
        rs_rating = compute_rs_rating(df, bench_df)
    else:
        rs_rating = 50

    setups = []

    if asset_class == 'fx':
        pb, pb_dir, pb_q = detect_pullback_to_ema(df, 21)
        if pb: setups.append(('trend-pullback', pb_dir, pb_q))
        bo, bo_dir, bo_q = detect_breakout(df, 20)
        if bo: setups.append(('structure-break', bo_dir, bo_q))
    else:
        vcp, vcp_q = detect_vcp(df)
        if vcp and stage == 2:
            setups.append(('sepa', 'long', vcp_q))
        bo, bo_dir, bo_q = detect_breakout(df, 20)
        if bo and ((bo_dir == 'long' and stage == 2) or (bo_dir == 'short' and stage == 4)):
            setups.append(('canslim', bo_dir, bo_q))
        pb, pb_dir, pb_q = detect_pullback_to_ema(df, 21)
        if pb and ((pb_dir == 'long' and stage == 2) or (pb_dir == 'short' and stage == 4)):
            setups.append(('pullback', pb_dir, pb_q))
        cn, cn_q = detect_connors(df)
        if cn and stage == 2:
            setups.append(('connors', 'long', cn_q))

    if setups:
        setups.sort(key=lambda x: x[2], reverse=True)
        strategy, bias, setup_quality = setups[0]
    else:
        strategy = 'no-setup'
        bias = bias_default
        setup_quality = 0.2

    if asset_class == 'fx':
        pip = 0.01 if 'JPY' in symbol_info.get('display', '') else 0.0001
        stop_distance = atr_v * 2.0
        entry = last_close
        if bias == 'long':
            stop = entry - stop_distance
            target = entry + stop_distance * 2.0
        else:
            stop = entry + stop_distance
            target = entry - stop_distance * 2.0
        stop_pips = round(stop_distance / pip)
        target_pips = round(stop_distance * 2.0 / pip)
        stop_pct = None
    else:
        atr_stop = atr_v * 2.5
        pct_stop = last_close * 0.10
        stop_distance = min(atr_stop, pct_stop)
        entry = last_close
        if bias == 'long':
            stop = entry - stop_distance
            target = entry + stop_distance * 2.0
        else:
            stop = entry + stop_distance
            target = entry - stop_distance * 2.0
        stop_pips = None
        stop_pct = round(stop_distance / entry * 100, 2)

    rr = round(abs(target - entry) / abs(entry - stop), 2) if abs(entry - stop) > 0 else 0

    score_breakdown = {}

    trend_score = 0
    if asset_class == 'fx':
        if sma200 and ema50:
            if bias == 'long' and last_close > sma200 and last_close > ema50 and ema50 > sma200:
                trend_score = 25
            elif bias == 'short' and last_close < sma200 and last_close < ema50 and ema50 < sma200:
                trend_score = 25
            elif bias == 'long' and last_close > sma200:
                trend_score = 16
            elif bias == 'short' and last_close < sma200:
                trend_score = 16
        if adx_v > 25: trend_score = min(28, trend_score + 3)
    else:
        if stage == 2 and bias == 'long':
            trend_score = 20
            if sma50 and sma150 and sma200 and sma50 > sma150 > sma200:
                trend_score = 26
            if adx_v > 25: trend_score = min(28, trend_score + 2)
        elif stage == 4 and bias == 'short':
            trend_score = 20
            if sma50 and sma150 and sma200 and sma50 < sma150 < sma200:
                trend_score = 26
            if adx_v > 25: trend_score = min(28, trend_score + 2)
        elif stage == 1 or stage == 3:
            trend_score = 4
    score_breakdown['Trend Quality'] = (round(trend_score), 28)

    rs_score = 0
    if asset_class == 'fx':
        roc_60 = ((last_close / close.iloc[-min(60, len(close)-1)]) - 1) * 100
        if (bias == 'long' and roc_60 > 5) or (bias == 'short' and roc_60 < -5):
            rs_score = 20
        elif (bias == 'long' and roc_60 > 2) or (bias == 'short' and roc_60 < -2):
            rs_score = 13
        else:
            rs_score = 6
    else:
        if rs_rating >= 85: rs_score = 22
        elif rs_rating >= 75: rs_score = 18
        elif rs_rating >= 65: rs_score = 13
        elif rs_rating >= 50: rs_score = 7
        else: rs_score = 2
    score_breakdown['Relative Strength'] = (round(rs_score), 22)

    setup_score = round(setup_quality * 12)
    if vol_ratio > 1.8:
        setup_score += 3
    elif vol_ratio > 1.4:
        setup_score += 1
    setup_score = min(15, setup_score)
    score_breakdown['Setup Quality'] = (round(setup_score), 15)

    if rr >= 3.0: rr_score = 15
    elif rr >= 2.5: rr_score = 13
    elif rr >= 2.0: rr_score = 10
    elif rr >= 1.5: rr_score = 5
    else: rr_score = 0
    score_breakdown['Risk:Reward'] = (round(rr_score), 15)

    mo_score = 0
    if asset_class != 'fx':
        if bias == 'long' and pct_off_high > -10: mo_score += 5
        elif bias == 'long' and pct_off_high > -20: mo_score += 3
        if bias == 'short' and pct_off_high < -30: mo_score += 5
    if bias == 'long' and 40 < rsi_v < 70: mo_score += 3
    elif bias == 'short' and 30 < rsi_v < 60: mo_score += 3
    elif bias == 'long' and rsi_v >= 70: mo_score += 1
    if last_close > ema8 and bias == 'long': mo_score += 2
    elif last_close < ema8 and bias == 'short': mo_score += 2
    mo_score = min(10, mo_score)
    score_breakdown['Momentum'] = (round(mo_score), 10)

    timing_score = 5
    earnings_days = None
    if asset_class != 'fx' and symbol_info.get('earnings_days') is not None:
        earnings_days = symbol_info['earnings_days']
        if 0 <= earnings_days <= 5:
            timing_score = 0
        elif earnings_days < 0 and earnings_days > -10:
            timing_score = 8
    if vol_ratio > 1.5: timing_score = min(10, timing_score + 2)
    score_breakdown['Catalyst/Timing'] = (round(timing_score), 10)

    priority_score = sum(s[0] for s in score_breakdown.values())

    if strategy == 'no-setup':
        priority_score = min(priority_score, 35)
        bias = 'flat'

    return {
        'symbol': symbol_info.get('display', symbol_info['symbol']),
        'yahoo_symbol': symbol_info['symbol'],
        'name': symbol_info.get('name', symbol_info.get('display', '')),
        'sector': symbol_info.get('sector', ''),
        'region': symbol_info.get('region', ''),
        'asset_class': asset_class,
        'price': round(last_close, 5 if asset_class == 'fx' else 2),
        'change_pct': round(daily_change, 2),
        'bias': bias,
        'strategy': strategy,
        'stage': stage,
        'stage_desc': stage_desc,
        'rs_rating': round(rs_rating, 1),
        'priority_score': priority_score,
        'score_breakdown': score_breakdown,
        'entry': round(entry, 5 if asset_class == 'fx' else 2),
        'stop': round(stop, 5 if asset_class == 'fx' else 2),
        'target': round(target, 5 if asset_class == 'fx' else 2),
        'rr': rr,
        'stop_pips': stop_pips,
        'stop_pct': stop_pct,
        'atr': round(atr_v, 5 if asset_class == 'fx' else 2),
        'rsi': round(rsi_v, 1),
        'adx': round(adx_v, 1),
        'vol_ratio': round(vol_ratio, 2) if avg_vol > 0 else None,
        'pct_off_high': round(pct_off_high, 1),
        'ema21': round(ema21, 5 if asset_class == 'fx' else 2),
        'sma50': round(sma50, 5 if asset_class == 'fx' else 2) if sma50 else None,
        'sma200': round(sma200, 5 if asset_class == 'fx' else 2) if sma200 else None,
        'high_52w': round(high_52w, 5 if asset_class == 'fx' else 2),
        'low_52w': round(low_52w, 5 if asset_class == 'fx' else 2),
        'earnings_days': earnings_days,
        'ohlc': format_ohlc(df.tail(180)),
        'ema21_series': format_series(ema(close, 21).tail(180)),
        'sma50_series': format_series(sma(close, 50).tail(180)) if len(close) >= 50 else [],
        'sma200_series': format_series(sma(close, 200).tail(180)) if len(close) >= 200 else [],
    }

def format_ohlc(df):
    """Format OHLC data for lightweight-charts library."""
    out = []
    for idx, row in df.iterrows():
        try:
            out.append({
                'time': idx.strftime('%Y-%m-%d'),
                'open': round(float(row['Open']), 4),
                'high': round(float(row['High']), 4),
                'low': round(float(row['Low']), 4),
                'close': round(float(row['Close']), 4),
                'volume': int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 0,
            })
        except (ValueError, TypeError):
            continue
    return out

def format_series(series):
    """Format MA series for lightweight-charts."""
    out = []
    for idx, val in series.items():
        if pd.notna(val):
            out.append({'time': idx.strftime('%Y-%m-%d'), 'value': round(float(val), 4)})
    return out

# =============================================================================
# MAIN
# =============================================================================

def get_earnings_days(symbol):
    """Try to fetch earnings date, return days until next earnings."""
    try:
        t = yf.Ticker(symbol)
        cal = t.calendar
        if cal is not None and not isinstance(cal, dict):
            return None
        if isinstance(cal, dict) and 'Earnings Date' in cal:
            ed = cal['Earnings Date']
            if ed and len(ed) > 0:
                if isinstance(ed, list): ed = ed[0]
                days = (ed - datetime.now().date()).days
                return days
    except Exception:
        pass
    return None

def fetch_one(symbol, period='1y'):
    """Fetch OHLCV for a single symbol with error handling."""
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if len(df) < 30:
            return None
        return df
    except Exception as e:
        print(f"  x Failed to fetch {symbol}: {e}")
        return None

def main():
    print("=" * 70)
    print("GLOBAL SWING DESK - Data Engine")
    print("=" * 70)

    print(f"\n[1/7] Fetching benchmarks...")
    bench_us    = fetch_one(BENCHMARK)
    bench_eu    = fetch_one(BENCH_EU)
    bench_asia  = fetch_one(BENCH_ASIA)
    bench_small = fetch_one(BENCH_SMALL)
    bench_em    = fetch_one(BENCH_EM)

    signals = []
    universes = [('fx', UNIVERSE['fx'], None),
                 ('us', UNIVERSE['us'], bench_us),
                 ('eu', UNIVERSE['eu'], bench_eu),
                 ('asia', UNIVERSE['asia'], bench_asia),
                 ('smallcap', UNIVERSE['smallcap'], bench_small),
                 ('em', UNIVERSE['em'], bench_em)]

    for i, (asset_class, items, bench) in enumerate(universes, start=2):
        print(f"\n[{i}/7] Processing {asset_class.upper()} ({len(items)} instruments)...")
        for item in items:
            sym = item['symbol']
            print(f"  -> {sym:12s}", end=' ')
            df = fetch_one(sym)
            if df is None:
                print("(skipped)")
                continue

            if asset_class != 'fx':
                item['earnings_days'] = get_earnings_days(sym)

            sig = compute_signal(item, df, bench, asset_class)
            if sig:
                signals.append(sig)
                bias_str = f"{sig['bias']:5s}"
                print(f"  ${sig['price']:>10}  {bias_str}  score: {sig['priority_score']:>3}/100  ({sig['strategy']})")
            else:
                print("(insufficient data)")

    signals.sort(key=lambda s: s['priority_score'], reverse=True)

    fx_signals = [s for s in signals if s['asset_class'] == 'fx']
    us_signals = [s for s in signals if s['asset_class'] == 'us']
    eu_signals = [s for s in signals if s['asset_class'] == 'eu']
    asia_signals = [s for s in signals if s['asset_class'] == 'asia']
    smallcap_signals = [s for s in signals if s['asset_class'] == 'smallcap']
    em_signals = [s for s in signals if s['asset_class'] == 'em']
    high_conv = [s for s in signals if s['priority_score'] >= 70]

    print("\n" + "=" * 70)
    print(f"COMPLETE: {len(signals)} instruments analyzed")
    print(f"  FX:        {len(fx_signals):3d}")
    print(f"  US:        {len(us_signals):3d}")
    print(f"  EU:        {len(eu_signals):3d}")
    print(f"  Asia:      {len(asia_signals):3d}")
    print(f"  Small Cap: {len(smallcap_signals):3d}")
    print(f"  EM:        {len(em_signals):3d}")
    print(f"\n  HIGH-CONVICTION SETUPS (score >= 70): {len(high_conv)}")
    for s in high_conv[:10]:
        print(f"    {s['priority_score']:>3}/100  {s['symbol']:12s}  {s['bias']:5s}  {s['strategy']:20s}  R:R 1:{s['rr']}")
    print("=" * 70)

    output = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'signals': signals,
        'summary': {
            'total': len(signals),
            'by_class': {
                'fx': len(fx_signals),
                'us': len(us_signals),
                'eu': len(eu_signals),
                'asia': len(asia_signals),
                'smallcap': len(smallcap_signals),
                'em': len(em_signals),
            },
            'high_conviction': len(high_conv),
        }
    }

    def clean_for_json(obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean_for_json(x) for x in obj]
        return obj

    with open('data.json', 'w') as f:
        json.dump(clean_for_json(output), f, indent=2, default=str)
    print(f"\nOK data.json written ({len(signals)} signals)")
    print("  Now open dashboard.html in your browser.\n")

if __name__ == '__main__':
    main()
