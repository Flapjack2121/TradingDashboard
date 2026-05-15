"""
Global configuration for the Trading Dashboard.

Centralizes every tunable parameter so no other module hardcodes values:
    - Region universes, index proxies, and timezones
    - Timeframe -> yfinance interval / period / resample mapping
    - Indicator periods (EMA, RSI, ATR, MACD, S/R, Fibonacci)
    - Scoring rules (confluence points, thresholds, final-score weights)
    - Risk management limits (per-trade %, R:R, daily loss, costs)
    - Backtesting setup (capital, walk-forward, risk-free rate)
    - ML hyperparameters and artifact paths
    - Cache TTLs
    - Alpaca paper-trading credentials (env-loaded)
    - UI colors and labels

All values exposed via frozen dataclasses or Enums for type safety.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional: python-dotenv is only needed for local .env loading. On
# managed hosts (Streamlit Cloud, etc.) env vars are injected directly
# and dotenv may not be installed — degrade to a no-op in that case.
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "_data_cache"
MODEL_DIR: Path = PROJECT_ROOT / "_models"
EXPORT_DIR: Path = PROJECT_ROOT / "_exports"
LOG_DIR: Path = PROJECT_ROOT / "_logs"

for _d in (DATA_DIR, MODEL_DIR, EXPORT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Load .env from project root (silent if missing).
load_dotenv(PROJECT_ROOT / ".env", override=False)


# =============================================================================
# Regions
# =============================================================================

class Region(str, Enum):
    """Supported trading regions exposed in the UI sidebar."""
    US = "US"
    EUROPE = "Europe"
    ASIA = "Asia"


# Default ticker universe per region (yfinance symbol conventions).
# Users can override via the UI ticker input.
REGION_TICKERS: Dict[Region, List[str]] = {
    Region.US: [
        # Mega-cap tech
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
        "AVGO", "ORCL", "CRM", "ADBE", "AMD", "QCOM", "TXN", "INTC",
        "MU", "AMAT", "LRCX", "KLAC", "MRVL",
        # Financials
        "JPM", "V", "MA", "BAC", "GS", "MS", "WFC", "BLK", "AXP", "C",
        # Healthcare / Pharma
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "BMY", "AMGN", "GILD",
        # Consumer
        "WMT", "HD", "PG", "COST", "MCD", "KO", "PEP", "NKE", "SBUX", "TGT",
        # Energy
        "XOM", "CVX", "COP", "SLB", "OXY",
        # Industrials
        "CAT", "DE", "HON", "BA", "GE", "RTX", "UPS", "FDX",
        # Communication / Media
        "NFLX", "DIS", "T", "VZ", "CMCSA",
        # Materials / Other
        "LIN", "FCX", "NEM",
    ],
    Region.EUROPE: [
        # DAX — Germany (XETRA)
        "SAP.DE", "SIE.DE", "ALV.DE", "BAS.DE", "BMW.DE",
        "VOW3.DE", "DTE.DE", "BAYN.DE", "DBK.DE", "MUV2.DE",
        "ADS.DE", "RWE.DE", "BEI.DE", "MRK.DE", "HEN3.DE",
        # FTSE 100 — UK
        "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "SHEL.L",
        "GSK.L", "RIO.L", "DGE.L", "LLOY.L", "BATS.L",
        "AAL.L", "NG.L", "PRU.L",
        # CAC 40 — France
        "MC.PA", "OR.PA", "AIR.PA", "BNP.PA", "SAN.PA",
        "TTE.PA", "RI.PA", "SU.PA", "EL.PA", "KER.PA",
        # AEX — Netherlands
        "ASML.AS", "PHIA.AS", "UNA.AS", "ADYEN.AS", "HEIA.AS",
        # SMI — Switzerland
        "NESN.SW", "ROG.SW", "NOVN.SW",
        # IBEX — Spain
        "SAN.MC", "IBE.MC", "ITX.MC",
    ],
    Region.ASIA: [
        # Nikkei 225 — Japan
        "7203.T", "6758.T", "9984.T", "8306.T", "6861.T",
        "6501.T", "9432.T", "7974.T", "4063.T", "6902.T",
        "7267.T", "8035.T", "6594.T", "2914.T", "9433.T",
        # Hang Seng — Hong Kong
        "0700.HK", "9988.HK", "0005.HK", "1299.HK",
        "0941.HK", "2318.HK", "0388.HK", "1113.HK", "1810.HK",
        # KOSPI — South Korea (yfinance)
        "005930.KS", "000660.KS", "035420.KS", "005380.KS",
        # SGX — Singapore
        "D05.SI", "O39.SI", "U11.SI",
        # Indian ADRs (NYSE/NASDAQ listed, no suffix)
        "INFY", "WIT", "HDB", "IBN", "TTM",
        # Chinese ADRs & Tech
        "TSM", "BABA", "JD", "BIDU", "PDD", "NIO",
    ],
}

# Index proxy per region (used for the "market context" ML feature).
REGION_INDEX_PROXY: Dict[Region, str] = {
    Region.US: "^GSPC",       # S&P 500
    Region.EUROPE: "^GDAXI",  # DAX
    Region.ASIA: "^N225",     # Nikkei 225
}

# IANA timezone per region (for market-open/close logic).
REGION_TIMEZONE: Dict[Region, str] = {
    Region.US: "America/New_York",
    Region.EUROPE: "Europe/London",
    Region.ASIA: "Asia/Tokyo",
}

# Local trading hours per region (HH:MM 24h).
REGION_MARKET_HOURS: Dict[Region, Tuple[str, str]] = {
    Region.US: ("09:30", "16:00"),
    Region.EUROPE: ("08:00", "16:30"),
    Region.ASIA: ("09:00", "15:00"),
}


# =============================================================================
# Timeframes
# =============================================================================

class Timeframe(str, Enum):
    """Supported chart timeframes (UI labels)."""
    D1 = "1D"
    H4 = "4H"
    H1 = "1H"


# yfinance native interval per timeframe.
# Note: yfinance has no native 4H bar, so we fetch 1H and resample.
YFINANCE_INTERVAL: Dict[Timeframe, str] = {
    Timeframe.D1: "1d",
    Timeframe.H4: "1h",
    Timeframe.H1: "1h",
}

# yfinance period string per timeframe (history depth).
YFINANCE_PERIOD: Dict[Timeframe, str] = {
    Timeframe.D1: "5y",
    Timeframe.H4: "60d",
    Timeframe.H1: "60d",
}

# Pandas resample rule per timeframe (None = no resampling needed).
RESAMPLE_RULE: Dict[Timeframe, Optional[str]] = {
    Timeframe.D1: None,
    Timeframe.H4: "4h",
    Timeframe.H1: None,
}

# Approx bars per trading year per yfinance interval (for annualization).
BARS_PER_YEAR: Dict[str, int] = {
    "1d": 252,
    "1h": 252 * 7,
    "4h": 252 * 2,
}


# =============================================================================
# Strategies
# =============================================================================

class StrategyName(str, Enum):
    """All implemented strategies (also used as UI labels)."""
    TREND_FOLLOWING = "Trend Following"
    BREAKOUT_RETEST = "Breakout & Retest"
    PRICE_ACTION = "Price Action"
    SMC = "SMC"
    SWING_TRADING = "Swing Trading"
    FIBONACCI = "Fibonacci"


STRATEGY_LIST: List[str] = [s.value for s in StrategyName]

# Ensemble weights — applied when multiple strategies confirm the same trade.
STRATEGY_WEIGHTS: Dict[StrategyName, float] = {
    StrategyName.TREND_FOLLOWING: 1.0,
    StrategyName.BREAKOUT_RETEST: 1.0,
    StrategyName.PRICE_ACTION: 0.9,
    StrategyName.SMC: 1.1,
    StrategyName.SWING_TRADING: 0.9,
    StrategyName.FIBONACCI: 1.0,
}


# =============================================================================
# Indicators (single source of truth)
# =============================================================================

@dataclass(frozen=True)
class IndicatorConfig:
    """Periods and tolerances for every technical indicator."""
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    atr_avg_period: int = 20
    volume_avg_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    sr_lookback: int = 50            # bars scanned for support/resistance
    sr_tolerance_pct: float = 0.5    # S/R zone half-width (% of price)
    fib_lookback: int = 100          # bars for swing high/low detection
    pullback_tolerance_pct: float = 1.0   # "near EMA 50" tolerance
    swing_tf_rsi_buy_low: float = 40.0
    swing_tf_rsi_buy_high: float = 50.0
    swing_tf_rsi_sell_low: float = 50.0
    swing_tf_rsi_sell_high: float = 60.0
    fib_entry_level: float = 0.618       # 61.8% retracement
    fib_entry_tolerance_pct: float = 1.0


INDICATORS = IndicatorConfig()


# =============================================================================
# Scoring & Trade Validation
# =============================================================================

@dataclass(frozen=True)
class ScoringConfig:
    """Confluence scoring and final-score thresholds (per the spec)."""
    # Point allocation (sum = 100)
    trend_alignment_pts: int = 25
    volume_confirmation_pts: int = 15
    support_resistance_pts: int = 20
    candlestick_confirmation_pts: int = 20
    ml_high_score_pts: int = 20

    # Gates
    min_strategy_score: float = 75.0   # confluence score gate
    min_ml_score: float = 0.65         # strict ML filter
    ml_high_threshold: float = 0.70    # awards the ML confluence points
    min_final_score: float = 0.70      # final-display gate (0-1 scale)

    # Final score blend
    strategy_weight: float = 0.5
    ml_weight: float = 0.5

    # Highlighting in the trade table (0-100 scale)
    strong_signal_score: float = 80.0


SCORING = ScoringConfig()

# ML confidence band labels (probability -> label).
ML_CONFIDENCE_BANDS: List[Tuple[float, str]] = [
    (0.70, "High"),
    (0.50, "Medium"),
    (0.00, "Low"),
]


# =============================================================================
# Risk Management
# =============================================================================

@dataclass(frozen=True)
class RiskConfig:
    """Per-trade and portfolio-level risk limits."""
    max_risk_per_trade_pct: float = 0.01   # 1% of account equity
    max_open_trades: int = 5
    daily_loss_limit_pct: float = 0.03     # 3% of account equity per day
    min_rr_ratio: float = 1.5              # minimum reward-to-risk
    default_atr_sl_multiplier: float = 1.5
    default_atr_tp_multiplier: float = 3.0   # ~1:2 R:R by default
    commission_per_trade_pct: float = 0.0005   # 5 bps per side
    slippage_pct: float = 0.0005               # 5 bps per side


RISK = RiskConfig()


# =============================================================================
# Backtesting
# =============================================================================

@dataclass(frozen=True)
class BacktestConfig:
    """Backtest engine settings."""
    initial_capital: float = 100_000.0
    walk_forward_train_pct: float = 0.70
    walk_forward_test_pct: float = 0.30
    walk_forward_windows: int = 4
    risk_free_rate_annual: float = 0.04


BACKTEST = BacktestConfig()


# =============================================================================
# Machine Learning
# =============================================================================

@dataclass(frozen=True)
class MLConfig:
    """ML training and artifact settings."""
    test_size: float = 0.30
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int = 8
    min_samples_leaf: int = 10
    class_weight: str = "balanced"   # handles class imbalance
    n_jobs: int = -1
    model_filename: str = "trade_ranker.joblib"
    feature_filename: str = "feature_columns.json"
    metrics_filename: str = "training_metrics.json"


ML = MLConfig()

MODEL_PATH: Path = MODEL_DIR / ML.model_filename
FEATURE_PATH: Path = MODEL_DIR / ML.feature_filename
METRICS_PATH: Path = MODEL_DIR / ML.metrics_filename


# =============================================================================
# Caching
# =============================================================================

@dataclass(frozen=True)
class CacheConfig:
    """Disk + in-memory cache settings for market data."""
    enabled: bool = True
    intraday_ttl_seconds: int = 60 * 15        # 15 min
    daily_ttl_seconds: int = 60 * 60 * 6       # 6 hours
    disk_cache_subdir: str = "yf_cache"

    @property
    def disk_cache_dir(self) -> Path:
        path = DATA_DIR / self.disk_cache_subdir
        path.mkdir(parents=True, exist_ok=True)
        return path


CACHE = CacheConfig()


# =============================================================================
# Alpaca (paper trading)
# =============================================================================

@dataclass(frozen=True)
class AlpacaConfig:
    """Alpaca credentials and endpoint config (env-driven)."""
    api_key: str
    secret_key: str
    paper: bool
    base_url: str

    @property
    def is_configured(self) -> bool:
        """True if both API key and secret are present."""
        return bool(self.api_key and self.secret_key)


@lru_cache(maxsize=1)
def get_alpaca_config() -> AlpacaConfig:
    """
    Build the Alpaca config from environment variables (cached).

    Required env vars:
        ALPACA_API_KEY, ALPACA_SECRET_KEY
    Optional env vars:
        ALPACA_PAPER (default "true")
        ALPACA_BASE_URL (default paper endpoint)
    """
    return AlpacaConfig(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        paper=os.getenv("ALPACA_PAPER", "true").strip().lower() == "true",
        base_url=os.getenv(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        ),
    )


# =============================================================================
# Alerts
# =============================================================================

@dataclass(frozen=True)
class AlertConfig:
    """Notification channels and thresholds."""
    console_enabled: bool = True
    telegram_enabled: bool = False
    discord_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    min_final_score_for_alert: float = 0.80


@lru_cache(maxsize=1)
def get_alert_config() -> AlertConfig:
    """Build the alert config from environment variables (cached)."""
    return AlertConfig(
        console_enabled=os.getenv("ALERTS_CONSOLE", "true").lower() == "true",
        telegram_enabled=os.getenv("ALERTS_TELEGRAM", "false").lower() == "true",
        discord_enabled=os.getenv("ALERTS_DISCORD", "false").lower() == "true",
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
    )


# =============================================================================
# UI
# =============================================================================

UI_COLORS: Dict[str, str] = {
    "bullish": "#22c55e",   # green
    "bearish": "#ef4444",   # red
    "neutral": "#3b82f6",   # blue
    "warning": "#f59e0b",   # amber
    "bg": "#0e1117",
    "panel": "#161a24",
    "text": "#e5e7eb",
    "muted": "#9ca3af",
    "border": "#2a2f3a",
}

SIGNAL_COLOR_MAP: Dict[str, str] = {
    "BUY": UI_COLORS["bullish"],
    "SELL": UI_COLORS["bearish"],
    "HOLD": UI_COLORS["warning"],
}

UI_PAGE_TITLE: str = "Global Trading Dashboard"
UI_PAGE_ICON: str = ":chart_with_upwards_trend:"


# =============================================================================
# Public helpers
# =============================================================================

def get_tickers_for_region(region: Region | str) -> List[str]:
    """Return the default ticker universe for the given region."""
    if isinstance(region, str):
        region = Region(region)
    return list(REGION_TICKERS[region])


def get_index_for_region(region: Region | str) -> str:
    """Return the reference index ticker for the given region."""
    if isinstance(region, str):
        region = Region(region)
    return REGION_INDEX_PROXY[region]


def get_timezone_for_region(region: Region | str) -> str:
    """Return the IANA timezone string for the given region."""
    if isinstance(region, str):
        region = Region(region)
    return REGION_TIMEZONE[region]


def get_market_hours_for_region(region: Region | str) -> Tuple[str, str]:
    """Return the (open, close) local-time tuple for the given region."""
    if isinstance(region, str):
        region = Region(region)
    return REGION_MARKET_HOURS[region]


def yf_interval_for_timeframe(tf: Timeframe | str) -> str:
    """Return the yfinance interval string for the given timeframe."""
    if isinstance(tf, str):
        tf = Timeframe(tf)
    return YFINANCE_INTERVAL[tf]


def yf_period_for_timeframe(tf: Timeframe | str) -> str:
    """Return the yfinance period string for the given timeframe."""
    if isinstance(tf, str):
        tf = Timeframe(tf)
    return YFINANCE_PERIOD[tf]


def resample_rule_for_timeframe(tf: Timeframe | str) -> Optional[str]:
    """Return the pandas resample rule (or None) for the given timeframe."""
    if isinstance(tf, str):
        tf = Timeframe(tf)
    return RESAMPLE_RULE[tf]


def bars_per_year_for_interval(interval: str) -> int:
    """Return the approx bars per trading year for the given yfinance interval."""
    return BARS_PER_YEAR.get(interval, 252)
