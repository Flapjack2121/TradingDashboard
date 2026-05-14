"""
Training pipeline for the TradeRanker.

End-to-end flow:
    1. Fetch OHLCV for each region's default universe.
    2. Run the FULL strategy roster across every ticker through the
       backtester with `apply_score_filter=False` so we collect both
       winners and losers (otherwise the label distribution would be
       dominated by high-score setups).
    3. For each closed trade, rebuild the feature vector from the entry
       bar's trailing window and attach the region one-hot column.
    4. Train/test split, fit RandomForest, evaluate, save artifacts.

Run from the project root:

    python -m ml.train --regions us europe asia --timeframe 1D
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.backtester import run_backtest
from config import (
    BACKTEST,
    FEATURE_PATH,
    METRICS_PATH,
    ML,
    MODEL_PATH,
    Region,
    Timeframe,
    get_tickers_for_region,
)
from data.data_fetcher import fetch_many
from ml.features import (
    FEATURE_COLUMNS,
    build_features_from_trade,
    features_to_dataframe,
)
from ml.model import TradeRanker
from strategies import REGISTRY, build_strategy
from utils.helpers import Trade, get_logger

logger = get_logger("ml.train")


# =============================================================================
# Result containers
# =============================================================================

@dataclass
class TrainingResult:
    """All outputs from a single training run."""
    model: Optional[TradeRanker] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    n_trades: int = 0
    n_features: int = 0
    saved: bool = False


# =============================================================================
# Stage 1: collect trades
# =============================================================================

def collect_trades(
    regions: List[Region],
    timeframe: Timeframe = Timeframe.D1,
    max_tickers_per_region: Optional[int] = None,
) -> Tuple[List[Tuple[Trade, str]], Dict[str, pd.DataFrame]]:
    """
    Run every strategy across every ticker in each region.

    Returns:
        trades_with_region:  list of (Trade, region_value) pairs
        dfs:                 dict ticker -> source OHLCV (for feature rebuild)
    """
    strategies = [build_strategy(name) for name in REGISTRY.keys()]
    logger.info("Running %d strategies across %d region(s)", len(strategies), len(regions))

    all_trades: List[Tuple[Trade, str]] = []
    dfs: Dict[str, pd.DataFrame] = {}

    for region in regions:
        tickers = get_tickers_for_region(region)
        if max_tickers_per_region:
            tickers = tickers[:max_tickers_per_region]

        logger.info("Region %s: fetching %d tickers", region.value, len(tickers))
        region_dfs = fetch_many(tickers, timeframe=timeframe)

        for ticker, df in region_dfs.items():
            if df is None or df.empty:
                continue
            dfs[ticker] = df
            try:
                result = run_backtest(
                    df=df,
                    ticker=ticker,
                    region=region.value,
                    strategies=strategies,
                    timeframe=timeframe,
                    apply_score_filter=False,    # collect all labels
                )
            except Exception:
                logger.exception("Backtest failed for %s", ticker)
                continue

            for trade in result.trades:
                all_trades.append((trade, region.value))

            logger.info(
                "  %s: %d trades", ticker, len(result.trades)
            )

    logger.info("Collected %d total trades across all regions", len(all_trades))
    return all_trades, dfs


# =============================================================================
# Stage 2: build training set
# =============================================================================

_REGION_COLUMN_MAP = {
    Region.US.value: "region_us",
    Region.EUROPE.value: "region_europe",
    Region.ASIA.value: "region_asia",
}


def build_training_set(
    trades_with_region: List[Tuple[Trade, str]],
    dfs: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Convert closed trades to (X, y) for sklearn."""
    rows: List[Dict[str, float]] = []
    labels: List[int] = []

    for trade, region_value in trades_with_region:
        df = dfs.get(trade.ticker)
        if df is None:
            continue

        feat = build_features_from_trade(trade, df)
        if feat is None:
            continue

        # build_features_from_trade does not know the region; patch it in.
        region_col = _REGION_COLUMN_MAP.get(region_value)
        if region_col:
            # Zero all region one-hots first (in case of stale values).
            for col in _REGION_COLUMN_MAP.values():
                feat[col] = 0.0
            feat[region_col] = 1.0

        rows.append(feat)
        labels.append(1 if trade.is_win else 0)

    X = features_to_dataframe(rows)
    y = np.asarray(labels, dtype=int)
    logger.info(
        "Training set: %d rows  pos_rate=%.3f",
        len(y), float(y.mean()) if len(y) else 0.0,
    )
    return X, y


# =============================================================================
# Stage 3: evaluate
# =============================================================================

def evaluate(
    model: TradeRanker,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Compute classification metrics on a held-out set."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    proba = model.predict_proba(X_test)
    preds = (proba >= 0.5).astype(int)

    try:
        roc_auc = float(roc_auc_score(y_test, proba))
    except ValueError:
        roc_auc = float("nan")

    cm = confusion_matrix(y_test, preds).tolist()

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "n_test": int(len(y_test)),
        "pos_rate_test": float(np.mean(y_test)) if len(y_test) else 0.0,
        "feature_importances": model.feature_importances(),
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    return metrics


# =============================================================================
# Top-level pipeline
# =============================================================================

def train_pipeline(
    regions: Optional[List[Region]] = None,
    timeframe: Timeframe = Timeframe.D1,
    max_tickers_per_region: Optional[int] = None,
    test_size: float = ML.test_size,
    save_artifacts: bool = True,
) -> TrainingResult:
    """
    Full pipeline: collect → features → split → train → eval → save.
    """
    regions = regions or [Region.US, Region.EUROPE, Region.ASIA]

    # 1. Collect closed trades from the backtester
    trades_with_region, dfs = collect_trades(
        regions=regions,
        timeframe=timeframe,
        max_tickers_per_region=max_tickers_per_region,
    )
    if not trades_with_region:
        logger.error("No trades collected — cannot train")
        return TrainingResult()

    # 2. Build the supervised learning matrix
    X, y = build_training_set(trades_with_region, dfs)
    if len(X) < 20 or len(np.unique(y)) < 2:
        logger.error(
            "Insufficient training data: n=%d  classes=%s",
            len(X), np.unique(y).tolist(),
        )
        return TrainingResult(n_trades=len(X))

    # 3. Split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=ML.random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    logger.info("Train/test split: %d / %d", len(X_train), len(X_test))

    # 4. Fit
    model = TradeRanker()
    model.fit(X_train, y_train)

    # 5. Evaluate
    metrics = evaluate(model, X_test, y_test)
    metrics["n_train"] = int(len(X_train))
    metrics["regions"] = [r.value for r in regions]
    metrics["timeframe"] = timeframe.value
    logger.info(
        "Eval: acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f  auc=%.3f",
        metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1"], metrics["roc_auc"],
    )

    # 6. Save
    saved = False
    if save_artifacts:
        try:
            model.save(MODEL_PATH, FEATURE_PATH)
            METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
            logger.info("Metrics written to %s", METRICS_PATH)
            saved = True
        except Exception:
            logger.exception("Failed to persist artifacts")

    return TrainingResult(
        model=model,
        metrics=metrics,
        n_trades=len(X),
        n_features=len(FEATURE_COLUMNS),
        saved=saved,
    )


# =============================================================================
# CLI
# =============================================================================

def _parse_regions(values: List[str]) -> List[Region]:
    out: List[Region] = []
    for v in values:
        key = v.strip().lower()
        match = {
            "us": Region.US,
            "europe": Region.EUROPE,
            "eu": Region.EUROPE,
            "asia": Region.ASIA,
        }.get(key)
        if match is None:
            raise argparse.ArgumentTypeError(f"Unknown region: {v!r}")
        out.append(match)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ml.train",
        description="Train the TradeRanker on backtested trades.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["us", "europe", "asia"],
        help="Regions to include (us / europe / asia).",
    )
    parser.add_argument(
        "--timeframe",
        choices=[tf.value for tf in Timeframe],
        default=Timeframe.D1.value,
        help="Bar timeframe.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Cap tickers per region (handy for a fast smoke run).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing model + metrics artifacts to disk.",
    )
    args = parser.parse_args()

    result = train_pipeline(
        regions=_parse_regions(args.regions),
        timeframe=Timeframe(args.timeframe),
        max_tickers_per_region=args.max_tickers,
        save_artifacts=not args.no_save,
    )

    if result.model is None:
        raise SystemExit("Training failed — see logs above.")

    summary = {
        "n_trades": result.n_trades,
        "n_features": result.n_features,
        "saved": result.saved,
        "metrics": {
            k: v for k, v in result.metrics.items()
            if k in ("accuracy", "precision", "recall", "f1", "roc_auc",
                     "n_train", "n_test", "pos_rate_test")
        },
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
