"""
ml - Machine-learning trade ranker.

Modules:
    features   Feature engineering. Builds a fixed-column numeric vector
               from a market snapshot (OHLCV window + Signal context).
               Same function used for training (over backtest trades) and
               for live inference (over the current signal).

    model      `TradeRanker`: RandomForestClassifier wrapper with
               class_weight="balanced" to handle the typical 30-50% win
               rate imbalance. Persists the model + the exact feature
               column order so reloads stay consistent.

    train      Pipeline that runs the backtester across each region's
               universe, collects all closed trades, labels them as
               wins / losses, fits the model, and writes the artifact
               set to `_models/`. CLI-callable: `python -m ml.train`.

Import directly:

    from ml.model import TradeRanker
    from ml.features import build_features_from_signal
"""
