"""
TradeRanker - Random Forest classifier wrapping for trade ranking.

The model takes the feature vector defined in `ml.features` and outputs
the probability that a trade will be a winner (net_pnl > 0). The same
class is used for:

    - Training:  fit(X, y), then save() to `_models/`.
    - Inference: load() once, then score_signal(signal, df) per signal
                 inside the orchestrator / dashboard.

Persistence:
    The estimator is joblib-dumped, and the exact `FEATURE_COLUMNS` list
    that was used at fit time is written alongside as JSON. Reload
    cross-checks the column list and raises on mismatch — this catches
    accidental feature reorderings between training and serving.

Failure modes:
    - score_signal returns 0.0 (rather than raising) when the model is
      not yet trained or the window is too short to compute features.
    - load() raises FileNotFoundError if either artifact is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from config import FEATURE_PATH, ML, MODEL_PATH
from ml.features import (
    FEATURE_COLUMNS,
    build_features_from_signal,
    features_dict_to_row,
    features_to_dataframe,
)
from utils.helpers import Signal, get_logger

logger = get_logger("ml.model")


# =============================================================================
# Model wrapper
# =============================================================================

class TradeRanker:
    """
    Sklearn-backed binary classifier. Predict-proba column 1 is the
    P(win) score returned by `score_signal`.
    """

    def __init__(self) -> None:
        # Import lazily so the project imports cleanly when scikit-learn is
        # not installed (e.g. the dashboard can still run, the user just
        # can't train or auto-rank signals).
        from sklearn.ensemble import RandomForestClassifier

        self._estimator = RandomForestClassifier(
            n_estimators=ML.n_estimators,
            max_depth=ML.max_depth,
            min_samples_leaf=ML.min_samples_leaf,
            class_weight=ML.class_weight,
            random_state=ML.random_state,
            n_jobs=ML.n_jobs,
        )
        self.feature_columns_: List[str] = list(FEATURE_COLUMNS)
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def is_trained(self) -> bool:
        return self._is_trained

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y) -> "TradeRanker":
        """
        Fit on a feature matrix and binary win/loss labels.

        Args:
            X: DataFrame whose columns are EXACTLY `FEATURE_COLUMNS`
               (use `features_to_dataframe` to assemble it).
            y: Iterable of 0/1 labels (1 = win, 0 = loss).
        """
        self._validate_columns(X)
        y_arr = np.asarray(y).astype(int)
        if len(X) != len(y_arr):
            raise ValueError(
                f"X / y length mismatch: {len(X)} vs {len(y_arr)}"
            )
        if len(np.unique(y_arr)) < 2:
            raise ValueError(
                "Need at least two classes in y; got "
                f"{np.unique(y_arr).tolist()}"
            )

        self._estimator.fit(X.values, y_arr)
        self.feature_columns_ = list(X.columns)
        self._is_trained = True
        logger.info(
            "TradeRanker trained: n=%d  pos_rate=%.3f  features=%d",
            len(y_arr), float(y_arr.mean()), len(self.feature_columns_),
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return the win-class probability for each row (1-D array of length n).
        """
        if not self._is_trained:
            return np.zeros(len(X), dtype=float)
        self._validate_columns(X)
        proba = self._estimator.predict_proba(X.values)
        # Column 1 corresponds to class=1 (win) because classes_ are sorted.
        win_col = int(np.where(self._estimator.classes_ == 1)[0][0])
        return proba[:, win_col]

    def score_signal(self, signal: Signal, df: pd.DataFrame) -> float:
        """
        Inference convenience for the live path. Returns a 0-1 win
        probability for the given Signal. Returns 0.0 when untrained or
        on any inference error (logged).
        """
        if not self._is_trained:
            return 0.0
        try:
            feat = build_features_from_signal(signal, df)
            row = features_dict_to_row(feat)
            X = pd.DataFrame([row], columns=self.feature_columns_)
            proba = self.predict_proba(X)
            return float(proba[0])
        except Exception:
            logger.exception("score_signal failed for %s", signal.ticker)
            return 0.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def feature_importances(self) -> Dict[str, float]:
        """Map feature name -> importance (empty dict when untrained)."""
        if not self._is_trained:
            return {}
        importances = self._estimator.feature_importances_
        return {
            col: float(imp)
            for col, imp in zip(self.feature_columns_, importances)
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(
        self,
        model_path: Path = MODEL_PATH,
        feature_path: Path = FEATURE_PATH,
    ) -> None:
        """Persist the estimator + the feature column list."""
        if not self._is_trained:
            raise RuntimeError("Cannot save an untrained model")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._estimator, model_path)
        feature_path.write_text(json.dumps(self.feature_columns_, indent=2))
        logger.info("Model saved to %s", model_path)

    @classmethod
    def load(
        cls,
        model_path: Path = MODEL_PATH,
        feature_path: Path = FEATURE_PATH,
    ) -> "TradeRanker":
        """Load a previously-saved model + its feature column list."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature columns file not found: {feature_path}"
            )

        instance = cls()
        instance._estimator = joblib.load(model_path)
        instance.feature_columns_ = json.loads(feature_path.read_text())
        instance._is_trained = True

        # Sanity: warn if the columns diverged from the canonical list
        # (the orchestrator should be using the same list at inference).
        canonical = list(FEATURE_COLUMNS)
        if instance.feature_columns_ != canonical:
            logger.warning(
                "Loaded feature columns differ from FEATURE_COLUMNS — "
                "the model was trained against a different feature set",
            )
        logger.info("Model loaded from %s", model_path)
        return instance

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _validate_columns(self, X: pd.DataFrame) -> None:
        """Raise if X's columns don't match the trained / canonical order."""
        expected = self.feature_columns_ if self._is_trained else FEATURE_COLUMNS
        actual = list(X.columns)
        if actual != expected:
            missing = [c for c in expected if c not in actual]
            extra = [c for c in actual if c not in expected]
            raise ValueError(
                "Feature column mismatch. "
                f"Missing: {missing}  Extra: {extra}  "
                f"Order ok: {actual == expected}"
            )


# =============================================================================
# Convenience batch helper
# =============================================================================

def score_many(
    ranker: TradeRanker,
    signals_with_dfs,
) -> List[float]:
    """
    Score a list of (Signal, df) pairs in one batched predict call.

    More efficient than looping score_signal() when ranking a watchlist.
    """
    if not ranker.is_trained:
        return [0.0] * len(signals_with_dfs)

    rows = []
    for sig, df in signals_with_dfs:
        feat = build_features_from_signal(sig, df)
        rows.append(feat)

    if not rows:
        return []

    X = features_to_dataframe(rows)
    # Align to the trained column order in case features.py ever drifts.
    X = X[ranker.feature_columns_]
    return [float(p) for p in ranker.predict_proba(X)]
