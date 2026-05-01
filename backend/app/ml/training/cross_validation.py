"""
app/ml/training/cross_validation.py
--------------------------------------
Walk-forward expanding-window cross-validator for time-series province-quarter data.

Temporal structure (no look-ahead bias):
  Fold 1: train on quarters 1..min_train,
          skip quarters min_train+1..min_train+forecast_gap (gap),
          test  on quarter min_train+forecast_gap+1
  Fold 2: train on quarters 1..min_train+1, ..., test on quarter min_train+forecast_gap+2
  ...
  Fold N: train on all but (1 + forecast_gap) trailing quarters, test on last quarter

forecast_gap controls how many quarters are left between the end of training and the test
quarter.  Set forecast_gap=0 to test on the immediately next quarter (standard walk-forward).
Set forecast_gap=3 to simulate a real-world 3-quarter-ahead prediction system (the rows in
the gap window are excluded from both train and test for that fold).

Province-aware: ALL CALABARZON provinces appear in both train and test sets for every fold.

StratifiedShuffleSplit and random CV are INVALID for time-series — any random split leaks
future label information into training, inflating all reported metrics.

Usage:
    from app.ml.training.cross_validation import WalkForwardSplitter
    splitter = WalkForwardSplitter(min_train_quarters=8, forecast_gap=3)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df)):
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
"""

from __future__ import annotations

import logging
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    """
    Walk-forward expanding-window splitter for province-quarter DataFrames.

    Parameters
    ----------
    min_train_quarters : int
        Minimum number of distinct quarters in the first training fold (default 8).
    forecast_gap : int
        Number of quarters to skip between the end of training and the test quarter.
        0 = predict the immediately next quarter.
        3 = predict 3 quarters ahead (rows in the gap are excluded from both sets).

    Attributes
    ----------
    n_folds_ : int
        Number of folds generated on the last call to split().
    fold_metadata_ : list[dict]
        Per-fold metadata (train_quarters, gap_quarters, test_quarter, n_train, n_test).
    """

    def __init__(
        self,
        min_train_quarters: int = 8,
        forecast_gap: int = 0,
    ) -> None:
        if forecast_gap < 0:
            raise ValueError(f"forecast_gap must be >= 0, got {forecast_gap}")
        self.min_train_quarters = min_train_quarters
        self.forecast_gap = forecast_gap
        self.n_folds_: int = 0
        self.fold_metadata_: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        df: pd.DataFrame,
        quarter_col: str = "quarter",
    ) -> Iterator[tuple[pd.Index, pd.Index]]:
        """
        Yield (train_indices, test_indices) for each walk-forward fold.

        Rows that fall in the gap window (between the last training quarter and the test
        quarter) are excluded from both train_idx and test_idx for that fold.

        Parameters
        ----------
        df         : DataFrame containing at least a `quarter_col` column.
                     Must be sorted by quarter before calling (or this method will sort).
        quarter_col: column name holding the quarter string (e.g. "2020-Q1").

        Yields
        ------
        train_idx : pd.Index — row indices for the training window
        test_idx  : pd.Index — row indices for the test quarter
        """
        quarters: list[str] = sorted(df[quarter_col].unique())
        n_quarters = len(quarters)
        gap = self.forecast_gap

        min_required = self.min_train_quarters + gap + 1
        if n_quarters < min_required:
            raise ValueError(
                f"Need at least {min_required} distinct quarters "
                f"(min_train={self.min_train_quarters} + gap={gap} + 1 test); "
                f"got {n_quarters}."
            )

        self.fold_metadata_ = []
        fold = 0

        # k is the number of training quarters in this fold.
        # The test quarter sits gap positions ahead of the last training quarter.
        # Valid k: min_train_quarters <= k <= n_quarters - gap - 1
        for k in range(self.min_train_quarters, n_quarters - gap):
            train_quarters = quarters[:k]
            test_quarter   = quarters[k + gap]
            gap_quarters   = quarters[k:k + gap]  # excluded from both sets

            train_idx = df.index[df[quarter_col].isin(train_quarters)]
            test_idx  = df.index[df[quarter_col] == test_quarter]

            meta = {
                "fold":           fold,
                "train_quarters": train_quarters,
                "gap_quarters":   list(gap_quarters),
                "test_quarter":   test_quarter,
                "n_train":        len(train_idx),
                "n_test":         len(test_idx),
            }
            self.fold_metadata_.append(meta)
            fold += 1

            logger.debug(
                "Fold %d: train=%d rows (%s→%s) | gap=%s | test=%d rows (%s)",
                fold, len(train_idx),
                train_quarters[0], train_quarters[-1],
                list(gap_quarters) if gap_quarters else "none",
                len(test_idx), test_quarter,
            )

            yield train_idx, test_idx

        self.n_folds_ = fold
        logger.info(
            "WalkForwardSplitter: %d folds | min_train=%d | gap=%d quarters",
            self.n_folds_, self.min_train_quarters, self.forecast_gap,
        )

    def get_n_splits(self, df: pd.DataFrame, quarter_col: str = "quarter") -> int:
        """Return the total number of folds without iterating."""
        n_quarters = df[quarter_col].nunique()
        return max(0, n_quarters - self.min_train_quarters - self.forecast_gap)
