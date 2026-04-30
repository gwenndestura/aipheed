"""
app/ml/training/cross_validation.py
--------------------------------------
Walk-forward expanding-window cross-validator for time-series province-quarter data.

Temporal structure (no look-ahead bias):
  Fold 1: train on quarters 1..min_train,   test on quarter min_train+1
  Fold 2: train on quarters 1..min_train+1, test on quarter min_train+2
  ...
  Fold N: train on all but last quarter,    test on last quarter

Province-aware: ALL 5 CALABARZON provinces appear in both train and test sets for
every fold (a fold operates on province-quarters, not raw quarters).

StratifiedShuffleSplit and random CV are INVALID for time-series — any random split
leaks future label information into training, inflating all reported metrics.

Usage:
    from app.ml.training.cross_validation import WalkForwardSplitter
    splitter = WalkForwardSplitter(min_train_quarters=8)
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
        Minimum number of distinct quarters to include in the first training fold.
        With 24 quarters (2020-Q1 to 2025-Q4), min_train_quarters=8 yields 16 folds.

    Attributes
    ----------
    n_folds_ : int
        Number of folds generated on the last call to split().
    fold_metadata_ : list[dict]
        Per-fold metadata (train_quarters, test_quarter, n_train, n_test).
    """

    def __init__(self, min_train_quarters: int = 8) -> None:
        self.min_train_quarters = min_train_quarters
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

        Parameters
        ----------
        df         : DataFrame containing at least a `quarter_col` column.
                     Must be indexed by default integer index (or any hashable index).
        quarter_col: column name holding the quarter string (e.g. "2020-Q1").

        Yields
        ------
        train_idx : pd.Index — row indices for the training window
        test_idx  : pd.Index — row indices for the test quarter
        """
        quarters: list[str] = sorted(df[quarter_col].unique())
        n_quarters = len(quarters)

        if n_quarters <= self.min_train_quarters:
            raise ValueError(
                f"Need > {self.min_train_quarters} distinct quarters; "
                f"got {n_quarters}."
            )

        self.fold_metadata_ = []
        fold = 0

        for k in range(self.min_train_quarters, n_quarters):
            train_quarters = quarters[:k]
            test_quarter = quarters[k]

            train_idx = df.index[df[quarter_col].isin(train_quarters)]
            test_idx = df.index[df[quarter_col] == test_quarter]

            meta = {
                "fold": fold,
                "train_quarters": train_quarters,
                "test_quarter": test_quarter,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
            self.fold_metadata_.append(meta)
            fold += 1

            logger.debug(
                "Fold %d: train=%d rows (%s→%s), test=%d rows (%s)",
                fold, len(train_idx), train_quarters[0], train_quarters[-1],
                len(test_idx), test_quarter,
            )

            yield train_idx, test_idx

        self.n_folds_ = fold
        logger.info(
            "WalkForwardSplitter: %d folds | min_train=%d quarters",
            self.n_folds_, self.min_train_quarters,
        )

    def get_n_splits(self, df: pd.DataFrame, quarter_col: str = "quarter") -> int:
        """Return the total number of folds without iterating."""
        n_quarters = df[quarter_col].nunique()
        return max(0, n_quarters - self.min_train_quarters)
