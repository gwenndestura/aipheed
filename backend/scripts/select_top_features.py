"""
scripts/select_top_features.py
-------------------------------
Rank the 45 model features by LightGBM gain importance and print the top 15.

Single fits on a 120-row matrix are noisy, so importance is averaged over
several random seeds (same discovery params the trainer uses). Prints the
ranked list; does NOT edit trainer.py (that's a deliberate, reviewed step).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.training.trainer import (  # noqa: E402
    FEATURE_COLS, LABEL_COL, PROVINCE_COL, QUARTER_COL,
    FEATURES_PATH, LABELS_PATH,
)

TOP_N = 15
SEEDS = [42, 7, 123, 2024, 99, 314, 1, 555, 88, 2026]


def main() -> None:
    feat = pd.read_parquet(FEATURES_PATH)
    lab = pd.read_parquet(LABELS_PATH)
    df = feat.merge(lab[[PROVINCE_COL, QUARTER_COL, LABEL_COL]],
                    on=[PROVINCE_COL, QUARTER_COL], how="inner")
    X = df[FEATURE_COLS]
    y = df[LABEL_COL]
    print(f"matrix: {X.shape[0]} rows x {X.shape[1]} features | "
          f"positives: {int(y.sum())}/{len(y)}")

    gains = np.zeros(len(FEATURE_COLS))
    for s in SEEDS:
        m = LGBMClassifier(objective="binary", verbosity=-1, boosting_type="gbdt",
                           class_weight="balanced", n_estimators=200,
                           learning_rate=0.05, num_leaves=63, max_depth=6,
                           random_state=s, importance_type="gain")
        m.fit(X, y)
        g = np.asarray(m.feature_importances_, dtype=float)
        if g.sum() > 0:
            g = g / g.sum()          # normalize per-seed so seeds weigh equally
        gains += g
    gains /= len(SEEDS)

    ranked = sorted(zip(FEATURE_COLS, gains), key=lambda kv: kv[1], reverse=True)
    print("\nrank  gain%   feature")
    for i, (f, g) in enumerate(ranked, 1):
        mark = "  <= keep" if i <= TOP_N else ""
        print(f"{i:>3}  {g*100:6.2f}  {f}{mark}")

    top = [f for f, _ in ranked[:TOP_N]]
    print("\nTOP_15 =", top)


if __name__ == "__main__":
    main()
