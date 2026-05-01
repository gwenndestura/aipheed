"""
app/ml/training/evaluator.py
------------------------------
Four-baseline benchmark on walk-forward folds.

Baselines evaluated on the SAME walk-forward CV folds as the LightGBM trainer,
including the same forecast_gap so the comparison is fair:

  1. Naive Persistence  : predict(t+k) = label(t) for each province
  2. PSA-Only Logistic  : LogisticRegression on economic/climate features only
  3. NLP-Only Logistic  : LogisticRegression on FSSI + 5-trigger features only
  4. Full-Feature Logistic: LogisticRegression on all 31 features (no pct_total_hunger)

Metrics per fold and aggregated:
  - Weighted F1, ROC-AUC, Accuracy, Precision (weighted), Recall (weighted)

pct_total_hunger is EXCLUDED from all feature sets: it is the raw SWS value
that directly determines label_fies (label = sws > cycle_median), so including
it would constitute label leakage.

Output:
  data/processed/eval_results.json

Usage:
    from app.ml.training.evaluator import run_baseline_evaluation
    results = run_baseline_evaluation()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from app.ml.training.cross_validation import WalkForwardSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

NLP_FEATURES: list[str] = [
    "FSSI", "FSSI_lag1", "FSSI_lag2", "FSSI_accel",
    "trigger_market", "trigger_climate", "trigger_employment",
    "trigger_ofw_remittance", "trigger_fish_kill",
]

PSA_FEATURES: list[str] = [
    "food_cpi", "food_cpi_yoy", "food_minus_headline_yoy", "headline_cpi",
    "rice_price_regular",
    "unemployment_rate", "poverty_incidence",
    "ofw_remit_yoy_pct", "fx_usd_php_avg",
    "diesel_php_per_l", "gasoline_php_per_l", "brent_usd_per_bbl",
    "tc_count", "tc_severe_flag", "rainfall_anomaly_pct",
    "drought_alert", "enso_numeric",
    "commodity_fruit_veg", "commodity_leafy_veg", "commodity_livestock",
    "commodity_poultry", "commodity_rootcrops",
]

ALL_FEATURES: list[str] = PSA_FEATURES + NLP_FEATURES

# Province code column — treated as one-hot categorical
PROVINCE_COL: str = "province_code"

# Columns excluded from features (identifier or leakage)
EXCLUDE_COLS: set[str] = {
    "province_code", "quarter",
    "pct_total_hunger",  # raw label source — leakage if included
}

LABEL_COL: str = "label_fies"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FEATURES_PATH = Path("data/processed/features_fused.parquet")
LABELS_PATH   = Path("data/processed/labels.parquet")
OUTPUT_PATH   = Path("data/processed/eval_results.json")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _load_merged(
    features_path: Path = FEATURES_PATH,
    labels_path: Path = LABELS_PATH,
) -> pd.DataFrame:
    """Merge features and labels on (province_code, quarter)."""
    feat_df = pd.read_parquet(features_path)
    lab_df  = pd.read_parquet(labels_path)[
        ["province_code", "quarter", LABEL_COL]
    ]
    df = feat_df.merge(lab_df, on=["province_code", "quarter"], how="inner")
    df = df.sort_values(["quarter", "province_code"]).reset_index(drop=True)

    logger.info(
        "_load_merged: %d rows | %d feature cols | label dist: %s",
        len(df), len(feat_df.columns) - 2,
        df[LABEL_COL].value_counts().to_dict(),
    )
    return df


def _build_pipeline(feature_cols: list[str]) -> Pipeline:
    """
    Build a sklearn Pipeline for logistic regression.

    Numeric features are standardized; province_code is one-hot encoded.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [PROVINCE_COL]),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])


# ---------------------------------------------------------------------------
# Baseline 1: Naive Persistence
# ---------------------------------------------------------------------------

def _naive_persistence_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict last known label per province.

    Returns (y_pred, y_prob) where y_prob is 0.95 for positive, 0.05 for negative.
    """
    last_labels = (
        train_df.sort_values("quarter")
        .groupby("province_code")[LABEL_COL]
        .last()
    )
    preds = test_df["province_code"].map(last_labels).fillna(0).astype(int)
    probs = preds.map({1: 0.95, 0: 0.05}).values
    return preds.values, probs


# ---------------------------------------------------------------------------
# Baseline 2–4: Logistic Regression variants
# ---------------------------------------------------------------------------

def _logistic_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit logistic regression on train, predict on test.

    Returns (y_pred, y_prob_positive).
    """
    available = set(train_df.columns)
    cols = [c for c in feature_cols if c in available]
    if len(cols) < len(feature_cols):
        logger.warning(
            "Missing feature columns (skipped): %s", set(feature_cols) - available
        )

    pipe = _build_pipeline(cols)

    X_train = train_df[cols + [PROVINCE_COL]]
    y_train = train_df[LABEL_COL]
    X_test  = test_df[cols + [PROVINCE_COL]]

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    return y_pred, y_prob


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute a standardized metrics dict for one fold or aggregate."""
    metrics: dict = {}

    metrics["accuracy"]    = float(accuracy_score(y_true, y_pred))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["precision"]   = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["recall"]      = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    metrics["classification_report"] = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return metrics


def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Compute mean ± std across folds for scalar metrics."""
    scalar_keys = ["accuracy", "weighted_f1", "precision", "recall", "roc_auc"]
    agg: dict = {}
    for key in scalar_keys:
        vals = [m[key] for m in fold_metrics if not np.isnan(m[key])]
        agg[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{key}_std"]  = float(np.std(vals))  if vals else float("nan")
    return agg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_baseline_evaluation(
    features_path: Path = FEATURES_PATH,
    labels_path: Path = LABELS_PATH,
    output_path: Path = OUTPUT_PATH,
    min_train_quarters: int = 8,
    forecast_gap: int = 3,
) -> dict:
    """
    Run 4-baseline evaluation on walk-forward CV folds.

    Uses the same forecast_gap as the LightGBM trainer so the comparison is fair —
    all baselines predict the same number of quarters ahead under the same gap constraint.

    Returns the full results dict (also saved to output_path).

    Parameters
    ----------
    features_path     : path to features_fused.parquet
    labels_path       : path to labels.parquet
    output_path       : where to save eval_results.json
    min_train_quarters: minimum quarters in first training fold (default 8 = 2 years)
    forecast_gap      : quarters between end of training and test quarter (default 3)
    """
    df = _load_merged(features_path, labels_path)
    splitter = WalkForwardSplitter(
        min_train_quarters=min_train_quarters,
        forecast_gap=forecast_gap,
    )

    baselines = {
        "naive_persistence":     {"feature_cols": None,         "type": "naive"},
        "psa_only_logistic":     {"feature_cols": PSA_FEATURES, "type": "logistic"},
        "nlp_only_logistic":     {"feature_cols": NLP_FEATURES, "type": "logistic"},
        "full_feature_logistic": {"feature_cols": ALL_FEATURES, "type": "logistic"},
    }

    fold_results: dict[str, list[dict]] = {name: [] for name in baselines}
    all_y_true:  dict[str, list]        = {name: [] for name in baselines}
    all_y_pred:  dict[str, list]        = {name: [] for name in baselines}
    all_y_prob:  dict[str, list]        = {name: [] for name in baselines}

    n_folds = splitter.get_n_splits(df)
    logger.info(
        "Starting evaluation: %d folds, %d baselines, gap=%d quarters",
        n_folds, len(baselines), forecast_gap,
    )

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(df)):
        train_df     = df.loc[train_idx]
        test_df      = df.loc[test_idx]
        y_true       = test_df[LABEL_COL].values
        test_quarter = splitter.fold_metadata_[fold_idx]["test_quarter"]

        for name, cfg in baselines.items():
            if cfg["type"] == "naive":
                y_pred, y_prob = _naive_persistence_fold(train_df, test_df)
            else:
                try:
                    y_pred, y_prob = _logistic_fold(train_df, test_df, cfg["feature_cols"])
                except Exception as exc:
                    logger.warning("Fold %d baseline %s failed: %s", fold_idx, name, exc)
                    y_pred = np.zeros(len(test_df), dtype=int)
                    y_prob = np.zeros(len(test_df))

            fold_metrics = _compute_metrics(y_true, y_pred, y_prob)
            fold_metrics["fold"]         = fold_idx
            fold_metrics["test_quarter"] = test_quarter
            fold_results[name].append(fold_metrics)

            all_y_true[name].extend(y_true.tolist())
            all_y_pred[name].extend(y_pred.tolist())
            all_y_prob[name].extend(y_prob.tolist())

    # Build final results
    results: dict = {
        "config": {
            "min_train_quarters":     min_train_quarters,
            "forecast_gap":           forecast_gap,
            "n_folds":                splitter.n_folds_,
            "n_provinces":            df["province_code"].nunique(),
            "n_quarters":             df["quarter"].nunique(),
            "label_col":              LABEL_COL,
            "label_distribution":     df[LABEL_COL].value_counts().to_dict(),
            "pct_total_hunger_excluded": True,
            "leakage_note": (
                "pct_total_hunger excluded from all feature sets — "
                "it is the raw SWS value that determines label_fies."
            ),
        },
        "feature_groups": {
            "nlp_features":       NLP_FEATURES,
            "psa_features":       PSA_FEATURES,
            "all_features_count": len(ALL_FEATURES),
        },
        "baselines": {},
        "target_metrics": {
            "lightgbm_weighted_f1_target": 0.75,
            "lightgbm_roc_auc_target":     0.80,
            "note": "LightGBM results added by trainer.py",
        },
    }

    for name, fold_list in fold_results.items():
        y_true_all = np.array(all_y_true[name])
        y_pred_all = np.array(all_y_pred[name])
        y_prob_all = np.array(all_y_prob[name])

        overall   = _compute_metrics(y_true_all, y_pred_all, y_prob_all)
        aggregate = _aggregate_fold_metrics(fold_list)

        results["baselines"][name] = {
            "overall": {
                k: v for k, v in overall.items()
                if k not in ("classification_report", "confusion_matrix")
            },
            "aggregate_across_folds":  aggregate,
            "confusion_matrix_overall": overall["confusion_matrix"],
            "fold_metrics": [
                {k: v for k, v in fm.items()
                 if k not in ("classification_report", "confusion_matrix")}
                for fm in fold_list
            ],
        }

        logger.info(
            "Baseline %-25s | weighted_F1=%.3f | precision=%.3f | recall=%.3f | ROC-AUC=%.3f",
            name,
            overall["weighted_f1"],
            overall["precision"],
            overall["recall"],
            overall["roc_auc"] if not np.isnan(overall["roc_auc"]) else -1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("eval_results.json saved → %s", output_path)

    return results
