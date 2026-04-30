"""
app/ml/training/trainer.py

LightGBM training with Walk-Forward Expanding-Window Cross-Validation.

Walk-Forward CV is MANDATORY for time-series data.
Fold k trains on 2020Q1..2020Q1+k, forecasts 2020Q1+k+3.
No future data ever enters a training window.

Usage:
    from app.ml.training.trainer import train_model
    result = train_model()
"""

import logging
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd

from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score
from app.ml.training.cross_validation import WalkForwardSplitter

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = Path("data/processed/features_fused.parquet")
LABELS_PATH   = Path("data/processed/labels.parquet")
MODEL_PATH    = Path("models/lgbm_best.pkl")
STUDY_PATH    = Path("models/optuna_study.pkl")

# ── Feature columns (32 features, no province_code, no quarter, no label) ────
FEATURE_COLS = [
    "FSSI", "FSSI_lag1", "FSSI_lag2", "FSSI_accel",
    "trigger_market", "trigger_climate", "trigger_employment",
    "trigger_ofw_remittance", "trigger_fish_kill",
    "food_cpi", "food_cpi_yoy", "rice_price_regular",
    "unemployment_rate", "poverty_incidence",
    "food_minus_headline_yoy", "headline_cpi",
    "ofw_remit_yoy_pct", "fx_usd_php_avg",
    "diesel_php_per_l", "gasoline_php_per_l", "brent_usd_per_bbl",
    "tc_count", "tc_severe_flag", "rainfall_anomaly_pct",
    "drought_alert", "enso_numeric",
    "commodity_fruit_veg", "commodity_leafy_veg",
    "commodity_livestock", "commodity_poultry", "commodity_rootcrops",
    "pct_total_hunger",
]

LABEL_COL   = "label_fies"
PROVINCE_COL = "province_code"
QUARTER_COL  = "quarter"

# ── Optuna search space ───────────────────────────────────────────────────────
N_TRIALS   = 100
RANDOM_SEED = 42

# ── Performance targets ───────────────────────────────────────────────────────
TARGET_F1      = 0.75
TARGET_ROC_AUC = 0.80


def _load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load features_fused.parquet and labels.parquet.
    Join on province_code + quarter.
    Returns X (features), y (labels), quarters (for walk-forward splits).
    """
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"features_fused.parquet not found at {FEATURES_PATH}. "
            "Run feature_matrix.py first."
        )
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"labels.parquet not found at {LABELS_PATH}. "
            "Run label_generator.py first."
        )

    features = pd.read_parquet(FEATURES_PATH)
    labels   = pd.read_parquet(LABELS_PATH)

    df = features.merge(
        labels[[PROVINCE_COL, QUARTER_COL, LABEL_COL]],
        on=[PROVINCE_COL, QUARTER_COL],
        how="inner",
    )

    df = df.sort_values([QUARTER_COL, PROVINCE_COL]).reset_index(drop=True)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X        = df[FEATURE_COLS]
    y        = df[LABEL_COL].astype(int)
    quarters = df[QUARTER_COL]

    logger.info(
        "Data loaded: %d rows | %d features | label distribution: %s",
        len(df), len(FEATURE_COLS),
        y.value_counts().to_dict()
    )
    return X, y, quarters


def _make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    quarters: pd.Series,
) -> callable:
    """
    Build Optuna objective using Walk-Forward CV.
    Maximizes mean weighted F1 across all folds.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "binary",
            "verbosity":         -1,
            "boosting_type":     "gbdt",
            "random_state":      RANDOM_SEED,
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 20),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        splitter = WalkForwardSplitter(quarters=quarters, forecast_gap=3)
        fold_f1_scores = []

        for train_idx, test_idx in splitter.split():
            if len(train_idx) < 5 or len(test_idx) < 1:
                continue

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_test  = y.iloc[test_idx]

            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            fold_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            fold_f1_scores.append(fold_f1)

        if not fold_f1_scores:
            return 0.0

        return float(np.mean(fold_f1_scores))

    return objective


def train_model() -> dict:
    """
    Full training pipeline:
    1. Load features_fused.parquet + labels.parquet
    2. Run Optuna 100-trial walk-forward search
    3. Retrain final model on full data with best params
    4. Serialize lgbm_best.pkl and optuna_study.pkl
    5. Log with MLflow

    Returns:
        dict with best_params, best_f1, final_f1, final_roc_auc
    """
    logger.info("=" * 60)
    logger.info("LIGHTGBM TRAINING — WALK-FORWARD CV")
    logger.info("=" * 60)

    X, y, quarters = _load_data()

    # ── Optuna search ─────────────────────────────────────────────────────
    logger.info("Starting Optuna: %d trials, walk-forward CV", N_TRIALS)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_aipheed",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        _make_objective(X, y, quarters),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_f1     = study.best_value

    logger.info("Best trial F1: %.4f", best_f1)
    logger.info("Best params: %s", best_params)

    # ── Final model — retrain on full data ───────────────────────────────
    final_params = {
        **best_params,
        "objective":     "binary",
        "boosting_type": "gbdt",
        "verbosity":     -1,
        "random_state":  RANDOM_SEED,
    }

    final_model = LGBMClassifier(**final_params)
    final_model.fit(X, y)

    y_pred     = final_model.predict(X)
    y_prob     = final_model.predict_proba(X)[:, 1]
    final_f1   = float(f1_score(y, y_pred, average="weighted", zero_division=0))
    final_auc  = float(roc_auc_score(y, y_prob))

    logger.info("Final model — F1: %.4f | ROC-AUC: %.4f", final_f1, final_auc)
    logger.info("F1 target (%s): %s", TARGET_F1, "PASS" if final_f1 >= TARGET_F1 else "FAIL")
    logger.info("AUC target (%s): %s", TARGET_ROC_AUC, "PASS" if final_auc >= TARGET_ROC_AUC else "FAIL")

    # ── Serialize ─────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(study, STUDY_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    # ── MLflow logging ────────────────────────────────────────────────────
    with mlflow.start_run(run_name="lgbm_aipheed"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_f1",  best_f1)
        mlflow.log_metric("final_f1",    final_f1)
        mlflow.log_metric("final_auc",   final_auc)
        mlflow.log_metric("n_trials",    N_TRIALS)
        mlflow.sklearn.log_model(final_model, "lgbm_model")

    return {
        "best_params":  best_params,
        "best_cv_f1":   round(best_f1, 4),
        "final_f1":     round(final_f1, 4),
        "final_roc_auc": round(final_auc, 4),
        "meets_f1_target":      final_f1 >= TARGET_F1,
        "meets_roc_auc_target": final_auc >= TARGET_ROC_AUC,
    }