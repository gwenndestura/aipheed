"""
app/ml/training/trainer.py

LightGBM training with Walk-Forward Expanding-Window Cross-Validation.

Walk-Forward CV is MANDATORY for time-series data.
With forecast_gap=3, fold k trains on the first k quarters and tests on the
quarter that is 3 positions ahead — matching real deployment (predict 3 quarters
into the future based on current data).

Overfitting controls:
  - forecast_gap=3   : no intermediate-quarter leakage into the training window
  - HOLDOUT_QUARTERS : last N quarters never seen during Optuna search; used for
                       the final honest out-of-sample evaluation
  - class_weight="balanced" : counters class imbalance without hand-tuning
  - min_child_samples in [10, 50] : prevents leaves with too few samples
  - reg_alpha / reg_lambda searched log-uniformly over [1e-8, 10]

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

# ── Feature columns (31 features, no province_code, no quarter, no label) ────
# pct_total_hunger is intentionally excluded: label_fies = (pct_total_hunger > cycle_median),
# so including it hands the model the answer directly (perfect label leakage → ~100% accuracy).
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
]

LABEL_COL    = "label_fies"
PROVINCE_COL = "province_code"
QUARTER_COL  = "quarter"

# ── Training configuration ────────────────────────────────────────────────────
N_TRIALS           = 100
RANDOM_SEED        = 42
FORECAST_GAP       = 3   # quarters between end of training window and test quarter
HOLDOUT_QUARTERS   = 4   # last N quarters withheld from Optuna; used for final eval
MIN_TRAIN_QUARTERS = 8   # minimum training window for the first CV fold

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
        y.value_counts().to_dict(),
    )
    return X, y, quarters


def _make_objective(
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    df_cv: pd.DataFrame,
) -> callable:
    """
    Build Optuna objective using Walk-Forward CV on the CV window only.
    df_cv is a thin DataFrame with only the 'quarter' column, sharing the
    same index as X_cv / y_cv.  Maximises mean weighted F1 across folds.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "binary",
            "verbosity":         -1,
            "boosting_type":     "gbdt",
            "random_state":      RANDOM_SEED,
            # class_weight="balanced" compensates for imbalanced labels without
            # manual scale_pos_weight tuning
            "class_weight":      "balanced",
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
            # min_child_samples lower-bound raised from 1→10 to prevent leaves
            # with too few samples (a common source of overfitting on small panels)
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        splitter = WalkForwardSplitter(
            min_train_quarters=MIN_TRAIN_QUARTERS,
            forecast_gap=FORECAST_GAP,
        )
        fold_f1_scores = []

        for train_idx, test_idx in splitter.split(df_cv):
            if len(train_idx) < 5 or len(test_idx) < 1:
                continue

            # Use .loc because train_idx / test_idx are actual index labels,
            # not positional integers (safe even if df_cv has a non-contiguous index)
            X_train = X_cv.loc[train_idx]
            y_train = y_cv.loc[train_idx]
            X_test  = X_cv.loc[test_idx]
            y_test  = y_cv.loc[test_idx]

            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)

            y_pred   = model.predict(X_test)
            fold_f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            fold_f1_scores.append(fold_f1)

        if not fold_f1_scores:
            return 0.0

        return float(np.mean(fold_f1_scores))

    return objective


def train_model() -> dict:
    """
    Full training pipeline:
    1.  Load features_fused.parquet + labels.parquet
    2.  Reserve last HOLDOUT_QUARTERS as a true out-of-sample test set
    3.  Run Optuna 100-trial walk-forward search on the CV window only
    4.  Evaluate the best params on the holdout window (honest final metrics)
    5.  Retrain production model on ALL data with best params
    6.  Serialize lgbm_best.pkl and optuna_study.pkl
    7.  Log with MLflow

    Returns:
        dict with best_params, best_cv_f1, holdout_f1, holdout_roc_auc
    """
    logger.info("=" * 60)
    logger.info(
        "LIGHTGBM TRAINING — WALK-FORWARD CV (gap=%d, holdout=%d quarters)",
        FORECAST_GAP, HOLDOUT_QUARTERS,
    )
    logger.info("=" * 60)

    X, y, quarters = _load_data()

    # ── Split CV window from holdout ──────────────────────────────────────
    all_quarters = sorted(quarters.unique())

    min_needed = MIN_TRAIN_QUARTERS + FORECAST_GAP + 1 + HOLDOUT_QUARTERS
    if len(all_quarters) < min_needed:
        raise ValueError(
            f"Not enough quarters for this configuration: need >= {min_needed}, "
            f"got {len(all_quarters)}."
        )

    holdout_qs   = set(all_quarters[-HOLDOUT_QUARTERS:])
    cv_mask      = ~quarters.isin(holdout_qs)
    holdout_mask = quarters.isin(holdout_qs)

    X_cv        = X[cv_mask]
    y_cv        = y[cv_mask]
    # Thin DataFrame used only so WalkForwardSplitter can look up the quarter column
    df_cv       = pd.DataFrame(
        {QUARTER_COL: quarters[cv_mask].values},
        index=X_cv.index,
    )
    X_holdout   = X[holdout_mask]
    y_holdout   = y[holdout_mask]

    logger.info(
        "CV window : %s → %s  (%d rows)",
        all_quarters[0], all_quarters[-HOLDOUT_QUARTERS - 1], len(X_cv),
    )
    logger.info(
        "Holdout   : %s → %s  (%d rows) — never seen during Optuna search",
        all_quarters[-HOLDOUT_QUARTERS], all_quarters[-1], len(X_holdout),
    )

    # ── Optuna search on CV window only ──────────────────────────────────
    logger.info(
        "Starting Optuna: %d trials, walk-forward CV, gap=%d quarters",
        N_TRIALS, FORECAST_GAP,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_aipheed",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        _make_objective(X_cv, y_cv, df_cv),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_cv_f1  = study.best_value

    logger.info("Best trial CV-F1 : %.4f", best_cv_f1)
    logger.info("Best params      : %s", best_params)

    # ── Honest holdout evaluation ─────────────────────────────────────────
    # Train on CV window only, evaluate on holdout — the model has never
    # seen the holdout rows at any point in the Optuna search.
    eval_params = {
        **best_params,
        "objective":     "binary",
        "boosting_type": "gbdt",
        "verbosity":     -1,
        "random_state":  RANDOM_SEED,
        "class_weight":  "balanced",
    }

    eval_model = LGBMClassifier(**eval_params)
    eval_model.fit(X_cv, y_cv)

    y_pred_h = eval_model.predict(X_holdout)
    y_prob_h = eval_model.predict_proba(X_holdout)[:, 1]

    holdout_f1  = float(f1_score(y_holdout, y_pred_h, average="weighted", zero_division=0))
    try:
        holdout_auc = float(roc_auc_score(y_holdout, y_prob_h))
    except ValueError:
        # Holdout may contain only one class in small datasets
        holdout_auc = float("nan")
        logger.warning("roc_auc_score undefined on holdout (single class present)")

    logger.info("Holdout F1      : %.4f", holdout_f1)
    logger.info("Holdout ROC-AUC : %.4f", holdout_auc)
    logger.info(
        "F1 target  (%s) : %s", TARGET_F1,
        "PASS" if holdout_f1 >= TARGET_F1 else "FAIL",
    )
    logger.info(
        "AUC target (%s) : %s", TARGET_ROC_AUC,
        "PASS" if holdout_auc >= TARGET_ROC_AUC else "FAIL",
    )

    # ── Production model: retrain on ALL data ────────────────────────────
    # Maximises training data for deployment; metrics above are the honest estimate.
    prod_model = LGBMClassifier(**eval_params)
    prod_model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(prod_model, MODEL_PATH)
    joblib.dump(study, STUDY_PATH)
    logger.info("Production model saved to %s", MODEL_PATH)

    # ── MLflow logging ────────────────────────────────────────────────────
    with mlflow.start_run(run_name="lgbm_aipheed"):
        mlflow.log_params(best_params)
        mlflow.log_param("forecast_gap",      FORECAST_GAP)
        mlflow.log_param("holdout_quarters",  HOLDOUT_QUARTERS)
        mlflow.log_metric("best_cv_f1",       best_cv_f1)
        mlflow.log_metric("holdout_f1",       holdout_f1)
        mlflow.log_metric("holdout_roc_auc",  holdout_auc)
        mlflow.log_metric("n_trials",         N_TRIALS)
        mlflow.sklearn.log_model(prod_model, "lgbm_model")

    return {
        "best_params":           best_params,
        "best_cv_f1":            round(best_cv_f1, 4),
        "holdout_f1":            round(holdout_f1, 4),
        "holdout_roc_auc":       round(holdout_auc, 4),
        "meets_f1_target":       holdout_f1 >= TARGET_F1,
        "meets_roc_auc_target":  holdout_auc >= TARGET_ROC_AUC,
    }
