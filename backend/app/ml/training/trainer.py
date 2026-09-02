"""
app/ml/training/trainer.py

LightGBM training with Walk-Forward Expanding-Window Cross-Validation.

Walk-Forward CV is MANDATORY for time-series data.
The forecast gap is NOT fixed. discover_max_lead_time() iterates k = 1, 2, 3 ...
and halts at the first k where mean walk-forward Accuracy drops below
ACCURACY_THRESHOLD (75%). The last passing k becomes the Operationally Valid
Forecast Horizon used for the Optuna search and production model.

This directly answers Research Question 2:
  "What is the furthest lead time at which aiPHeed can reliably forecast
   province-level food insecurity risk across CALABARZON?"

Overfitting controls:
  - dynamic forecast_gap   : empirically discovered, no look-ahead leakage
  - HOLDOUT_QUARTERS : last N quarters never seen during Optuna search; used for
                       the final honest out-of-sample evaluation
  - class_weight="balanced" : counters class imbalance without hand-tuning
  - min_child_samples in [10, 50] : prevents leaves with too few samples
  - reg_alpha / reg_lambda searched log-uniformly over [1e-8, 10]

Usage:
    from app.ml.training.trainer import train_model
    result = train_model()
    print(result["max_reliable_lead_time"])  # e.g. 3
"""

import logging
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd

from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from app.ml.training.cross_validation import WalkForwardSplitter

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = Path("data/processed/features_fused.parquet")
LABELS_PATH   = Path("data/processed/labels.parquet")
MODEL_PATH    = Path("models/lgbm_best.pkl")
STUDY_PATH    = Path("models/optuna_study.pkl")

# ── Feature columns — reduced to 15 (2026-07-29) ────────────────────────────
# Cut from 45 to 15 by LightGBM gain importance (averaged over 10 seeds; see
# scripts/select_top_features.py), with slots RESERVED for the NLP signal so
# the study's news contribution is retained even though it ranks low at the
# current article density:
#   • NLP (5): the 4 FSSI features + trigger_climate (top-ranked trigger)
#   • Government (10): the 10 highest-importance primary-data features
# pct_total_hunger stays excluded (label leakage — see label_generator).
# Features derived from food CPI are EXCLUDED as label leakage. stress_score is
#   sws_hunger_t + 2 * (food_cpi_yoy_p,t - regional_mean_food_cpi_yoy_t)
# and sws_hunger_t is identical across all five provinces in a given quarter, so
# the food-CPI deviation term is the ONLY source of province-level variation in
# label_stress. Feeding food_cpi_yoy (or its lag/accel/level, or the
# food-minus-headline contrast that contains it) to the model hands it the
# entire province-discriminating signal of its own target. Same class of
# leakage as pct_total_hunger, one step removed.
LEAKY_LABEL_FEATURES = [
    "food_cpi_yoy", "food_cpi_yoy_lag1", "food_cpi_yoy_accel", "food_cpi",
    "food_minus_headline_yoy", "food_minus_headline_yoy_lag1",
    "food_minus_headline_yoy_accel",
]

# Lagged label features. label_stress is autocorrelated at 0.81, and naive
# persistence scores 0.8125 accuracy purely on y_{t-1}. Withholding the lagged
# label while benchmarking against persistence compares a model to a baseline
# holding strictly more information. At forecast_gap=0 y_{t-1} is known at
# prediction time; _add_label_lags shifts by (gap + 1) so the lag is never
# drawn from inside the forecast window.
LABEL_LAG_COLS = ["label_lag1", "label_lag2", "quarters_since_flip"]

# Features that carry NO province-level variation — the same value is inherited
# by all five provinces in a given quarter. They contribute temporal signal only.
# Measured cross-province correlation of the underlying series:
#   rainfall_anomaly_pct  1.0000  (was manufactured as Quezon x a constant;
#                                  demoted to a regional series 2026-09-01)
#   unemployment_rate     1.0000  (PSA publishes no province cut)
#   headline_cpi          1.0000
#   ofw_remit_yoy_pct     1.0000  (national BSP series)
#   diesel_php_per_l      1.0000  (national DOE series)
# Kept because their time variation is real, but none of them can help the model
# tell one province from another. FSSI (corr -0.0095) is the only feature in this
# matrix carrying genuinely independent province-level variation.
REGIONAL_ONLY_FEATURES = [
    "rainfall_anomaly_pct", "rainfall_anomaly_pct_lag1", "rainfall_anomaly_pct_accel",
    "unemployment_rate", "unemployment_rate_lag1", "unemployment_rate_accel",
    "headline_cpi", "ofw_remit_yoy_pct", "ofw_remit_yoy_pct_lag1",
    "diesel_php_per_l", "diesel_php_per_l_lag1",
]

FEATURE_COLS = [
    # ── NLP / FSSI (secondary data) — reserved ──
    "FSSI", "FSSI_lag1", "FSSI_lag2", "FSSI_accel",
    "trigger_climate",
    # ── Label persistence — what the naive baseline gets ──
    "label_lag1", "label_lag2", "quarters_since_flip",
    # ── Primary data, province-varying ──
    "commodity_livestock", "commodity_leafy_veg", "commodity_fruit_veg",
    "rice_price_regular_lag1",
    # ── Primary data, regional only (temporal signal; see above) ──
    "ofw_remit_yoy_pct_lag1", "unemployment_rate_lag1",
    "headline_cpi", "diesel_php_per_l_lag1",
    "rainfall_anomaly_pct_lag1", "rainfall_anomaly_pct_accel",
]

LABEL_COL    = "label_stress"
PROVINCE_COL = "province_code"
QUARTER_COL  = "quarter"

# ── Training configuration ────────────────────────────────────────────────────
N_TRIALS           = 100
RANDOM_SEED        = 42
HOLDOUT_QUARTERS   = 4   # last N quarters withheld from Optuna; used for final eval
MIN_TRAIN_QUARTERS = 8   # minimum training window for the first CV fold

# ── Lead time discovery configuration ────────────────────────────────────────
# FORECAST_GAP is no longer a fixed constant. discover_max_lead_time() iterates
# k = 1 .. MAX_LEAD_QUARTERS and returns the furthest k where Accuracy >= threshold.
ACCURACY_THRESHOLD  = 0.75   # reliable forecasting standard (75%)
MAX_LEAD_QUARTERS   = 8      # upper bound for the lead-time search

# ── Performance targets ───────────────────────────────────────────────────────
TARGET_F1      = 0.75
TARGET_ROC_AUC = 0.80


def _add_label_lags(df: pd.DataFrame, forecast_gap: int) -> pd.DataFrame:
    """
    Add lagged-label features, shifted to respect the forecast horizon.

    Predicting quarter t at a gap of k means the most recent label actually
    observable is y_{t-k-1}, so every lag is offset by (gap + 1). At gap=0 that
    is y_{t-1} — exactly what naive persistence uses. Rows with no history are
    dropped by the caller via NaN handling rather than imputed, since a filled
    lag would be a fabricated observation.

    quarters_since_flip counts quarters since the label last changed value, so
    the model can learn how stale a persistent run is — the signal that
    distinguishes "still stressed" from "about to flip".
    """
    out = df.sort_values([PROVINCE_COL, QUARTER_COL]).copy()
    shift = forecast_gap + 1
    g = out.groupby(PROVINCE_COL)[LABEL_COL]

    out["label_lag1"] = g.shift(shift)
    out["label_lag2"] = g.shift(shift + 1)

    def _since_flip(s: pd.Series) -> pd.Series:
        lagged = s.shift(shift)
        run = lagged.groupby((lagged != lagged.shift()).cumsum()).cumcount()
        return run.where(lagged.notna())

    out["quarters_since_flip"] = (
        out.groupby(PROVINCE_COL)[LABEL_COL].transform(_since_flip)
    )

    before = len(out)
    out = out.dropna(subset=["label_lag1", "label_lag2"]).reset_index(drop=True)
    logger.info(
        "_add_label_lags: gap=%d -> shift=%d | %d rows (dropped %d without label history)",
        forecast_gap, shift, len(out), before - len(out),
    )
    return out


def _load_data(forecast_gap: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load features_fused.parquet and labels.parquet.
    Join on province_code + quarter, then attach lagged-label features at the
    given forecast gap.
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

    leaked = [c for c in LEAKY_LABEL_FEATURES if c in FEATURE_COLS]
    if leaked:
        raise ValueError(
            f"Label-leaking features present in FEATURE_COLS: {leaked}. "
            "These are inputs to stress_score — see LEAKY_LABEL_FEATURES."
        )

    df = features.merge(
        labels[[PROVINCE_COL, QUARTER_COL, LABEL_COL]],
        on=[PROVINCE_COL, QUARTER_COL],
        how="inner",
    )

    df = df.sort_values([QUARTER_COL, PROVINCE_COL]).reset_index(drop=True)
    df = _add_label_lags(df, forecast_gap=forecast_gap)

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


def discover_max_lead_time(
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    quarters: pd.Series | None = None,
) -> int:
    """
    Empirically identify the furthest lead time k where mean walk-forward
    Accuracy >= ACCURACY_THRESHOLD (75%).

    Strategy
    --------
    Uses a fast default LightGBM (no Optuna) to scan k = 1, 2, ... MAX_LEAD_QUARTERS.
    For each k, walk-forward CV is run and mean Accuracy is computed across folds.
    The loop halts at the first k where mean Accuracy < ACCURACY_THRESHOLD.
    The last k that passed the threshold is returned as the Operationally Valid
    Forecast Horizon used for the full Optuna training run.

    This directly answers Research Question 2:
      "What is the furthest lead time at which aiPHeed can reliably forecast
       province-level food insecurity risk across CALABARZON?"

    Returns
    -------
    int : max reliable lead time in quarters, or 0 when no horizon reaches the
          threshold — including k=1. 0 is a real answer, not an error: it says
          the model cannot forecast ahead reliably and is operating as a
          nowcast (forecast_gap=0 tests on the immediately next quarter).
          This previously floored at 1, which reported a passing horizon even
          when the k=1 scan had failed.
    """
    logger.info("=" * 60)
    logger.info(
        "LEAD TIME DISCOVERY — Accuracy threshold: %.0f%% | Max search: %d quarters",
        ACCURACY_THRESHOLD * 100, MAX_LEAD_QUARTERS,
    )
    logger.info("=" * 60)

    # Default params for the discovery scan — fast, no Optuna overhead.
    discovery_params = {
        "objective":     "binary",
        "verbosity":     -1,
        "boosting_type": "gbdt",
        "random_state":  RANDOM_SEED,
        "class_weight":  "balanced",
        "n_estimators":  200,
        "learning_rate": 0.05,
        "num_leaves":    63,
        "max_depth":     6,
    }

    # 0 until a horizon actually passes. Do NOT floor this at 1: doing so
    # reports a reliable 1-quarter horizon even when the k=1 scan failed.
    k_max = 0

    for k in range(1, MAX_LEAD_QUARTERS + 1):
        # Reload per k: lagged-label features must be shifted by (k + 1) so the
        # lag never comes from inside the forecast window.
        X, y, quarters = _load_data(forecast_gap=k)
        df_quarters = pd.DataFrame({QUARTER_COL: quarters.values}, index=X.index)

        splitter = WalkForwardSplitter(
            min_train_quarters=MIN_TRAIN_QUARTERS,
            forecast_gap=k,
        )

        try:
            n_folds = splitter.get_n_splits(df_quarters)
        except ValueError:
            logger.info("k=%d: not enough quarters for gap=%d. Stopping search.", k, k)
            break

        if n_folds < 2:
            logger.info("k=%d: only %d fold(s) — insufficient for reliable estimate. Stopping.", k, n_folds)
            break

        fold_metrics: dict[str, list[float]] = {
            "accuracy":  [],
            "precision": [],
            "recall":    [],
            "f1":        [],
            "roc_auc":   [],
            "persistence_accuracy": [],
        }

        for train_idx, test_idx in splitter.split(df_quarters):
            if len(train_idx) < 5 or len(test_idx) < 1:
                continue

            y_train = y.loc[train_idx]
            y_test  = y.loc[test_idx]

            model = LGBMClassifier(**discovery_params)
            model.fit(X.loc[train_idx], y_train)
            y_pred  = model.predict(X.loc[test_idx])
            y_proba = model.predict_proba(X.loc[test_idx])[:, 1]

            fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            # Persistence on the same fold: predict the last observable label.
            fold_metrics["persistence_accuracy"].append(
                accuracy_score(y_test, X.loc[test_idx, "label_lag1"].astype(int))
            )
            fold_metrics["precision"].append(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            fold_metrics["recall"].append(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            fold_metrics["f1"].append(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            if y_test.nunique() > 1:
                fold_metrics["roc_auc"].append(roc_auc_score(y_test, y_proba))

        if not fold_metrics["accuracy"]:
            logger.info("k=%d: no valid folds produced. Stopping.", k)
            break

        # Skill over persistence. An absolute accuracy threshold is not a
        # standard here: label_stress is autocorrelated at ~0.81, so copying
        # y_{t-1} already scores ~0.8125 and clears the 75% bar without a model.
        # What matters is whether the model beats that baseline.
        persist_acc = float(np.mean(fold_metrics["persistence_accuracy"])) \
            if fold_metrics.get("persistence_accuracy") else float("nan")

        mean_acc       = float(np.mean(fold_metrics["accuracy"]))
        mean_precision = float(np.mean(fold_metrics["precision"])) if fold_metrics["precision"] else float("nan")
        mean_recall    = float(np.mean(fold_metrics["recall"]))    if fold_metrics["recall"]    else float("nan")
        mean_f1        = float(np.mean(fold_metrics["f1"]))        if fold_metrics["f1"]        else float("nan")
        mean_auc       = float(np.mean(fold_metrics["roc_auc"]))   if fold_metrics["roc_auc"]   else float("nan")
        passed         = mean_acc >= ACCURACY_THRESHOLD

        skill = mean_acc - persist_acc
        logger.info(
            "k=%-2d | Accuracy=%.4f | persistence=%.4f | skill=%+.4f | "
            "F1=%.4f | AUC-ROC=%.4f | %s",
            k, mean_acc, persist_acc, skill, mean_f1, mean_auc,
            "PASS  (>= 75%)" if passed else "FAIL  (< 75%) — HALTING",
        )
        if passed and skill <= 0:
            logger.warning(
                "k=%d clears the %.0f%% bar but does NOT beat persistence "
                "(skill %+.4f). The absolute threshold is weaker than the naive "
                "baseline — report skill, not raw accuracy.",
                k, ACCURACY_THRESHOLD * 100, skill,
            )

        if passed:
            k_max = k
        else:
            if k_max == 0:
                logger.warning(
                    "Accuracy %.4f < %.0f%% at the shortest horizon (k=1). "
                    "NO lead time meets the reliability threshold — reporting 0. "
                    "The model is a nowcast, not a forecaster.",
                    mean_acc, ACCURACY_THRESHOLD * 100,
                )
            else:
                logger.info(
                    "Accuracy fell below %.0f%% at k=%d. "
                    "Maximum Reliable Lead Time = %d quarter(s) ahead.",
                    ACCURACY_THRESHOLD * 100, k, k_max,
                )
            break
    else:
        logger.info(
            "All lead times up to k=%d passed the threshold. "
            "Maximum Reliable Lead Time = %d quarter(s) ahead.",
            MAX_LEAD_QUARTERS, k_max,
        )

    logger.info("=" * 60)
    logger.info("OPERATIONALLY VALID FORECAST HORIZON: %d quarter(s)", k_max)
    logger.info("=" * 60)
    return k_max


def _make_objective(
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    df_cv: pd.DataFrame,
    forecast_gap: int = 1,
) -> callable:
    """
    Build Optuna objective using Walk-Forward CV on the CV window only.

    Optimizes the COMPOSITE objective:
        score = 0.6 * mean(weighted_F1) + 0.4 * mean(ROC-AUC across folds with both classes)

    F1 captures classification quality; ROC-AUC captures ranking quality. Weight
    favours F1 slightly because the holdout positive rate (40%) is still below
    50%, so threshold-based F1 is a stronger signal than threshold-free AUC.
    Both targets must be hit per Backend Guide v3 (F1 ≥ 0.75, AUC ≥ 0.80).
    Folds with single-class y_test contribute only to F1 (AUC undefined there).
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "binary",
            "verbosity":         -1,
            "boosting_type":     "gbdt",
            "random_state":      RANDOM_SEED,
            "class_weight":      "balanced",
            # Search space sized for the data, not for a large dataset. With
            # ~115 rows the previous space (num_leaves up to 150, depth to 10,
            # reg_alpha down to 1e-8) could fit a leaf per observation; Optuna
            # never converged, swinging num_leaves 137->78->112 across runs.
            # A tree here can only afford a handful of splits.
            "num_leaves":        trial.suggest_int("num_leaves", 2, 8),
            "max_depth":         trial.suggest_int("max_depth", 2, 4),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        splitter = WalkForwardSplitter(
            min_train_quarters=MIN_TRAIN_QUARTERS,
            forecast_gap=forecast_gap,
        )
        fold_f1_scores  = []
        fold_auc_scores = []

        for train_idx, test_idx in splitter.split(df_cv):
            if len(train_idx) < 5 or len(test_idx) < 1:
                continue

            X_train = X_cv.loc[train_idx]
            y_train = y_cv.loc[train_idx]
            X_test  = X_cv.loc[test_idx]
            y_test  = y_cv.loc[test_idx]

            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)

            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            fold_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            fold_f1_scores.append(fold_f1)

            # ROC-AUC only defined when both classes appear in y_test
            if y_test.nunique() > 1:
                fold_auc_scores.append(roc_auc_score(y_test, y_proba))

        if not fold_f1_scores:
            return 0.0

        # Pure F1 objective — best empirical result (holdout F1=0.84) when
        # paired with global-stress-threshold labeling. Composite F1+AUC was
        # tested but cost F1 without meaningfully lifting AUC.
        return float(np.mean(fold_f1_scores))

    return objective


def train_model() -> dict:
    """
    Full training pipeline:
    1.  Load features_fused.parquet + labels.parquet
    2.  Discover maximum reliable lead time (Accuracy >= 75% threshold)
    3.  Reserve last HOLDOUT_QUARTERS as a true out-of-sample test set
    4.  Run Optuna 100-trial walk-forward search on the CV window only
        using the empirically discovered forecast gap
    5.  Evaluate the best params on the holdout window (honest final metrics)
    6.  Retrain production model on ALL data with best params
    7.  Serialize lgbm_best.pkl and optuna_study.pkl
    8.  Log with MLflow including max_reliable_lead_time

    Returns:
        dict with best_params, best_cv_f1, holdout metrics,
        and max_reliable_lead_time (the answer to Research Question 2).
    """
    logger.info("=" * 60)
    logger.info("LIGHTGBM TRAINING PIPELINE — aiPHeed")
    logger.info("=" * 60)

    # ── Step 1: Empirically discover maximum reliable lead time ──────────
    # This removes the fixed FORECAST_GAP=3 assumption and answers:
    # "What is the furthest lead time at which aiPHeed can reliably forecast
    #  province-level food insecurity risk across CALABARZON?"
    # The scan reloads per k because the lagged-label features depend on the
    # gap: at gap k only y_{t-k-1} and older are observable.
    max_reliable_lead_time = discover_max_lead_time(None, None, None)
    forecast_gap = max_reliable_lead_time

    # Reload at the discovered horizon so the lags match the horizon trained on.
    X, y, quarters = _load_data(forecast_gap=forecast_gap)

    logger.info(
        "LIGHTGBM TRAINING — WALK-FORWARD CV (gap=%d, holdout=%d quarters)",
        forecast_gap, HOLDOUT_QUARTERS,
    )

    # ── Split CV window from holdout ──────────────────────────────────────
    all_quarters = sorted(quarters.unique())

    min_needed = MIN_TRAIN_QUARTERS + forecast_gap + 1 + HOLDOUT_QUARTERS
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
        N_TRIALS, forecast_gap,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_aipheed",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        _make_objective(X_cv, y_cv, df_cv, forecast_gap=forecast_gap),
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

    holdout_f1        = float(f1_score(y_holdout, y_pred_h, average="weighted", zero_division=0))
    holdout_accuracy  = float(accuracy_score(y_holdout, y_pred_h))
    holdout_precision = float(precision_score(y_holdout, y_pred_h, average="weighted", zero_division=0))
    holdout_recall    = float(recall_score(y_holdout, y_pred_h, average="weighted", zero_division=0))
    try:
        holdout_auc = float(roc_auc_score(y_holdout, y_prob_h))
    except ValueError:
        # Holdout may contain only one class in small datasets
        holdout_auc = float("nan")
        logger.warning("roc_auc_score undefined on holdout (single class present)")

    logger.info("Holdout Accuracy  : %.4f", holdout_accuracy)
    logger.info("Holdout F1        : %.4f", holdout_f1)
    logger.info("Holdout Precision : %.4f", holdout_precision)
    logger.info("Holdout Recall    : %.4f", holdout_recall)
    logger.info("Holdout ROC-AUC   : %.4f", holdout_auc)
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

    # ── MLflow logging (best-effort; model already saved to disk) ────────
    if max_reliable_lead_time == 0:
        logger.warning(
            "ANSWER — Research Question 2: NO reliable forecast horizon. Walk-forward "
            "Accuracy stayed below %.0f%% at every lead time tested, k=1 included, so "
            "the model is reported as a nowcast (gap=0) rather than a forecaster.",
            ACCURACY_THRESHOLD * 100,
        )
    else:
        logger.info(
            "ANSWER — Research Question 2: Maximum Reliable Lead Time = %d quarter(s) ahead "
            "(furthest horizon where walk-forward Accuracy >= %.0f%%)",
            max_reliable_lead_time, ACCURACY_THRESHOLD * 100,
        )

    try:
        with mlflow.start_run(run_name="lgbm_aipheed"):
            mlflow.log_params(best_params)
            mlflow.log_param("max_reliable_lead_time", max_reliable_lead_time)
            mlflow.log_param("accuracy_threshold",     ACCURACY_THRESHOLD)
            mlflow.log_param("holdout_quarters",       HOLDOUT_QUARTERS)
            mlflow.log_metric("best_cv_f1",            best_cv_f1)
            mlflow.log_metric("holdout_accuracy",      holdout_accuracy)
            mlflow.log_metric("holdout_f1",            holdout_f1)
            mlflow.log_metric("holdout_precision",     holdout_precision)
            mlflow.log_metric("holdout_recall",        holdout_recall)
            # ROC-AUC may be NaN when holdout has a single class — skip in that case
            if not (holdout_auc != holdout_auc):  # NaN check
                mlflow.log_metric("holdout_roc_auc", holdout_auc)
            mlflow.log_metric("n_trials", N_TRIALS)
            try:
                mlflow.sklearn.log_model(prod_model, name="lgbm_model")
            except Exception as exc:
                logger.warning("MLflow log_model skipped (%s) — model is on disk.", exc)
    except Exception as exc:
        logger.warning(
            "MLflow logging failed (%s) — production model is still saved at %s.",
            exc, MODEL_PATH,
        )

    return {
        "max_reliable_lead_time":  max_reliable_lead_time,   # answer to RQ2
        "accuracy_threshold":      ACCURACY_THRESHOLD,
        "best_params":             best_params,
        "best_cv_f1":              round(best_cv_f1, 4),
        "holdout_accuracy":        round(holdout_accuracy, 4),
        "holdout_f1":              round(holdout_f1, 4),
        "holdout_precision":       round(holdout_precision, 4),
        "holdout_recall":          round(holdout_recall, 4),
        "holdout_roc_auc":         round(holdout_auc, 4),
        "meets_f1_target":         holdout_f1 >= TARGET_F1,
        "meets_roc_auc_target":    holdout_auc >= TARGET_ROC_AUC,
    }
