"""
app/ml/inference/predictor.py
------------------------------
Province-level food insecurity forecast predictor.

Loads the trained LightGBM model (models/lgbm_best.pkl) and produces
province-level forecast for a given quarter's feature row.

Usage:
    from app.ml.inference.predictor import Predictor
    predictor = Predictor()
    forecasts = predictor.forecast_quarter("2026-Q1")
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH    = Path("models/lgbm_best.pkl")
FEATURES_PATH = Path("data/processed/features_fused.parquet")
BIAS_WEIGHTS_PATH = Path("data/processed/bias_weights.parquet")

PROVINCE_NAMES: dict[str, str] = {
    "PH040100000": "Cavite",
    "PH040200000": "Laguna",
    "PH040300000": "Quezon",
    "PH040400000": "Rizal",
    "PH040500000": "Batangas",
}

# Must match trainer.py FEATURE_COLS exactly
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
    "food_cpi_yoy_lag1",         "food_cpi_yoy_accel",
    "food_minus_headline_yoy_lag1", "food_minus_headline_yoy_accel",
    "unemployment_rate_lag1",    "unemployment_rate_accel",
    "ofw_remit_yoy_pct_lag1",    "ofw_remit_yoy_pct_accel",
    "rainfall_anomaly_pct_lag1", "rainfall_anomaly_pct_accel",
    "rice_price_regular_lag1",   "rice_price_regular_accel",
    "diesel_php_per_l_lag1",     "diesel_php_per_l_accel",
]

DATA_SUFFICIENCY_MIN_ARTICLES = 5

# ---------------------------------------------------------------------------
# Alert thresholds — sudden-rise detection
# ---------------------------------------------------------------------------
# An alert is staged when BOTH conditions are true:
#   1. risk_probability rose by >= ALERT_DELTA_THRESHOLD from the previous quarter
#   2. current risk_probability >= ALERT_FLOOR
#
# Rationale for 0.15 delta:
#   A 15-point quarter-on-quarter jump represents a fast deterioration in the
#   model's predicted risk — the signal that DSWD needs to act on early.
#   Flat or slowly rising probabilities are informational, not urgent.
#
# Rationale for 0.35 floor:
#   Prevents noise alerts from very-low-risk provinces (e.g. 0.05 → 0.21 is
#   a big jump but still a low absolute risk).  0.35 is below the historical
#   median risk (≈0.54 baseline) but above the noise floor.
ALERT_DELTA_THRESHOLD: float = 0.15   # minimum quarter-on-quarter rise
ALERT_FLOOR: float = 0.35             # minimum current probability for alert to fire


class Predictor:
    """
    Singleton predictor: loads model once, forecasts on demand.

    Thread-safe for FastAPI concurrent requests (read-only after init).
    """

    _instance: Predictor | None = None
    _model = None
    _features: pd.DataFrame | None = None
    _bias_weights: pd.DataFrame | None = None

    def __new__(cls) -> Predictor:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        """Load model and feature matrix into memory (idempotent)."""
        if self._model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Trained model not found at {MODEL_PATH}. "
                    "Run scripts/run_w12_training.py first."
                )
            self._model = joblib.load(MODEL_PATH)
            logger.info("Predictor: LightGBM model loaded from %s", MODEL_PATH)

        if self._features is None and FEATURES_PATH.exists():
            self._features = pd.read_parquet(FEATURES_PATH)
            logger.info(
                "Predictor: feature matrix loaded (%d rows)", len(self._features)
            )

        if self._bias_weights is None and BIAS_WEIGHTS_PATH.exists():
            self._bias_weights = pd.read_parquet(BIAS_WEIGHTS_PATH)

    def forecast_quarter(self, quarter: str) -> list[dict]:
        """
        Produce province-level forecasts for a specific quarter.

        Parameters
        ----------
        quarter : str  e.g. "2026-Q1"

        Returns
        -------
        list[dict] — one dict per province (5 CALABARZON provinces), sorted by
                     risk_probability descending. Each dict has:
            province_code, province_name, quarter,
            risk_probability, risk_label, data_sufficiency_flag
        """
        self.load()

        if self._features is None:
            raise RuntimeError("Feature matrix not loaded.")

        df = self._features[self._features["quarter"] == quarter].copy()
        if df.empty:
            logger.warning(
                "forecast_quarter: no feature rows for quarter=%s. "
                "Returning zero-probability forecasts.",
                quarter,
            )
            return self._zero_forecasts(quarter)

        # Ensure all feature columns exist (fill missing with 0)
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

        X = df[FEATURE_COLS]
        proba = self._model.predict_proba(X)[:, 1]

        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            province_code = row["province_code"]
            prob = float(proba[i])
            # Display label only — alerts use detect_alerts() delta logic, not this
            risk_label = "HIGH" if prob >= 0.5 else "LOW"

            # Check data sufficiency
            data_flag = self._get_data_sufficiency(province_code, quarter)

            results.append({
                "province_code":        province_code,
                "province_name":        PROVINCE_NAMES.get(province_code, province_code),
                "quarter":              quarter,
                "risk_probability":     round(prob, 4),
                "risk_label":           risk_label,
                "data_sufficiency_flag": data_flag,
            })

        # Sort by risk probability descending
        results.sort(key=lambda x: x["risk_probability"], reverse=True)
        logger.info(
            "forecast_quarter(%s): %d provinces forecast", quarter, len(results)
        )
        return results

    def detect_alerts(
        self,
        current_forecasts: list[dict],
        prev_forecasts: list[dict],
    ) -> list[dict]:
        """
        Compare current quarter forecasts against the previous quarter and
        return alert dicts for provinces with a sudden rise.

        A sudden rise is defined as:
            risk_delta >= ALERT_DELTA_THRESHOLD  (>= 0.15 points)
            AND current risk_probability >= ALERT_FLOOR  (>= 0.35)

        Parameters
        ----------
        current_forecasts : list[dict]  — output of forecast_quarter() for this quarter
        prev_forecasts    : list[dict]  — output of forecast_quarter() for last quarter
                                         (or DB records cast to dicts)

        Returns
        -------
        list[dict] — one dict per triggered province, ready for AlertRepository.insert()
            keys: quarter, province_code, threshold_exceeded,
                  prev_risk_probability, risk_delta, alert_reason
        """
        prev_map = {f["province_code"]: f.get("risk_probability", 0.0) for f in prev_forecasts}

        alerts = []
        for fc in current_forecasts:
            province_code = fc["province_code"]
            current_prob  = fc["risk_probability"]
            prev_prob     = prev_map.get(province_code, 0.0)
            delta         = round(current_prob - prev_prob, 4)

            if delta >= ALERT_DELTA_THRESHOLD and current_prob >= ALERT_FLOOR:
                alerts.append({
                    "quarter":               fc["quarter"],
                    "province_code":         province_code,
                    "threshold_exceeded":    True,
                    "prev_risk_probability": round(prev_prob, 4),
                    "risk_delta":            delta,
                    "alert_reason":          "SUDDEN_RISE",
                })
                logger.info(
                    "ALERT staged — %s | %s | prob %.3f → %.3f (Δ=%.3f)",
                    fc["quarter"], fc.get("province_name", province_code),
                    prev_prob, current_prob, delta,
                )

        return alerts

    def forecast_all_quarters(self) -> list[dict]:
        """Forecast all quarters present in the feature matrix."""
        self.load()
        if self._features is None:
            return []
        quarters = sorted(self._features["quarter"].unique())
        all_results = []
        for q in quarters:
            all_results.extend(self.forecast_quarter(q))
        return all_results

    def _get_data_sufficiency(self, province_code: str, quarter: str) -> str | None:
        """Return 'LIMITED_SIGNAL' if below article threshold, else None."""
        if self._bias_weights is None:
            return None
        mask = (
            (self._bias_weights["province_code"] == province_code)
            & (self._bias_weights["quarter"] == quarter)
        )
        row = self._bias_weights[mask]
        if row.empty:
            return "LIMITED_SIGNAL"
        article_count = row.iloc[0].get("article_count", 0)
        if article_count < DATA_SUFFICIENCY_MIN_ARTICLES:
            return "LIMITED_SIGNAL"
        return None

    def _zero_forecasts(self, quarter: str) -> list[dict]:
        """Return zero-probability forecasts for all 5 provinces."""
        return [
            {
                "province_code":         code,
                "province_name":         name,
                "quarter":               quarter,
                "risk_probability":      0.0,
                "risk_label":            "LOW",
                "data_sufficiency_flag": "LIMITED_SIGNAL",
            }
            for code, name in PROVINCE_NAMES.items()
        ]
