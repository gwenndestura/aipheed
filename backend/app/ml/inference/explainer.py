"""
app/ml/inference/explainer.py
------------------------------
SHAP TreeExplainer for LightGBM province-level forecasts.

Computes per-feature SHAP values for a specific province-quarter,
enabling transparent feature importance for DSWD decision-makers.

Usage:
    from app.ml.inference.explainer import Explainer
    explainer = Explainer()
    shap_records = explainer.explain_quarter("2026-Q1")
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH    = Path("models/lgbm_best.pkl")
FEATURES_PATH = Path("data/processed/features_fused.parquet")

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

PROVINCE_NAMES: dict[str, str] = {
    "PH040100000": "Cavite",
    "PH040200000": "Laguna",
    "PH040300000": "Quezon",
    "PH040400000": "Rizal",
    "PH040500000": "Batangas",
}


class Explainer:
    """
    Lazy-loading SHAP TreeExplainer singleton.
    """

    _model = None
    _explainer = None
    _features: pd.DataFrame | None = None

    def _load(self):
        if self._model is not None:
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run training first."
            )
        self._model = joblib.load(MODEL_PATH)

        import shap
        self._explainer = shap.TreeExplainer(self._model)
        logger.info("Explainer: SHAP TreeExplainer loaded.")

        if FEATURES_PATH.exists():
            self._features = pd.read_parquet(FEATURES_PATH)

    def explain_province_quarter(
        self,
        province_code: str,
        quarter: str,
    ) -> list[dict]:
        """
        Compute SHAP values for one province-quarter.

        Returns
        -------
        list[dict] — sorted by |shap_value| descending, each dict has:
            quarter, province_code, feature_name, shap_value, mean_abs_shap
        """
        self._load()

        if self._features is None:
            return []

        mask = (
            (self._features["province_code"] == province_code)
            & (self._features["quarter"] == quarter)
        )
        row = self._features[mask]
        if row.empty:
            logger.warning(
                "explain_province_quarter: no data for %s / %s", province_code, quarter
            )
            return []

        for col in FEATURE_COLS:
            if col not in row.columns:
                row = row.copy()
                row[col] = 0.0
        X = row[FEATURE_COLS].fillna(0.0)

        shap_values = self._explainer.shap_values(X)
        # For binary classification, shap_values may be a list [neg_class, pos_class]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]   # positive class, first (only) row
        else:
            sv = shap_values[0]

        mean_abs = float(np.abs(sv).mean())

        records = []
        for feat, val in zip(FEATURE_COLS, sv):
            records.append({
                "quarter":        quarter,
                "province_code":  province_code,
                "feature_name":   feat,
                "shap_value":     round(float(val), 6),
                "mean_abs_shap":  round(mean_abs, 6),
            })

        records.sort(key=lambda r: abs(r["shap_value"]), reverse=True)
        return records

    def explain_quarter(self, quarter: str) -> list[dict]:
        """Explain all 5 provinces for a given quarter."""
        all_records = []
        for province_code in PROVINCE_NAMES:
            all_records.extend(self.explain_province_quarter(province_code, quarter))
        return all_records
