"""
app/ml/inference/explainer.py
------------------------------
SHAP TreeExplainer for LightGBM province-level forecasts.

Computes per-feature SHAP values (probability space) for a specific
province-quarter, enabling transparent feature importance for DSWD
decision-makers.

Outputs two record types:

1. shap_records  (stored in DB, served as SHAPResponse)
   One row per feature per province-quarter. Includes display_name,
   feature_group, feature_value, unit, baseline, and final_rfii.

2. driver_records  (derived on-the-fly from shap_records, served as DriversResponse)
   One row per driver group per province-quarter.

Usage:
    from app.ml.inference.explainer import Explainer
    explainer = Explainer()
    shap_records   = explainer.explain_quarter("2026-Q1")
    driver_records = explainer.build_drivers("PH040300000", "2026-Q1", shap_records)
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.ml.inference.feature_display import (
    DRIVER_GROUP_FEATURES,
    DRIVER_GROUP_LABELS,
    FEATURE_DISPLAY_MAP,
)

logger = logging.getLogger(__name__)

MODEL_PATH    = Path("models/lgbm_best.pkl")
FEATURES_PATH = Path("data/processed/features_fused.parquet")
TRIGGERS_PATH = Path("data/processed/trigger_proportions.parquet")

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

    Uses interventional perturbation with model_output='probability' so that
    SHAP values are directly interpretable as probability contributions:
        baseline + sum(shap_values) ≈ predicted risk_probability

    baseline ≈ 0.54  (mean training probability on the balanced 50/50 dataset).
    """

    _model = None
    _explainer = None
    _features: pd.DataFrame | None = None
    _triggers: pd.DataFrame | None = None
    _baseline: float | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run training first."
            )
        self._model = joblib.load(MODEL_PATH)
        logger.info("Explainer: LightGBM model loaded from %s", MODEL_PATH)

        if FEATURES_PATH.exists():
            self._features = pd.read_parquet(FEATURES_PATH)
            logger.info(
                "Explainer: feature matrix loaded (%d rows)", len(self._features)
            )

        if TRIGGERS_PATH.exists():
            self._triggers = pd.read_parquet(TRIGGERS_PATH)

        # Build interventional SHAP explainer with probability output.
        # Background dataset: 50-row sample of training features.
        import shap
        if self._features is not None:
            bg = self._features[FEATURE_COLS].fillna(0.0).sample(
                min(50, len(self._features)), random_state=42
            )
        else:
            bg = None

        self._explainer = shap.TreeExplainer(
            self._model,
            bg,
            model_output="probability",
            feature_perturbation="interventional",
        )
        self._baseline = float(self._explainer.expected_value)
        logger.info(
            "Explainer: SHAP explainer ready | baseline=%.4f", self._baseline
        )

    # ── Province-quarter SHAP ─────────────────────────────────────────────

    def explain_province_quarter(
        self,
        province_code: str,
        quarter: str,
    ) -> list[dict]:
        """
        Compute SHAP values for one province-quarter.

        Returns
        -------
        list[dict]  — one dict per feature (all 45), sorted by |shap_value|.
        Each dict contains:
            quarter, province_code,
            feature_name, display_name, feature_group, unit,
            shap_value, mean_abs_shap, feature_value,
            baseline, final_rfii
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
                "explain_province_quarter: no data for %s / %s",
                province_code, quarter,
            )
            return []

        for col in FEATURE_COLS:
            if col not in row.columns:
                row = row.copy()
                row[col] = 0.0
        X = row[FEATURE_COLS].fillna(0.0)

        sv_raw = self._explainer.shap_values(X)
        sv = sv_raw[0] if isinstance(sv_raw, list) else sv_raw[0]
        # sv shape: (1, n_features) → flatten to (n_features,)
        if sv.ndim == 2:
            sv = sv[0]

        baseline    = self._baseline or 0.0
        final_rfii  = round(baseline + float(sv.sum()), 4)
        mean_abs    = round(float(np.abs(sv).mean()), 6)

        records = []
        for feat, val in zip(FEATURE_COLS, sv):
            meta = FEATURE_DISPLAY_MAP.get(feat, {})
            fval = X[feat].iloc[0] if feat in X.columns else None
            records.append({
                "quarter":         quarter,
                "province_code":   province_code,
                "feature_name":    feat,
                "display_name":    meta.get("display_name", feat),
                "feature_group":   meta.get("feature_group", "other"),
                "unit":            meta.get("unit"),
                "shap_value":      round(float(val), 6),
                "mean_abs_shap":   mean_abs,
                "feature_value":   round(float(fval), 4) if fval is not None else None,
                "baseline":        round(baseline, 4),
                "final_rfii":      final_rfii,
            })

        records.sort(key=lambda r: abs(r["shap_value"]), reverse=True)
        return records

    def explain_quarter(self, quarter: str) -> list[dict]:
        """Explain all 5 provinces for a given quarter."""
        all_records = []
        for province_code in PROVINCE_NAMES:
            all_records.extend(
                self.explain_province_quarter(province_code, quarter)
            )
        return all_records

    # ── Driver group aggregation ──────────────────────────────────────────

    def build_drivers(
        self,
        province_code: str,
        quarter: str,
        shap_records: list[dict],
    ) -> dict:
        """
        Aggregate SHAP records into 5 driver groups for the Risk Drivers panel.

        Parameters
        ----------
        province_code  : str
        quarter        : str
        shap_records   : output of explain_province_quarter() — all 45 features.

        Returns
        -------
        dict  matching DriversResponse schema:
            province_code, quarter,
            drivers: list[DriverRecord dicts],
            article_count: int
        """
        self._load()

        sv_dict = {r["feature_name"]: r["shap_value"] for r in shap_records}

        # ── Group SHAP ────────────────────────────────────────────────────
        group_shap: dict[str, float] = {}
        for group, feats in DRIVER_GROUP_FEATURES.items():
            group_shap[group] = sum(sv_dict.get(f, 0.0) for f in feats)

        # ── Trigger proportions from NLP corpus ───────────────────────────
        trigger_map: dict[str, float] = {}
        article_count = 0
        if self._triggers is not None:
            t_row = self._triggers[
                (self._triggers["province_code"] == province_code)
                & (self._triggers["quarter"] == quarter)
            ]
            if not t_row.empty:
                r = t_row.iloc[0]
                article_count = int(r.get("article_count", 0))
                trigger_map = {
                    "market":        float(r.get("trigger_market", 0.0)),
                    "climate":       float(r.get("trigger_climate", 0.0)),
                    "employment":    float(r.get("trigger_employment", 0.0)),
                    "macro_ofw":     float(r.get("trigger_ofw_remittance", 0.0)),
                    "nlp_sentiment": float(r.get("trigger_fish_kill", 0.0)),
                }

        # ── Build driver records ───────────────────────────────────────────
        total_abs = sum(abs(v) for v in group_shap.values()) or 1.0

        drivers = []
        for group in DRIVER_GROUP_FEATURES:
            gs = group_shap[group]
            drivers.append({
                "driver_group":       group,
                "driver_label":       DRIVER_GROUP_LABELS[group],
                "group_shap":         round(gs, 4),
                "direction":          "increases_risk" if gs >= 0 else "protective",
                "display_pct":        round(abs(gs) / total_abs * 100, 1),
                "trigger_proportion": trigger_map.get(group),
            })

        # Sort by abs(group_shap) descending
        drivers.sort(key=lambda d: abs(d["group_shap"]), reverse=True)

        return {
            "province_code": province_code,
            "quarter":       quarter,
            "drivers":       drivers,
            "article_count": article_count,
        }
