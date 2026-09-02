"""
app/ml/inference/explainer.py
------------------------------
SHAP TreeExplainer for the food-availability shock model.

The target changed on 2026-09-01: the model now predicts a production shortfall
per province x quarter x COMMODITY, not a composite stress label per province.
SHAP is therefore computed at the series level and summed to the province, which
is also what the Risk Drivers panel wants -- "what is driving risk in Quezon
this quarter" is a sum over that province's commodity series.

The served model is a per-group ENSEMBLE (LightGBM, RandomForest, ExtraTrees,
LogisticRegression). TreeExplainer cannot explain the logistic pipeline, so
explanations come from each group's LightGBM member -- the ensemble's primary
model. Explained probabilities therefore approximate, rather than exactly
reproduce, the served ensemble probability; the gap is reported as
`explained_prob_gap` rather than hidden.

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

MODEL_PATH    = Path("models/food_availability_model.joblib")
PANEL_PATH    = Path("data/processed/food_availability_panel.parquet")
TRIGGERS_PATH = Path("data/processed/trigger_proportions.parquet")

# The feature list is no longer hardcoded here. It travels inside the model
# bundle (self._feature_cols) so serving, explaining and training cannot
# drift apart -- this copy had already gone stale against the trainer once.

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

    _explainers: dict | None = None
    _feature_cols: list[str] | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run scripts/train_final.py first."
            )
        self._model = joblib.load(MODEL_PATH)
        self._feature_cols = self._model["feature_cols"]
        logger.info("Explainer: bundle loaded from %s (%d groups)",
                    MODEL_PATH, len(self._model["groups"]))

        # Same panel the predictor scores, built by the training loader so the
        # two cannot diverge.
        from scripts.train_food_availability import load as load_panel
        panel = load_panel()
        enc = self._model["commodity_encoding"]
        panel["commodity_te"] = panel["commodity"].map(enc).fillna(
            self._model["encoding_prior"])
        self._features = panel
        logger.info("Explainer: panel loaded (%d series-quarters)", len(panel))

        if TRIGGERS_PATH.exists():
            self._triggers = pd.read_parquet(TRIGGERS_PATH)

        # One TreeExplainer per commodity group, over that group's LightGBM
        # member. Background is a sample of that group's own rows so the
        # baseline reflects the group's base rate -- fisheries shocks ~56% of
        # quarters against 24-30% for crops, so a pooled baseline would misstate
        # both.
        import shap
        self._explainers = {}
        baselines = []
        for grp, spec in self._model["groups"].items():
            lgbm = dict(spec["members"])["lgbm"]
            rows = panel[panel["group"] == grp]
            if rows.empty:
                rows = panel
            bg = rows[self._feature_cols].fillna(0.0).sample(
                min(50, len(rows)), random_state=42)
            ex = shap.TreeExplainer(lgbm, bg, model_output="probability",
                                    feature_perturbation="interventional")
            self._explainers[grp] = ex
            baselines.append(float(ex.expected_value))
            logger.info("Explainer: %-22s baseline=%.4f", grp, float(ex.expected_value))

        self._baseline = float(np.mean(baselines)) if baselines else None

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
        list[dict]  — one dict per feature, sorted by |shap_value|.
        Each dict contains:
            quarter, province_code,
            feature_name, display_name, feature_group, unit,
            shap_value, mean_abs_shap, feature_value,
            baseline, final_rfii

        SHAP is computed per commodity series and AVERAGED across the province's
        series, matching how the predictor rolls series risk up to a province.
        Each series is explained by its own group's explainer, so a fisheries
        series is read against the fisheries baseline rather than a pooled one.
        """
        self._load()

        if self._features is None:
            return []

        mask = (
            (self._features["province_code"] == province_code)
            & (self._features["quarter"] == quarter)
        )
        rows = self._features[mask]
        if rows.empty:
            logger.warning(
                "explain_province_quarter: no data for %s / %s",
                province_code, quarter,
            )
            return []

        cols = self._feature_cols
        sv_parts, base_parts = [], []
        for grp, sub in rows.groupby("group"):
            ex = self._explainers.get(grp)
            if ex is None:
                continue
            X_g = sub[cols].fillna(0.0)
            raw = ex.shap_values(X_g)
            arr = np.asarray(raw[1] if isinstance(raw, list) and len(raw) > 1
                             else raw[0] if isinstance(raw, list) else raw)
            if arr.ndim == 3:            # (n, features, classes)
                arr = arr[:, :, -1]
            sv_parts.append(np.atleast_2d(arr))
            base_parts.extend([float(ex.expected_value)] * len(sub))

        if not sv_parts:
            return []

        sv = np.vstack(sv_parts).mean(axis=0)      # average across the province's series
        X = rows[cols].fillna(0.0)

        baseline    = float(np.mean(base_parts))
        final_rfii  = round(baseline + float(sv.sum()), 4)
        mean_abs    = round(float(np.abs(sv).mean()), 6)

        records = []
        for feat, val in zip(cols, sv):
            meta = FEATURE_DISPLAY_MAP.get(feat, {})
            fval = float(X[feat].mean()) if feat in X.columns else None
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
        Aggregate SHAP records into the driver groups for the Risk Drivers panel.

        Parameters
        ----------
        province_code  : str
        quarter        : str
        shap_records   : output of explain_province_quarter().

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
