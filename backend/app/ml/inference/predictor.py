"""
app/ml/inference/predictor.py
------------------------------
Province-level food insecurity forecast predictor.

Serves the food-availability shock model: for a given quarter, scores every
province-commodity series and rolls the results up to a province-level risk.

The target changed on 2026-09-01. The former model predicted a composite
SWS + food-CPI stress label at province-quarter resolution; that label was
withdrawn once its inputs were found to be fabricated and its only province-
varying term traced to a leaking feature. The current model predicts a
production shortfall against a series' own seasonal baseline, at province x
quarter x COMMODITY resolution, from PSA OpenStat volumes.

A province therefore no longer has one prediction but many -- one per commodity
series it grows or fishes. `risk_probability` is the share of that province's
series flagged as at risk, which is both interpretable for DSWD ("a third of
Quezon's monitored commodities are at risk this quarter") and usable for the
existing sudden-rise alert logic unchanged.

Loads models/food_availability_model.joblib, written by scripts/train_final.py.
That bundle carries the per-group ensembles, the commodity target-encoding map
and the per-group decision thresholds, so serving cannot drift from training.

Usage:
    from app.ml.inference.predictor import Predictor
    predictor = Predictor()
    forecasts = predictor.forecast_quarter("2026-Q1")          # by province
    detail    = predictor.forecast_commodities("2026-Q1")      # by series
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

MODEL_PATH    = Path("models/food_availability_model.joblib")
PANEL_PATH    = Path("data/processed/food_availability_panel.parquet")
FEATURES_PATH = Path("data/processed/features_fused.parquet")
BIAS_WEIGHTS_PATH = Path("data/processed/bias_weights.parquet")

PROVINCE_NAMES: dict[str, str] = {
    "PH040100000": "Cavite",
    "PH040200000": "Laguna",
    "PH040300000": "Quezon",
    "PH040400000": "Rizal",
    "PH040500000": "Batangas",
}

# The feature list travels inside the bundle rather than being imported, so
# serving cannot drift from training. It previously imported trainer.FEATURE_COLS
# and drifted anyway when the trainer changed underneath it.
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
    _panel: pd.DataFrame | None = None
    _bias_weights: pd.DataFrame | None = None

    def __new__(cls) -> Predictor:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        """Load the model bundle and scored panel into memory (idempotent)."""
        if self._model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Trained model not found at {MODEL_PATH}. "
                    "Run scripts/train_final.py first."
                )
            self._model = joblib.load(MODEL_PATH)
            logger.info(
                "Predictor: bundle loaded from %s (%d groups, trained through %s)",
                MODEL_PATH, len(self._model["groups"]),
                self._model.get("trained_through", "?"),
            )

        if self._panel is None and PANEL_PATH.exists():
            self._panel = self._prepare_panel()
            logger.info("Predictor: panel prepared (%d series-quarters)", len(self._panel))

        if self._bias_weights is None and BIAS_WEIGHTS_PATH.exists():
            self._bias_weights = pd.read_parquet(BIAS_WEIGHTS_PATH)

    def _prepare_panel(self) -> pd.DataFrame:
        """
        Rebuild the exact feature frame training used.

        Delegates to the training loader so the two cannot diverge: it derives
        the same series lags, seasonal terms and commodity-matched news counts,
        then joins the government feature matrix.
        """
        from scripts.train_food_availability import load as load_panel

        df = load_panel()
        enc = self._model["commodity_encoding"]
        prior = self._model["encoding_prior"]
        df["commodity_te"] = df["commodity"].map(enc).fillna(prior)
        return df

    def _score(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble probability and the group's decision threshold for each row."""
        cols = self._model["feature_cols"]
        prob = np.zeros(len(rows))
        thr = np.zeros(len(rows))
        for grp, spec in self._model["groups"].items():
            mask = (rows["group"] == grp).to_numpy()
            if not mask.any():
                continue
            sub = rows.loc[mask, cols]
            member_probs = [
                m.predict_proba(sub.fillna(-999) if name in ("rf", "et") else sub)[:, 1]
                for name, m in spec["members"]
            ]
            prob[mask] = np.mean(member_probs, axis=0)
            thr[mask] = spec["threshold"]
        return prob, thr

    def forecast_commodities(self, quarter: str) -> list[dict]:
        """
        Score every province-commodity series for one quarter.

        This is the model's native resolution; forecast_quarter() rolls it up.
        """
        self.load()
        if self._panel is None:
            raise RuntimeError("Panel not loaded.")

        rows = self._panel[self._panel["quarter"] == quarter]
        if rows.empty:
            logger.warning("forecast_commodities: no rows for quarter=%s", quarter)
            return []

        prob, thr = self._score(rows)
        out = [
            {
                "province_code": r["province_code"],
                "province_name": PROVINCE_NAMES.get(r["province_code"], r["province_code"]),
                "quarter": quarter,
                "commodity_group": r["group"],
                "commodity": r["commodity"],
                "shock_probability": round(float(p), 4),
                "at_risk": bool(p >= t),
            }
            for (_, r), p, t in zip(rows.iterrows(), prob, thr)
        ]
        out.sort(key=lambda d: d["shock_probability"], reverse=True)
        return out

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

        if self._panel is None:
            raise RuntimeError("Panel not loaded.")

        rows = self._panel[self._panel["quarter"] == quarter]
        if rows.empty:
            logger.warning(
                "forecast_quarter: no panel rows for quarter=%s. "
                "Returning zero-probability forecasts.",
                quarter,
            )
            return self._zero_forecasts(quarter)

        prob, thr = self._score(rows)
        scored = rows[["province_code", "group", "commodity"]].copy()
        scored["prob"] = prob
        scored["at_risk"] = prob >= thr

        results = []
        for province_code, d in scored.groupby("province_code"):
            # Province risk = share of that province's monitored series flagged.
            # The model predicts per commodity, so a province has many
            # predictions rather than one; the share is what a DSWD reader can
            # act on ("a third of Quezon's monitored commodities are at risk").
            share = float(d["at_risk"].mean())
            data_flag = self._get_data_sufficiency(province_code, quarter)
            at_risk = d[d["at_risk"]].nlargest(3, "prob")["commodity"].tolist()

            results.append({
                "province_code":        province_code,
                "province_name":        PROVINCE_NAMES.get(province_code, province_code),
                "quarter":              quarter,
                "risk_probability":     round(share, 4),
                "risk_label":           "HIGH" if share >= 0.5 else "LOW",
                "series_monitored":     int(len(d)),
                "series_at_risk":       int(d["at_risk"].sum()),
                "mean_shock_probability": round(float(d["prob"].mean()), 4),
                "top_at_risk_commodities": at_risk,
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
