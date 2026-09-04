"""
app/ml/inference/forecaster.py
-------------------------------
Serves the one-quarter-ahead forecast model.

The difference from Predictor is timing, not method. Predictor serves the
nowcast: it scores a quarter using that quarter's CPI, typhoon count, ENSO
phase and news volume, none of which exist until the quarter is over. It
answers "what happened", and it can never run ahead of the data.

This serves the variant trained by scripts/train_forecast_model.py, where the
government/climate matrix is attached a quarter late (load(feature_lag=1)) and
the two panel-built news counts are shifted. Every input therefore predates the
quarter being scored, so the model can answer "what to watch next".

Two consequences worth knowing:

* It reaches one quarter further than the nowcast. 2026-Q1 is scoreable from
  2025-Q4 features; the nowcast needs 2026-Q1 features that PSA, BSP, DOE and
  PAGASA have not all published.
* It is slightly less accurate and does NOT beat seasonal persistence on binary
  accuracy at full coverage. Its value is the forward RANKING -- which series to
  look at first -- which a persistence rule cannot express at all. Callers
  should surface `roc_auc` and the baselines alongside any score.

Usage:
    from app.ml.inference.forecaster import Forecaster
    f = Forecaster()
    f.available_quarters()          # scoreable quarters, oldest first
    f.forecast_quarter("2026-Q1")   # province-level, one quarter ahead
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/food_availability_forecast_model.joblib")
RESULTS_PATH = Path("data/processed/forecast_results.json")

PROVINCE_NAMES: dict[str, str] = {
    "PH040100000": "Cavite",
    "PH040200000": "Laguna",
    "PH040300000": "Quezon",
    "PH040400000": "Rizal",
    "PH040500000": "Batangas",
}

RISK_CUTOFF = 0.50
TOP_COMMODITIES = 3


class Forecaster:
    """Singleton: the bundle is ~50 MB and the panel takes seconds to build."""

    _instance: Forecaster | None = None

    def __new__(cls) -> Forecaster:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._panel = None
        return cls._instance

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Forecast model not found at {MODEL_PATH}. "
                    "Run scripts/train_forecast_model.py first."
                )
            self._model = joblib.load(MODEL_PATH)
            logger.info(
                "Forecaster: bundle loaded (%s horizon, trained through %s)",
                self._model.get("horizon", "?"), self._model.get("trained_through", "?"),
            )

        if self._panel is None:
            self._panel = self._prepare_panel()
            logger.info("Forecaster: panel prepared (%d series-quarters, %s..%s)",
                        len(self._panel), self._panel["quarter"].min(),
                        self._panel["quarter"].max())

    def _prepare_panel(self) -> pd.DataFrame:
        """
        Rebuild the exact frame training used.

        Delegates to the training loader with the same feature lag, so serving
        cannot drift from training: any change to the panel or the lag applies
        to both.
        """
        from scripts.train_food_availability import load as load_panel
        from scripts.train_forecast_model import (
            CONTEMPORANEOUS, FEATURE_LAG, LAG_SUFFIX,
        )

        df = load_panel(feature_lag=FEATURE_LAG)
        df = df.sort_values(["province_code", "commodity", "quarter"])
        g = df.groupby(["province_code", "commodity"], sort=False)
        for c in CONTEMPORANEOUS:
            if c in df.columns:
                df[f"{c}{LAG_SUFFIX}"] = g[c].shift(1)

        enc = self._model["commodity_encoding"]
        prior = self._model["encoding_prior"]
        df["commodity_te"] = df["commodity"].map(enc).fillna(prior)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
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

    def available_quarters(self) -> list[str]:
        """
        Quarters this model can score, oldest first.

        Reaches one quarter past Predictor.available_quarters(), because the
        feature matrix is attached at t-1.
        """
        self.load()
        if self._panel is None:
            return []
        return sorted(self._panel["quarter"].unique().tolist())

    def forecast_commodities(self, quarter: str) -> list[dict]:
        """Score every province-commodity series for one quarter."""
        self.load()
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
        Province-level forecast for one quarter, ranked by risk descending.

        `risk_probability` matches the nowcast's definition -- the share of a
        province's monitored series flagged at risk -- so the two models'
        numbers are directly comparable.
        """
        self.load()
        rows = self._panel[self._panel["quarter"] == quarter]
        if rows.empty:
            logger.warning("forecast_quarter: no panel rows for quarter=%s", quarter)
            return []

        prob, thr = self._score(rows)
        scored = rows.assign(_p=prob, _at_risk=(prob >= thr))

        out = []
        for code, grp in scored.groupby("province_code"):
            at_risk = grp[grp["_at_risk"]]
            share = round(float(len(at_risk) / len(grp)), 4) if len(grp) else 0.0
            top = (at_risk.nlargest(TOP_COMMODITIES, "_p")["commodity"].tolist()
                   if not at_risk.empty else [])
            out.append({
                "province_code": code,
                "province_name": PROVINCE_NAMES.get(code, code),
                "quarter": quarter,
                "risk_probability": share,
                "risk_label": "HIGH" if share >= RISK_CUTOFF else "LOW",
                "series_monitored": int(len(grp)),
                "series_at_risk": int(len(at_risk)),
                "mean_shock_probability": round(float(grp["_p"].mean()), 4),
                "top_at_risk_commodities": top,
                "horizon": self._model.get("horizon", "one-quarter-ahead"),
            })
        out.sort(key=lambda d: d["risk_probability"], reverse=True)
        return out
