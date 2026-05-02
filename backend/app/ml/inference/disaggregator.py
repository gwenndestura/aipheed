"""
app/ml/inference/disaggregator.py
-----------------------------------
Municipal-level spatial disaggregation of province forecasts.

Formula (Backend Guide v3):
    y_m = y_province * (0.6 * poverty_m + 0.4 * density_m)

where:
    y_province  = province risk probability from LightGBM (Predictor)
    poverty_m   = LGU poverty incidence (PSA 2021 SAE), normalized within province
    density_m   = LGU population density (PSA 2020 CPH), normalized within province

The output is a ranked DataFrame with 142 municipal risk indices per quarter.
Every output row MUST carry disaggregation_label (Backend Guide Rule 8).

Usage:
    from app.ml.inference.disaggregator import Disaggregator
    lgu_df = Disaggregator().disaggregate(province_forecasts, quarter="2026-Q1")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LGU_CENSUS_PATH  = Path("data/processed/lgu_census.parquet")
LGU_POVERTY_PATH = Path("data/processed/lgu_poverty.parquet")

DISAGGREGATION_LABEL = (
    "Spatially disaggregated estimate derived from province-level LightGBM forecast "
    "using PSA 2021 SAE poverty incidence (60%) and PSA 2020 CPH population density (40%). "
    "Municipal-level uncertainty is higher than province-level uncertainty. "
    "Not an independent municipal forecast."
)

# Vulnerability weight split (Backend Guide v3)
POVERTY_WEIGHT  = 0.6
DENSITY_WEIGHT  = 0.4


class Disaggregator:
    """
    Loads LGU census and poverty data once, then disaggregates on demand.
    """

    _lgu: pd.DataFrame | None = None
    _pov: pd.DataFrame | None = None

    def _load(self):
        if self._lgu is not None:
            return
        if not LGU_CENSUS_PATH.exists() or not LGU_POVERTY_PATH.exists():
            raise FileNotFoundError(
                "LGU census or poverty parquets missing. Run fix_primary_data.py."
            )
        self._lgu = pd.read_parquet(LGU_CENSUS_PATH)
        self._pov = pd.read_parquet(LGU_POVERTY_PATH)
        logger.info(
            "Disaggregator: loaded %d LGUs (census) + %d LGUs (poverty)",
            len(self._lgu), len(self._pov),
        )

    def disaggregate(
        self,
        province_forecasts: list[dict],
        quarter: str,
    ) -> pd.DataFrame:
        """
        Disaggregate province-level forecasts to 142 LGUs.

        Parameters
        ----------
        province_forecasts : list[dict]
            Output from Predictor.forecast_quarter() — one dict per province.
        quarter : str  e.g. "2026-Q1"

        Returns
        -------
        pd.DataFrame  — 142 rows (all CALABARZON LGUs), columns:
            quarter, province_code, lgu_code, municipality_name,
            risk_index, disaggregation_label, data_sufficiency_flag
        Sorted by risk_index descending.
        """
        self._load()

        # Build province → forecast lookup
        prov_map = {f["province_code"]: f for f in province_forecasts}

        lgu = self._lgu.copy()
        pov = self._pov[["lgu_code", "poverty_incidence_pct"]].copy() if self._pov is not None else pd.DataFrame()

        # Merge census + poverty
        merged = lgu.merge(pov, on="lgu_code", how="left")
        merged["poverty_incidence_pct"] = merged["poverty_incidence_pct"].fillna(
            merged.groupby("province_code")["poverty_incidence_pct"].transform("mean")
        ).fillna(0.0)

        # Compute population density (persons / km²)
        if "land_area_km2" in merged.columns and "population_2020" in merged.columns:
            merged["density"] = merged["population_2020"] / merged["land_area_km2"].replace(0, np.nan)
        elif "population_density" in merged.columns:
            merged["density"] = merged["population_density"]
        else:
            merged["density"] = 1.0
        merged["density"] = merged["density"].fillna(merged.groupby("province_code")["density"].transform("mean")).fillna(1.0)

        # Normalize within province (min-max scaling)
        for col in ["poverty_incidence_pct", "density"]:
            grp = merged.groupby("province_code")[col]
            mn = grp.transform("min")
            mx = grp.transform("max")
            rng = (mx - mn).replace(0, 1)
            merged[f"{col}_norm"] = (merged[col] - mn) / rng

        # Vulnerability weight
        merged["vulnerability"] = (
            POVERTY_WEIGHT  * merged["poverty_incidence_pct_norm"]
            + DENSITY_WEIGHT * merged["density_norm"]
        )

        # Apply province forecast
        rows = []
        for _, lgu_row in merged.iterrows():
            province_code = lgu_row["province_code"]
            forecast = prov_map.get(province_code, {})
            prov_prob = forecast.get("risk_probability", 0.0)
            data_flag = forecast.get("data_sufficiency_flag")

            risk_index = float(prov_prob * lgu_row["vulnerability"])
            risk_index = round(min(max(risk_index, 0.0), 1.0), 4)

            lgu_code = lgu_row.get("lgu_code", "")
            lgu_name = lgu_row.get("lgu_name", "")

            rows.append({
                "quarter":              quarter,
                "province_code":        province_code,
                "lgu_code":             lgu_code,
                "municipality_name":    lgu_name,
                "risk_index":           risk_index,
                "disaggregation_label": DISAGGREGATION_LABEL,
                "data_sufficiency_flag": data_flag,
            })

        df = pd.DataFrame(rows).sort_values("risk_index", ascending=False).reset_index(drop=True)
        logger.info(
            "disaggregate(%s): %d LGUs | risk_index range [%.4f, %.4f]",
            quarter, len(df),
            df["risk_index"].min() if len(df) else 0,
            df["risk_index"].max() if len(df) else 0,
        )
        return df
