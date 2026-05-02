"""
app/ml/features/feature_matrix.py
------------------------------------
Multi-scale fused feature matrix builder (province-quarter level).

Fuses ALL eleven primary-data parquets with NLP features into a single
training-ready DataFrame. This is the input to LightGBM.

Feature groups and sources:
  1. NLP / FSSI      : FSSI, FSSI_lag1, FSSI_lag2, FSSI_accel
  2. Triggers (5-cat): trigger_market, trigger_climate, trigger_employment,
                       trigger_ofw_remittance, trigger_fish_kill (proportions)
  3. Food CPI        : food_cpi, food_cpi_yoy, food_minus_headline_yoy
  4. Rice prices     : rice_price_regular (PSA 2020-21 + Ricelytics 2022-25)
  5. Macro / Labour  : unemployment_rate, poverty_incidence
  6. BSP             : ofw_remit_yoy_pct, fx_usd_php_avg
  7. Fuel            : diesel_php_per_l, brent_usd_per_bbl
  8. Climate         : tc_count, rainfall_anomaly_pct, enso_numeric, drought_alert
  9. Commodity basket: veg_price_mean, fish_price_mean, livestock_price_mean
 10. BERTopic props  : topic_N_pct columns (supplementary, if available)

NOTE: pct_total_hunger (SWS) is NOT included here. The primary label label_fies is defined as
(pct_total_hunger > cycle_median), so including it as a feature is direct label leakage.
The label generator reads sws_hunger.parquet independently.

LightGBM formula reference (Backend Guide v3):
    y_hat_p,t+3 = F(FSSI_t, FSSI_t-1, FSSI_t-2, dFSSI_t,
                    trigger_5cat_t, BERTopic_t, food_CPI_t, headline_CPI_t,
                    food_minus_headline_yoy_t, rice_t, commodity_basket_t,
                    tc_count_t, rainfall_anom_t, ENSO_t, drought_t,
                    ofw_remit_t, fx_php_usd_t, diesel_t, gasoline_t,
                    brent_t, unemployment_t, poverty_incidence_t)

Usage:
    from app.ml.features.feature_matrix import build_feature_matrix
    features_df = build_feature_matrix()
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

OUTPUT_PATH = Path("data/processed/features_fused.parquet")

CALABARZON_PROVINCES = [
    "PH040100000", "PH040200000", "PH040300000", "PH040400000", "PH040500000",
]

MODEL_QUARTERS: list[str] = [
    f"{yr}-Q{q}" for yr in range(2020, 2026) for q in range(1, 5)
]

# ENSO phase → numeric encoding (for LightGBM ordinal)
ENSO_ENCODE: dict[str, int] = {
    "LA_NINA":   -1,
    "NEUTRAL":    0,
    "EL_NINO":    1,
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _fix_cpi_quarter(q: str) -> str:
    """Fix cpi_full.parquet quarter format '2020.0-Q1.0' → '2020-Q1'."""
    try:
        parts = str(q).split("-")
        year = parts[0].split(".")[0]          # '2020.0' → '2020'
        qpart = parts[1].split(".")[0]         # 'Q1.0'  → 'Q1'
        return f"{year}-{qpart}"
    except Exception:
        return q


def _quarter_to_int(q: str) -> int:
    try:
        year, qpart = q.split("-")
        return int(year) * 4 + int(qpart[1]) - 1
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Individual source loaders
# ---------------------------------------------------------------------------

def _load_fssi(path: Path) -> pd.DataFrame:
    """Load FSSI features: FSSI, FSSI_lag1, FSSI_lag2, FSSI_accel."""
    df = pd.read_parquet(path)
    cols = ["province_code", "quarter", "FSSI", "FSSI_lag1", "FSSI_lag2", "FSSI_accel"]
    return df[[c for c in cols if c in df.columns]].copy()


def _load_triggers(path: Path) -> pd.DataFrame:
    """Load 5-category trigger proportions."""
    df = pd.read_parquet(path)
    trigger_cols = [
        "trigger_market", "trigger_climate", "trigger_employment",
        "trigger_ofw_remittance", "trigger_fish_kill",
    ]
    keep = ["province_code", "quarter"] + [c for c in trigger_cols if c in df.columns]
    return df[keep].copy()


def _load_psa(path: Path) -> pd.DataFrame:
    """Load PSA OpenStat indicators: food_cpi, food_cpi_yoy, rice_price, unemployment, poverty."""
    df = pd.read_parquet(path)
    keep = [
        "province_code", "quarter",
        "food_cpi", "food_cpi_yoy", "rice_price_regular",
        "unemployment_rate", "poverty_incidence",
    ]
    return df[[c for c in keep if c in df.columns]].copy()


def _load_cpi_full(path: Path) -> pd.DataFrame:
    """Load headline CPI and food-vs-headline gap feature."""
    df = pd.read_parquet(path)
    df["quarter"] = df["quarter"].apply(_fix_cpi_quarter)
    keep = ["province_code", "quarter", "cpi_food_minus_general_yoy", "cpi_all_items"]
    out = df[[c for c in keep if c in df.columns]].copy()
    out = out.rename(columns={
        "cpi_food_minus_general_yoy": "food_minus_headline_yoy",
        "cpi_all_items": "headline_cpi",
    })
    return out


def _load_bsp(path: Path) -> pd.DataFrame:
    """Load BSP OFW remittance growth and FX rate."""
    df = pd.read_parquet(path)
    keep = ["province_code", "quarter", "ofw_remit_yoy_pct", "fx_usd_php_avg"]
    return df[[c for c in keep if c in df.columns]].copy()


def _load_oil(path: Path) -> pd.DataFrame:
    """Load DOE/EIA fuel prices."""
    df = pd.read_parquet(path)
    keep = [
        "province_code", "quarter",
        "diesel_php_per_l", "gasoline_php_per_l", "brent_usd_per_bbl",
    ]
    return df[[c for c in keep if c in df.columns]].copy()


def _load_pagasa(path: Path) -> pd.DataFrame:
    """Load PAGASA climate features: tc_count, rainfall_anomaly, ENSO, drought."""
    df = pd.read_parquet(path)
    keep = [
        "province_code", "quarter",
        "tc_count", "tc_severe_flag", "rainfall_anomaly_pct",
        "enso_phase", "drought_alert",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    if "enso_phase" in out.columns:
        out["enso_numeric"] = out["enso_phase"].map(ENSO_ENCODE).fillna(0).astype(int)
        out = out.drop(columns=["enso_phase"])
    return out


def _load_commodity(path: Path) -> pd.DataFrame:
    """
    Load PSA NRP commodity prices and pivot to wide format.

    commodity_group → mean price per province-quarter.
    Groups: fruit_veg, fish, livestock/poultry, others.
    """
    df = pd.read_parquet(path)
    if "province_code" not in df.columns or "commodity_group" not in df.columns:
        logger.warning("_load_commodity: unexpected schema, returning empty.")
        return pd.DataFrame()

    # Pivot: mean price per commodity_group per province-quarter
    pivot = (
        df.groupby(["province_code", "quarter", "commodity_group"])["price_php_per_kg"]
        .mean()
        .reset_index()
        .pivot_table(
            index=["province_code", "quarter"],
            columns="commodity_group",
            values="price_php_per_kg",
            aggfunc="mean",
        )
        .reset_index()
    )

    # Rename columns to avoid conflicts
    pivot.columns = [
        f"commodity_{c}" if c not in ("province_code", "quarter") else c
        for c in pivot.columns
    ]
    return pivot


def _load_sws(path: Path) -> pd.DataFrame:
    """
    Intentionally returns an empty DataFrame.

    pct_total_hunger is the raw SWS value from which label_fies is derived
    (label = pct_total_hunger > cycle_median). Including it as a training feature
    constitutes perfect label leakage and will produce ~100% accuracy.
    The label generator reads sws_hunger.parquet directly and independently.
    """
    return pd.DataFrame()


def _load_ricelytics(path: Path) -> pd.DataFrame:
    """
    Load PhilRice Ricelytics 2022-2025 rice prices.

    The PSA psa_indicators.parquet already contains rice prices for 2020-21.
    Ricelytics covers 2022-25. We extract regular_milled price to fill the gap.
    """
    df = pd.read_parquet(path)
    rice = df[df.get("rice_class", pd.Series(dtype=str)) == "regular_milled"].copy() if "rice_class" in df.columns else df.copy()
    if rice.empty:
        return pd.DataFrame()
    rice = rice.rename(columns={"price_php_per_kg": "ricelytics_price_regular"})
    keep = ["province_code", "quarter", "ricelytics_price_regular"]
    return rice[[c for c in keep if c in rice.columns]].copy()


def _load_topic_proportions(path: Path) -> pd.DataFrame | None:
    """Load BERTopic province-quarter proportions (optional)."""
    if not path.exists():
        logger.info("_load_topic_proportions: file not found (%s) — skipping BERTopic features.", path)
        return None
    df = pd.read_parquet(path)
    logger.info("_load_topic_proportions: loaded %d rows, %d topic columns", len(df), len(df.columns) - 2)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_matrix(
    fssi_path:       Path = Path("data/processed/fssi_quarterly.parquet"),
    triggers_path:   Path = Path("data/processed/trigger_proportions.parquet"),
    psa_path:        Path = Path("data/processed/psa_indicators.parquet"),
    cpi_full_path:   Path = Path("data/processed/cpi_full.parquet"),
    bsp_path:        Path = Path("data/processed/bsp_macro.parquet"),
    oil_path:        Path = Path("data/processed/oil_prices.parquet"),
    pagasa_path:     Path = Path("data/processed/pagasa_climate.parquet"),
    commodity_path:  Path = Path("data/processed/commodity_prices.parquet"),
    sws_path:        Path = Path("data/processed/sws_hunger.parquet"),
    ricelytics_path: Path = Path("data/processed/ricelytics_prices.parquet"),
    topic_path:      Path = Path("data/processed/topic_proportions.parquet"),
    save_path:       Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Fuse all eleven primary-data sources + NLP features into features_fused.parquet.

    All sources are merged on (province_code, quarter) using left joins from
    a backbone grid of all 5-province × 24-quarter combinations. NaN values
    from missing coverage are forward-filled where safe (macro series), then
    zero-filled as a last resort with a flag.

    Returns
    -------
    pd.DataFrame
        province_code, quarter (join keys) + all feature columns.
        Filtered to 2020-Q1 → 2025-Q4, 5 CALABARZON provinces only.
        120 rows (5 × 24) if all sources have full coverage.
    """

    # --- Backbone grid: 5 provinces × 24 quarters ---
    backbone = pd.DataFrame([
        {"province_code": p, "quarter": q}
        for p in CALABARZON_PROVINCES
        for q in MODEL_QUARTERS
    ])
    backbone["_q_int"] = backbone["quarter"].apply(_quarter_to_int)
    backbone = backbone.sort_values(["province_code", "_q_int"])

    def _left_merge(base: pd.DataFrame, right: pd.DataFrame, tag: str) -> pd.DataFrame:
        if right.empty:
            logger.warning("_left_merge[%s]: empty DataFrame — skipping.", tag)
            return base
        before = len(base)
        merged = base.merge(right, on=["province_code", "quarter"], how="left")
        new_cols = [c for c in merged.columns if c not in base.columns]
        na_counts = {c: merged[c].isna().sum() for c in new_cols if merged[c].isna().any()}
        if na_counts:
            logger.debug("_left_merge[%s]: NaN counts after merge: %s", tag, na_counts)
        return merged

    # Build merged DataFrame step by step
    df = backbone.copy()

    # 1. FSSI
    try:
        df = _left_merge(df, _load_fssi(fssi_path), "FSSI")
    except FileNotFoundError:
        logger.warning("FSSI parquet not found at %s — FSSI features will be NaN.", fssi_path)

    # 2. Trigger proportions
    try:
        df = _left_merge(df, _load_triggers(triggers_path), "Triggers")
    except FileNotFoundError:
        logger.warning("Trigger proportions not found — trigger features will be NaN.")

    # 3. PSA indicators
    df = _left_merge(df, _load_psa(psa_path), "PSA")

    # 4. CPI full (food-vs-headline gap)
    df = _left_merge(df, _load_cpi_full(cpi_full_path), "CPI_Full")

    # 5. BSP macro
    df = _left_merge(df, _load_bsp(bsp_path), "BSP")

    # 6. Oil prices
    df = _left_merge(df, _load_oil(oil_path), "Oil")

    # 7. PAGASA climate
    df = _left_merge(df, _load_pagasa(pagasa_path), "PAGASA")

    # 8. Commodity prices (PSA NRP — wide pivot)
    try:
        df = _left_merge(df, _load_commodity(commodity_path), "Commodity")
    except Exception as exc:
        logger.warning("Commodity prices failed to load: %s", exc)

    # 9. Ricelytics 2022-25 (gap fill)
    try:
        rice_ext = _load_ricelytics(ricelytics_path)
        if not rice_ext.empty:
            df = _left_merge(df, rice_ext, "Ricelytics")
            # Fill rice_price_regular NaN (post-2021) with ricelytics values
            if "rice_price_regular" in df.columns and "ricelytics_price_regular" in df.columns:
                df["rice_price_regular"] = df["rice_price_regular"].fillna(
                    df["ricelytics_price_regular"]
                )
    except Exception as exc:
        logger.warning("Ricelytics load failed: %s", exc)

    # 11. BERTopic proportions (optional)
    topic_df = _load_topic_proportions(topic_path)
    if topic_df is not None:
        df = _left_merge(df, topic_df, "BERTopic")

    # --- Post-merge cleanup ---

    # Forward-fill macro series within each province (fills gaps from sparse surveys)
    # pct_total_hunger is excluded — see module docstring.
    ff_cols = [
        "FSSI", "food_cpi", "food_cpi_yoy", "unemployment_rate", "poverty_incidence",
        "ofw_remit_yoy_pct", "fx_usd_php_avg", "diesel_php_per_l",
        "brent_usd_per_bbl", "headline_cpi", "food_minus_headline_yoy",
    ]
    for col in ff_cols:
        if col in df.columns:
            df[col] = df.groupby("province_code")[col].transform(
                lambda s: s.ffill().bfill()
            )

    # ── Temporal lag features for primary data (momentum signal) ─────────
    # Mirrors the FSSI_lag1/lag2/accel pattern that the model already uses.
    # Adds 1-quarter lag and acceleration (Δ vs t-1) for the most informative
    # primary signals: food prices, labor, climate, remittances, fuel.
    df = df.sort_values(["province_code", "quarter"]).reset_index(drop=True)
    LAG_FEATURES = [
        "food_cpi_yoy",
        "food_minus_headline_yoy",
        "unemployment_rate",
        "ofw_remit_yoy_pct",
        "rainfall_anomaly_pct",
        "rice_price_regular",
        "diesel_php_per_l",
    ]
    for col in LAG_FEATURES:
        if col in df.columns:
            df[f"{col}_lag1"]  = df.groupby("province_code")[col].shift(1)
            df[f"{col}_accel"] = df[col] - df[f"{col}_lag1"]

    # Fill remaining NaN → 0 (categorical flags, trigger proportions already 0-1)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # Drop helper column
    df = df.drop(columns=["_q_int"], errors="ignore")
    # Drop metadata columns that shouldn't be features
    df = df.drop(columns=["ricelytics_price_regular"], errors="ignore")

    logger.info(
        "build_feature_matrix: final shape %s | %d feature columns",
        df.shape,
        len(df.columns) - 2,  # subtract province_code + quarter
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        logger.info("Feature matrix saved → %s", save_path)

    return df
