"""
app/ml/features/label_generator.py
------------------------------------
Composite food-stress label generator for the LightGBM classifier.

PRIMARY LABEL — Composite stress score (SWS quarterly hunger + PSA food CPI deviation)
---------------------------------------------------------------------------------------
For every (province, quarter) cell in the 2020-Q1 → 2025-Q4 window:

    stress_score_p,t = SWS_hunger_t + α · (food_cpi_yoy_p,t − regional_mean_food_cpi_yoy_t)

    label_stress_p,t = 1 if stress_score_p,t > global_median(stress_score) else 0

Sources:
  - SWS quarterly hunger (Social Weather Stations) — temporal signal, regional resolution.
    Continuous quarterly series since 1998; cited by NEDA, DSWD, BSP, PIDS, ADB.
  - PSA OpenStat food CPI YoY — spatial signal, province-quarter resolution.

The composite combines a household-experience indicator (SWS hunger) with an
economic-access indicator (food CPI deviation), giving both temporal AND spatial
variation that any single source alone cannot provide. Threshold = global median
across the 120 (province, quarter) pairs guarantees a balanced 50/50 class split
across the full window.

CONTEXT (NOT used in label generation):
  DOST-FNRI ENNS FIES (NNS 2021: 50.9%, NNS 2023: 51.6% moderate-or-severe)
  is the FAO SDG 2.1.2 regional anchor that confirms CALABARZON has elevated
  food insecurity prevalence. It is biennial and regional-only, so it cannot
  produce province-quarter labels at the resolution this model operates at.
  See get_fies_regional_baseline() for citation in thesis intro / README.

ROBUSTNESS LABEL (CPI-deviation — sanity check)
-----------------------------------------------
Source: data/processed/psa_indicators.parquet (food_cpi)
Formula: y = 1 if food_CPI_t > rolling_mean_24q + σ for >= 2 consecutive quarters

Used as a cross-check that label_stress tracks economic food-stress signals.

Outputs:
  data/processed/labels.parquet
  data/processed/label_distribution.json

Usage:
    from app.ml.features.label_generator import generate_labels
    labels_df = generate_labels()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CALABARZON_PROVINCES: dict[str, str] = {
    "PH040100000": "Cavite",
    "PH040200000": "Laguna",
    "PH040300000": "Quezon",
    "PH040400000": "Rizal",
    "PH040500000": "Batangas",
}

# All model quarters (2020-Q1 → 2025-Q4 = 24 quarters)
MODEL_QUARTERS: list[str] = [
    f"{yr}-Q{q}"
    for yr in range(2020, 2026)
    for q in range(1, 5)
]

# Stress-score formula coefficient: alpha amplifies province deviation from regional mean
ALPHA: float = 2.0

LABELS_PATH = Path("data/processed/labels.parquet")
LABEL_DIST_PATH = Path("data/processed/label_distribution.json")


# ---------------------------------------------------------------------------
# Quarter utilities
# ---------------------------------------------------------------------------

def _quarter_to_int(q: str) -> int:
    try:
        year, qpart = q.split("-")
        return int(year) * 4 + int(qpart[1]) - 1
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Primary composite-stress label
# ---------------------------------------------------------------------------

def _build_stress_labels(
    sws_path: Path,
    psa_path: Path,
) -> pd.DataFrame:
    """
    Build province-quarter labels from a composite SWS + PSA stress score.

    Formula:
        stress_p,t = sws_hunger_t + ALPHA * (food_cpi_yoy_p,t − regional_mean_t)
        label_stress = 1 if stress_p,t > global_median(stress) else 0

    Returns DataFrame with columns:
        province_code, province_name, quarter,
        sws_hunger_pct, food_cpi_yoy, regional_food_cpi_yoy,
        stress_score, global_threshold, label_stress
    """
    sws_df = pd.read_parquet(sws_path)
    psa_df = pd.read_parquet(psa_path)

    # Quarterly regional SWS hunger (regional series → mean across provinces)
    sws_quarterly = (
        sws_df.groupby("quarter")["pct_total_hunger"]
        .mean()
        .reset_index()
        .rename(columns={"pct_total_hunger": "sws_hunger_pct"})
    )

    # Province-quarter food CPI YoY
    psa_sub = psa_df[psa_df["province_code"].isin(CALABARZON_PROVINCES)][
        ["province_code", "quarter", "food_cpi_yoy"]
    ].copy()
    psa_sub["food_cpi_yoy"] = pd.to_numeric(
        psa_sub["food_cpi_yoy"], errors="coerce"
    ).fillna(0.0)

    # Cross-province mean food_cpi_yoy per quarter (regional baseline)
    quarter_cpi_mean = (
        psa_sub.groupby("quarter")["food_cpi_yoy"]
        .mean()
        .reset_index()
        .rename(columns={"food_cpi_yoy": "regional_food_cpi_yoy"})
    )

    rows: list[dict] = []
    for province_code in CALABARZON_PROVINCES:
        for quarter in MODEL_QUARTERS:
            sws_q = sws_quarterly[
                sws_quarterly["quarter"] == quarter
            ]["sws_hunger_pct"]
            sws_val = float(sws_q.iloc[0]) if len(sws_q) > 0 else 0.0

            cpi_q = psa_sub[
                (psa_sub["province_code"] == province_code)
                & (psa_sub["quarter"] == quarter)
            ]["food_cpi_yoy"]
            cpi_val = float(cpi_q.iloc[0]) if len(cpi_q) > 0 else 0.0

            cpi_reg = quarter_cpi_mean[
                quarter_cpi_mean["quarter"] == quarter
            ]["regional_food_cpi_yoy"]
            cpi_reg_val = float(cpi_reg.iloc[0]) if len(cpi_reg) > 0 else 0.0

            stress = sws_val + ALPHA * (cpi_val - cpi_reg_val)

            rows.append({
                "province_code":         province_code,
                "province_name":         CALABARZON_PROVINCES[province_code],
                "quarter":               quarter,
                "sws_hunger_pct":        sws_val,
                "food_cpi_yoy":          cpi_val,
                "regional_food_cpi_yoy": cpi_reg_val,
                "stress_score":          round(stress, 4),
            })

    df = pd.DataFrame(rows)

    # Global median threshold over all 120 (province, quarter) pairs.
    # Guarantees ~50/50 class split across the full window.
    global_threshold = float(df["stress_score"].median())
    df["global_threshold"] = round(global_threshold, 4)
    df["label_stress"] = (df["stress_score"] > global_threshold).astype(int)

    logger.info(
        "_build_stress_labels: %d province-quarter rows | label_stress distribution: %s | "
        "global stress threshold: %.4f | stress range [%.4f, %.4f]",
        len(df),
        df["label_stress"].value_counts().to_dict(),
        global_threshold,
        df["stress_score"].min(),
        df["stress_score"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# CPI-deviation robustness label
# ---------------------------------------------------------------------------

def _build_cpi_labels(psa_path: Path) -> pd.DataFrame:
    """
    Build CPI-deviation binary label as a robustness sanity-check.

    Rule: y_cpi = 1 if food_CPI_t > rolling_mean_24q + 1*sigma
                 for at least 2 consecutive quarters.

    Returns DataFrame with columns:
        province_code, quarter, label_cpi
    """
    psa_df = pd.read_parquet(psa_path)
    psa_df = psa_df[psa_df["province_code"].isin(CALABARZON_PROVINCES)].copy()
    psa_df = psa_df[psa_df["quarter"].isin(MODEL_QUARTERS)].copy()

    psa_df["_q_int"] = psa_df["quarter"].apply(_quarter_to_int)
    psa_df = psa_df.sort_values(["province_code", "_q_int"])

    grp = psa_df.groupby("province_code")["food_cpi"]
    psa_df["cpi_rolling_mean"] = grp.transform(
        lambda s: s.rolling(window=24, min_periods=4).mean()
    )
    psa_df["cpi_rolling_std"] = grp.transform(
        lambda s: s.rolling(window=24, min_periods=4).std()
    )
    psa_df["cpi_deviation_flag"] = (
        psa_df["food_cpi"] > psa_df["cpi_rolling_mean"] + psa_df["cpi_rolling_std"]
    ).astype(int)

    psa_df["cpi_dev_prev"] = (
        psa_df.groupby("province_code")["cpi_deviation_flag"]
        .shift(1).fillna(0).astype(int)
    )
    psa_df["label_cpi"] = (
        (psa_df["cpi_deviation_flag"] == 1) & (psa_df["cpi_dev_prev"] == 1)
    ).astype(int)

    logger.info(
        "_build_cpi_labels: label_cpi distribution: %s",
        psa_df["label_cpi"].value_counts().to_dict(),
    )
    return psa_df[["province_code", "quarter", "label_cpi"]].copy()


# ---------------------------------------------------------------------------
# FAO SDG 2.1.2 regional anchor (citation only — not used in label generation)
# ---------------------------------------------------------------------------

def get_fies_regional_baseline(
    nns_fies_path: Path = Path("data/processed/nns_fies.parquet"),
) -> dict:
    """
    Read DOST-FNRI ENNS FIES values for thesis intro / README citation.

    NOT used in label generation. FIES is biennial and published only at the
    regional level (Region IV-A / CALABARZON), so it cannot produce
    province-quarter labels at the resolution this model operates at. It is
    retained as the FAO SDG 2.1.2 regional anchor that confirms CALABARZON
    has elevated food insecurity prevalence (motivating the forecasting work).

    Returns a dict like:
        {
          "NNS_2021": {"moderate_or_severe_pct": 50.9, "severe_pct": 11.7},
          "NNS_2023": {"moderate_or_severe_pct": 51.6, "severe_pct": 12.4},
        }
    """
    if not nns_fies_path.exists():
        logger.warning("nns_fies.parquet not found at %s — skipping FIES anchor.", nns_fies_path)
        return {}
    fies_df = pd.read_parquet(nns_fies_path)
    out: dict = {}
    for cycle in fies_df["survey_cycle"].unique():
        row = fies_df[fies_df["survey_cycle"] == cycle].iloc[0]
        out[str(cycle)] = {
            "moderate_or_severe_pct": float(row.get("fies_moderate_severe_pct", float("nan"))),
            "severe_pct":             float(row.get("fies_severe_pct", float("nan"))),
        }
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_labels(
    psa_path: Path = Path("data/processed/psa_indicators.parquet"),
    sws_path: Path = Path("data/processed/sws_hunger.parquet"),
    nns_fies_path: Path = Path("data/processed/nns_fies.parquet"),
    labels_path: Path = LABELS_PATH,
    dist_path: Path = LABEL_DIST_PATH,
) -> pd.DataFrame:
    """
    Generate composite-stress labels for all province-quarter cells in the
    2020-Q1 → 2025-Q4 model window.

    Parameters
    ----------
    psa_path      : path to psa_indicators.parquet (food_cpi_yoy source)
    sws_path      : path to sws_hunger.parquet     (pct_total_hunger source)
    nns_fies_path : path to nns_fies.parquet       (FIES anchor — context only)
    labels_path   : output path for labels.parquet
    dist_path     : output path for label_distribution.json

    Returns
    -------
    pd.DataFrame
        Columns: province_code, province_name, quarter,
                 sws_hunger_pct, food_cpi_yoy, regional_food_cpi_yoy,
                 stress_score, global_threshold,
                 label_stress (PRIMARY composite stress label),
                 label_cpi  (ROBUSTNESS CPI-deviation cross-check),
                 label_agreement
    """
    stress_labels = _build_stress_labels(sws_path=sws_path, psa_path=psa_path)
    cpi_labels    = _build_cpi_labels(psa_path)

    labels_df = stress_labels.merge(
        cpi_labels, on=["province_code", "quarter"], how="left"
    )
    labels_df["label_cpi"] = labels_df["label_cpi"].fillna(0).astype(int)
    labels_df["label_agreement"] = (
        labels_df["label_stress"] == labels_df["label_cpi"]
    )

    agreement_pct = labels_df["label_agreement"].mean() * 100
    logger.info(
        "generate_labels: stress/CPI agreement = %.1f%% across %d rows",
        agreement_pct, len(labels_df),
    )

    # Save labels parquet
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(labels_path, index=False)
    logger.info("Labels saved → %s", labels_path)

    # FIES anchor (context only — for thesis intro citation)
    fies_anchor = get_fies_regional_baseline(nns_fies_path)

    # Save distribution JSON
    dist: dict = {
        "total_rows": len(labels_df),
        "provinces":  list(labels_df["province_code"].unique()),
        "quarters":   sorted(labels_df["quarter"].unique()),
        "label_stress": {
            "positive":      int((labels_df["label_stress"] == 1).sum()),
            "negative":      int((labels_df["label_stress"] == 0).sum()),
            "positive_rate": round(labels_df["label_stress"].mean(), 4),
        },
        "label_cpi_robustness": {
            "positive":      int((labels_df["label_cpi"] == 1).sum()),
            "negative":      int((labels_df["label_cpi"] == 0).sum()),
            "positive_rate": round(labels_df["label_cpi"].mean(), 4),
        },
        "label_agreement_pct": round(agreement_pct, 2),
        "label_definition": {
            "primary":  "stress_score_p,t = sws_hunger_t + 2 * (food_cpi_yoy_p,t - regional_mean_food_cpi_yoy_t); "
                        "label_stress = 1 if stress_score > global_median",
            "robustness": "label_cpi = 1 if food_cpi > rolling_24q_mean + 1σ for >= 2 consecutive quarters",
        },
        "sources": {
            "primary_label_temporal":  "SWS quarterly Hunger Survey (1998-present)",
            "primary_label_spatial":   "PSA OpenStat food CPI YoY by province",
            "robustness_label":        "PSA OpenStat food CPI rolling deviation",
        },
        "fao_sdg_2_1_2_anchor": {
            "note": ("DOST-FNRI ENNS FIES regional baseline — NOT used in label generation. "
                     "Cited as context in thesis intro to establish CALABARZON's elevated "
                     "food insecurity prevalence per FAO SDG 2.1.2 methodology."),
            "values": fies_anchor,
        },
    }
    dist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dist_path, "w", encoding="utf-8") as f:
        json.dump(dist, f, indent=2, default=str)
    logger.info("Label distribution saved → %s", dist_path)

    return labels_df
