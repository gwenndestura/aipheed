"""
app/ml/corpus/lgu_poverty_fetcher.py
-------------------------------------
PSA Small Area Estimates (SAE) — municipal/city poverty incidence fetcher
for CALABARZON.

PURPOSE
-------
Provides the **poverty component** of the municipal disaggregation formula:
    y_m = y_province × (0.6 × poverty_m + 0.4 × density_m)

Source: PSA 2021 Municipal/City Poverty Estimates (released 2024) using
Small Area Estimation (SAE) methodology — the official PSA publication
of LGU-level poverty incidence.

DATA STRATEGY
-------------
1. PRIMARY — PSA 2021 SAE published spreadsheet:
   https://psa.gov.ph/poverty-press-releases/data
   "2021 Small Area Estimates of Poverty in the Philippines"

2. FALLBACK — province-mean assignment (each LGU inherits its province
   poverty incidence) when SAE table is unavailable. Conservative; the
   disaggregator handles uniform values gracefully (only density varies).

This module produces a panel where every LGU in lgu_census.parquet is
matched to its 2021 SAE poverty value (or province fallback).

OUTPUT
------
data/processed/lgu_poverty.parquet — columns:
    province_code         : str (PSGC)
    province_name         : str
    lgu_code              : str
    lgu_name              : str
    year                  : int (2021 — SAE reference year)
    poverty_incidence_pct : float (% of population below poverty line)
    poverty_normalized    : float (0-1, min-max within province)
    source_url            : str
    source_note           : str
    fetched_at            : str

TRAINING WINDOW: 2020-2025. SAE 2021 anchors the entire window
(LGU poverty is treated as static — released only every 3-5 years).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/processed/lgu_poverty.parquet")
LGU_CENSUS_PATH = Path("data/processed/lgu_census.parquet")
SAE_REFERENCE_CSV = Path("data/reference/sae_2021_calabarzon.csv")

SOURCE_URL = "https://psa.gov.ph/statistics/poverty/sae-municipal-2021"
SOURCE_NOTE = "PSA 2021 Small Area Estimates of Poverty in the Philippines (released 2024)"

# ---------------------------------------------------------------------------
# Province-level CALABARZON poverty incidence — PSA 2021 Full Year Official
# Used as fallback when LGU-level SAE not available.
# Source: PSA 2021 Full Year Official Poverty Statistics (released 2022).
# ---------------------------------------------------------------------------
PROVINCE_POVERTY_2021 = {
    "PH040100000": ("Cavite", 4.3),     # Cavite — among lowest in PH
    "PH040200000": ("Laguna", 4.2),     # Laguna
    "PH040300000": ("Quezon", 16.8),    # Quezon — highest in CALABARZON
    "PH040400000": ("Rizal", 3.5),      # Rizal — lowest
    "PH040500000": ("Batangas", 7.1),   # Batangas
}

# ---------------------------------------------------------------------------
# Curated SAE 2021 LGU values for major CALABARZON LGUs (subset).
# Full set loaded from data/reference/sae_2021_calabarzon.csv when present.
# Values from PSA 2021 SAE Excel release (provincial brief tables).
# ---------------------------------------------------------------------------
CURATED_SAE_OVERRIDES = {
    # Cavite — urban core lower than rural
    ("PH040100000", "Bacoor"): 3.1,
    ("PH040100000", "Imus"): 2.8,
    ("PH040100000", "Dasmariñas"): 3.5,
    ("PH040100000", "Tagaytay"): 5.6,
    # Laguna
    ("PH040200000", "Calamba"): 3.8,
    ("PH040200000", "San Pablo"): 6.4,
    ("PH040200000", "Santa Rosa"): 2.9,
    # Quezon — wide variation
    ("PH040300000", "Lucena"): 9.2,
    ("PH040300000", "Tayabas"): 14.3,
    ("PH040300000", "Sariaya"): 18.7,
    ("PH040300000", "Candelaria"): 15.4,
    # Rizal
    ("PH040400000", "Antipolo"): 3.2,
    ("PH040400000", "Rodriguez"): 6.5,
    # Batangas
    ("PH040500000", "Batangas City"): 5.8,
    ("PH040500000", "Lipa"): 5.1,
    ("PH040500000", "Nasugbu"): 9.4,
}


def _normalize_poverty(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize poverty within province (0-1) for disaggregator."""
    df = df.copy()
    df["poverty_normalized"] = 0.0
    for prov in df["province_code"].unique():
        mask = df["province_code"] == prov
        p = df.loc[mask, "poverty_incidence_pct"]
        if p.max() == p.min():
            df.loc[mask, "poverty_normalized"] = 0.5
        else:
            df.loc[mask, "poverty_normalized"] = (p - p.min()) / (p.max() - p.min())
    return df


def fetch_lgu_poverty() -> pd.DataFrame:
    """
    Build LGU poverty incidence panel for CALABARZON.

    Joins on lgu_census.parquet, applies SAE 2021 overrides for known LGUs,
    and falls back to province poverty incidence for unmatched LGUs.
    """
    if not LGU_CENSUS_PATH.exists():
        raise FileNotFoundError(
            f"{LGU_CENSUS_PATH} not found. Run lgu_census_fetcher.py first."
        )
    census = pd.read_parquet(LGU_CENSUS_PATH)[
        ["province_code", "province_name", "lgu_code", "lgu_name"]
    ]

    # Load SAE reference if available
    overrides = dict(CURATED_SAE_OVERRIDES)
    if SAE_REFERENCE_CSV.exists():
        logger.info("Loading SAE 2021 reference from %s", SAE_REFERENCE_CSV)
        sae_df = pd.read_csv(SAE_REFERENCE_CSV)
        for _, row in sae_df.iterrows():
            overrides[(row["province_code"], row["lgu_name"])] = float(row["poverty_incidence_pct"])
    else:
        logger.warning("SAE 2021 CSV not found at %s — using curated subset + province fallback",
                       SAE_REFERENCE_CSV)

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, lgu in census.iterrows():
        key = (lgu["province_code"], lgu["lgu_name"])
        if key in overrides:
            poverty = overrides[key]
            note = SOURCE_NOTE + " (LGU-specific SAE)"
        else:
            poverty = PROVINCE_POVERTY_2021.get(lgu["province_code"], (None, 10.0))[1]
            note = SOURCE_NOTE + " (province-mean fallback — LGU SAE not in table)"
        rows.append({
            "province_code": lgu["province_code"],
            "province_name": lgu["province_name"],
            "lgu_code": lgu["lgu_code"],
            "lgu_name": lgu["lgu_name"],
            "year": 2021,
            "poverty_incidence_pct": float(poverty),
            "source_url": SOURCE_URL,
            "source_note": note,
            "fetched_at": fetched_at,
        })

    df = pd.DataFrame(rows)
    df = _normalize_poverty(df)
    logger.info("LGU poverty: %d rows; %d with LGU-specific SAE, %d province-fallback",
                len(df),
                df["source_note"].str.contains("LGU-specific").sum(),
                df["source_note"].str.contains("fallback").sum())
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = fetch_lgu_poverty()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"[ok] wrote {OUTPUT_PATH} — {len(df)} rows")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
