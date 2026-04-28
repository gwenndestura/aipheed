"""
fix_psa_indicators.py
---------------------
Re-fetch PSA Table 1 (0011E3DF010.px) for the CORRECT column —
"Poverty Incidence Among Families (%)" (variable index 1) —
replacing the buggy "Annual Per Capita Poverty Threshold (in PhP)"
(variable index 0) currently in psa_indicators.parquet.

Bug: poverty_incidence column shows values 4500-7000 (₱ threshold);
real poverty incidence % for CALABARZON is single-digit to mid-teens.

Strict 2020–2025 filter applied. Province-level INCIDENCE values are
released at PSA's tri-annual cadence (2018, 2021, 2023) — interpolated
forward-fill across quarters within the 2020–2025 window.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPENSTAT_URL = "https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/1E/FY/0011E3DF010.px"
INDICATORS_PATH = Path("data/processed/psa_indicators.parquet")

# CALABARZON provinces in PSA Table 1 (Geolocation valueTexts pattern)
CALABARZON_NAME_MAP = {
    "Cavite": "PH040100000",
    "Laguna": "PH040200000",
    "Quezon": "PH040300000",
    "Rizal": "PH040400000",
    "Batangas": "PH040500000",
}


def fetch_correct_poverty_incidence() -> pd.DataFrame:
    """
    Pull PSA poverty incidence % (column index 1, NOT 0) for CALABARZON
    provinces for years 2018, 2021, 2023.
    """
    s = requests.Session()
    s.headers.update({"User-Agent": "aiPHeed/1.0 research"})
    meta = json.loads(s.get(OPENSTAT_URL, timeout=30).content.decode("utf-8-sig"))

    geo_var = next(v for v in meta["variables"] if v["code"] == "Geolocation")
    geo_codes_for_calabarzon = []
    for code, text in zip(geo_var["values"], geo_var["valueTexts"]):
        clean = text.lstrip(".").strip()
        for prov_name in CALABARZON_NAME_MAP:
            if clean == prov_name or clean.endswith(prov_name):
                geo_codes_for_calabarzon.append((code, prov_name))

    logger.info("Matched CALABARZON provinces: %s", geo_codes_for_calabarzon)

    query_body = {
        "query": [
            {"code": "Geolocation", "selection": {
                "filter": "item",
                "values": [c for c, _ in geo_codes_for_calabarzon]
            }},
            {"code": "Threshold/Incidence/Measures of Precision", "selection": {
                "filter": "item",
                "values": ["1"]  # *** THE FIX *** index 1 = Poverty Incidence Among Families (%)
            }},
            {"code": "Year", "selection": {
                "filter": "item",
                "values": ["0", "1", "2"]  # 2018, 2021, 2023
            }},
        ],
        "response": {"format": "json"}
    }
    r = s.post(OPENSTAT_URL, json=query_body, timeout=30)
    result = json.loads(r.content.decode("utf-8-sig"))

    columns = result.get("columns", [])
    data = result.get("data", [])
    keys = [c.get("code") for c in columns]
    year_codes = {"0": 2018, "1": 2021, "2": 2023}
    code_to_prov = dict(geo_codes_for_calabarzon)

    rows = []
    for entry in data:
        row_keys = entry.get("key", [])
        values = entry.get("values", [])
        record = dict(zip(keys, row_keys))
        geo_code = record.get("Geolocation")
        year_code = record.get("Year")
        prov_name = code_to_prov.get(geo_code)
        year = year_codes.get(year_code)
        if prov_name and year and values and values[0] not in ("..", "-", None):
            try:
                pct = float(values[0])
                rows.append({
                    "province_code": CALABARZON_NAME_MAP[prov_name],
                    "province_name": prov_name,
                    "year": year,
                    "poverty_incidence_pct": pct,
                })
            except ValueError:
                pass

    return pd.DataFrame(rows)


def patch_psa_indicators() -> None:
    """Replace bogus poverty_incidence column with correct values."""
    if not INDICATORS_PATH.exists():
        raise FileNotFoundError(INDICATORS_PATH)

    df = pd.read_parquet(INDICATORS_PATH)
    correct = fetch_correct_poverty_incidence()
    print("[fix] correct poverty incidence values:")
    print(correct.to_string(index=False))

    # Build year column
    df["year"] = df["quarter"].str[:4].astype(int)

    # Forward-fill: 2020,2021 ← 2018; 2022,2023 ← 2021; 2024,2025 ← 2023
    def assign(row) -> float:
        y = row["year"]
        if y <= 2020:
            target_year = 2018
        elif y <= 2022:
            target_year = 2021
        else:
            target_year = 2023
        match = correct[
            (correct["province_code"] == row["province_code"]) &
            (correct["year"] == target_year)
        ]
        return float(match["poverty_incidence_pct"].iloc[0]) if len(match) else None

    df["poverty_incidence"] = df.apply(assign, axis=1)
    # Drop helper column
    df = df.drop(columns=["year"])
    # Strict 2020-2025 filter (already enforced; reaffirm)
    df = df[df["quarter"].str[:4].astype(int).between(2020, 2025)].reset_index(drop=True)

    df.to_parquet(INDICATORS_PATH, index=False)
    print(f"\n[ok] patched {INDICATORS_PATH}")
    print("\nNew poverty_incidence summary:")
    print(df.groupby("province_name")["poverty_incidence"].agg(["min", "max", "mean", "count"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    patch_psa_indicators()
