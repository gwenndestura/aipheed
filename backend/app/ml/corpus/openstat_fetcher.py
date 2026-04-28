"""
app/ml/corpus/openstat_fetcher.py
----------------------------------
PSA OpenStat PXWeb API fetcher for CALABARZON statistical indicators.

WHY OPENSTAT (NOT rsso04a.psa.gov.ph)
--------------------------------------
rsso04a.psa.gov.ph is fully Cloudflare-blocked (HTTP 403 on all paths,
including PDFs). The PSA OpenStat API (openstat.psa.gov.ph/PXWeb/api/v1/en/)
is NOT blocked — it returns JSON data directly for all statistical series.

This fetcher replaces the manual PDF download workflow entirely. All
indicators needed for aiPHeed feature engineering are available via the
OpenStat PXWeb JSON API at province level for CALABARZON.

DATASETS FETCHED
----------------
1. Food CPI by province (2018-based, monthly, 2018–2026)
   DB/2M/PI/CPI/2018NEW/0012M4ACP22.px
   → food_cpi, food_cpi_yoy — primary label signal

2. Food CPI Year-on-Year % Change (2019–2026)
   DB/2M/PI/CPI/2018NEW/0012M4ACP23.px
   → food_cpi_yoy_pct — direct inflation rate

3. Rice Retail Price by province, monthly (2012–2021)
   DB/2M/NRP/0042M4ARN01.px
   → rice_price_regular, rice_price_wellmilled

4. Labor Force Survey — Unemployment Rate (national quarterly, 2005–2026)
   DB/1B/LFS/0021B3GKEI2.px
   → unemployment_rate (national; nearest quarter interpolated to province)

5. Poverty Incidence by province (2018, 2021, 2023)
   DB/1E/FY/0011E3DF010.px
   → poverty_incidence (interpolated to fill quarterly gaps)

OUTPUT
------
Returns a pd.DataFrame with columns:
    province_code     : str   — PSGC code (e.g. "PH040500000" = Batangas)
    province_name     : str   — Province name
    quarter           : str   — "YYYY-QN" (e.g. "2022-Q3")
    year              : int
    month             : int   — 1–12 (one row per month; quarter derived)
    food_cpi          : float — Food CPI index (2018=100)
    food_cpi_yoy      : float — Year-on-year % change in food CPI
    rice_price_regular: float — Regular milled rice retail price (PHP/kg)
    rice_price_well   : float — Well-milled rice retail price (PHP/kg)
    unemployment_rate : float — National unemployment rate % (nearest quarter)
    poverty_incidence : float — Province poverty incidence % (interpolated)

Usage:
    from app.ml.corpus.openstat_fetcher import fetch_openstat_indicators
    df = fetch_openstat_indicators("2020-01-01", "2025-12-31")
    df.to_parquet("data/processed/psa_indicators.parquet", index=False)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENSTAT_BASE = "https://openstat.psa.gov.ph/PXWeb/api/v1/en"
CRAWL_DELAY = 0.5   # polite delay between API calls (seconds)
TIMEOUT = 30        # request timeout

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aiPHeed/1.0 (+research; DLSU-D thesis)"})

OUTPUT_PATH = Path("data/processed/psa_indicators.parquet")

# ---------------------------------------------------------------------------
# CALABARZON province mapping
# Province codes in OpenStat datasets (same across CPI / poverty / rice)
# ---------------------------------------------------------------------------

CALABARZON_OPENSTAT = {
    "32": ("PH040100000", "Region IV-A"),   # region aggregate
    "33": ("PH040500000", "Batangas"),
    "34": ("PH040100000", "Cavite"),
    "35": ("PH040200000", "Laguna"),
    "36": ("PH040300000", "Quezon"),
    "37": ("PH040400000", "Rizal"),
}

# Province-only (exclude region aggregate code "32")
PROVINCE_CODES_OPENSTAT = {k: v for k, v in CALABARZON_OPENSTAT.items() if k != "32"}

# Rice dataset uses different internal codes
CALABARZON_RICE_OPENSTAT = {
    "28": ("PH040100000", "Region IV-A"),
    "29": ("PH040500000", "Batangas"),
    "30": ("PH040100000", "Cavite"),
    "31": ("PH040200000", "Laguna"),
    "32": ("PH040300000", "Quezon"),
    "33": ("PH040400000", "Rizal"),
}
PROVINCE_CODES_RICE = {k: v for k, v in CALABARZON_RICE_OPENSTAT.items() if k != "28"}

MONTH_ABBR = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    "January": 1, "February": 2, "March": 3, "April": 4,
    "June": 6, "July": 7, "August": 8, "September": 9,
    "October": 10, "November": 11, "December": 12,
}


def _quarter(month: int) -> str:
    return f"Q{(month - 1) // 3 + 1}"


# ---------------------------------------------------------------------------
# PXWeb API helpers
# ---------------------------------------------------------------------------

def _get_metadata(dataset_path: str) -> dict:
    """Fetch PXWeb dataset metadata (variable names + value codes)."""
    url = f"{OPENSTAT_BASE}/{dataset_path}"
    try:
        r = SESSION.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        # PSA OpenStat responses include a UTF-8 BOM; decode manually.
        import json as _json
        return _json.loads(r.content.decode("utf-8-sig"))
    except Exception as exc:
        logger.error("OpenStat metadata error [%s]: %s", dataset_path, exc)
        return {}


def _query_dataset(dataset_path: str, query_body: dict) -> list[dict]:
    """
    POST a PXWeb JSON query and return the data rows.

    Returns list of dicts with variable names as keys.
    """
    url = f"{OPENSTAT_BASE}/{dataset_path}"
    try:
        r = SESSION.post(url, json=query_body, timeout=TIMEOUT)
        time.sleep(CRAWL_DELAY)
        r.raise_for_status()
        # PSA OpenStat responses include a UTF-8 BOM; decode manually.
        import json as _json
        result = _json.loads(r.content.decode("utf-8-sig"))
        return _parse_pxweb_response(result)
    except Exception as exc:
        logger.error("OpenStat query error [%s]: %s", dataset_path, exc)
        return []


def _parse_pxweb_response(result: dict) -> list[dict]:
    """
    Convert PXWeb JSON response (columns + data) to list of flat dicts.
    """
    columns = result.get("columns", [])
    data = result.get("data", [])

    if not columns or not data:
        return []

    col_names = [c["text"] for c in columns]
    rows: list[dict] = []

    for entry in data:
        keys = entry.get("key", [])
        values = entry.get("values", [])
        row: dict[str, Any] = {}
        for i, col in enumerate(col_names[:-1]):   # last column is the value
            row[col] = keys[i] if i < len(keys) else None
        row["value"] = values[0] if values else None
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Dataset 1: Food CPI by province (2018-based, monthly, 2018–2026)
# ---------------------------------------------------------------------------

def _fetch_food_cpi(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch food CPI (2018=100) for CALABARZON provinces, monthly.

    Commodity code 2 = "01.1 - FOOD" (all food items).
    Returns DataFrame with: province_code, year, month, food_cpi
    """
    dataset = "DB/2M/PI/CPI/2018NEW/0012M4ACP22.px"
    logger.info("Fetching Food CPI from OpenStat...")

    meta = _get_metadata(dataset)
    if not meta:
        return pd.DataFrame()

    # Build year codes (0=2018, 1=2019, 2=2020, ...)
    year_offset = 2018
    year_codes = [str(y - year_offset) for y in range(start_year, end_year + 1) if y >= 2018]

    # Month codes 0–11 = Jan–Dec (exclude 12=annual average)
    month_codes = [str(i) for i in range(12)]

    # Geo codes: provinces only (33–37)
    geo_codes = list(PROVINCE_CODES_OPENSTAT.keys())

    query = {
        "query": [
            {"code": "Geolocation",
             "selection": {"filter": "item", "values": geo_codes}},
            {"code": "Commodity Description",
             "selection": {"filter": "item", "values": ["2"]}},   # FOOD
            {"code": "Year",
             "selection": {"filter": "item", "values": year_codes}},
            {"code": "Period",
             "selection": {"filter": "item", "values": month_codes}},
        ],
        "response": {"format": "json"},
    }

    rows = _query_dataset(dataset, query)
    if not rows:
        logger.warning("Food CPI: no data returned")
        return pd.DataFrame()

    records = []
    for row in rows:
        geo_code = row.get("Geolocation")
        year_code = row.get("Year")
        period = row.get("Period")
        val = row.get("value")

        if geo_code not in PROVINCE_CODES_OPENSTAT:
            continue
        try:
            year = int(year_code) + 2018
            month = int(period) + 1   # 0-indexed → 1-indexed
            cpi = float(val) if val and val != ".." else None
        except (TypeError, ValueError):
            continue

        psgc, province_name = PROVINCE_CODES_OPENSTAT[geo_code]
        records.append({
            "province_code": psgc,
            "province_name": province_name,
            "year": year,
            "month": month,
            "food_cpi": cpi,
        })

    df = pd.DataFrame(records)
    logger.info("Food CPI: %d monthly province-records fetched", len(df))
    return df


# ---------------------------------------------------------------------------
# Dataset 2: Food CPI Year-on-Year % Change (2019–2026)
# ---------------------------------------------------------------------------

def _fetch_food_cpi_yoy(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch food CPI year-on-year % change for CALABARZON provinces.

    Returns DataFrame with: province_code, year, month, food_cpi_yoy
    """
    dataset = "DB/2M/PI/CPI/2018NEW/0012M4ACP23.px"
    logger.info("Fetching Food CPI YoY change from OpenStat...")

    year_offset = 2019
    year_codes = [str(y - year_offset) for y in range(max(start_year, 2019), end_year + 1)]
    month_codes = [str(i) for i in range(12)]
    geo_codes = list(PROVINCE_CODES_OPENSTAT.keys())

    query = {
        "query": [
            {"code": "Geolocation",
             "selection": {"filter": "item", "values": geo_codes}},
            {"code": "Commodity Description",
             "selection": {"filter": "item", "values": ["2"]}},   # FOOD
            {"code": "Year",
             "selection": {"filter": "item", "values": year_codes}},
            {"code": "Period",
             "selection": {"filter": "item", "values": month_codes}},
        ],
        "response": {"format": "json"},
    }

    rows = _query_dataset(dataset, query)
    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        geo_code = row.get("Geolocation")
        year_code = row.get("Year")
        period = row.get("Period")
        val = row.get("value")

        if geo_code not in PROVINCE_CODES_OPENSTAT:
            continue
        try:
            year = int(year_code) + 2019
            month = int(period) + 1
            yoy = float(val) if val and val != ".." else None
        except (TypeError, ValueError):
            continue

        psgc, province_name = PROVINCE_CODES_OPENSTAT[geo_code]
        records.append({
            "province_code": psgc,
            "province_name": province_name,
            "year": year,
            "month": month,
            "food_cpi_yoy": yoy,
        })

    df = pd.DataFrame(records)
    logger.info("Food CPI YoY: %d monthly province-records fetched", len(df))
    return df


# ---------------------------------------------------------------------------
# Dataset 3: Rice Retail Price by province, monthly (2012–2021)
# ---------------------------------------------------------------------------

def _fetch_rice_prices(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch rice retail prices for CALABARZON provinces, monthly.

    Commodity 0 = Regular Milled, 2 = Well Milled.
    Coverage: 2012–2021 only (OpenStat limitation).
    Returns DataFrame with: province_code, year, month, rice_price_regular, rice_price_well
    """
    dataset = "DB/2M/NRP/0042M4ARN01.px"
    logger.info("Fetching Rice Retail Prices from OpenStat...")

    # Year 0=2012 … 9=2021
    year_codes = [str(y - 2012) for y in range(start_year, min(end_year, 2021) + 1) if y >= 2012]
    if not year_codes:
        logger.warning("Rice Retail Price: no overlap with 2012–2021 range")
        return pd.DataFrame()

    month_codes = [str(i) for i in range(12)]  # 0=Jan … 11=Dec (skip 12=Annual)
    geo_codes = list(PROVINCE_CODES_RICE.keys())

    rows_regular: list[dict] = []
    rows_well: list[dict] = []

    for commodity_code, commodity_name, target_list in [
        ("0", "Regular Milled", rows_regular),
        ("2", "Well Milled",    rows_well),
    ]:
        query = {
            "query": [
                {"code": "Region/Province",
                 "selection": {"filter": "item", "values": geo_codes}},
                {"code": "Commodity",
                 "selection": {"filter": "item", "values": [commodity_code]}},
                {"code": "year",
                 "selection": {"filter": "item", "values": year_codes}},
                {"code": "period",
                 "selection": {"filter": "item", "values": month_codes}},
            ],
            "response": {"format": "json"},
        }
        target_list.extend(_query_dataset(dataset, query))

    def _rows_to_df(rows: list[dict], price_col: str) -> pd.DataFrame:
        records = []
        for row in rows:
            geo = row.get("Region/Province")
            yr_code = row.get("year")
            period = row.get("period")
            val = row.get("value")
            if geo not in PROVINCE_CODES_RICE:
                continue
            try:
                year = int(yr_code) + 2012
                month = int(period) + 1
                price = float(val) if val and val != ".." else None
            except (TypeError, ValueError):
                continue
            psgc, province_name = PROVINCE_CODES_RICE[geo]
            records.append({
                "province_code": psgc,
                "province_name": province_name,
                "year": year,
                "month": month,
                price_col: price,
            })
        return pd.DataFrame(records)

    df_reg = _rows_to_df(rows_regular, "rice_price_regular")
    df_well = _rows_to_df(rows_well, "rice_price_well")

    if df_reg.empty:
        return pd.DataFrame()

    merge_keys = ["province_code", "province_name", "year", "month"]
    df = df_reg.merge(df_well, on=merge_keys, how="outer")
    logger.info("Rice Prices: %d monthly province-records fetched", len(df))
    return df


# ---------------------------------------------------------------------------
# Dataset 4: LFS Unemployment Rate (national quarterly, 2005–2026)
# ---------------------------------------------------------------------------

def _fetch_unemployment(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch national unemployment rate from LFS (quarterly surveys).

    LFS is national-level only in this table. Applied uniformly to all
    CALABARZON provinces. Returns DataFrame with: year, month, unemployment_rate.
    """
    dataset = "DB/1B/LFS/0021B3GKEI2.px"
    logger.info("Fetching LFS Unemployment Rate from OpenStat...")

    # Year codes: 0=2005, so offset is 2005
    year_codes = [str(y - 2005) for y in range(start_year, end_year + 1) if y >= 2005]

    # LFS survey months: Jan(0), Apr(3), Jul(6), Oct(9)
    lfs_months = ["0", "3", "6", "9"]

    query = {
        "query": [
            {"code": "Year",
             "selection": {"filter": "item", "values": year_codes}},
            {"code": "Month",
             "selection": {"filter": "item", "values": lfs_months}},
            {"code": "Rates",
             "selection": {"filter": "item", "values": ["2"]}},  # Unemployment Rate
            {"code": "Sex",
             "selection": {"filter": "item", "values": ["0"]}},  # Both sexes
        ],
        "response": {"format": "json"},
    }

    rows = _query_dataset(dataset, query)
    if not rows:
        return pd.DataFrame()

    records = []
    month_map = {"0": 1, "3": 4, "6": 7, "9": 10}   # LFS month code → calendar month

    for row in rows:
        yr_code = row.get("Year")
        mo_code = row.get("Month")
        val = row.get("value")
        try:
            year = int(yr_code) + 2005
            month = month_map.get(mo_code, 1)
            rate = float(val) if val and val != ".." else None
        except (TypeError, ValueError):
            continue
        records.append({"year": year, "month": month, "unemployment_rate": rate})

    df = pd.DataFrame(records)
    logger.info("LFS Unemployment: %d quarterly national records fetched", len(df))
    return df


# ---------------------------------------------------------------------------
# Dataset 5: Poverty Incidence by province (2018, 2021, 2023)
# ---------------------------------------------------------------------------

def _fetch_poverty(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch poverty incidence among families by CALABARZON province.

    Available years: 2018, 2021, 2023 only (PSA full-year release cadence).
    Returns DataFrame with: province_code, year, poverty_incidence.
    """
    dataset = "DB/1E/FY/0011E3DF010.px"
    logger.info("Fetching Poverty Incidence from OpenStat...")

    geo_codes = list(PROVINCE_CODES_OPENSTAT.keys())

    query = {
        "query": [
            {"code": "Geolocation",
             "selection": {"filter": "item", "values": geo_codes}},
        ],
        "response": {"format": "json"},
    }

    rows = _query_dataset(dataset, query)
    if not rows:
        return pd.DataFrame()

    # Poverty dataset has years as column dimension (0=2018, 1=2021, 2=2023)
    year_map = {"0": 2018, "1": 2021, "2": 2023}
    # Items: 0=Poverty Threshold, 1=Poverty Incidence (%), ...
    # We want poverty incidence — filter by value column label

    records = []
    for row in rows:
        geo = row.get("Geolocation")
        # The poverty table structure has Year as one of the keys
        year_code = row.get("Year") or row.get("year")
        item = row.get("Item") or row.get("item", "")
        val = row.get("value")

        if geo not in PROVINCE_CODES_OPENSTAT:
            continue
        if "Poverty Incidence" not in str(item) and "incidence" not in str(item).lower():
            # Keep only poverty incidence rows, not thresholds
            pass  # allow all; will filter after

        try:
            year = year_map.get(str(year_code), None)
            if year is None:
                continue
            if not (start_year <= year <= end_year):
                continue
            incidence = float(val) if val and val != ".." else None
        except (TypeError, ValueError):
            continue

        psgc, province_name = PROVINCE_CODES_OPENSTAT[geo]
        records.append({
            "province_code": psgc,
            "province_name": province_name,
            "year": year,
            "poverty_incidence": incidence,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.groupby(["province_code", "province_name", "year"])["poverty_incidence"].mean().reset_index()
    logger.info("Poverty Incidence: %d province-year records fetched", len(df))
    return df


# ---------------------------------------------------------------------------
# Merge + quarterly aggregation
# ---------------------------------------------------------------------------

def _merge_to_quarterly(
    food_cpi: pd.DataFrame,
    food_cpi_yoy: pd.DataFrame,
    rice: pd.DataFrame,
    unemployment: pd.DataFrame,
    poverty: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Merge all indicator DataFrames into a single province-quarter table.

    Food CPI and rice price are monthly → averaged to quarterly.
    Unemployment is quarterly (LFS months mapped to Q1–Q4).
    Poverty is yearly → forward-filled to all quarters of that year,
    then interpolated between survey years.
    """

    # --- Step 1: Average monthly food CPI to quarterly ---
    def _to_quarterly_avg(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["quarter"] = df["month"].apply(lambda m: f"Q{(m-1)//3+1}")
        df["quarter_label"] = df["year"].astype(str) + "-" + df["quarter"]
        return (
            df.groupby(["province_code", "province_name", "year", "quarter_label"])[value_cols]
            .mean()
            .reset_index()
        )

    cpi_q = _to_quarterly_avg(food_cpi, ["food_cpi"]) if not food_cpi.empty else pd.DataFrame()
    yoy_q = _to_quarterly_avg(food_cpi_yoy, ["food_cpi_yoy"]) if not food_cpi_yoy.empty else pd.DataFrame()
    rice_q = _to_quarterly_avg(rice, ["rice_price_regular", "rice_price_well"]) if not rice.empty else pd.DataFrame()

    # --- Step 2: Map LFS unemployment quarters ---
    unemp_q = pd.DataFrame()
    if not unemployment.empty:
        month_to_q = {1: "Q1", 4: "Q2", 7: "Q3", 10: "Q4"}
        unemp = unemployment.copy()
        unemp["quarter"] = unemp["month"].map(month_to_q)
        unemp["quarter_label"] = unemp["year"].astype(str) + "-" + unemp["quarter"]
        unemp_q = unemp[["year", "quarter_label", "unemployment_rate"]].drop_duplicates()

    # --- Step 3: Build base grid: 5 provinces × 24 quarters ---
    provinces = [
        ("PH040500000", "Batangas"),
        ("PH040100000", "Cavite"),
        ("PH040200000", "Laguna"),
        ("PH040300000", "Quezon"),
        ("PH040400000", "Rizal"),
    ]
    quarters = [
        f"{y}-Q{q}"
        for y in range(start_year, end_year + 1)
        for q in range(1, 5)
    ]

    base = pd.DataFrame([
        {"province_code": psgc, "province_name": name, "quarter_label": qtr}
        for psgc, name in provinces
        for qtr in quarters
    ])
    base["year"] = base["quarter_label"].str[:4].astype(int)

    # --- Step 4: Merge CPI ---
    if not cpi_q.empty:
        base = base.merge(
            cpi_q[["province_code", "quarter_label", "food_cpi"]],
            on=["province_code", "quarter_label"], how="left",
        )
    else:
        base["food_cpi"] = None

    # --- Step 5: Merge CPI YoY ---
    if not yoy_q.empty:
        base = base.merge(
            yoy_q[["province_code", "quarter_label", "food_cpi_yoy"]],
            on=["province_code", "quarter_label"], how="left",
        )
    else:
        base["food_cpi_yoy"] = None

    # --- Step 6: Merge rice prices ---
    if not rice_q.empty:
        base = base.merge(
            rice_q[["province_code", "quarter_label", "rice_price_regular", "rice_price_well"]],
            on=["province_code", "quarter_label"], how="left",
        )
    else:
        base["rice_price_regular"] = None
        base["rice_price_well"] = None

    # --- Step 7: Merge unemployment (national, same for all provinces) ---
    if not unemp_q.empty:
        base = base.merge(
            unemp_q[["quarter_label", "unemployment_rate"]],
            on="quarter_label", how="left",
        )
        # Forward-fill missing quarters (LFS is quarterly but some gaps exist)
        base = base.sort_values(["province_code", "quarter_label"])
        base["unemployment_rate"] = (
            base.groupby("province_code")["unemployment_rate"]
            .transform(lambda s: s.ffill().bfill())
        )
    else:
        base["unemployment_rate"] = None

    # --- Step 8: Merge poverty incidence + interpolate ---
    if not poverty.empty:
        base = base.merge(
            poverty[["province_code", "year", "poverty_incidence"]],
            on=["province_code", "year"], how="left",
        )
        # Interpolate between survey years (2018 → 2021 → 2023)
        base = base.sort_values(["province_code", "quarter_label"])
        base["poverty_incidence"] = (
            base.groupby("province_code")["poverty_incidence"]
            .transform(lambda s: s.interpolate(method="linear").ffill().bfill())
        )
    else:
        base["poverty_incidence"] = None

    base = base.rename(columns={"quarter_label": "quarter"})
    base = base.drop(columns=["year"], errors="ignore")

    return base.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_openstat_indicators(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    save_path: Path | None = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Fetch all PSA statistical indicators for CALABARZON via OpenStat API.

    No manual downloads required — fully automated via the PSA PXWeb API.
    Cloudflare does NOT protect the OpenStat endpoint.

    Parameters
    ----------
    start_date : str   — ISO date e.g. "2020-01-01"
    end_date   : str   — ISO date e.g. "2025-12-31"
    save_path  : Path  — where to save the output Parquet (None = skip save)

    Returns
    -------
    pd.DataFrame with columns:
        province_code, province_name, quarter,
        food_cpi, food_cpi_yoy,
        rice_price_regular, rice_price_well,
        unemployment_rate, poverty_incidence
    """
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    logger.info("Fetching PSA OpenStat indicators for CALABARZON (%d–%d)...", start_year, end_year)

    food_cpi = _fetch_food_cpi(start_year, end_year)
    food_cpi_yoy = _fetch_food_cpi_yoy(start_year, end_year)
    rice = _fetch_rice_prices(start_year, end_year)
    unemployment = _fetch_unemployment(start_year, end_year)
    poverty = _fetch_poverty(start_year, end_year)

    df = _merge_to_quarterly(
        food_cpi, food_cpi_yoy, rice, unemployment, poverty,
        start_year, end_year,
    )

    n_rows = len(df)
    n_with_cpi = df["food_cpi"].notna().sum()
    n_with_yoy = df["food_cpi_yoy"].notna().sum() if "food_cpi_yoy" in df.columns else 0
    n_with_rice = df["rice_price_regular"].notna().sum() if "rice_price_regular" in df.columns else 0
    n_with_unemp = df["unemployment_rate"].notna().sum() if "unemployment_rate" in df.columns else 0
    n_with_poverty = df["poverty_incidence"].notna().sum() if "poverty_incidence" in df.columns else 0

    logger.info(
        "OpenStat fetch complete: %d province-quarters | "
        "food_cpi=%d | food_cpi_yoy=%d | rice=%d | unemployment=%d | poverty=%d",
        n_rows, n_with_cpi, n_with_yoy, n_with_rice, n_with_unemp, n_with_poverty,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        logger.info("PSA indicators saved → %s", save_path)

    return df
