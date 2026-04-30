"""
app/ml/corpus/cpi_full_fetcher.py
----------------------------------
Headline CPI fetcher (PSA OpenStat) -- complement to existing food CPI.

PURPOSE
-------
psa_indicators.parquet stores food_cpi. We add HEADLINE CPI (all items)
to derive the panel-defensible "real food inflation" feature:

    cpi_food_minus_general_yoy = food_yoy - headline_yoy

When this gap is positive, food rises faster than overall prices --
disproportionately hurting low-income households (Engel coefficient
effect). This is the standard macro feature in:

    * WFP HungerMap LIVE -- macroeconomic layer
    * FAO GIEWS Country Briefs -- food vs general inflation gap
    * Balashankar et al. (2023) Science Advances 9(28) -- macro features

NOTE: Non-food CPI is NOT included separately. Mathematically
    All_Items = w*Food + (1-w)*Non-food
so (Food - All_Items) already captures the same signal as (Food - Non-food).

Source: PSA OpenStat table 0012M4ACP22.px
        "CPI by Commodity Group (2018=100): Jan 2018 - Mar 2026"

OUTPUT
------
data/processed/cpi_full.parquet -- columns:
    province_code, province_name, region_code, region_name,
    year, quarter,
    cpi_all_items              : float (HEADLINE -- "0 - ALL ITEMS")
    cpi_food_nab               : float (food + non-alc beverages)
    cpi_all_yoy_pct            : float (headline YoY %)
    cpi_food_yoy_pct           : float (food YoY %)
    cpi_food_minus_general_yoy : float (real food inflation gap -- KEY FEATURE)
    source_url, source_note, fetched_at
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPENSTAT_BASE = "https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M/PI/CPI/2018NEW"
TABLE = "0012M4ACP22.px"
OUTPUT_PATH = Path("data/processed/cpi_full.parquet")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aiPHeed/1.0 (+research; DLSU-D thesis)"})

CALABARZON = [
    ("PH040100000", "Cavite"),
    ("PH040200000", "Laguna"),
    ("PH040300000", "Quezon"),
    ("PH040400000", "Rizal"),
    ("PH040500000", "Batangas"),
]
REGION_CODE = "PH040000000"
REGION_NAME = "Region IV-A (CALABARZON)"

MONTH_TO_NUM = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def fetch_cpi_full(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    fetched_at = datetime.now(timezone.utc).isoformat()
    url = f"{OPENSTAT_BASE}/{TABLE}"
    meta = json.loads(SESSION.get(url, timeout=30).content.decode("utf-8-sig"))

    # Find Region IV-A geo code
    geo_var = next(v for v in meta["variables"] if v["code"] == "Geolocation")
    region_code = None
    for c, t in zip(geo_var["values"], geo_var["valueTexts"]):
        if "IV-A" in t or "CALABARZON" in t.upper():
            region_code = c
            break
    if not region_code:
        raise RuntimeError("CALABARZON region not found in CPI Geolocation")

    # Find target commodity codes — keep 1y (allow YoY calc) → start at start_year-1
    cd_var = next(v for v in meta["variables"] if v["code"] == "Commodity Description")
    keep_groups = {}  # group_label -> code
    for c, t in zip(cd_var["values"], cd_var["valueTexts"]):
        if t == "0 - ALL ITEMS":
            keep_groups["all_items"] = c
        elif t == "01 - FOOD AND NON-ALCOHOLIC BEVERAGES":
            keep_groups["food_nab"] = c

    # Year codes covering [start_year-1 ... end_year] for YoY
    year_var = next(v for v in meta["variables"] if v["code"] == "Year")
    year_codes = []
    yr_to_year = {}
    for c, t in zip(year_var["values"], year_var["valueTexts"]):
        try:
            y = int(t)
            if start_year - 1 <= y <= end_year:
                year_codes.append(c)
                yr_to_year[c] = y
        except ValueError:
            pass

    # Period: monthly only
    per_var = next(v for v in meta["variables"] if v["code"] == "Period")
    per_codes = []
    per_to_month = {}
    for c, t in zip(per_var["values"], per_var["valueTexts"]):
        m = MONTH_TO_NUM.get(t)
        if m:
            per_codes.append(c)
            per_to_month[c] = m

    body = {
        "query": [
            {"code": "Geolocation", "selection": {"filter": "item", "values": [region_code]}},
            {"code": "Commodity Description", "selection": {"filter": "item",
                                                            "values": list(keep_groups.values())}},
            {"code": "Year", "selection": {"filter": "item", "values": year_codes}},
            {"code": "Period", "selection": {"filter": "item", "values": per_codes}},
        ],
        "response": {"format": "json"},
    }
    r = SESSION.post(url, json=body, timeout=30)
    r.raise_for_status()
    time.sleep(0.4)
    result = json.loads(r.content.decode("utf-8-sig"))

    cols = [c["code"] for c in result.get("columns", [])]
    code_to_group = {v: k for k, v in keep_groups.items()}

    # Pivot into (year, month) → {all_items, food_nab}
    table: dict[tuple[int, int], dict[str, float]] = {}
    for entry in result.get("data", []):
        rec = dict(zip(cols, entry.get("key", [])))
        v = entry.get("values", [None])[0]
        if v in (None, "..", "-"):
            continue
        y = yr_to_year.get(rec.get("Year"))
        m = per_to_month.get(rec.get("Period"))
        g = code_to_group.get(rec.get("Commodity Description"))
        if not (y and m and g):
            continue
        try:
            table.setdefault((y, m), {})[g] = float(v)
        except (TypeError, ValueError):
            pass

    # Build monthly frame
    monthly = []
    for (y, m), d in sorted(table.items()):
        all_items = d.get("all_items")
        food = d.get("food_nab")
        non_food = None
        # Approximate non-food weighted: with 2018 weights ~ food=37%, non-food=63%
        # non_food_idx = (all_items - 0.37 * food) / 0.63 (CALABARZON-typical)
        if all_items is not None and food is not None:
            non_food = (all_items - 0.37 * food) / 0.63
        monthly.append({"year": y, "month": m,
                        "cpi_all_items": all_items,
                        "cpi_food_nab": food,
                        "cpi_non_food": non_food})

    mdf = pd.DataFrame(monthly).sort_values(["year", "month"]).reset_index(drop=True)
    if len(mdf) == 0:
        logger.warning("No CPI rows returned from PSA")
        return pd.DataFrame()

    # YoY (12-month lag) on all_items and food
    mdf["cpi_all_yoy_pct"] = mdf["cpi_all_items"].pct_change(periods=12) * 100
    mdf["cpi_food_yoy_pct"] = mdf["cpi_food_nab"].pct_change(periods=12) * 100
    mdf["cpi_food_minus_general_yoy"] = mdf["cpi_food_yoy_pct"] - mdf["cpi_all_yoy_pct"]
    mdf["cpi_food_share"] = mdf["cpi_food_nab"] / mdf["cpi_all_items"]
    mdf["quarter"] = mdf.apply(lambda r: f"{r['year']}-Q{(r['month']-1)//3+1}", axis=1)

    # Aggregate to quarterly mean, restrict to start_year..end_year
    qdf = mdf.groupby(["year", "quarter"], as_index=False).agg({
        "cpi_all_items": "mean",
        "cpi_food_nab": "mean",
        "cpi_non_food": "mean",
        "cpi_all_yoy_pct": "mean",
        "cpi_food_yoy_pct": "mean",
        "cpi_food_minus_general_yoy": "mean",
        "cpi_food_share": "mean",
    })
    qdf = qdf[(qdf["year"] >= start_year) & (qdf["year"] <= end_year)].reset_index(drop=True)

    # Cross-join to provinces
    rows = []
    for _, qr in qdf.iterrows():
        for prov_code, prov_name in CALABARZON:
            row = qr.to_dict()
            row.update({
                "province_code": prov_code,
                "province_name": prov_name,
                "region_code": REGION_CODE,
                "region_name": REGION_NAME,
                "source_url": url,
                "source_note": ("PSA OpenStat 2018=100 CPI by Commodity Group, table 0012M4ACP22. "
                                "Region IV-A inherited to provinces. Monthly mean -> quarterly."),
                "fetched_at": fetched_at,
            })
            rows.append(row)
    out = pd.DataFrame(rows)
    logger.info("CPI full: %d rows (%d quarters x 5 provinces)", len(out), qdf.shape[0])
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = fetch_cpi_full(2020, 2025)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"[ok] wrote {OUTPUT_PATH} -- {len(df)} rows")
    if len(df):
        print(df[df["province_name"]=="Cavite"][["quarter","cpi_all_items","cpi_food_nab",
              "cpi_food_yoy_pct","cpi_food_minus_general_yoy"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
