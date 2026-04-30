"""
app/ml/corpus/commodity_prices_fetcher.py
------------------------------------------
Multi-commodity retail price fetcher (PSA OpenStat NRP 2012-based series).

PURPOSE
-------
Beyond rice, food insecurity prediction needs basket-level price signals.
Households substitute when one staple spikes. Fish stress (T1b in your
HungerGist taxonomy) and vegetable price volatility are early indicators
of climate-shock-driven insecurity.

PRECEDENT (legitimate references for thesis defense):
    • WFP VAM (Vulnerability Analysis & Mapping) — multi-commodity baskets
    • FAO GIEWS Food Price Monitoring & Analysis (FPMA) — same approach
    • IFPRI Food Price Monitor — uses 6-10 staple commodities per country
    • PSA Monthly Bulletin of Statistics — official commodity coverage

PSA NRP TABLES IN /DB/2M/NRP
----------------------------
    0042M4ARN01.px — Cereals (RICE regular/special/well, CORN GRITS variants)
    0042M4ARN02.px — Rootcrops
    0042M4ARN03.px — Beans and Legumes
    0042M4ARN04.px — Condiments
    0042M4ARN05.px — Fruit Vegetables
    0042M4ARN06.px — Leafy Vegetables
    0042M4ARN07.px — Fruits
    0042M4ARN08.px — Commercial Crops
    0042M4ARN09.px — Livestock (PORK, BEEF, etc.)
    0042M4ARN10.px — Poultry (CHICKEN)
    0042M4ARN11.px — Fish (T1b trigger)

NOTE: The PSA NRP series ENDS AT 2021. For 2022-2025 prices we already
fetch rice via Ricelytics (ricelytics_prices.parquet). For other commodities
(corn, fish, vegetables) post-2021, the model uses the 2020-2021 historical
trajectory as feature time-series anchor; the fitted boosting model does
not require continuous coverage of every commodity.

OUTPUT
------
data/processed/commodity_prices.parquet — columns:
    province_code     : str
    province_name     : str
    year              : int
    quarter           : str
    commodity_group   : str (cereals/fish/leafy_veg/livestock/poultry)
    commodity         : str (specific item, e.g. "PORK_LEAN")
    price_php_per_kg  : float
    source_url        : str
    fetched_at        : str

TRAINING WINDOW: 2020-2021 (PSA NRP coverage); curated regional means
allow extension. Strict 2020-2025 boundary respected.
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

OPENSTAT_BASE = "https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M/NRP"
OUTPUT_PATH = Path("data/processed/commodity_prices.parquet")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aiPHeed/1.0 (+research; DLSU-D thesis)"})

CALABARZON_NAMES = {
    "Cavite": "PH040100000",
    "Laguna": "PH040200000",
    "Quezon": "PH040300000",
    "Rizal": "PH040400000",
    "Batangas": "PH040500000",
}

# Commodity tables to fetch + which specific items (key signals)
TABLES = {
    "0042M4ARN02.px": ("rootcrops", ["KAMOTE", "POTATO", "CASSAVA"]),
    "0042M4ARN05.px": ("fruit_veg", ["TOMATO", "EGGPLANT", "AMPALAYA", "SQUASH"]),
    "0042M4ARN06.px": ("leafy_veg", ["KANGKONG", "PECHAY", "CABBAGE"]),
    "0042M4ARN09.px": ("livestock", ["PORK", "BEEF"]),
    "0042M4ARN10.px": ("poultry", ["CHICKEN"]),
    "0042M4ARN11.px": ("fish", ["BANGUS", "TILAPIA", "GALUNGGONG"]),
}


def _meta(table_id: str) -> dict:
    r = SESSION.get(f"{OPENSTAT_BASE}/{table_id}", timeout=30)
    return json.loads(r.content.decode("utf-8-sig"))


def _post(table_id: str, body: dict) -> list[dict]:
    r = SESSION.post(f"{OPENSTAT_BASE}/{table_id}", json=body, timeout=30)
    r.raise_for_status()
    time.sleep(0.4)
    result = json.loads(r.content.decode("utf-8-sig"))
    cols = [c["code"] for c in result.get("columns", [])]
    rows = []
    for entry in result.get("data", []):
        rec = dict(zip(cols, entry.get("key", [])))
        rec["_value"] = entry.get("values", [None])[0]
        rows.append(rec)
    return rows


def _find_geo_codes(meta: dict) -> dict[str, str]:
    """Return {province_name: geo_code} for CALABARZON entries in this table."""
    out = {}
    for v in meta.get("variables", []):
        if v.get("code") in ("Geolocation", "Region/Province"):
            for code, text in zip(v.get("values", []), v.get("valueTexts", [])):
                clean = text.lstrip(".").strip()
                for prov_name in CALABARZON_NAMES:
                    if clean == prov_name:
                        out[prov_name] = code
            break
    return out


def _quarter(month_text: str) -> str | None:
    months = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
              "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
    m = months.get(month_text)
    return f"Q{(m - 1) // 3 + 1}" if m else None


def fetch_commodity_prices(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    """Fetch CALABARZON commodity prices across PSA NRP tables 02-11."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    all_rows = []

    for table_id, (group_name, target_items) in TABLES.items():
        try:
            meta = _meta(table_id)
            geo_map = _find_geo_codes(meta)
            if not geo_map:
                logger.warning("No CALABARZON geos in %s — skipping", table_id)
                continue

            # find year codes within window
            year_var = next((v for v in meta["variables"] if v["code"] == "year"), None)
            yr_filter = []
            yr_to_year = {}
            if year_var:
                for c, t in zip(year_var["values"], year_var["valueTexts"]):
                    try:
                        y = int(t)
                        if start_year <= y <= end_year:
                            yr_filter.append(c)
                            yr_to_year[c] = y
                    except ValueError:
                        pass
            if not yr_filter:
                logger.info("%s has no data in %d-%d (max year covered = %s)",
                            table_id, start_year, end_year,
                            year_var["valueTexts"][-1] if year_var else "?")
                continue

            # commodity filter — match target items
            comm_var = next((v for v in meta["variables"] if v["code"] == "Commodity"), None)
            if not comm_var:
                continue
            comm_codes = []
            comm_text_by_code = {}
            for c, t in zip(comm_var["values"], comm_var["valueTexts"]):
                if any(item in t.upper() for item in target_items):
                    comm_codes.append(c)
                    comm_text_by_code[c] = t

            # period — keep monthly (skip Annual)
            per_var = next((v for v in meta["variables"] if v["code"] == "period"), None)
            per_codes = []
            per_text_by_code = {}
            if per_var:
                for c, t in zip(per_var["values"], per_var["valueTexts"]):
                    if t != "Annual":
                        per_codes.append(c)
                        per_text_by_code[c] = t

            # NRP tables use "Region/Province" as the geo variable code
            body = {
                "query": [
                    {"code": "Region/Province", "selection": {"filter": "item",
                                                              "values": list(geo_map.values())}},
                    {"code": "Commodity", "selection": {"filter": "item", "values": comm_codes}},
                    {"code": "year", "selection": {"filter": "item", "values": yr_filter}},
                    {"code": "period", "selection": {"filter": "item", "values": per_codes}},
                ],
                "response": {"format": "json"},
            }
            rows = _post(table_id, body)
            geo_to_prov = {v: k for k, v in geo_map.items()}

            for r in rows:
                geo = r.get("Region/Province") or r.get("Geolocation")
                yr = yr_to_year.get(r.get("year"))
                comm = comm_text_by_code.get(r.get("Commodity"), "?")
                per_text = per_text_by_code.get(r.get("period"))
                v = r.get("_value")
                prov_name = geo_to_prov.get(geo)
                quarter = _quarter(per_text) if per_text else None
                if not (prov_name and yr and quarter and v not in (None, "..", "-")):
                    continue
                try: price = float(v)
                except (TypeError, ValueError): continue
                all_rows.append({
                    "province_code": CALABARZON_NAMES[prov_name],
                    "province_name": prov_name,
                    "year": yr,
                    "quarter": f"{yr}-{quarter}",
                    "commodity_group": group_name,
                    "commodity": comm.strip(),
                    "price_php_per_kg": price,
                    "source_url": f"{OPENSTAT_BASE}/{table_id}",
                    "fetched_at": fetched_at,
                })
            logger.info("%s (%s): %d rows fetched", table_id, group_name,
                        sum(1 for r in all_rows if r["commodity_group"] == group_name))
        except Exception as exc:
            logger.warning("Table %s failed: %s", table_id, exc)

    df = pd.DataFrame(all_rows)
    if len(df):
        # Aggregate monthly → quarterly mean per (province, group, commodity, quarter)
        df = df.groupby(
            ["province_code", "province_name", "year", "quarter",
             "commodity_group", "commodity", "source_url", "fetched_at"],
            as_index=False)["price_php_per_kg"].mean()
        df["source_note"] = "PSA OpenStat NRP 2012-based series — quarterly mean of monthly retail prices"
    logger.info("Commodity prices: %d rows across %d groups", len(df),
                df["commodity_group"].nunique() if len(df) else 0)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = fetch_commodity_prices(2020, 2025)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"[ok] wrote {OUTPUT_PATH} — {len(df)} rows")
    if len(df):
        print(df.groupby("commodity_group")["price_php_per_kg"].agg(["count", "mean"]).round(2))


if __name__ == "__main__":
    main()
