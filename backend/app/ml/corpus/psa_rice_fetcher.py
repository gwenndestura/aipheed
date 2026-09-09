"""
app/ml/corpus/psa_rice_fetcher.py
----------------------------------
CALABARZON province-quarter rice prices, fetched live from PSA OpenStat.

WHY THIS REPLACES THE OLD RICE PATH
-----------------------------------
rice_price_regular used to be assembled from two sources spliced at 2021:

    2020-2021   PSA NRP retail prices (real)
    2022-2025   Ricelytics -- in practice a curated fallback table baked into
                app/ml/corpus/ricelytics_fetcher.py, declared window 2022-2025

That left 2021 with no value at all once the retail series was joined on the
province grid, and it stopped dead at 2025-Q4. The PSA NRP retail table really
does end at 2021 -- it has not been extended since the 2012-based series was
retired -- so no amount of re-fetching would have fixed it.

PSA publishes a successor that does cover the whole window:

    DB/2M/NWSNEW/0052M4AWB01.px
    "Cereals: Wholesale Selling Prices of Agricultural Commodities"
    Regular Milled Rice (RMR), by province, monthly, 2010 to present.

This module uses that single series for the entire window. Wholesale rather
than retail is a deliberate choice: one consistent, real, live series beats a
retail-then-curated splice with a level break at the join. The feature is a
price-level covariate, so what matters is that the series is internally
consistent and actually observed.

RETAIL COMPARISON
-----------------
Wholesale RMR runs roughly PHP 3-6/kg below retail RMR. Anything reading this
feature as an absolute consumer price should adjust for that; the model sees
only relative movement, lagged one quarter.

OUTPUT
------
data/processed/psa_rice_prices.parquet:
    province_code, province_name, year, quarter, price_php_per_kg,
    rice_class, source_url, source_note, fetched_at
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

OPENSTAT = "https://openstat.psa.gov.ph/PXWeb/api/v1/en"
TABLE = "DB/2M/NWSNEW/0052M4AWB01.px"
COMMODITY_TEXT = "Regular Milled Rice (RMR)"

PROVINCES = {"Batangas": "PH040500000", "Cavite": "PH040100000",
             "Laguna": "PH040200000", "Quezon": "PH040300000",
             "Rizal": "PH040400000"}

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

REQUEST_DELAY = 1.5      # OpenStat returns 429 under rapid batching
MAX_RETRIES = 5

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aiPHeed/1.0 (+research; DLSU-D thesis)"})


def _json(r: requests.Response):
    r.raise_for_status()
    return json.loads(r.content.decode("utf-8-sig"))


def _request(method: str, url: str, **kw):
    """GET/POST with backoff on the 429 and 503 OpenStat uses for throttling."""
    for attempt in range(MAX_RETRIES):
        time.sleep(REQUEST_DELAY)
        try:
            r = SESSION.request(method, url, **kw)
            if r.status_code in (429, 503):
                wait = 5 * (attempt + 1)
                logger.info("throttled (%s) -- waiting %ds", r.status_code, wait)
                time.sleep(wait)
                continue
            return _json(r)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (429, 503):
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"{url} still throttled after {MAX_RETRIES} attempts")


def fetch_psa_rice_prices(start_year: int = 2021,
                          end_year: int | None = None) -> pd.DataFrame:
    """
    Quarterly mean wholesale Regular Milled Rice price per CALABARZON province.

    end_year defaults to the current calendar year, so a re-run picks up
    whatever PSA has published since the last one.
    """
    end_year = end_year or datetime.now(timezone.utc).year

    meta = _request("GET", f"{OPENSTAT}/{TABLE}", timeout=90)
    variables = meta["variables"]
    codes = [v["code"] for v in variables]

    geo_i = next(i for i, v in enumerate(variables) if "geo" in v["code"].lower())
    com_i = next(i for i, v in enumerate(variables)
                 if v["code"].lower() == "commodity")
    year_i = next(i for i, v in enumerate(variables) if v["code"].lower() == "year")
    per_i = next(i for i, v in enumerate(variables)
                 if v["code"].lower() in ("period", "quarter"))

    geo = variables[geo_i]
    prov_code_by_value = {c: t.strip(". ")
                          for c, t in zip(geo["values"], geo["valueTexts"])
                          if t.strip(". ") in PROVINCES}
    if len(prov_code_by_value) != len(PROVINCES):
        missing = set(PROVINCES) - set(prov_code_by_value.values())
        raise RuntimeError(f"OpenStat geolocation list is missing {missing} -- "
                           "the province coding has changed")

    com = variables[com_i]
    rmr = [c for c, t in zip(com["values"], com["valueTexts"])
           if t.strip() == COMMODITY_TEXT]
    if not rmr:
        raise RuntimeError(f"{COMMODITY_TEXT!r} is no longer in the commodity "
                           f"list: {com['valueTexts']}")

    yv = variables[year_i]
    years = [c for c, t in zip(yv["values"], yv["valueTexts"])
             if t.strip().isdigit() and start_year <= int(t) <= end_year]
    ytxt = dict(zip(yv["values"], yv["valueTexts"]))
    if not years:
        raise RuntimeError(f"OpenStat has no years in {start_year}..{end_year}")

    pv = variables[per_i]
    months = [c for c, t in zip(pv["values"], pv["valueTexts"]) if t.strip() in MONTHS]
    ptxt = dict(zip(pv["values"], pv["valueTexts"]))

    query = []
    for i, code in enumerate(codes):
        if i == geo_i:
            sel = list(prov_code_by_value)
        elif i == com_i:
            sel = rmr
        elif i == year_i:
            sel = years
        elif i == per_i:
            sel = months
        else:
            sel = [variables[i]["values"][0]]
        query.append({"code": code, "selection": {"filter": "item", "values": sel}})

    data = _request("POST", f"{OPENSTAT}/{TABLE}",
                    json={"query": query, "response": {"format": "json"}},
                    timeout=180)

    rows = []
    for item in data["data"]:
        key = item["key"]
        try:
            price = float(item["values"][0])
        except (TypeError, ValueError):
            continue          # OpenStat writes ".." for a suppressed cell
        if price <= 0:
            continue
        pname = prov_code_by_value[key[geo_i]]
        month = ptxt[key[per_i]].strip()
        rows.append({"province_name": pname,
                     "province_code": PROVINCES[pname],
                     "year": int(ytxt[key[year_i]]),
                     "month": MONTHS.index(month) + 1,
                     "price_php_per_kg": price})

    monthly = pd.DataFrame(rows)
    if monthly.empty:
        raise RuntimeError("OpenStat returned no rice price observations -- "
                           "the table structure has probably changed")

    monthly["quarter_num"] = (monthly["month"] - 1) // 3 + 1
    q = (monthly.groupby(["province_code", "province_name", "year", "quarter_num"],
                         as_index=False)["price_php_per_kg"]
                .mean())
    q["price_php_per_kg"] = q["price_php_per_kg"].round(2)
    q["quarter"] = q["year"].astype(str) + "-Q" + q["quarter_num"].astype(str)
    q = q.drop(columns=["quarter_num"])

    q["rice_class"] = "regular_milled"
    q["source_url"] = f"{OPENSTAT}/{TABLE}"
    q["source_note"] = (
        "PSA OpenStat, Cereals: Wholesale Selling Prices of Agricultural "
        f"Commodities (table {TABLE.rsplit('/', 1)[-1]}), {COMMODITY_TEXT}, "
        "quarterly mean of monthly province-level wholesale prices. Fetched "
        "live. Wholesale runs roughly PHP 3-6/kg below retail; the PSA NRP "
        "retail series was retired after 2021 and cannot cover this window."
    )
    q["fetched_at"] = datetime.now(timezone.utc).isoformat()

    q = q.sort_values(["province_code", "quarter"]).reset_index(drop=True)
    logger.info("psa_rice: %d province-quarters, %s .. %s, from %d monthly obs",
                len(q), q["quarter"].min(), q["quarter"].max(), len(monthly))
    return q


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    frame = fetch_psa_rice_prices(2021)
    dest = Path("data/processed/psa_rice_prices.parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest, index=False)
    print(frame.pivot_table(index="quarter", columns="province_name",
                            values="price_php_per_kg").to_string())
    print(f"\nwrote {len(frame)} rows -> {dest}")
