"""
scripts/add_area_yield_features.py
----------------------------------
Add area and yield features to the food-availability panel.

Production = area x yield. The model has never seen either term, only the
product. PSA publishes area planted/harvested at the same province-quarter-crop
resolution as production:

    DB/2E/CS/0112E4EAHM3.px   vegetables and root crops
    DB/2E/CS/0102E4EAHM2.px   fruit crops

Whether a shortfall comes from less land or from worse yield per hectare is
mechanically informative, and the two behave differently: area responds to
planting decisions and land conversion, yield to weather, pests and disease.

Leakage discipline. Area harvested is measured alongside the harvest itself, so
the CURRENT quarter's area is not safely available when nowcasting the current
quarter's production. Only LAGGED terms are used -- t-1 and t-4, plus lagged
yield and lagged area deviation from the series' own seasonal norm. Every
feature added here is known before the quarter being predicted.

Features added:
    area_lag1, area_lag4          area in metric-equivalent units, lagged
    area_dev_lag1                 area vs its own seasonal baseline, lagged
    yield_lag1, yield_lag4        production / area, lagged
    yield_dev_lag1                yield vs its own seasonal baseline, lagged
    area_trend                    area_lag1 / area_lag4
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("area_yield")

OPENSTAT = "https://openstat.psa.gov.ph/PXWeb/api/v1/en"
PANEL = Path("data/processed/food_availability_panel.parquet")
OUT = Path("data/processed/food_availability_panel_v2.parquet")

SOURCES = [("DB/2E/CS/0112E4EAHM3.px", "vegetables_rootcrops"),
           ("DB/2E/CS/0102E4EAHM2.px", "fruit_crops")]

PROVINCES = {"Batangas": "PH040500000", "Cavite": "PH040100000",
             "Laguna": "PH040200000", "Quezon": "PH040300000",
             "Rizal": "PH040400000"}
START_YEAR, END_YEAR = 2020, 2026
DELAY, RETRIES = 1.5, 5

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aiPHeed/1.0 (+research; DLSU-D thesis)"})


def _req(method: str, url: str, **kw):
    for attempt in range(RETRIES):
        time.sleep(DELAY)
        r = SESSION.request(method, url, **kw)
        if r.status_code in (429, 503):
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return json.loads(r.content.decode("utf-8-sig"))
    raise RuntimeError(f"throttled: {url}")


def fetch_area(table: str, group: str, chunk: int = 30) -> pd.DataFrame:
    meta = _req("GET", f"{OPENSTAT}/{table}", timeout=90)
    variables = meta["variables"]
    codes = [v["code"] for v in variables]
    crop, geo, yr, per = variables

    prov = {c: t.strip(". ") for c, t in zip(geo["values"], geo["valueTexts"])
            if t.strip(". ") in PROVINCES}
    years = [c for c, t in zip(yr["values"], yr["valueTexts"])
             if t.strip().isdigit() and START_YEAR <= int(t) <= END_YEAR]
    ytxt = dict(zip(yr["values"], yr["valueTexts"]))
    quarters = [c for c, t in zip(per["values"], per["valueTexts"]) if "uarter" in t]
    qtxt = dict(zip(per["values"], per["valueTexts"]))

    cand = [(c, t) for c, t in zip(crop["values"], crop["valueTexts"])
            if not t.strip().startswith("..")]
    ctxt = dict(cand)
    ids = [c for c, _ in cand]

    rows = []
    for i in range(0, len(ids), chunk):
        batch = ids[i:i + chunk]
        query = [
            {"code": codes[0], "selection": {"filter": "item", "values": batch}},
            {"code": codes[1], "selection": {"filter": "item", "values": list(prov)}},
            {"code": codes[2], "selection": {"filter": "item", "values": years}},
            {"code": codes[3], "selection": {"filter": "item", "values": quarters}},
        ]
        try:
            data = _req("POST", f"{OPENSTAT}/{table}",
                        json={"query": query, "response": {"format": "json"}}, timeout=180)
        except Exception as exc:
            log.warning("%s batch skipped (%s)", group, str(exc)[:60])
            continue
        for item in data["data"]:
            k = item["key"]
            try:
                area = float(item["values"][0])
            except (TypeError, ValueError):
                continue
            rows.append({"group": group, "commodity": ctxt[k[0]].strip(". "),
                         "province_code": PROVINCES[prov[k[1]]],
                         "year": int(ytxt[k[2]]),
                         "quarter_num": int("".join(ch for ch in qtxt[k[3]] if ch.isdigit())),
                         "area": area})
    df = pd.DataFrame(rows)
    log.info("%-22s %5d area rows | %3d commodities", group, len(df),
             df["commodity"].nunique() if len(df) else 0)
    return df


def main() -> None:
    panel = pd.read_parquet(PANEL)
    area = pd.concat([fetch_area(t, g) for t, g in SOURCES], ignore_index=True)

    key = ["group", "commodity", "province_code"]
    df = panel.merge(area, on=key + ["year", "quarter_num"], how="left")
    log.info("area matched for %d of %d panel rows (%.0f%%)",
             int(df["area"].notna().sum()), len(df),
             100 * df["area"].notna().mean())

    df = df.sort_values(key + ["year", "quarter_num"]).reset_index(drop=True)
    g = df.groupby(key)

    # Yield = production per unit area, both from the same quarter.
    df["yield_raw"] = np.where((df["area"] > 0) & df["area"].notna(),
                               df["volume"] / df["area"], np.nan)

    # Seasonal baselines for area and yield, prior in-window years only.
    for col in ("area", "yield_raw"):
        df[f"{col}_base"] = (df.groupby(key + ["quarter_num"])[col]
                               .transform(lambda s: s.shift(1).expanding(min_periods=1).mean()))
        df[f"{col}_dev"] = 100.0 * (df[col] - df[f"{col}_base"]) / df[f"{col}_base"]

    # Only lagged terms: current-quarter area is measured with the harvest and is
    # not safely available when nowcasting that quarter's production.
    df["area_lag1"] = g["area"].shift(1)
    df["area_lag4"] = g["area"].shift(4)
    df["area_dev_lag1"] = g["area_dev"].shift(1)
    df["yield_lag1"] = g["yield_raw"].shift(1)
    df["yield_lag4"] = g["yield_raw"].shift(4)
    df["yield_dev_lag1"] = g["yield_raw_dev"].shift(1)
    df["area_trend"] = df["area_lag1"] / df["area_lag4"].replace(0, np.nan)

    drop = ["area", "yield_raw", "area_base", "area_dev",
            "yield_raw_base", "yield_raw_dev"]
    df = df.drop(columns=[c for c in drop if c in df.columns])

    df.to_parquet(OUT, index=False)
    added = ["area_lag1", "area_lag4", "area_dev_lag1", "yield_lag1",
             "yield_lag4", "yield_dev_lag1", "area_trend"]
    log.info("saved %d rows -> %s", len(df), OUT)
    print("\ncoverage of the new features:")
    for c in added:
        print(f"  {c:18s} {df[c].notna().mean() * 100:5.1f}% non-null")


if __name__ == "__main__":
    main()
