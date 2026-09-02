"""
scripts/build_agroclimate.py
----------------------------
Full agro-climatic panel for CALABARZON provinces from NASA POWER.

build_province_rainfall.py pulled precipitation only. Rainfall alone does not
explain a crop shortfall: heat stress, soil moisture in the root zone, humidity
(disease pressure) and solar radiation (photosynthesis) are the other standard
drivers, and the same API serves all of them at the same points.

Parameters
    T2M                mean air temperature at 2 m
    T2M_MAX            mean daily maximum -- heat stress
    T2M_MIN            mean daily minimum
    RH2M               relative humidity -- fungal and disease pressure
    GWETROOT           root-zone soil wetness -- what a crop actually draws on
    GWETTOP            surface soil wetness -- germination and establishment
    ALLSKY_SFC_SW_DWN  incident shortwave -- photosynthetically active energy

Same geometry as the rainfall build: interior sample points per province from
geoBoundaries gbOpen PHL ADM2 (NAMRIA / PSA), averaged to a province-month, then
to province-quarters, with anomalies against the 1991-2020 WMO normal for that
province and calendar month.

Output: data/processed/province_agroclimate.parquet
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agroclimate")

POINTS = Path("data/reference/calabarzon_rainfall_points.csv")
OUT = Path("data/processed/province_agroclimate.parquet")
POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

PARAMS = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "GWETROOT", "GWETTOP",
          "ALLSKY_SFC_SW_DWN"]
BASELINE_START, BASELINE_END = 1991, 2020
# NASA POWER monthly data currently ends 2025-12-31.
SERIES_START, SERIES_END = 1991, 2025

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aiPHeed/1.0 (+research; DLSU-D thesis)"})


def fetch_point(lon: float, lat: float) -> dict[str, dict[str, float]]:
    q = {"parameters": ",".join(PARAMS), "community": "AG",
         "longitude": lon, "latitude": lat,
         "start": SERIES_START, "end": SERIES_END, "format": "JSON"}
    for attempt in range(4):
        try:
            r = SESSION.get(POWER_URL, params=q, timeout=150)
            if r.status_code == 200:
                return r.json()["properties"]["parameter"]
            log.warning("POWER http %s at (%.3f,%.3f)", r.status_code, lon, lat)
        except Exception as exc:
            log.warning("POWER error (%.3f,%.3f): %s", lon, lat, str(exc)[:70])
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"NASA POWER failed at ({lon}, {lat})")


def main() -> None:
    if not POINTS.exists():
        raise SystemExit(f"{POINTS} not found — run build_province_rainfall.py first")
    pts = pd.read_csv(POINTS)
    log.info("sampling %d points across %d provinces",
             len(pts), pts["province_name"].nunique())

    records = []
    for n, row in enumerate(pts.itertuples(), 1):
        data = fetch_point(row.lon, row.lat)
        for param, series in data.items():
            for ym, val in series.items():
                if len(ym) != 6 or ym[4:] == "13" or val is None or val <= -900:
                    continue
                records.append({"province_code": row.province_code,
                                "province_name": row.province_name,
                                "year": int(ym[:4]), "month": int(ym[4:]),
                                "param": param, "value": float(val)})
        log.info("[%d/%d] %-9s (%.3f, %.3f)", n, len(pts), row.province_name,
                 row.lon, row.lat)
        time.sleep(0.5)

    raw = pd.DataFrame(records)
    pm = (raw.groupby(["province_code", "province_name", "param", "year", "month"])
             ["value"].mean().reset_index())

    base = (pm[pm["year"].between(BASELINE_START, BASELINE_END)]
              .groupby(["province_code", "param", "month"])["value"]
              .mean().reset_index().rename(columns={"value": "normal"}))
    pm = pm.merge(base, on=["province_code", "param", "month"], how="left")
    pm["anomaly_pct"] = 100.0 * (pm["value"] - pm["normal"]) / pm["normal"]

    pm["quarter"] = pm["year"].astype(str) + "-Q" + ((pm["month"] - 1) // 3 + 1).astype(str)
    q = (pm.groupby(["province_code", "province_name", "quarter", "param"])
           .agg(level=("value", "mean"), anomaly_pct=("anomaly_pct", "mean"))
           .reset_index())

    wide = q.pivot_table(index=["province_code", "province_name", "quarter"],
                         columns="param", values=["level", "anomaly_pct"])
    wide.columns = [f"{p.lower()}_{a}" for a, p in wide.columns]
    wide = wide.reset_index()
    wide["source_url"] = POWER_URL
    wide["source_note"] = (
        "NASA POWER monthly agro-climatic parameters averaged over interior sample "
        "points per province (geoBoundaries gbOpen PHL ADM2, NAMRIA/PSA); anomalies "
        "vs the 1991-2020 WMO normal for the same province and calendar month.")
    wide["fetched_at"] = datetime.now(timezone.utc).isoformat()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUT, index=False)
    log.info("saved %d province-quarters, %d columns -> %s",
             len(wide), len(wide.columns), OUT)
    print("\ncolumns:", [c for c in wide.columns if c.endswith(("_level", "_anomaly_pct"))])


if __name__ == "__main__":
    main()
