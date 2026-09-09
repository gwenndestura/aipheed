"""
scripts/build_climate_extremes.py
----------------------------------
Province-quarter agro-climatic EXTREMES from NASA POWER daily data.

WHY THIS EXISTS
---------------
The model currently sees rainfall as `rainfall_anomaly_pct`: a quarterly mean of
monthly means. That average is close to blind to the thing that actually
destroys a harvest.

A quarter with three catastrophic typhoon days and otherwise ordinary weather
has almost the same quarterly mean as a uniformly damp quarter. So does a
quarter with a five-week dry spell followed by heavy rain that restores the
average. Crop loss comes from extremes and their timing; a mean erases both.

build_province_rainfall.py requests PRECTOTCORR at temporal/monthly. The same
POWER endpoint serves DAILY data and more parameters, which is all that is
needed to compute the standard extremes indices. This script does that and
leaves the existing monthly rainfall series untouched.

INDICES (ETCCDI-style, the conventional set for agro-climatic stress)
---------------------------------------------------------------------
    cdd_max      longest run of consecutive dry days (< 1 mm) -- drought stress
    rx5day       largest 5-day precipitation total -- flood and washout events
    r20mm        count of very heavy rain days (>= 20 mm)
    wet_days     count of wet days (>= 1 mm)
    heat_days    count of days with T2M_MAX >= 35 C -- heat stress
    precip_total total quarterly precipitation, for reference

Each is also expressed as an anomaly against that province's own normal for the
SAME quarter of the year, computed over NORMAL_START..NORMAL_END. Comparing a
Q3 against other Q3s is the same reasoning that made shock_rate_qoy work: a
pooled comparison across quarters is meaningless in a monsoon climate.

The normal period ends before the model window begins, so no anomaly is computed
against data the model is being scored on. This mirrors the 1991-2020 WMO normal
already used by build_province_rainfall.py.

OUTPUT
------
data/processed/province_climate_extremes.parquet:
    province_code, province_name, year, quarter, <index>, <index>_anom, ...
    source_url, source_note, fetched_at

USAGE
-----
    python scripts/build_climate_extremes.py
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)
log = logging.getLogger("extremes")

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POINTS = Path("data/reference/calabarzon_rainfall_points.csv")
CACHE = Path("data/raw/power_daily.parquet")
OUT = Path("data/processed/province_climate_extremes.parquet")

SERIES_START = "20110101"
NORMAL_START, NORMAL_END = 2011, 2020      # ends before the 2021 model window

DRY_MM = 1.0        # a "dry day" in the ETCCDI definition
HEAVY_MM = 20.0     # R20mm, very heavy precipitation
HEAT_C = 35.0       # heat-stress threshold for tropical field crops

PARAMETERS = "PRECTOTCORR,T2M_MAX"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aiPHeed/1.0 (+research; DLSU-D thesis)"})

INDICES = ["cdd_max", "rx5day", "r20mm", "wet_days", "heat_days", "precip_total"]


def fetch_point_daily(lon: float, lat: float, end: str) -> pd.DataFrame:
    """Daily precipitation and max temperature for one sample point."""
    params = {
        "parameters": PARAMETERS, "community": "AG",
        "longitude": lon, "latitude": lat,
        "start": SERIES_START, "end": end, "format": "JSON",
    }
    for attempt in range(5):
        try:
            r = SESSION.get(POWER_URL, params=params, timeout=180)
            if r.status_code in (429, 503):
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            block = r.json()["properties"]["parameter"]
            df = pd.DataFrame({
                "date": pd.to_datetime(list(block["PRECTOTCORR"]), format="%Y%m%d"),
                "precip_mm": list(block["PRECTOTCORR"].values()),
                "tmax_c": list(block["T2M_MAX"].values()),
            })
            # POWER writes -999 for a missing value; it must not become a
            # zero-rainfall day, which would fabricate a dry spell.
            df.loc[df["precip_mm"] < -100, "precip_mm"] = np.nan
            df.loc[df["tmax_c"] < -100, "tmax_c"] = np.nan
            return df
        except requests.RequestException as exc:
            log.warning("point (%.3f, %.3f) attempt %d failed: %s",
                        lon, lat, attempt + 1, str(exc)[:80])
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"NASA POWER daily failed for ({lon}, {lat})")


def load_daily(end: str) -> pd.DataFrame:
    """All sample points, cached so a re-run does not re-download 30 series."""
    if CACHE.exists():
        cached = pd.read_parquet(CACHE)
        if cached["date"].max() >= pd.Timestamp(end):
            log.info("daily cache is current (%s rows through %s)",
                     len(cached), cached["date"].max().date())
            return cached
        log.info("daily cache ends %s, refetching", cached["date"].max().date())

    pts = pd.read_csv(POINTS)
    frames = []
    for i, row in pts.iterrows():
        d = fetch_point_daily(row["lon"], row["lat"], end)
        d["province_code"] = row["province_code"]
        d["province_name"] = row["province_name"]
        frames.append(d)
        log.info("[%2d/%d] %-10s (%.3f, %.3f) %d days",
                 i + 1, len(pts), row["province_name"], row["lon"], row["lat"], len(d))
        time.sleep(1.0)
    daily = pd.concat(frames, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(CACHE, index=False)
    log.info("cached -> %s (%d rows)", CACHE, len(daily))
    return daily


def _max_consecutive_dry(precip: pd.Series) -> float:
    """Longest run of days below DRY_MM. NaN days break neither run nor count."""
    dry = (precip < DRY_MM).to_numpy()
    best = run = 0
    for d in dry:
        run = run + 1 if d else 0
        best = max(best, run)
    return float(best)


def quarterly_indices(daily: pd.DataFrame) -> pd.DataFrame:
    """Collapse daily point data to province-quarter extremes."""
    # Average the points to a single province-day first, so an index is computed
    # on the province's weather rather than averaged across six separate runs of
    # dry days that never coincided.
    prov = (daily.groupby(["province_code", "province_name", "date"], as_index=False)
                 [["precip_mm", "tmax_c"]].mean())
    prov["year"] = prov["date"].dt.year
    prov["quarter_num"] = prov["date"].dt.quarter

    rows = []
    for (pc, pn, yr, qn), g in prov.groupby(
            ["province_code", "province_name", "year", "quarter_num"]):
        g = g.sort_values("date")
        p = g["precip_mm"]
        rows.append({
            "province_code": pc, "province_name": pn, "year": yr, "quarter_num": qn,
            "cdd_max": _max_consecutive_dry(p),
            "rx5day": float(p.rolling(5, min_periods=5).sum().max()),
            "r20mm": float((p >= HEAVY_MM).sum()),
            "wet_days": float((p >= DRY_MM).sum()),
            "heat_days": float((g["tmax_c"] >= HEAT_C).sum()),
            "precip_total": float(p.sum()),
        })
    q = pd.DataFrame(rows)
    q["quarter"] = q["year"].astype(str) + "-Q" + q["quarter_num"].astype(str)
    return q


def add_anomalies(q: pd.DataFrame) -> pd.DataFrame:
    """Express each index against the province's normal for the same quarter."""
    base = q[(q["year"] >= NORMAL_START) & (q["year"] <= NORMAL_END)]
    normal = (base.groupby(["province_code", "quarter_num"])[INDICES]
                  .mean().add_suffix("_normal").reset_index())
    out = q.merge(normal, on=["province_code", "quarter_num"], how="left")
    for c in INDICES:
        # Difference, not percentage: several of these are counts that are
        # legitimately zero in a normal quarter, and a ratio would explode.
        out[f"{c}_anom"] = out[c] - out[f"{c}_normal"]
    return out.drop(columns=[f"{c}_normal" for c in INDICES])


def main() -> None:
    end = datetime.now(timezone.utc).strftime("%Y%m%d")
    daily = load_daily(end)
    log.info("daily rows: %d | %s .. %s", len(daily),
             daily["date"].min().date(), daily["date"].max().date())

    q = quarterly_indices(daily)
    q = add_anomalies(q)

    q["source_url"] = POWER_URL
    q["source_note"] = (
        "NASA POWER daily PRECTOTCORR and T2M_MAX, averaged over 6 interior "
        "sample points per province, collapsed to ETCCDI-style quarterly "
        f"extremes. Dry day < {DRY_MM} mm, very heavy rain >= {HEAVY_MM} mm, "
        f"heat stress T2M_MAX >= {HEAT_C} C. Anomalies are differences against "
        f"the same province and same quarter-of-year over {NORMAL_START}-"
        f"{NORMAL_END}, a normal period ending before the model window."
    )
    q["fetched_at"] = datetime.now(timezone.utc).isoformat()

    q = q.sort_values(["province_code", "quarter"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    q.to_parquet(OUT, index=False)

    recent = q[q["year"] >= 2024]
    print(recent.pivot_table(index="quarter", columns="province_name",
                             values="cdd_max").to_string())
    log.info("wrote %d province-quarters (%s .. %s) -> %s",
             len(q), q["quarter"].min(), q["quarter"].max(), OUT)


if __name__ == "__main__":
    main()
