"""
scripts/build_province_rainfall.py
----------------------------------
Real per-province rainfall for CALABARZON, replacing the manufactured series.

pagasa_climate_fetcher.py held data for Quezon only and derived the other four
provinces as `anom_qz * (0.5 + 0.5 * shielding_factor)`, producing a
cross-province correlation of exactly 1.0000. This script measures each province
independently instead.

Sources
-------
Boundaries : geoBoundaries gbOpen PHL ADM2 (provinces), sourced by geoBoundaries
             from NAMRIA (National Mapping and Resource Information Authority)
             and the Philippine Statistics Authority.
             https://www.geoboundaries.org/api/current/gbOpen/PHL/ADM2/
Rainfall   : NASA POWER, parameter PRECTOTCORR (Precipitation Corrected, mm/day),
             monthly temporal API, MERRA-2 based.
             https://power.larc.nasa.gov/api/temporal/monthly/point

Method
------
1. Extract each province polygon; take its largest ring by area.
2. Sample an interior point grid (point-in-polygon by ray casting) so a province
   is represented by its area, not a single point.
3. Query NASA POWER monthly precipitation at each sample point.
4. Average points to a province-month series (mm/day).
5. Baseline = 1991-2020 climatological normal for that province and calendar
   month (the WMO standard normals period).
6. rainfall_anomaly_pct = 100 * (month - baseline) / baseline, averaged to
   quarters.

Nothing is scaled, inherited, or inferred from another province.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("province_rainfall")
logging.getLogger("urllib3").setLevel(logging.WARNING)

GEOJSON = Path("data/reference/geoBoundaries-PHL-ADM2.geojson")
OUT = Path("data/processed/province_rainfall.parquet")
POINTS_OUT = Path("data/reference/calabarzon_rainfall_points.csv")

POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
BASELINE_START, BASELINE_END = 1991, 2020     # WMO standard normals
SERIES_START, SERIES_END = 1991, 2025

PROVINCES = {
    "Batangas": "PH040500000",
    "Cavite":   "PH040100000",
    "Laguna":   "PH040200000",
    "Quezon":   "PH040300000",
    "Rizal":    "PH040400000",
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aiPHeed/1.0 (+research; DLSU-D thesis)"})


# ---------------------------------------------------------------------------
# Geometry (no shapely available; these are the two primitives needed)
# ---------------------------------------------------------------------------

def _ring_area(ring: list[list[float]]) -> float:
    """Shoelace area of a lon/lat ring, in squared degrees (sign ignored)."""
    a = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[i + 1][0], ring[i + 1][1]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _largest_ring(geom: dict) -> list[list[float]]:
    """Outer ring of the largest polygon in a Polygon / MultiPolygon."""
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    best, best_area = None, -1.0
    for poly in polys:
        outer = poly[0]
        area = _ring_area(outer)
        if area > best_area:
            best, best_area = outer, area
    return best


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon."""
    inside = False
    n = len(ring)
    for i in range(n - 1):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[i + 1][0], ring[i + 1][1]
        if (y1 > lat) != (y2 > lat):
            xint = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lon < xint:
                inside = not inside
    return inside


def sample_points(ring: list[list[float]], target: int = 6) -> list[tuple[float, float]]:
    """Interior points on a regular grid, so the province is sampled by area."""
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    lo_lon, hi_lon, lo_lat, hi_lat = min(lons), max(lons), min(lats), max(lats)
    for n in range(3, 13):                      # densify until enough land points
        step_lon = (hi_lon - lo_lon) / (n + 1)
        step_lat = (hi_lat - lo_lat) / (n + 1)
        pts = []
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                lon = lo_lon + i * step_lon
                lat = lo_lat + j * step_lat
                if _point_in_ring(lon, lat, ring):
                    pts.append((round(lon, 4), round(lat, 4)))
        if len(pts) >= target:
            # thin evenly to `target` so query cost stays bounded
            stride = max(1, len(pts) // target)
            return pts[::stride][:target]
    return pts


# ---------------------------------------------------------------------------
# NASA POWER
# ---------------------------------------------------------------------------

def fetch_power_monthly(lon: float, lat: float) -> dict[str, float]:
    """Monthly PRECTOTCORR (mm/day) keyed 'YYYYMM'. Retries on transient errors."""
    params = {
        "parameters": "PRECTOTCORR", "community": "AG",
        "longitude": lon, "latitude": lat,
        "start": SERIES_START, "end": SERIES_END, "format": "JSON",
    }
    for attempt in range(4):
        try:
            r = SESSION.get(POWER_URL, params=params, timeout=120)
            if r.status_code == 200:
                data = r.json()["properties"]["parameter"]["PRECTOTCORR"]
                # POWER emits a '13' pseudo-month (annual mean) — drop it.
                return {k: v for k, v in data.items()
                        if len(k) == 6 and k[4:] != "13" and v is not None and v > -900}
            log.warning("POWER http %s at (%.3f, %.3f), retry %d", r.status_code, lon, lat, attempt + 1)
        except Exception as exc:
            log.warning("POWER error at (%.3f, %.3f): %s", lon, lat, str(exc)[:80])
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"NASA POWER failed at ({lon}, {lat})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points-per-province", type=int, default=6)
    args = ap.parse_args()

    if not GEOJSON.exists():
        raise SystemExit(
            f"{GEOJSON} not found. Download from geoBoundaries first:\n"
            "  https://www.geoboundaries.org/api/current/gbOpen/PHL/ADM2/"
        )

    log.info("loading boundaries (large file, this takes a moment)...")
    with GEOJSON.open("r", encoding="utf-8") as fh:
        gj = json.load(fh)

    rings: dict[str, list] = {}
    for feat in gj["features"]:
        name = feat["properties"].get("shapeName")
        if name in PROVINCES:
            rings[name] = _largest_ring(feat["geometry"])
    missing = set(PROVINCES) - set(rings)
    if missing:
        raise SystemExit(f"provinces not found in boundary file: {missing}")

    # --- sample points -----------------------------------------------------
    point_rows = []
    province_points: dict[str, list[tuple[float, float]]] = {}
    for name, ring in rings.items():
        pts = sample_points(ring, target=args.points_per_province)
        province_points[name] = pts
        log.info("%-9s %d boundary vertices -> %d interior sample points",
                 name, len(ring), len(pts))
        for lon, lat in pts:
            point_rows.append({"province_name": name, "province_code": PROVINCES[name],
                               "lon": lon, "lat": lat})
    POINTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(point_rows).to_csv(POINTS_OUT, index=False)
    log.info("sample points -> %s", POINTS_OUT)

    # --- fetch rainfall ----------------------------------------------------
    records = []
    total = sum(len(p) for p in province_points.values())
    done = 0
    for name, pts in province_points.items():
        for lon, lat in pts:
            series = fetch_power_monthly(lon, lat)
            done += 1
            log.info("[%d/%d] %-9s (%.3f, %.3f) -> %d months",
                     done, total, name, lon, lat, len(series))
            for ym, mm_day in series.items():
                records.append({
                    "province_name": name, "province_code": PROVINCES[name],
                    "lon": lon, "lat": lat,
                    "year": int(ym[:4]), "month": int(ym[4:]),
                    "precip_mm_day": float(mm_day),
                })
            time.sleep(0.5)

    raw = pd.DataFrame(records)

    # --- province-month mean over sample points ----------------------------
    pm = (raw.groupby(["province_code", "province_name", "year", "month"])
             ["precip_mm_day"].mean().reset_index())

    # --- 1991-2020 climatological normal per province and calendar month ----
    base = (pm[pm["year"].between(BASELINE_START, BASELINE_END)]
              .groupby(["province_code", "month"])["precip_mm_day"]
              .mean().reset_index().rename(columns={"precip_mm_day": "normal_mm_day"}))
    pm = pm.merge(base, on=["province_code", "month"], how="left")
    pm["anomaly_pct"] = 100.0 * (pm["precip_mm_day"] - pm["normal_mm_day"]) / pm["normal_mm_day"]

    # --- quarterly ---------------------------------------------------------
    pm["quarter"] = pm["year"].astype(str) + "-Q" + ((pm["month"] - 1) // 3 + 1).astype(str)
    q = (pm.groupby(["province_code", "province_name", "year", "quarter"])
           .agg(precip_mm_day=("precip_mm_day", "mean"),
                normal_mm_day=("normal_mm_day", "mean"),
                rainfall_anomaly_pct=("anomaly_pct", "mean"),
                n_months=("month", "count"))
           .reset_index())
    q["rainfall_anomaly_pct"] = q["rainfall_anomaly_pct"].round(2)
    q["geographic_level"] = "province_measured"
    q["province_varying"] = True
    q["source_url"] = POWER_URL
    q["source_note"] = (
        "NASA POWER PRECTOTCORR (Precipitation Corrected, mm/day), monthly, averaged over "
        f"{args.points_per_province} interior sample points per province from geoBoundaries "
        "gbOpen PHL ADM2 (NAMRIA / PSA). Anomaly vs 1991-2020 WMO climatological normal for "
        "the same province and calendar month."
    )
    q["fetched_at"] = datetime.now(timezone.utc).isoformat()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    q.to_parquet(OUT, index=False)
    log.info("saved %d province-quarters -> %s", len(q), OUT)

    # --- report independence ----------------------------------------------
    w = q[q["year"].between(2020, 2025)].pivot_table(
        index="quarter", columns="province_name", values="rainfall_anomaly_pct")
    corr = w.corr()
    off = [corr.iloc[i, j] for i in range(len(corr)) for j in range(len(corr)) if i != j]
    print("\ncross-province correlation of measured rainfall anomaly (2020-2025):")
    print(corr.round(3).to_string())
    print(f"\nmean off-diagonal correlation: {sum(off)/len(off):.4f}  "
          f"(the manufactured series was exactly 1.0000)")


if __name__ == "__main__":
    main()
