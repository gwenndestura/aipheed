"""
scripts/extend_province_rainfall.py
------------------------------------
Extend province_rainfall.parquet past the NASA POWER monthly cut-off.

build_province_rainfall.py uses the POWER *monthly* endpoint, which reports
"The data is available to 2025/12/31" and returns HTTP 422 beyond it. The
*daily* endpoint carries the same MERRA-2 PRECTOTCORR parameter and is current
through mid-2026, so the recent quarters are measurable -- they just have to be
aggregated here rather than server-side.

Method, deliberately identical to the monthly build except for the endpoint:

  1. Reuse the exact sample points in data/reference/calabarzon_rainfall_points.csv
     so a province is represented by the same geometry as its history.
  2. Query daily PRECTOTCORR (mm/day) at each point.
  3. Mean the days within a month -> point-month mm/day; mean the points ->
     province-month; mean the months -> province-quarter. A mean of daily
     mm/day over a month is the same quantity the monthly endpoint returns.
  4. Anomaly against the SAME 1991-2020 WMO normal already stored per
     province-quarter -- not recomputed, so the new rows sit on the identical
     baseline as every historical row.

Only whole quarters are written. A quarter with a missing month is dropped
rather than annualised from partial data.

Nothing is inherited, scaled or interpolated between provinces. Rows carry a
source_note naming the daily endpoint so the provenance difference is visible
in the data, not just in this docstring.

Usage
-----
    python scripts/extend_province_rainfall.py --through 2026-Q2
    python scripts/extend_province_rainfall.py --through 2026-Q2 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extend_rainfall")
logging.getLogger("urllib3").setLevel(logging.WARNING)

RAINFALL = ROOT / "data" / "processed" / "province_rainfall.parquet"
POINTS = ROOT / "data" / "reference" / "calabarzon_rainfall_points.csv"
BACKUP = ROOT / "data" / "processed" / "province_rainfall.premonthly_extend.parquet"

DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAM = "PRECTOTCORR"

SOURCE_NOTE = (
    "NASA POWER PRECTOTCORR (Precipitation Corrected, mm/day), DAILY endpoint "
    "averaged to province-month then quarter, over the same interior sample "
    "points as the 1991-2025 monthly series. Used because the POWER monthly "
    "endpoint stops at 2025-12-31. Anomaly is against the same 1991-2020 WMO "
    "normal as every historical row."
)


def quarter_of(month: int) -> int:
    return (month - 1) // 3 + 1


def fetch_point(lon: float, lat: float, start: str, end: str) -> pd.Series:
    """Daily mm/day for one point, indexed by date. -999 is POWER's null."""
    params = dict(parameters=PARAM, community="AG", longitude=lon, latitude=lat,
                  start=start, end=end, format="JSON")
    for attempt in range(4):
        try:
            r = requests.get(DAILY_URL, params=params, timeout=120)
            if r.status_code == 200:
                raw = r.json()["properties"]["parameter"][PARAM]
                s = pd.Series(raw, dtype="float64")
                s.index = pd.to_datetime(s.index, format="%Y%m%d")
                return s[s > -900]
            if r.status_code == 422:
                raise SystemExit(f"POWER rejected the range: {r.text[:200]}")
            log.warning("point (%s,%s) HTTP %s, retrying", lon, lat, r.status_code)
        except requests.RequestException as exc:
            log.warning("point (%s,%s) %s, retrying", lon, lat, type(exc).__name__)
        time.sleep(2 * (attempt + 1))
    raise SystemExit(f"POWER unreachable for point ({lon},{lat}); aborting rather "
                     "than writing a province mean from fewer points than its history.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip())
    ap.add_argument("--through", default="2026-Q2", help="last quarter to add, e.g. 2026-Q2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    hist = pd.read_parquet(RAINFALL)
    last_hist = max(hist["quarter"])
    log.info("existing series: %s .. %s (%d rows)", min(hist["quarter"]), last_hist, len(hist))

    end_year, end_q = int(args.through[:4]), int(args.through[-1])
    start_year = int(last_hist[:4]) + 1
    if f"{start_year}-Q1" > args.through:
        log.info("nothing to add; %s already covers %s", last_hist, args.through)
        return

    start = f"{start_year}0101"
    end_month = end_q * 3
    end_day = pd.Timestamp(year=end_year, month=end_month, day=1).days_in_month
    end = f"{end_year}{end_month:02d}{end_day:02d}"
    log.info("fetching daily %s .. %s", start, end)

    points = pd.read_csv(POINTS)
    log.info("sample points: %d across %d provinces",
             len(points), points["province_code"].nunique())

    rows = []
    for i, p in enumerate(points.itertuples(index=False), 1):
        s = fetch_point(p.lon, p.lat, start, end)
        m = s.groupby([s.index.year, s.index.month]).agg(["mean", "size"])
        for (yr, mo), rec in m.iterrows():
            rows.append({"province_code": p.province_code,
                         "province_name": p.province_name,
                         "year": yr, "month": mo,
                         "mm_day": rec["mean"], "days": int(rec["size"])})
        log.info("  point %d/%d done", i, len(points))

    daily = pd.DataFrame(rows)

    # A month must be complete. A partial month would bias the quarter mean.
    expected = daily.apply(
        lambda r: pd.Timestamp(year=int(r["year"]), month=int(r["month"]), day=1).days_in_month,
        axis=1)
    incomplete = daily[daily["days"] < expected]
    if not incomplete.empty:
        drop = set(zip(incomplete["year"], incomplete["month"]))
        log.warning("dropping %d incomplete month(s): %s", len(drop), sorted(drop))
        daily = daily[~daily.set_index(["year", "month"]).index.isin(drop)]

    prov_month = (daily.groupby(["province_code", "province_name", "year", "month"])
                  ["mm_day"].mean().reset_index())
    prov_month["q"] = prov_month["month"].map(quarter_of)

    q = (prov_month.groupby(["province_code", "province_name", "year", "q"])
         .agg(precip_mm_day=("mm_day", "mean"), n_months=("month", "nunique"))
         .reset_index())
    q = q[q["n_months"] == 3]                        # whole quarters only
    q["quarter"] = q["year"].astype(str) + "-Q" + q["q"].astype(str)
    q = q[q["quarter"] <= args.through]

    # Same 1991-2020 normal the history uses, keyed by province and quarter-of-year.
    normals = (hist.assign(qq=hist["quarter"].str[-1])
               .groupby(["province_code", "qq"])["normal_mm_day"].first().reset_index())
    q = q.merge(normals, left_on=["province_code", q["q"].astype(str)],
                right_on=["province_code", "qq"], how="left").drop(columns=["qq", "key_1"],
                                                                   errors="ignore")
    if q["normal_mm_day"].isna().any():
        raise SystemExit("missing a 1991-2020 normal for a province-quarter; aborting.")

    q["rainfall_anomaly_pct"] = (
        100 * (q["precip_mm_day"] - q["normal_mm_day"]) / q["normal_mm_day"]).round(2)
    q["geographic_level"] = "province_measured"
    q["province_varying"] = True
    q["source_url"] = DAILY_URL
    q["source_note"] = SOURCE_NOTE
    q["fetched_at"] = datetime.now(timezone.utc).isoformat()

    out = q[[c for c in hist.columns if c in q.columns]]
    log.info("new rows:\n%s", out[["province_name", "quarter", "precip_mm_day",
                                  "normal_mm_day", "rainfall_anomaly_pct"]].to_string(index=False))

    if args.dry_run:
        log.info("--dry-run: nothing written.")
        return

    if not BACKUP.exists():
        shutil.copy2(RAINFALL, BACKUP)
        log.info("backed up -> %s", BACKUP.name)

    combined = pd.concat([hist, out], ignore_index=True).sort_values(
        ["province_code", "quarter"]).reset_index(drop=True)
    combined.to_parquet(RAINFALL, index=False)
    log.info("wrote %s | %s .. %s (%d rows)", RAINFALL.name,
             min(combined["quarter"]), max(combined["quarter"]), len(combined))


if __name__ == "__main__":
    main()
