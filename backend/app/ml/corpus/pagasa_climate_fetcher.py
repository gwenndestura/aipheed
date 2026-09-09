"""
app/ml/corpus/pagasa_climate_fetcher.py
----------------------------------------
CALABARZON province-quarter climate features, fetched live.

WHAT CHANGED AND WHY
--------------------
This module used to return a hand-typed table: 48 literal province-quarter rows
plus a literal ENSO timeline, both ending at 2025-Q4. Nothing was fetched. That
is why every downstream series stopped on the same quarter, and why the feature
matrix could not be advanced without someone editing Python by hand.

Both signals now come from the authoritative archives:

    ENSO   NOAA CPC Oceanic Nino Index (ONI), the 3-month running mean sea
           surface temperature anomaly in the Nino 3.4 region. This is the
           index PAGASA cites when it declares an El Nino or La Nina alert.
           https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

    TCs    NOAA IBTrACS v04r01, Western Pacific basin -- the WMO-endorsed
           best-track archive, which ingests the JTWC and JMA tracks that
           PAGASA bulletins are built from. Storm positions are matched
           against each province rather than assigned by a hand-written
           exposure ranking.

Both archives update continuously, so end_year defaults to the current year and
the series extends itself as they do.

RAINFALL
--------
rainfall_anomaly_pct is emitted as null here, deliberately. The feature matrix
prefers data/processed/province_rainfall.parquet (NASA POWER, measured, province
level) and falls back to this column only when that file is absent, so writing a
guess here would displace a real measurement. See _load_pagasa in
app/ml/features/feature_matrix.py.

OUTPUT
------
data/processed/pagasa_climate.parquet, schema unchanged:
    province_code, province_name, year, quarter, tc_count, tc_max_signal,
    tc_severe_flag, rainfall_anomaly_pct, enso_phase, enso_intensity,
    drought_alert, geographic_level, province_varying, source_url,
    source_note, fetched_at
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
IBTRACS_URL = ("https://www.ncei.noaa.gov/data/"
               "international-best-track-archive-for-climate-stewardship-ibtracs/"
               "v04r01/access/csv/ibtracs.WP.list.v04r01.csv")

# The WP best-track file is ~115 MB, so it is cached on disk and refreshed only
# when the cache is older than this. A same-week re-run costs nothing.
CACHE = Path("data/raw/ibtracs_wp.csv")
CACHE_MAX_AGE_DAYS = 7

# Province centroids (WGS84). Geographic constants, not statistics -- there is
# nothing here that goes out of date.
PROVINCE_CENTROIDS = {
    "PH040100000": ("Cavite", 14.28, 120.88),
    "PH040200000": ("Laguna", 14.17, 121.33),
    "PH040300000": ("Quezon", 13.93, 122.11),
    "PH040400000": ("Rizal", 14.60, 121.30),
    "PH040500000": ("Batangas", 13.79, 121.06),
}

# A storm counts for a province when its track passes within this distance of
# the province centroid. 300 km approximates the radius over which a Western
# Pacific tropical cyclone produces damaging wind and rain, and is the scale at
# which PAGASA places whole provinces under a wind signal.
TC_RADIUS_KM = 300.0

# NOAA declares an episode at |ONI| >= 0.5 sustained over five overlapping
# seasons; the intensity bands are CPC definitions.
ENSO_THRESHOLD = 0.5
INTENSITY_BANDS = [(2.0, "VERY_STRONG"), (1.5, "STRONG"),
                   (1.0, "MODERATE"), (0.5, "WEAK")]

# PAGASA Tropical Cyclone Wind Signal by 10-minute sustained wind, converted
# from the km/h bands in the TCWS definition to the knots IBTrACS reports.
TCWS_BANDS_KT = [(100, 5), (64, 4), (49, 3), (34, 2), (21, 1)]

# ONI rows are 3-month running means labelled by their centre month.
SEASON_CENTRE_MONTH = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                       "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aiPHeed/1.0 (+research; DLSU-D thesis)"})


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _classify_enso(oni: float) -> tuple[str, str]:
    if oni >= ENSO_THRESHOLD:
        phase = "EL_NINO"
    elif oni <= -ENSO_THRESHOLD:
        phase = "LA_NINA"
    else:
        return "NEUTRAL", "NEUTRAL"
    mag = abs(oni)
    for cut, name in INTENSITY_BANDS:
        if mag >= cut:
            return phase, name
    return phase, "WEAK"


def fetch_oni(start_year: int, end_year: int) -> pd.DataFrame:
    """Quarterly ENSO state from the NOAA CPC Oceanic Nino Index."""
    r = SESSION.get(ONI_URL, timeout=60)
    r.raise_for_status()

    rows = []
    for line in r.text.splitlines()[1:]:
        parts = line.split()
        if len(parts) != 4 or parts[0] not in SEASON_CENTRE_MONTH:
            continue
        try:
            rows.append({"year": int(parts[1]),
                         "month": SEASON_CENTRE_MONTH[parts[0]],
                         "oni": float(parts[3])})
        except ValueError:
            continue

    m = pd.DataFrame(rows)
    if m.empty:
        raise RuntimeError(f"ONI feed at {ONI_URL} returned no parsable rows -- "
                           "the format has changed")
    m["quarter"] = "Q" + ((m["month"] - 1) // 3 + 1).astype(str)

    q = m.groupby(["year", "quarter"], as_index=False)["oni"].mean()
    q = q[(q["year"] >= start_year) & (q["year"] <= end_year)].reset_index(drop=True)
    q[["enso_phase", "enso_intensity"]] = q["oni"].apply(
        lambda v: pd.Series(_classify_enso(v)))
    # PAGASA issues drought advisories on a sustained moderate-or-stronger El
    # Nino, so the alert tracks that band rather than being a separate judgement.
    q["drought_alert"] = ((q["enso_phase"] == "EL_NINO") & (q["oni"] >= 1.0)).astype(int)

    logger.info("ONI: %d quarters, %s-%s .. %s-%s", len(q),
                q["year"].min(), q["quarter"].min(),
                q["year"].max(), q["quarter"].max())
    return q


def _ibtracs_frame() -> pd.DataFrame:
    """Best-track positions, cached on disk because the feed is ~115 MB."""
    stale = True
    if CACHE.exists():
        age_days = (datetime.now(timezone.utc).timestamp()
                    - CACHE.stat().st_mtime) / 86400
        stale = age_days > CACHE_MAX_AGE_DAYS
        logger.info("IBTrACS cache is %s (%.1f days old)",
                    "stale" if stale else "fresh", age_days)

    if stale:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading IBTrACS WP best track (~115 MB)...")
        with SESSION.get(IBTRACS_URL, timeout=900, stream=True) as r:
            r.raise_for_status()
            tmp = CACHE.with_suffix(".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            tmp.replace(CACHE)
        logger.info("cached -> %s (%.0f MB)", CACHE, CACHE.stat().st_size / 1e6)

    # Row 0 is the header; row 1 is a units row that must be skipped.
    df = pd.read_csv(CACHE, skiprows=[1], low_memory=False,
                     usecols=["SID", "SEASON", "NAME", "ISO_TIME",
                              "LAT", "LON", "WMO_WIND"])
    for c in ("LAT", "LON", "WMO_WIND", "SEASON"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["LAT", "LON"])


def fetch_tropical_cyclones(start_year: int, end_year: int) -> pd.DataFrame:
    """Province-quarter storm counts and peak wind signal from IBTrACS."""
    df = _ibtracs_frame()
    df = df[(df["SEASON"] >= start_year - 1) & (df["SEASON"] <= end_year)].copy()

    # Cheap bounding box before the per-point distance computation.
    df = df[df["LAT"].between(10.0, 19.0) & df["LON"].between(116.0, 127.0)]
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df = df.dropna(subset=["ISO_TIME"])
    logger.info("IBTrACS: %d track points in the Philippine box", len(df))

    frames = []
    for pcode, (pname, plat, plon) in PROVINCE_CENTROIDS.items():
        d = df.copy()
        d["dist_km"] = [_haversine_km(plat, plon, la, lo)
                        for la, lo in zip(d["LAT"], d["LON"])]
        near = d[d["dist_km"] <= TC_RADIUS_KM]
        if near.empty:
            continue
        near = near.assign(year=near["ISO_TIME"].dt.year,
                           quarter="Q" + near["ISO_TIME"].dt.quarter.astype(str))
        # A single storm sits in the radius for many 3-hourly points, so count
        # distinct storm ids rather than points.
        agg = (near.groupby(["year", "quarter"])
                   .agg(tc_count=("SID", "nunique"), peak_kt=("WMO_WIND", "max"))
                   .reset_index())
        agg["province_code"] = pcode
        agg["province_name"] = pname
        frames.append(agg)

    if not frames:
        raise RuntimeError("IBTrACS returned no storms near CALABARZON -- the "
                           "feed format has probably changed")
    tc = pd.concat(frames, ignore_index=True)

    def signal(kt: float) -> int:
        if pd.isna(kt):
            return 0
        for cut, sig in TCWS_BANDS_KT:
            if kt >= cut:
                return sig
        return 0

    tc["tc_max_signal"] = tc["peak_kt"].apply(signal)
    tc["tc_severe_flag"] = (tc["tc_max_signal"] >= 3).astype(int)
    return tc.drop(columns=["peak_kt"])


def fetch_pagasa_climate(start_year: int = 2021,
                         end_year: int | None = None) -> pd.DataFrame:
    """
    Build the CALABARZON province-quarter climate frame.

    end_year defaults to the current calendar year, so a re-run picks up every
    quarter the upstream archives have published since the last one.
    """
    end_year = end_year or datetime.now(timezone.utc).year
    oni = fetch_oni(start_year, end_year)
    tc = fetch_tropical_cyclones(start_year, end_year)

    grid = pd.MultiIndex.from_product(
        [list(PROVINCE_CENTROIDS), range(start_year, end_year + 1),
         ["Q1", "Q2", "Q3", "Q4"]],
        names=["province_code", "year", "quarter"]).to_frame(index=False)
    grid["province_name"] = grid["province_code"].map(
        {k: v[0] for k, v in PROVINCE_CENTROIDS.items()})

    out = (grid
           .merge(tc, on=["province_code", "province_name", "year", "quarter"],
                  how="left")
           .merge(oni.drop(columns=["oni"]), on=["year", "quarter"], how="left"))

    # No storm inside the radius is a real zero. A missing ENSO state is not
    # neutral -- it means the index has not been published for that quarter yet,
    # so those rows are dropped rather than filled.
    for c in ("tc_count", "tc_max_signal", "tc_severe_flag"):
        out[c] = out[c].fillna(0).astype(int)

    unresolved = int(out["enso_phase"].isna().sum())
    if unresolved:
        logger.warning("dropping %d province-quarters with no published ONI yet",
                       unresolved)
        out = out.dropna(subset=["enso_phase"])

    out["drought_alert"] = out["drought_alert"].astype(int)
    out["rainfall_anomaly_pct"] = pd.NA
    out["geographic_level"] = "province"
    out["province_varying"] = True
    out["source_url"] = f"{ONI_URL} ; {IBTRACS_URL}"
    out["source_note"] = (
        "ENSO from NOAA CPC Oceanic Nino Index (3-month running Nino 3.4 SST "
        "anomaly), averaged to quarters. Tropical cyclones from NOAA IBTrACS "
        "v04r01 Western Pacific best track: distinct storms passing within "
        f"{TC_RADIUS_KM:.0f} km of the province centroid, peak 10-minute wind "
        "mapped to the PAGASA TCWS band. rainfall_anomaly_pct is null here and "
        "supplied by province_rainfall.parquet (NASA POWER, measured)."
    )
    out["fetched_at"] = datetime.now(timezone.utc).isoformat()

    # The quarter column is the join key against the feature-matrix backbone,
    # which uses the full "YYYY-Qn" label. Composed last because the internal
    # merges above key on (year, quarter) with the bare quarter.
    out["quarter"] = out["year"].astype(str) + "-" + out["quarter"]

    out = out.sort_values(["province_code", "year", "quarter"]).reset_index(drop=True)
    logger.info("pagasa_climate: %d rows, %s-%s .. %s-%s", len(out),
                out["year"].min(), out["quarter"].min(),
                out["year"].max(), out["quarter"].max())
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    frame = fetch_pagasa_climate(2021)
    dest = Path("data/processed/pagasa_climate.parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest, index=False)
    print(frame.groupby(["year", "quarter"])[["tc_count", "tc_severe_flag"]]
               .sum().to_string())
    print(f"\nwrote {len(frame)} rows -> {dest}")
