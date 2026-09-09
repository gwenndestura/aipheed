"""
app/ml/corpus/oil_price_fetcher.py
-----------------------------------
Quarterly fuel-price features for CALABARZON.

TWO SOURCES, TWO DIFFERENT SITUATIONS
-------------------------------------
Brent crude  LIVE. Pulled from FRED series DCOILBRENTEU (Europe Brent spot,
             sourced by FRED from the US EIA), averaged to quarters. Free, no
             API key, updates daily, so this column extends itself.

DOE pump     NOT LIVE. Philippine retail diesel and gasoline come from the DOE
prices       Oil Industry Update. doe.gov.ph sits behind a WAF that serves the
             first request and then times out, so it cannot be scraped
             dependably. The values below are curated from DOE weekly Oil
             Monitor bulletins and cover 2020-Q1 to 2025-Q4 only.

WHY THIS MODULE NOW REFUSES TO GUESS
------------------------------------
Previously every column here, Brent included, was a hand-typed literal ending
at 2025-Q4, and asking for a later quarter simply returned a shorter frame. The
caller could not tell a complete series from a truncated one, so the whole
feature matrix silently stopped at 2025-Q4 and nobody noticed until the panel
outran it.

fetch_oil_prices now raises when the requested window extends past the curated
DOE coverage. A caller that genuinely wants the truncated series has to say so
with strict=False, which logs a prominent warning and records the gap in
source_note. Degraded output is still available; it just cannot happen by
accident.

TO EXTEND THE DOE SERIES
------------------------
Add rows to DOE_PUMP_QUARTERLY from the quarterly means published in the DOE
Oil Industry Update, then bump the range check. That is manual by necessity --
but it is now the only thing that is.

NOTE: NCR retail pump prices are used as the CALABARZON proxy. CALABARZON is
supplied from the same Petron/Shell/Caltex Pandacan-Tabangao terminals as NCR
and provincial pump prices typically sit within about PHP 0.50/L of it.

OUTPUT
------
data/processed/oil_prices.parquet:
    province_code, province_name, year, quarter, diesel_php_per_l,
    gasoline_php_per_l, diesel_yoy_pct, gasoline_yoy_pct, brent_usd_per_bbl,
    brent_yoy_pct, source_url, source_note, fetched_at
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from app.ml.corpus._coverage import CuratedCoverageError, check_coverage  # noqa: F401

logger = logging.getLogger(__name__)

FRED_BRENT_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
DOE_SOURCE_URL = "https://www.doe.gov.ph/oil-monitor"

PROVINCES = [
    ("PH040100000", "Cavite"),
    ("PH040200000", "Laguna"),
    ("PH040300000", "Quezon"),
    ("PH040400000", "Rizal"),
    ("PH040500000", "Batangas"),
]

# Curated quarterly mean retail pump prices, DOE Oil Industry Update (Common
# Pump Prices in Metro Manila/NCR, weekly Oil Monitor bulletins).
# Format: (year, "Q#", diesel_php_per_l, gasoline_php_per_l)
DOE_PUMP_QUARTERLY = [
    (2020, "Q1", 36.50, 45.20),
    (2020, "Q2", 27.40, 36.10),
    (2020, "Q3", 30.20, 39.50),
    (2020, "Q4", 33.10, 42.80),
    (2021, "Q1", 38.50, 49.20),
    (2021, "Q2", 42.30, 52.80),
    (2021, "Q3", 45.20, 56.10),
    (2021, "Q4", 49.80, 61.20),
    (2022, "Q1", 60.20, 71.80),
    (2022, "Q2", 80.50, 87.40),
    (2022, "Q3", 73.20, 81.50),
    (2022, "Q4", 65.80, 74.20),
    (2023, "Q1", 60.50, 72.80),
    (2023, "Q2", 56.80, 68.50),
    (2023, "Q3", 64.20, 72.50),
    (2023, "Q4", 62.80, 70.20),
    (2024, "Q1", 60.50, 67.50),
    (2024, "Q2", 62.80, 69.20),
    (2024, "Q3", 58.50, 65.20),
    (2024, "Q4", 56.50, 62.80),
    (2025, "Q1", 55.20, 61.50),
    (2025, "Q2", 53.80, 59.80),
    (2025, "Q3", 54.50, 60.50),
    (2025, "Q4", 53.20, 59.20),
]

DOE_COVERAGE_END = (2025, 4)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aiPHeed/1.0 (+research; DLSU-D thesis)"})


def fetch_brent(start_year: int, end_year: int) -> pd.DataFrame:
    """Quarterly mean Brent spot price from FRED (EIA series DCOILBRENTEU)."""
    r = SESSION.get(FRED_BRENT_URL, timeout=90)
    r.raise_for_status()
    if "DOCTYPE" in r.text[:200]:
        raise RuntimeError("FRED returned HTML rather than CSV -- series "
                           "DCOILBRENTEU may have been renamed or retired")

    raw = pd.read_csv(io.StringIO(r.text))
    date_col, value_col = raw.columns[0], raw.columns[1]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    raw = raw.dropna()

    raw["year"] = raw[date_col].dt.year
    raw["quarter"] = "Q" + raw[date_col].dt.quarter.astype(str)
    q = (raw.groupby(["year", "quarter"], as_index=False)[value_col]
            .mean()
            .rename(columns={value_col: "brent_usd_per_bbl"}))
    q["brent_usd_per_bbl"] = q["brent_usd_per_bbl"].round(2)
    q = q[(q["year"] >= start_year) & (q["year"] <= end_year)].reset_index(drop=True)

    logger.info("Brent: %d quarters, %s-%s .. %s-%s", len(q),
                q["year"].min(), q["quarter"].min(),
                q["year"].max(), q["quarter"].max())
    return q


def _check_doe_coverage(end_year: int, strict: bool) -> None:
    check_coverage(
        series="DOE pump prices (diesel_php_per_l, gasoline_php_per_l)",
        coverage_end=DOE_COVERAGE_END, end_year=end_year,
        reason="doe.gov.ph is behind a WAF that serves the first request and "
               "then times out, so it cannot be scraped reliably.",
        source_url=f"the DOE Oil Industry Update ({DOE_SOURCE_URL})",
        extend_hint="raise DOE_COVERAGE_END",
        strict=strict, logger=logger)


def fetch_oil_prices(start_year: int = 2021,
                     end_year: int | None = None,
                     strict: bool = True) -> pd.DataFrame:
    """
    Build the province-quarter fuel frame.

    Brent is fetched live. DOE pump prices are curated, so a request past their
    coverage raises CuratedCoverageError unless strict=False.
    """
    end_year = end_year or datetime.now(timezone.utc).year
    _check_doe_coverage(end_year, strict)

    doe = pd.DataFrame(DOE_PUMP_QUARTERLY,
                       columns=["year", "quarter", "diesel_php_per_l",
                                "gasoline_php_per_l"])
    doe = doe[(doe["year"] >= start_year) & (doe["year"] <= end_year)]

    brent = fetch_brent(start_year, end_year)

    # Outer join: Brent legitimately runs past the curated DOE range, and those
    # quarters should appear with the pump columns null rather than vanish.
    q = doe.merge(brent, on=["year", "quarter"], how="outer")
    q = q.sort_values(["year", "quarter"]).reset_index(drop=True)

    for col, out in (("diesel_php_per_l", "diesel_yoy_pct"),
                     ("gasoline_php_per_l", "gasoline_yoy_pct"),
                     ("brent_usd_per_bbl", "brent_yoy_pct")):
        # fill_method=None: a missing pump price must not be padded forward
        # into a fabricated year-on-year change.
        q[out] = (q[col].pct_change(4, fill_method=None) * 100).round(2)

    frames = []
    for code, name in PROVINCES:
        f = q.copy()
        f.insert(0, "province_name", name)
        f.insert(0, "province_code", code)
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)

    doe_gap = out["diesel_php_per_l"].isna().sum()
    out["source_url"] = f"{FRED_BRENT_URL} ; {DOE_SOURCE_URL}"
    out["source_note"] = (
        "brent_usd_per_bbl fetched live from FRED series DCOILBRENTEU (US EIA "
        "Europe Brent spot), quarterly mean. diesel_php_per_l and "
        "gasoline_php_per_l are curated from DOE Oil Industry Update weekly "
        f"bulletins, covering {DOE_PUMP_QUARTERLY[0][0]}-{DOE_PUMP_QUARTERLY[0][1]} "
        f"to {DOE_COVERAGE_END[0]}-Q{DOE_COVERAGE_END[1]} only; doe.gov.ph is "
        "WAF-protected and cannot be scraped reliably. NCR retail used as the "
        "CALABARZON proxy (same Pandacan/Tabangao terminal supply)."
        + (f" {doe_gap} rows in this frame have no DOE pump price."
           if doe_gap else "")
    )
    # The quarter column is the join key against the feature-matrix backbone,
    # which uses the full "YYYY-Qn" label. Composed last because the internal
    # merges above key on (year, quarter) with the bare quarter.
    out["quarter"] = out["year"].astype(str) + "-" + out["quarter"]

    out["fetched_at"] = datetime.now(timezone.utc).isoformat()

    logger.info("oil_prices: %d rows, %s-%s .. %s-%s (%d rows lack DOE pump prices)",
                len(out), out["year"].min(), out["quarter"].min(),
                out["year"].max(), out["quarter"].max(), doe_gap)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    frame = fetch_oil_prices(2021, strict=False)
    dest = Path("data/processed/oil_prices.parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest, index=False)
    print(frame[frame["province_name"] == "Cavite"]
          [["year", "quarter", "diesel_php_per_l", "brent_usd_per_bbl"]]
          .to_string(index=False))
    print(f"\nwrote {len(frame)} rows -> {dest}")
