"""
app/ml/corpus/bsp_macro_fetcher.py
-----------------------------------
Quarterly macro features for CALABARZON: OFW remittances and the USD/PHP rate.

WHY THESE TWO
-------------
Remittances are the single largest source of household food-purchasing power in
CALABARZON, and the peso rate sets the landed cost of imported rice, wheat and
fuel. Both are used as macro food-security covariates by the IMF, World Bank and
ADB Philippines country reports, and by Balashankar et al. (2023) Science
Advances as macro features in a food-price prediction setting.

TWO SOURCES, TWO DIFFERENT SITUATIONS
-------------------------------------
USD/PHP      LIVE. Pulled from the Frankfurter API, which serves European
             Central Bank reference rates, and averaged to quarters. Free, no
             API key, updates every business day.

OFW cash     NOT LIVE. BSP is the only publisher of the OFW cash remittance
remittances  series. www.bsp.gov.ph returns HTTP 403 to programmatic clients,
             including with a browser user agent, so it cannot be fetched. The
             values below are curated from BSP quarterly remittance press
             releases and cover 2020-Q1 to 2025-Q4 only.

WHY THIS MODULE NOW REFUSES TO GUESS
------------------------------------
This file used to be a 24-row literal table with no network call at all, ending
at 2025-Q4, and asking for later quarters silently returned a shorter frame.
That is a large part of why the whole feature matrix stopped at 2025-Q4 while
the production panel ran on to 2026-Q2.

fetch_bsp_macro now raises when the requested window runs past the curated
remittance coverage. A caller that wants the truncated series must pass
strict=False, which logs a prominent warning and records the gap in source_note.

TO EXTEND THE REMITTANCE SERIES
-------------------------------
Add rows to OFW_REMIT_QUARTERLY from the BSP quarterly remittance press release
and raise REMIT_COVERAGE_END.

OUTPUT
------
data/processed/bsp_macro.parquet:
    province_code, province_name, year, quarter, ofw_cash_remit_usd_bn,
    ofw_remit_yoy_pct, fx_usd_php_avg, fx_usd_php_yoy_pct, source_url,
    source_note, fetched_at
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from app.ml.corpus._coverage import CuratedCoverageError, check_coverage  # noqa: F401

logger = logging.getLogger(__name__)

FRANKFURTER_URL = "https://api.frankfurter.app"
BSP_SOURCE_URL = "https://www.bsp.gov.ph/SitePages/Statistics/External.aspx"

PROVINCES = [
    ("PH040100000", "Cavite"),
    ("PH040200000", "Laguna"),
    ("PH040300000", "Quezon"),
    ("PH040400000", "Rizal"),
    ("PH040500000", "Batangas"),
]

# Curated BSP OFW cash remittances, USD billions per quarter, from BSP
# quarterly remittance press releases.
# Format: (year, "Q#", ofw_cash_remit_usd_bn)
OFW_REMIT_QUARTERLY = [
    (2020, "Q1", 7.46), (2020, "Q2", 7.06), (2020, "Q3", 7.83), (2020, "Q4", 7.84),
    (2021, "Q1", 7.92), (2021, "Q2", 7.96), (2021, "Q3", 8.04), (2021, "Q4", 8.27),
    (2022, "Q1", 8.20), (2022, "Q2", 8.32), (2022, "Q3", 8.31), (2022, "Q4", 8.94),
    (2023, "Q1", 8.38), (2023, "Q2", 8.31), (2023, "Q3", 8.55), (2023, "Q4", 9.27),
    (2024, "Q1", 8.51), (2024, "Q2", 8.46), (2024, "Q3", 8.81), (2024, "Q4", 9.45),
    (2025, "Q1", 8.65), (2025, "Q2", 8.70), (2025, "Q3", 8.95), (2025, "Q4", 9.55),
]

REMIT_COVERAGE_END = (2025, 4)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aiPHeed/1.0 (+research; DLSU-D thesis)"})


def fetch_usd_php(start_year: int, end_year: int) -> pd.DataFrame:
    """Quarterly mean USD/PHP from ECB reference rates via the Frankfurter API."""
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"
    r = SESSION.get(f"{FRANKFURTER_URL}/{start}..{end}",
                    params={"from": "USD", "to": "PHP"}, timeout=90)
    r.raise_for_status()
    payload = r.json()

    rates = payload.get("rates") or {}
    if not rates:
        raise RuntimeError(f"Frankfurter returned no USD/PHP rates for "
                           f"{start}..{end} -- the API contract may have changed")

    rows = [{"date": d, "fx": v["PHP"]} for d, v in rates.items() if "PHP" in v]
    fx = pd.DataFrame(rows)
    fx["date"] = pd.to_datetime(fx["date"])
    fx["year"] = fx["date"].dt.year
    fx["quarter"] = "Q" + fx["date"].dt.quarter.astype(str)

    q = (fx.groupby(["year", "quarter"], as_index=False)["fx"]
           .mean()
           .rename(columns={"fx": "fx_usd_php_avg"}))
    q["fx_usd_php_avg"] = q["fx_usd_php_avg"].round(3)
    # The ECB range endpoint returns the last business day before the start
    # date, which would otherwise leak a quarter from before the window.
    q = q[(q["year"] >= start_year) & (q["year"] <= end_year)].reset_index(drop=True)

    logger.info("USD/PHP: %d quarters from %d daily ECB rates, %s-%s .. %s-%s",
                len(q), len(fx), q["year"].min(), q["quarter"].min(),
                q["year"].max(), q["quarter"].max())
    return q


def _check_remit_coverage(end_year: int, strict: bool) -> None:
    check_coverage(
        series="OFW cash remittances (ofw_cash_remit_usd_bn)",
        coverage_end=REMIT_COVERAGE_END, end_year=end_year,
        reason="www.bsp.gov.ph returns HTTP 403 to programmatic clients, "
               "including with a browser user agent.",
        source_url=f"the BSP quarterly remittance release ({BSP_SOURCE_URL})",
        extend_hint="raise REMIT_COVERAGE_END",
        strict=strict, logger=logger)


def fetch_bsp_macro(start_year: int = 2021,
                    end_year: int | None = None,
                    strict: bool = True) -> pd.DataFrame:
    """
    Build the province-quarter macro frame.

    USD/PHP is fetched live. OFW remittances are curated, so a request past
    their coverage raises CuratedCoverageError unless strict=False.
    """
    end_year = end_year or datetime.now(timezone.utc).year
    _check_remit_coverage(end_year, strict)

    remit = pd.DataFrame(OFW_REMIT_QUARTERLY,
                         columns=["year", "quarter", "ofw_cash_remit_usd_bn"])
    remit = remit[(remit["year"] >= start_year) & (remit["year"] <= end_year)]

    fx = fetch_usd_php(start_year, end_year)

    # Outer join: FX legitimately runs past the curated remittance range, and
    # those quarters belong in the frame with the remittance column null.
    q = remit.merge(fx, on=["year", "quarter"], how="outer")
    q = q.sort_values(["year", "quarter"]).reset_index(drop=True)

    for col, out in (("ofw_cash_remit_usd_bn", "ofw_remit_yoy_pct"),
                     ("fx_usd_php_avg", "fx_usd_php_yoy_pct")):
        # fill_method=None: a missing quarter must not be padded forward into a
        # fabricated year-on-year change.
        q[out] = (q[col].pct_change(4, fill_method=None) * 100).round(2)

    frames = []
    for code, name in PROVINCES:
        f = q.copy()
        f.insert(0, "province_name", name)
        f.insert(0, "province_code", code)
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)

    remit_gap = int(out["ofw_cash_remit_usd_bn"].isna().sum())
    out["source_url"] = f"{FRANKFURTER_URL} ; {BSP_SOURCE_URL}"
    out["source_note"] = (
        "fx_usd_php_avg fetched live from ECB reference rates via the "
        "Frankfurter API, quarterly mean of daily rates. "
        "ofw_cash_remit_usd_bn is curated from BSP quarterly remittance "
        f"releases, covering {OFW_REMIT_QUARTERLY[0][0]}-{OFW_REMIT_QUARTERLY[0][1]} "
        f"to {REMIT_COVERAGE_END[0]}-Q{REMIT_COVERAGE_END[1]} only; "
        "www.bsp.gov.ph returns 403 to programmatic clients. National values "
        "inherited to all CALABARZON provinces."
        + (f" {remit_gap} rows in this frame have no remittance value."
           if remit_gap else "")
    )
    # The quarter column is the join key against the feature-matrix backbone,
    # which uses the full "YYYY-Qn" label. Composed last because the internal
    # merges above key on (year, quarter) with the bare quarter.
    out["quarter"] = out["year"].astype(str) + "-" + out["quarter"]

    out["fetched_at"] = datetime.now(timezone.utc).isoformat()

    logger.info("bsp_macro: %d rows, %s-%s .. %s-%s (%d rows lack remittances)",
                len(out), out["year"].min(), out["quarter"].min(),
                out["year"].max(), out["quarter"].max(), remit_gap)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    frame = fetch_bsp_macro(2021, strict=False)
    dest = Path("data/processed/bsp_macro.parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest, index=False)
    print(frame[frame["province_name"] == "Cavite"]
          [["year", "quarter", "ofw_cash_remit_usd_bn", "fx_usd_php_avg"]]
          .to_string(index=False))
    print(f"\nwrote {len(frame)} rows -> {dest}")
