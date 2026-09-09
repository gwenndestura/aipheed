"""
app/ml/corpus/sws_hunger_fetcher.py
------------------------------------
Social Weather Stations (SWS) Hunger Survey — quarterly self-reported
involuntary hunger fetcher.

PURPOSE
-------
The SWS Hunger Survey is the **canonical Philippine quarterly food
insecurity indicator** — administered every quarter since 1998. It asks:
"In the last 3 months, did your family experience involuntary hunger
(nagutom) because there was nothing to eat?"

Reported as % of families experiencing:
    • Total hunger (overall)
    • Moderate hunger (sometimes / a few times)
    • Severe hunger (often / always)

WHY ESSENTIAL FOR aiPHeed
--------------------------
1. **Quarterly cadence** — bridges the 2-year gap between NNS 2021 and NNS
   2023 (the DOST-FNRI cycles). Provides 24 quarterly observations
   2020Q1–2025Q4 vs only 2 NNS data points.
2. **Region-level granularity** — Balance of Luzon (where CALABARZON sits)
   reported separately from NCR / Visayas / Mindanao.
3. **Methodologically independent** of FIES — provides cross-validation
   anchor for the model's quarterly forecasts.
4. **Used by FAO, World Bank, ADB Philippines country reports** as the
   official high-frequency hunger indicator. Citing SWS is panel-defensible.

PRECEDENT REFERENCES
--------------------
- World Bank (2024) Philippines Poverty Assessment — uses SWS Hunger
- FAO (2023) Country Profile Philippines — cites SWS quarterly hunger
- Asian Development Bank (2022) PH Country Strategy — SWS Hunger as core KPI
- WFP HungerMap LIVE methodology — equivalent to SWS-style self-report

DATA STRATEGY
-------------
SWS releases each quarter's results via press release on sws.org.ph.
Page contents are static HTML — scrapeable but format varies. This
fetcher uses curated values from public SWS press releases (2020Q1–2025Q4)
with each quarter's source URL recorded per row.

OUTPUT
------
data/processed/sws_hunger.parquet — columns:
    region_code              : str (Balance of Luzon → applied to CALABARZON)
    region_name              : str
    province_code            : str (PSGC, region inherited)
    province_name            : str
    year                     : int
    quarter                  : str ("YYYY-QN")
    survey_quarter_label     : str (e.g. "2020 Q3 SWS Hunger Survey")
    pct_total_hunger         : float (% of families)
    pct_moderate_hunger      : float
    pct_severe_hunger        : float
    source_url               : str
    source_note              : str
    fetched_at               : str

TRAINING WINDOW: 2020Q1–2025Q4 strict.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.ml.corpus._coverage import CuratedCoverageError, check_coverage  # noqa: F401

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/processed/sws_hunger.parquet")

# Region IV-A is reported under "Balance of Luzon" (rest of Luzon outside NCR)
REGION_CODE = "PH040000000"
REGION_NAME = "Region IV-A (CALABARZON) [SWS Balance of Luzon]"

PROVINCES = [
    ("PH040100000", "Cavite"),
    ("PH040200000", "Laguna"),
    ("PH040300000", "Quezon"),
    ("PH040400000", "Rizal"),
    ("PH040500000", "Batangas"),
]

# ---------------------------------------------------------------------------
# Curated SWS Hunger Survey quarterly results — Balance of Luzon
# Source: SWS press releases at sws.org.ph (one per quarter).
# These are PUBLIC PRIMARY DATA released by SWS each quarter.
# Values reflect involuntary hunger experienced by families in past 3 months.
#
# Citation pattern: SWS Survey conducted [date]; press release URL recorded
# in source_url. Original surveys publicly archived under sws.org.ph/swsmain/.
# ---------------------------------------------------------------------------
# ===========================================================================
# REBUILT 2026-09-01 — every prior value removed as unverifiable.
#
# The former table did not match SWS. Seven quarters were checked against press
# releases and none agreed: 2024-Q4 was recorded at 10.3% against a published
# Balance Luzon figure of 25.3%; 2025-Q4 at 9.5% against an actual 16.7%. Two
# quarters moved in the opposite direction from the real series, and the 2025
# block was commented "projected continuation of moderating trend" while being
# written out as "March 2025 SWS Survey" etc. 2025 is the model holdout.
#
# Only values read from an SWS release appear below, each with the release URL
# it came from. SWS migrated their website during 2025-26 and the pre-December-
# 2024 archive is not published — the Hunger category holds exactly four
# releases. Those quarters are therefore ABSENT, not estimated. Requesting the
# 2010-2025 series from sws_info@sws.org.ph is the outstanding action.
#
# Stratum: Balance Luzon (Luzon outside NCR) — the stratum containing
# CALABARZON. National figures are recorded separately for cross-checking only.
#
# Format: (year, "Q#", total%, moderate%, severe%, survey_label, source_url)
# ===========================================================================
SWS_BAL_LUZON_VERIFIED = [
    (2024, "Q3", 18.1, 13.0, 5.1, "September 2024 SWS Survey",
     "https://sws.org.ph/social-weather-report-fourth-quarter-2024-social-weather-survey-"
     "hunger-rises-to-25-9-in-december-2024-from-22-9-in-september-2024/"),
    (2024, "Q4", 25.3, 16.6, 8.7, "December 2024 SWS Survey",
     "https://sws.org.ph/social-weather-report-fourth-quarter-2024-social-weather-survey-"
     "hunger-rises-to-25-9-in-december-2024-from-22-9-in-september-2024/"),
    (2025, "Q1", 24.0, 16.5, 7.4, "Stratbase-SWS March 15-20, 2025 National Survey",
     "https://sws.org.ph/social-weather-report-stratbase-sws-march-15-20-2025-national-"
     "survey-hunger-rises-to-27-2-in-march-2025/"),
    (2025, "Q4", 16.7, 12.7, 4.0, "November 2025 SWS Survey",
     "https://sws.org.ph/social-weather-report-hunger-rises-from-20-1-in-november-2025-"
     "to-23-2-in-march-2026/"),
    # These were previously excluded by a default end_year of 2025 while the
    # model window ran to 2026-Q2 -- verified data sitting unused.
    (2026, "Q1", 22.4, 16.6, 5.8, "March 2026 SWS Survey",
     "https://sws.org.ph/social-weather-report-hunger-rises-from-20-1-in-november-2025-"
     "to-23-2-in-march-2026/"),
    (2026, "Q2", 17.3, 13.0, 4.3, "June 2026 SWS Survey",
     "https://sws.org.ph/social-weather-report-hunger-falls-to-20-3-in-june-2026-"
     "from-23-2-in-march-2026/"),
]

# National totals from the same releases, for sanity checks only. Balance Luzon
# and National are DIFFERENT series and must never be mixed — conflating them is
# a likely origin of the discarded table.
SWS_NATIONAL_REFERENCE = {
    (2024, "Q3"): 22.9, (2024, "Q4"): 25.9, (2025, "Q1"): 27.2,
    (2025, "Q4"): 20.1, (2026, "Q1"): 23.2, (2026, "Q2"): 20.3,
}

# Retained only to document what was removed. NOT data. Do not reinstate.
_DISCARDED_UNVERIFIED_2020_2025 = "24 quarters; see git history and iteration log 09"

_LEGACY_REMOVED = [
    # 2020 — pandemic year, hunger spiked dramatically
    (2020, "Q1", 8.2, 6.5, 1.7,  "March 2020 SWS Survey"),
    (2020, "Q2", 16.7, 13.8, 2.9, "May 2020 SWS Survey — pandemic spike"),
    (2020, "Q3", 26.5, 21.0, 5.5, "September 2020 SWS Survey"),
    (2020, "Q4", 14.8, 12.0, 2.8, "November 2020 SWS Survey"),
    # 2021 — sustained pandemic stress
    (2021, "Q1", 14.5, 11.5, 3.0, "May 2021 SWS Survey"),
    (2021, "Q2", 13.2, 10.5, 2.7, "July 2021 SWS Survey"),
    (2021, "Q3", 14.0, 11.0, 3.0, "September 2021 SWS Survey"),
    (2021, "Q4", 11.6, 9.2, 2.4,  "December 2021 SWS Survey"),
    # 2022 — economic reopening, moderating hunger
    (2022, "Q1", 11.5, 9.2, 2.3,  "April 2022 SWS Survey"),
    (2022, "Q2", 11.6, 9.3, 2.3,  "June 2022 SWS Survey"),
    (2022, "Q3", 11.3, 8.9, 2.4,  "September 2022 SWS Survey"),
    (2022, "Q4", 10.4, 8.2, 2.2,  "December 2022 SWS Survey"),
    # 2023 — rice price spike, hunger ticked up
    (2023, "Q1", 10.4, 8.0, 2.4,  "March 2023 SWS Survey"),
    (2023, "Q2", 10.7, 8.1, 2.6,  "June 2023 SWS Survey"),
    (2023, "Q3", 11.5, 8.8, 2.7,  "September 2023 SWS Survey — rice price spike"),
    (2023, "Q4", 11.0, 8.3, 2.7,  "December 2023 SWS Survey"),
    # 2024 — moderate easing
    (2024, "Q1", 11.7, 8.9, 2.8,  "March 2024 SWS Survey"),
    (2024, "Q2", 12.5, 9.5, 3.0,  "June 2024 SWS Survey"),
    (2024, "Q3", 11.8, 9.0, 2.8,  "September 2024 SWS Survey"),
    (2024, "Q4", 10.3, 7.8, 2.5,  "December 2024 SWS Survey"),
    # 2025 — projected continuation of moderating trend
    (2025, "Q1", 9.8, 7.5, 2.3,   "March 2025 SWS Survey"),
    (2025, "Q2", 9.5, 7.3, 2.2,   "June 2025 SWS Survey"),
    (2025, "Q3", 9.7, 7.4, 2.3,   "September 2025 SWS Survey"),
    (2025, "Q4", 9.5, 7.2, 2.3,   "December 2025 SWS Survey"),
]

SOURCE_BASE = "https://www.sws.org.ph/swsmain/artcldisppage/"
SOURCE_NOTE = ("SWS quarterly Hunger Survey, Balance of Luzon stratum. "
               "Self-reported involuntary hunger (nagutom) past 3 months. "
               "Cited by World Bank PH Poverty Assessment, FAO PH Country Profile, "
               "ADB PH Country Strategy as canonical quarterly hunger indicator.")


SWS_COVERAGE_END = (2026, 2)


def fetch_sws_hunger(start_year: int = 2021,
                     end_year: int | None = None,
                     strict: bool = True) -> pd.DataFrame:
    """
    Build the CALABARZON SWS Hunger panel from VERIFIED quarters only.

    The Balance Luzon value is inherited to all five provinces: SWS publishes no
    province cut, so these rows are not independent province observations.
    Quarters with no published release are absent from the output entirely — they
    are never interpolated, projected, or carried forward.
    """
    end_year = end_year or datetime.now(timezone.utc).year
    check_coverage(
        series="SWS hunger (Balance Luzon)",
        coverage_end=SWS_COVERAGE_END, end_year=end_year,
        reason="SWS publishes each round as a press release whose HTML layout "
               "varies between releases, so it is not safely scrapeable.",
        source_url="the release at sws.org.ph (SWS_BAL_LUZON_VERIFIED)",
        extend_hint="raise SWS_COVERAGE_END",
        strict=strict, logger=logger)

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for year, q, total, mod, sev, label, url in SWS_BAL_LUZON_VERIFIED:
        if not (start_year <= year <= end_year):
            continue
        for prov_code, prov_name in PROVINCES:
            rows.append({
                "region_code": REGION_CODE,
                "region_name": REGION_NAME,
                "province_code": prov_code,
                "province_name": prov_name,
                "year": year,
                "quarter": f"{year}-{q}",
                "survey_quarter_label": label,
                "pct_total_hunger": float(total),
                "pct_moderate_hunger": float(mod),
                "pct_severe_hunger": float(sev),
                "stratum": "balance_luzon",
                "geographic_level": "stratum_inherited",
                "verified": True,
                "source_url": url,
                "source_note": SOURCE_NOTE,
                "fetched_at": fetched_at,
            })

    df = pd.DataFrame(rows)

    expected = {f"{y}-Q{i}" for y in range(start_year, end_year + 1) for i in range(1, 5)}
    have = set(df["quarter"]) if len(df) else set()
    missing = sorted(expected - have)
    logger.info(
        "SWS Hunger: %d rows | %d of %d quarters verified in %d-%d",
        len(df), len(have), len(expected), start_year, end_year,
    )
    if missing:
        logger.warning(
            "%d quarters have NO published SWS release and are absent (not estimated): %s",
            len(missing), ", ".join(missing),
        )
        logger.warning(
            "SWS migrated their site; the pre-Dec-2024 archive is unpublished. "
            "Request the 2010-2025 Balance Luzon series from sws_info@sws.org.ph."
        )
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = fetch_sws_hunger(2020, 2025)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"[ok] wrote {OUTPUT_PATH} — {len(df)} rows")
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
