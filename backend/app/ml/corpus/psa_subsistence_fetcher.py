"""
app/ml/corpus/psa_subsistence_fetcher.py
----------------------------------------
PSA subsistence incidence (food poverty) and per-capita food threshold, by
CALABARZON province, from the OpenStat PXWeb API.

Subsistence incidence is the proportion of families whose income cannot cover
even basic FOOD needs -- the closest thing in Philippine official statistics to
a province-level food-insecurity measure. Distinct from poverty_incidence
(income poverty) already carried in psa_indicators.parquet.

    DB/1F/FY/0051F3DF030.px
    Table 3. Annual Per Capita Food Threshold and Subsistence Incidence Among
    Families with Measures of Precision, by Region and Province

Two outputs, because the two columns have very different standing:

  food_threshold_php   -- USABLE AS A FEATURE. Province-varying, annual, an
                          official cost-of-basic-food-basket figure. It is NOT
                          an input to stress_score, so unlike the food-CPI
                          columns (trainer.LEAKY_LABEL_FEATURES) it introduces
                          no label leakage.

  subsistence_pct      -- VALIDATION ONLY, and only at REGION level. PSA's own
                          coefficients of variation for the five CALABARZON
                          provinces run 24-59%; above 30% these are not fit for
                          inference, and every province confidence interval
                          overlaps nearly every other. The regional figure
                          (CV 12-16%) is reliable. Province rows are emitted
                          with their CV and a `discriminating` flag so the
                          distinction cannot be lost downstream.

Years are whatever OpenStat serves -- 2018/2021/2023 as of writing. PSA released
2025 Full Year poverty statistics on 21 August 2026; when OpenStat ingests it
this fetcher picks it up with no code change.

Nothing here is interpolated, projected, or carried forward.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPENSTAT_BASE = "https://openstat.psa.gov.ph/PXWeb/api/v1/en"
TABLE = "DB/1F/FY/0051F3DF030.px"
OUTPUT_PATH = Path("data/processed/psa_subsistence.parquet")
TIMEOUT = 60

# CV above this is not fit for inference (standard small-domain convention).
CV_UNRELIABLE = 30.0

# Geolocation codes within this PXWeb table.
GEO = {
    "32": ("PH040000000", "Region IV-A (CALABARZON)", "region"),
    "33": ("PH040500000", "Batangas", "province"),
    "34": ("PH040100000", "Cavite", "province"),
    "35": ("PH040200000", "Laguna", "province"),
    "36": ("PH040300000", "Quezon", "province"),
    "37": ("PH040400000", "Rizal", "province"),
}

# Measure indices within the table's second variable.
M_THRESHOLD, M_INCIDENCE, M_CV, M_SE, M_CI_LO, M_CI_HI = "0", "1", "2", "3", "4", "5"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aiPHeed/1.0 (+research; DLSU-D thesis)"})


def _json(resp: requests.Response) -> dict | list:
    """PXWeb serves UTF-8 with a BOM, which json.loads rejects."""
    return json.loads(resp.content.decode("utf-8-sig"))


def fetch_psa_subsistence(save_path: Path | None = OUTPUT_PATH) -> pd.DataFrame:
    """Fetch food threshold + subsistence incidence with precision measures."""
    meta = _json(SESSION.get(f"{OPENSTAT_BASE}/{TABLE}", timeout=TIMEOUT))
    geo_var, measure_var, year_var = (v["code"] for v in meta["variables"])
    years = meta["variables"][2]["valueTexts"]
    year_codes = meta["variables"][2]["values"]
    logger.info("OpenStat table %s | years available: %s", TABLE, years)

    body = {
        "query": [
            {"code": geo_var, "selection": {"filter": "item", "values": list(GEO)}},
            {"code": measure_var, "selection": {"filter": "item",
                                                "values": [M_THRESHOLD, M_INCIDENCE,
                                                           M_CV, M_CI_LO, M_CI_HI]}},
            {"code": year_var, "selection": {"filter": "item", "values": year_codes}},
        ],
        "response": {"format": "json"},
    }
    data = _json(SESSION.post(f"{OPENSTAT_BASE}/{TABLE}", json=body, timeout=TIMEOUT))

    cell: dict[tuple[str, str], dict[str, str]] = {}
    for item in data["data"]:
        geo_code, measure, year_idx = item["key"]
        cell.setdefault((geo_code, years[int(year_idx)]), {})[measure] = item["values"][0]

    def _num(v: str | None) -> float | None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for (geo_code, year), measures in cell.items():
        psgc, name, level = GEO[geo_code]
        cv = _num(measures.get(M_CV))
        incidence = _num(measures.get(M_INCIDENCE))
        reliable = cv is not None and cv <= CV_UNRELIABLE
        rows.append({
            "psgc_code": psgc,
            "area_name": name,
            "geographic_level": level,
            "year": int(year),
            "food_threshold_php": _num(measures.get(M_THRESHOLD)),
            "subsistence_pct": incidence,
            "subsistence_cv": cv,
            "subsistence_ci_low": _num(measures.get(M_CI_LO)),
            "subsistence_ci_high": _num(measures.get(M_CI_HI)),
            # False => this estimate cannot separate one province from another.
            "discriminating": bool(reliable),
            "usable_as_feature": True,          # food_threshold_php always is
            "usable_for_validation": bool(reliable),
            "source_url": f"https://openstat.psa.gov.ph/PXWeb/pxweb/en/DB/DB__1F__FY/",
            "source_note": (
                "PSA Full Year Official Poverty Statistics, Table 3: Annual Per Capita "
                "Food Threshold and Subsistence Incidence Among Families, by Region and "
                "Province. Retrieved live from OpenStat PXWeb."
            ),
            "fetched_at": fetched_at,
        })

    df = pd.DataFrame(rows).sort_values(["year", "geographic_level", "area_name"])

    prov = df[df["geographic_level"] == "province"]
    unreliable = int((~prov["discriminating"]).sum())
    logger.info(
        "PSA subsistence: %d rows | years %s | province estimates unfit for "
        "discrimination: %d of %d (CV > %.0f%%)",
        len(df), sorted(df["year"].unique()), unreliable, len(prov), CV_UNRELIABLE,
    )
    if unreliable:
        logger.warning(
            "Province subsistence CVs range %.1f-%.1f%%. Use the REGION row for "
            "validation; province rows are retained for completeness only and must "
            "not be used to rank provinces.",
            prov["subsistence_cv"].min(), prov["subsistence_cv"].max(),
        )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        logger.info("saved -> %s", save_path)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = fetch_psa_subsistence()
    print(f"\n[ok] wrote {OUTPUT_PATH} — {len(df)} rows\n")
    show = ["area_name", "geographic_level", "year", "food_threshold_php",
            "subsistence_pct", "subsistence_cv", "discriminating"]
    print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
