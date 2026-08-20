"""
app/ml/corpus/psgc_gazetteer_fetcher.py
----------------------------------------
Full PSGC gazetteer for Region IV-A (CALABARZON) — province → city /
municipality → barangay.

PURPOSE
-------
The corpus pipeline historically geocoded articles only to the PROVINCE level
(``province_code``). The thesis requires geographically specific coverage of
ALL local areas in Region IV-A — every city, municipality, and barangay across
Cavite, Laguna, Batangas, Rizal, and Quezon. This fetcher builds the
authoritative place-name reference the sub-province geocoder matches against.

SOURCE
------
Official PSA Philippine Standard Geographic Code (PSGC), served as static JSON
by the community mirror ``psgc.gitlab.io`` (a 1:1 publication of the PSA PSGC
tables). Region IV-A carries the 9-digit PSGC code ``040000000``.

We fetch the hierarchy in three tiers:
  regions/040000000/provinces.json
  provinces/{code}/cities-municipalities.json
  cities-municipalities/{code}/barangays.json

OUTPUT
------
data/processed/psgc_gazetteer.parquet — one row per barangay, columns:
    region_code        : str  ("040000000")
    region_name        : str  ("Region IV-A (CALABARZON)")
    province_psgc      : str  (real PSGC, e.g. "042100000")
    province_code      : str  (repo-internal, e.g. "PH040100000")
    province_name      : str  ("Cavite")
    lgu_psgc           : str  (real PSGC 9-digit)
    lgu_name           : str  (cleaned, e.g. "Bacoor")
    lgu_name_official   : str  (API form, e.g. "City of Bacoor")
    lgu_type           : str  ("city" | "municipality")
    barangay_psgc      : str  (real PSGC 9-digit)
    barangay_name      : str  (cleaned, e.g. "Molino III")
    barangay_generic   : bool (True for ambiguous "Barangay I (Pob.)" style
                               names that must never geocode on their own)

The repo-internal ``province_code`` mirrors the scheme already used across the
pipeline (geocoder.PROVINCE_PSGC) so the gazetteer joins cleanly onto the
existing corpus.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/processed/psgc_gazetteer.parquet")
API_BASE = "https://psgc.gitlab.io/api"
REGION_CODE = "040000000"  # Region IV-A (CALABARZON)
REGION_NAME = "Region IV-A (CALABARZON)"

# Real PSGC province code → repo-internal province_code (geocoder.PROVINCE_PSGC).
# Keeps the gazetteer joinable to the corpus, which already tags province_code
# in this internal scheme rather than raw PSGC.
PSGC_TO_INTERNAL: dict[str, str] = {
    "041000000": "PH040500000",  # Batangas
    "042100000": "PH040100000",  # Cavite
    "043400000": "PH040200000",  # Laguna
    "045600000": "PH040300000",  # Quezon
    "045800000": "PH040400000",  # Rizal
}

# Generic barangay names that repeat inside almost every LGU (numbered/named
# poblacion districts). They carry no disambiguating power and must never
# geocode an article on their own.
_GENERIC_BRGY = re.compile(
    r"^(barangay|brgy\.?|poblacion|pob\.?)\b"
    r"|\(pob\.?\)$"
    r"|^(district|zone|ward)\s|^bgy",
    re.I,
)


# CALABARZON province names — an LGU cleaned to one of these would collide
# with the province, so cities get their common "X City" form instead.
_PROVINCE_NAMES_LC = {"batangas", "cavite", "laguna", "quezon", "rizal"}


def _clean_lgu_name(name: str) -> str:
    """'City of Bacoor' → 'Bacoor'; '(Capital)' stripped. When the bare form
    would collide with a province name ('City of Cavite' → 'Cavite'), keep the
    common '<Name> City' form instead so it never masquerades as the province."""
    n = name.strip()
    m = re.match(r"^City of (.+)$", n, re.I)
    if m:
        base = m.group(1).strip()
        n = f"{base} City" if base.lower() in _PROVINCE_NAMES_LC else base
    n = re.sub(r"\s*\((Capital|Pob\.?)\)\s*$", "", n, flags=re.I).strip()
    return n


def _clean_brgy_name(name: str) -> str:
    return re.sub(r"\s*\(Pob\.?\)\s*$", "", name.strip(), flags=re.I).strip()


def _get(url: str) -> list[dict]:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_psgc_gazetteer() -> pd.DataFrame:
    """Fetch the full Region IV-A province → LGU → barangay hierarchy."""
    rows: list[dict] = []
    provinces = _get(f"{API_BASE}/regions/{REGION_CODE}/provinces.json")
    logger.info("Region IV-A: %d provinces", len(provinces))

    for prov in sorted(provinces, key=lambda p: p["name"]):
        p_psgc = prov["code"]
        p_name = prov["name"]
        internal = PSGC_TO_INTERNAL.get(p_psgc)
        if internal is None:
            logger.warning("province %s (%s) has no internal code mapping — skip",
                           p_name, p_psgc)
            continue
        lgus = _get(f"{API_BASE}/provinces/{p_psgc}/cities-municipalities.json")
        logger.info("  %s: %d cities/municipalities", p_name, len(lgus))

        for lgu in sorted(lgus, key=lambda x: x["name"]):
            l_psgc = lgu["code"]
            l_official = lgu["name"]
            l_type = "city" if lgu.get("isCity") else "municipality"
            l_clean = _clean_lgu_name(l_official)
            brgys = _get(f"{API_BASE}/cities-municipalities/{l_psgc}/barangays.json")

            for b in brgys:
                b_clean = _clean_brgy_name(b["name"])
                rows.append({
                    "region_code": REGION_CODE,
                    "region_name": REGION_NAME,
                    "province_psgc": p_psgc,
                    "province_code": internal,
                    "province_name": p_name,
                    "lgu_psgc": l_psgc,
                    "lgu_name": l_clean,
                    "lgu_name_official": l_official,
                    "lgu_type": l_type,
                    "barangay_psgc": b["code"],
                    "barangay_name": b_clean,
                    "barangay_generic": bool(_GENERIC_BRGY.search(b["name"])),
                })

    df = pd.DataFrame(rows)
    df.attrs["fetched_at"] = datetime.now(timezone.utc).isoformat()
    logger.info(
        "PSGC gazetteer: %d barangays across %d LGUs / %d provinces",
        len(df), df["lgu_psgc"].nunique(), df["province_psgc"].nunique(),
    )
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    df = fetch_psgc_gazetteer()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"[ok] wrote {OUTPUT_PATH} — {len(df)} barangays")
    print("\nBarangays per province:")
    print(df.groupby("province_name")
            .agg(lgus=("lgu_psgc", "nunique"), barangays=("barangay_psgc", "count"))
            .to_string())


if __name__ == "__main__":
    main()
