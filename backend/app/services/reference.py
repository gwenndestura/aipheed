"""
app/services/reference.py
--------------------------
Static reference data that the dashboard API needs but the model does not
produce: the province slug <-> PSGC map the frontend addresses subjects by,
the 142-LGU roster, and the display thresholds.

The frontend addresses every subject by slug ("quezon", "quezon-infanta").
The model and the DB address them by PSGC code ("PH040300000", "043424000").
This module is the only place that translation lives.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from pathlib import Path

import pandas as pd

LGU_REFERENCE_PATH = Path("data/reference/calabarzon_lgus.csv")

REGION_ID = "calabarzon"
REGION_NAME = "CALABARZON"

# ---------------------------------------------------------------------------
# Provinces
# ---------------------------------------------------------------------------
# Centroids are the map-marker anchors the frontend already uses; they are
# geography, not model output, so they are constants rather than served data.

PROVINCES: dict[str, dict] = {
    "cavite":   {"code": "PH040100000", "name": "Cavite",   "lat": 14.2456, "lng": 120.8786},
    "laguna":   {"code": "PH040200000", "name": "Laguna",   "lat": 14.1407, "lng": 121.4692},
    "quezon":   {"code": "PH040300000", "name": "Quezon",   "lat": 14.0313, "lng": 122.1106},
    "rizal":    {"code": "PH040400000", "name": "Rizal",    "lat": 14.6037, "lng": 121.3084},
    "batangas": {"code": "PH040500000", "name": "Batangas", "lat": 13.9073, "lng": 121.1517},
}

CODE_TO_SLUG: dict[str, str] = {v["code"]: k for k, v in PROVINCES.items()}


def province_code(slug: str) -> str | None:
    """PSGC code for a province slug, or None if the slug is unknown."""
    entry = PROVINCES.get(slug.lower())
    return entry["code"] if entry else None


def province_slug(code: str) -> str | None:
    """Province slug for a PSGC code, or None if the code is unknown."""
    return CODE_TO_SLUG.get(code)


def province_name(slug: str) -> str | None:
    entry = PROVINCES.get(slug.lower())
    return entry["name"] if entry else None


# ---------------------------------------------------------------------------
# Municipalities
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    """
    'Gen. Mariano Alvarez' -> 'gen-mariano-alvarez'.

    Accents are folded and punctuation dropped so the id stays URL-safe and
    stable across the CSV, the GeoJSON and the frontend's own municipality list.
    """
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", folded.lower())).strip("-")


@functools.lru_cache(maxsize=1)
def lgu_roster() -> pd.DataFrame:
    """
    The 142 CALABARZON cities and municipalities, with the frontend's
    composite id ("quezon-infanta") and a derived population density.

    Cached: the file is static reference data read once per process.
    """
    df = pd.read_csv(LGU_REFERENCE_PATH, dtype={"lgu_psgc": str})
    df["provinceId"] = df["province_code"].map(CODE_TO_SLUG)
    df["muniSlug"] = df["lgu_name"].map(slugify)
    df["id"] = df["provinceId"] + "-" + df["muniSlug"]
    df["classification"] = df["lgu_type"].str.title()
    df["densityPerKm2"] = (
        df["population_2020"] / df["land_area_km2"].replace(0, pd.NA)
    ).round(1)
    return df


def municipality_row(municipality_id: str) -> pd.Series | None:
    """Look up one LGU by the frontend's composite id."""
    roster = lgu_roster()
    hit = roster[roster["id"] == municipality_id.lower()]
    return None if hit.empty else hit.iloc[0]


# ---------------------------------------------------------------------------
# Display thresholds
# ---------------------------------------------------------------------------
# Served through GET /api/v1/config so the frontend stops hardcoding them.

RISK_DISPLAY_CUTOFF = 0.50          # score >= => HIGH pill, else LOW
ALERT_THRESHOLD = 0.60              # score >= => active-alert badge
TRIGGER_RED_CUTOFF = 20             # a driver above this share renders red
LIMITED_SIGNAL_MIN_ARTICLES = 5     # below this article count => limited signal

# Only two bands are ever produced. The frontend type carries four; the two
# unused ones are reported as inactive rather than silently never appearing.
ACTIVE_RISK_LEVELS = ["low", "high"]


def risk_level(score: float) -> str:
    return "high" if score >= RISK_DISPLAY_CUTOFF else "low"


# ---------------------------------------------------------------------------
# Quarters
# ---------------------------------------------------------------------------

QUARTER_MONTHS = {
    1: "Jan-Mar",
    2: "Apr-Jun",
    3: "Jul-Sep",
    4: "Oct-Dec",
}


def quarter_index(quarter: str) -> int:
    """Chronological sort key: 2025-Q4 -> 8103."""
    year, q = quarter.split("-Q")
    return int(year) * 4 + (int(q) - 1)


def quarter_parts(quarter: str) -> tuple[int, int]:
    year, q = quarter.split("-Q")
    return int(year), int(q)


def is_valid_quarter(quarter: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-Q[1-4]", quarter or ""))
