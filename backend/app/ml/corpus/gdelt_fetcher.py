"""
app/ml/corpus/gdelt_fetcher.py
-------------------------------
GDELT Project DOC 2.0 API fetcher for CALABARZON food insecurity corpus.

WHY GDELT
----------
GDELT (Global Database of Events, Language, and Tone) monitors the world's
broadcast, print, and web news in 65 languages. For Philippines coverage:
  - Indexes PNA, Rappler, Inquirer, GMA, ABS-CBN, Manila Bulletin, etc.
  - Full Boolean search with date range and source domain filters
  - Free, no API key required
  - Supports up to 250 articles per request
  - Covers 2015 to present, ~15-minute update lag

GDELT DOC 2.0 API:
  https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

COVERAGE STRATEGY
-----------------
We query ALL 147 CALABARZON LGUs (municipalities + cities) individually
with focused food insecurity sub-queries, then also run province-level
and regional queries. Each query is limited by a 3-month date window so
250-article caps are less likely to truncate relevant articles.

Queries are structured as:
  "<municipality> food" OR "<municipality> hunger" OR "<municipality> rice"

RATE LIMITS
-----------
GDELT asks for polite access. We enforce a 1-second delay between requests.
Empirically, GDELT returns 0 results for very narrow municipality queries —
for those, we broaden to province-level automatically.

Usage:
    from app.ml.corpus.gdelt_fetcher import fetch_gdelt_articles
    articles = fetch_gdelt_articles("2020-01-01", "2025-12-31")
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests

from app.ml.corpus.rss_fetcher import CALABARZON_FOOD_SIGNALS, CREDIBLE_DOMAINS, _is_credible

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GDELT_DOC_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
REQUEST_DELAY = 1.5
MAX_RECORDS = 250
GDELT_CREDIBLE_DOMAINS = [
    "pna.gov.ph", "rappler.com", "inquirer.net", "philstar.com",
    "mb.com.ph", "manilatimes.net", "gmanetwork.com", "news.abs-cbn.com",
    "cnnphilippines.com", "businessmirror.com.ph", "bworldonline.com",
    "sunstar.com.ph", "ptvnews.ph", "pia.gov.ph",
]

# ---------------------------------------------------------------------------
# ALL 147 CALABARZON LGUs
# ---------------------------------------------------------------------------

CALABARZON_LGUS: dict[str, list[str]] = {
    "Batangas": [
        "Batangas City", "Lipa City", "Tanauan City", "Santo Tomas Batangas",
        "Agoncillo", "Alitagtag", "Balayan", "Balete Batangas", "Bauan",
        "Calaca Batangas", "Calatagan", "Cuenca Batangas", "Ibaan",
        "Laurel Batangas", "Lemery", "Lian", "Lobo Batangas", "Mabini Batangas",
        "Malvar", "Mataas na Kahoy", "Nasugbu", "Padre Garcia",
        "Rosario Batangas", "San Jose Batangas", "San Juan Batangas",
        "San Luis Batangas", "San Nicolas Batangas", "San Pascual Batangas",
        "Santa Teresita Batangas", "Taysan", "Tingloy", "Tuy Batangas",
    ],
    "Cavite": [
        "Bacoor", "Cavite City", "Dasmariñas", "General Trias Cavite", "Imus Cavite",
        "Tagaytay City", "Trece Martires",
        "Alfonso Cavite", "Amadeo Cavite", "Carmona Cavite",
        "General Emilio Aguinaldo", "General Mariano Alvarez",
        "Indang Cavite", "Kawit", "Magallanes Cavite", "Maragondon",
        "Mendez Cavite", "Naic", "Noveleta", "Rosario Cavite",
        "Silang", "Tanza", "Ternate Cavite",
    ],
    "Laguna": [
        "Biñan City", "Cabuyao City", "Calamba City", "San Pablo City Laguna",
        "San Pedro Laguna", "Santa Rosa Laguna",
        "Alaminos Laguna", "Bay Laguna", "Calauan", "Cavinti",
        "Famy Laguna", "Kalayaan Laguna", "Liliw", "Los Baños",
        "Luisiana", "Lumban", "Mabitac", "Magdalena Laguna",
        "Majayjay", "Nagcarlan", "Paete", "Pagsanjan",
        "Pakil", "Pangil Laguna", "Pila Laguna", "Rizal Laguna",
        "Santa Cruz Laguna", "Santa Maria Laguna", "Siniloan", "Victoria Laguna",
    ],
    "Quezon": [
        "Candelaria Quezon", "Lucena City", "Tayabas City",
        "Agdangan", "Alabat", "Atimonan", "Buenavista Quezon", "Burdeos",
        "Calauag", "Catanauan", "Dolores Quezon", "General Luna Quezon",
        "General Nakar", "Guinayangan", "Gumaca", "Infanta Quezon",
        "Jomalig", "Lopez Quezon", "Lucban", "Macalelon", "Mulanay",
        "Padre Burgos Quezon", "Panukulan", "Patnanungan", "Perez Quezon",
        "Pitogo Quezon", "Plaridel Quezon", "Polillo", "Real Quezon",
        "Sampaloc Quezon", "San Andres Quezon", "San Antonio Quezon",
        "San Francisco Quezon", "San Narciso Quezon", "Sariaya",
        "Tagkawayan", "Tiaong", "Unisan",
    ],
    "Rizal": [
        "Antipolo City",
        "Angono", "Baras Rizal", "Binangonan", "Cainta", "Cardona",
        "Jala-Jala", "Morong Rizal", "Pililla", "Rodriguez Rizal",
        "San Mateo Rizal", "Tanay", "Taytay Rizal", "Teresa Rizal",
    ],
}

# ---------------------------------------------------------------------------
# Food insecurity sub-queries — appended to each LGU name
# ---------------------------------------------------------------------------

FOOD_SUBQUERIES: list[str] = [
    "food prices",
    "food insecurity",
    "hunger malnutrition",
    "rice supply shortage",
    "food assistance relief",
    "crop damage harvest",
    "food poverty livelihood",
    "typhoon flood food",
    "fish kill livelihood",
    "feeding program malnutrition",
    "presyo pagkain gutom",
    "food security program",
    "rice distribution NFA",
    "DSWD food aid",
]

REGIONAL_QUERIES: list[str] = [
    "CALABARZON food insecurity",
    "CALABARZON food prices rice",
    "CALABARZON hunger malnutrition",
    "CALABARZON typhoon food",
    "CALABARZON DSWD food relief",
    "CALABARZON fish kill lake",
    "CALABARZON crop damage palay",
    "Region IVA food security Philippines",
    "Batangas food insecurity",
    "Cavite food insecurity",
    "Laguna food insecurity",
    "Quezon Province food insecurity",
    "Rizal Province food insecurity",
    "Laguna de Bay fish kill food",
    "Taal Lake fish kill Batangas",
    "Taal Volcano food Batangas",
    "El Nino drought rice CALABARZON",
    "typhoon Quezon Batangas food damage",
    "typhoon Rizal Laguna food relief",
    "Kadiwa food prices CALABARZON",
    "4Ps pantawid food beneficiaries CALABARZON",
    "NFA rice distribution CALABARZON",
    "community pantry food CALABARZON",
    "pork ban ASF Cavite Laguna",
    "oil spill fishermen Cavite Batangas",
    "Bantilan bridge transport goods Quezon Batangas",
    "inflation food prices Calabarzon families",
    "onion sugar price crisis Philippines CALABARZON",
    "fertilizer price farmers CALABARZON",
    "OFW remittance food poor CALABARZON",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quarter_windows(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split date range into quarterly windows (GDELT date format: YYYYMMDDHHMMSS)."""
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    windows: list[tuple[str, str]] = []
    current = start
    while current < end:
        q_end = min(current + timedelta(days=90), end)
        windows.append((
            current.strftime("%Y%m%d%H%M%S"),
            q_end.strftime("%Y%m%d%H%M%S"),
        ))
        current = q_end
    return windows


def _build_url(query: str, start_dt: str, end_dt: str) -> str:
    q_encoded = quote(query)
    return (
        f"{GDELT_DOC_BASE}?query={q_encoded}"
        f"&mode=ArtList"
        f"&maxrecords={MAX_RECORDS}"
        f"&startdatetime={start_dt}"
        f"&enddatetime={end_dt}"
        f"&sort=DateDesc"
        f"&format=json"
        f"&sourcelang=english%20tagalog"
    )


def _parse_gdelt_article(item: dict) -> dict | None:
    url = item.get("url") or item.get("sourceurl") or ""
    if not url:
        return None
    if not _is_credible(url):
        return None

    title = (item.get("title") or "").strip()
    if not title:
        return None

    # Reject articles with no food insecurity signal in the title.
    if not any(kw in title.lower() for kw in CALABARZON_FOOD_SIGNALS):
        return None

    # GDELT seendate format: YYYYMMDDTHHMMSSZ
    seendate = item.get("seendate", "")
    try:
        published = datetime.strptime(seendate[:15], "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc
        ).isoformat()
    except Exception:
        published = seendate

    domain = item.get("domain") or url.split("/")[2]
    article_id = hashlib.md5(url.encode()).hexdigest()

    return {
        "title": title,
        "link": url,
        "article_id": article_id,
        "published": published,
        "summary": "",
        "source_domain": domain,
        "fetcher_source": "gdelt",
    }


def _fetch_gdelt(url: str) -> list[dict]:
    try:
        r = requests.get(url, timeout=30)
        time.sleep(REQUEST_DELAY)
        if r.status_code == 200:
            data = r.json()
            return data.get("articles") or []
        logger.debug("GDELT HTTP %d: %s", r.status_code, url[:80])
    except Exception as exc:
        logger.debug("GDELT request error: %s", exc)
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_gdelt_articles(
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Fetch CALABARZON food insecurity articles from the GDELT DOC 2.0 API.

    Queries every municipality/city in all 5 provinces against 14 food
    insecurity sub-queries, plus regional and national food queries —
    producing dense, location-specific food insecurity coverage.

    Parameters
    ----------
    start_date : "2020-01-01"
    end_date   : "2025-12-31"

    Returns
    -------
    list[dict]  — standard corpus records
    """
    windows = _quarter_windows(start_date, end_date)
    records: list[dict] = []
    seen_ids: set[str] = set()

    # ── Build per-LGU query list ──────────────────────────────────────────
    lgu_queries: list[str] = []
    for province, lgus in CALABARZON_LGUS.items():
        for lgu in lgus:
            for sub in FOOD_SUBQUERIES:
                lgu_queries.append(f'"{lgu}" {sub}')
        # Also pure province-level
        for sub in FOOD_SUBQUERIES:
            lgu_queries.append(f'"{province}" {sub}')

    all_queries = lgu_queries + REGIONAL_QUERIES

    logger.info(
        "GDELT: %d queries x %d windows = %d requests",
        len(all_queries), len(windows), len(all_queries) * len(windows),
    )

    for start_dt, end_dt in windows:
        window_count = 0
        for query in all_queries:
            url = _build_url(query, start_dt, end_dt)
            articles = _fetch_gdelt(url)
            for item in articles:
                record = _parse_gdelt_article(item)
                if record is None:
                    continue
                key = record["article_id"]
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                records.append(record)
                window_count += 1

        logger.info(
            "GDELT window %s–%s: +%d articles (total %d)",
            start_dt[:8], end_dt[:8], window_count, len(records),
        )

    logger.info(
        "fetch_gdelt_articles: %d credible articles (%s to %s)",
        len(records), start_date, end_date,
    )
    return records
