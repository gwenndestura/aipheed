"""
app/ml/corpus/google_cse_fetcher.py
-------------------------------------
Google Custom Search API fetcher for CALABARZON food insecurity corpus.

WHY GOOGLE CUSTOM SEARCH
-------------------------
Google Custom Search Engine (CSE) provides access to Google's full web index
with precise site: restrictions. Unlike Google News RSS (capped at ~100 per
query), CSE supports:
  - Exact Boolean queries with site: operators
  - Date restriction (dateRestrict parameter)
  - Pagination up to 100 results per query (10 per page × 10 pages)
  - Searching within specific credible Philippine news domains

The CSE is configured at https://cse.google.com to search:
  ALL Philippine credible news sites simultaneously OR
  Individual domains via the cx (engine ID).

SETUP
-----
1. Go to https://cse.google.com → New search engine
2. Add all CREDIBLE_DOMAINS as "Sites to search"
3. Copy the Search engine ID (cx)
4. Enable "Custom Search API" in Google Cloud Console
5. Create an API key
6. Set in backend/.env:
     GOOGLE_CSE_API_KEY=AIzaSy...
     GOOGLE_CSE_CX=xxxxxxxxxxxxxxxxx

Free tier: 100 queries/day; $5 per 1,000 additional queries.

COVERAGE
--------
- All 147 CALABARZON LGUs × 6 focused food queries
- All 5 provinces × 10 food insecurity queries
- Regional/thematic queries

Usage:
    from app.ml.corpus.google_cse_fetcher import fetch_google_cse_articles
    articles = fetch_google_cse_articles("2020-01-01", "2025-12-31")
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests

from app.ml.corpus.rss_fetcher import CALABARZON_FOOD_SIGNALS, CREDIBLE_DOMAINS, _is_credible

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOOGLE_CSE_BASE = "https://www.googleapis.com/customsearch/v1"
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY", "")
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX", "")

REQUEST_DELAY = 1.0
RESULTS_PER_PAGE = 10
MAX_PAGES = 10  # 100 results max per query


# ---------------------------------------------------------------------------
# ALL 147 CALABARZON LGUs (imported from GDELT fetcher to avoid duplication)
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
        "Bacoor City", "Cavite City", "Dasmariñas City", "General Trias Cavite",
        "Imus City", "Tagaytay City", "Trece Martires City",
        "Alfonso Cavite", "Amadeo Cavite", "Carmona Cavite",
        "General Emilio Aguinaldo Cavite", "General Mariano Alvarez Cavite",
        "Indang Cavite", "Kawit", "Magallanes Cavite", "Maragondon",
        "Mendez Cavite", "Naic", "Noveleta", "Rosario Cavite",
        "Silang", "Tanza", "Ternate Cavite",
    ],
    "Laguna": [
        "Biñan City", "Cabuyao City", "Calamba City", "San Pablo City Laguna",
        "San Pedro City Laguna", "Santa Rosa City Laguna",
        "Alaminos Laguna", "Bay Laguna", "Calauan", "Cavinti Laguna",
        "Famy Laguna", "Kalayaan Laguna", "Liliw", "Los Baños Laguna",
        "Luisiana", "Lumban", "Mabitac", "Magdalena Laguna",
        "Majayjay", "Nagcarlan", "Paete", "Pagsanjan",
        "Pakil", "Pangil Laguna", "Pila Laguna", "Rizal Laguna",
        "Santa Cruz Laguna", "Santa Maria Laguna", "Siniloan", "Victoria Laguna",
    ],
    "Quezon": [
        "Candelaria Quezon Province", "Lucena City", "Tayabas City",
        "Agdangan", "Alabat", "Atimonan", "Buenavista Quezon Province",
        "Burdeos Quezon Province", "Calauag", "Catanauan",
        "Dolores Quezon Province", "General Luna Quezon Province",
        "General Nakar", "Guinayangan", "Gumaca Quezon",
        "Infanta Quezon Province", "Jomalig", "Lopez Quezon Province",
        "Lucban Quezon", "Macalelon", "Mulanay",
        "Padre Burgos Quezon Province", "Panukulan",
        "Patnanungan Quezon Province", "Perez Quezon Province",
        "Pitogo Quezon Province", "Plaridel Quezon Province",
        "Polillo Quezon Province", "Real Quezon Province",
        "Sampaloc Quezon Province", "San Andres Quezon Province",
        "San Antonio Quezon Province", "San Francisco Quezon Province",
        "San Narciso Quezon Province", "Sariaya",
        "Tagkawayan Quezon Province", "Tiaong Quezon Province",
        "Unisan Quezon Province",
    ],
    "Rizal": [
        "Antipolo City Rizal",
        "Angono Rizal", "Baras Rizal", "Binangonan Rizal", "Cainta Rizal",
        "Cardona Rizal", "Jala-Jala Rizal", "Morong Rizal", "Pililla Rizal",
        "Rodriguez Rizal", "San Mateo Rizal", "Tanay Rizal",
        "Taytay Rizal", "Teresa Rizal",
    ],
}

# ---------------------------------------------------------------------------
# Food insecurity query templates (appended to LGU name)
# ---------------------------------------------------------------------------

LGU_FOOD_QUERIES: list[str] = [
    'food insecurity OR "food prices" OR "food shortage"',
    '"hunger" OR "malnutrition" OR "stunting"',
    '"food relief" OR "food assistance" OR "relief goods"',
    '"crop damage" OR "fish kill" OR "flood damage"',
    '"food security" OR "rice supply" OR "food program"',
    '"livelihood" OR "poverty" OR "food distribution"',
]

PROVINCE_EXTRA_QUERIES: list[str] = [
    '"food prices" market rice vegetables',
    '"malnutrition" OR "hunger" children school',
    '"typhoon" OR "flood" food crop damage',
    '"DSWD" OR "4Ps" food assistance program',
    '"fish kill" OR "oil spill" fishermen livelihood',
    '"kadiwa" OR "NFA rice" distribution',
    '"crop damage" OR "harvest loss" palay',
    '"food security" program government',
    '"poverty" food access households',
    '"feeding program" malnutrition stunting',
]

THEMATIC_QUERIES: list[str] = [
    'site:pna.gov.ph CALABARZON food',
    'site:pna.gov.ph CALABARZON rice prices',
    'site:pna.gov.ph Batangas food insecurity',
    'site:pna.gov.ph Cavite food prices',
    'site:pna.gov.ph Laguna food relief',
    'site:pna.gov.ph Quezon Province food',
    'site:pna.gov.ph Rizal food poverty',
    'site:rappler.com CALABARZON food prices rice',
    'site:inquirer.net CALABARZON food insecurity',
    'site:mb.com.ph CALABARZON food assistance',
    'site:philstar.com CALABARZON food prices',
    'site:gmanetwork.com CALABARZON food',
    'CALABARZON "food insecurity" 2020 2021',
    'CALABARZON "food insecurity" 2022 2023',
    'CALABARZON "food insecurity" 2024 2025',
    'CALABARZON "malnutrition" children Philippines',
    'CALABARZON "fish kill" Laguna Batangas',
    'CALABARZON "crop damage" typhoon flood',
    'Batangas Cavite Laguna Quezon Rizal "food prices"',
    '"Region IV-A" food insecurity Philippines',
    '"Laguna de Bay" fish kill food',
    '"Taal Lake" fish kill food Batangas',
    'CALABARZON "oil spill" fishermen food',
    'CALABARZON "Kadiwa" rice food program',
    'CALABARZON "4Ps" "pantawid" food beneficiaries',
    'CALABARZON El Nino drought rice food',
    '"presyo ng pagkain" CALABARZON Batangas Laguna',
    '"gutom" "kahirapan" CALABARZON',
    '"tulong pagkain" OR "ayuda pagkain" CALABARZON',
    'CALABARZON "community pantry" food 2021',
    'CALABARZON "onion prices" crisis 2022 2023',
    'CALABARZON "fertilizer" price farmers rice',
    'CALABARZON "ASF" pork prices food',
]


# ---------------------------------------------------------------------------
# Date restriction helper
# ---------------------------------------------------------------------------

def _date_restrict(start_date: str, end_date: str) -> str:
    """
    Google CSE dateRestrict format: 'd[number]' 'w[n]' 'm[n]' 'y[n]'.
    We use 'y' range approximation — for finer control, caller should loop.
    """
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    years = max(1, int((end - start).days / 365) + 1)
    return f"y{years}"


def _fetch_page(query: str, start_index: int, date_restrict: str) -> dict:
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_CX:
        return {}
    params = {
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "num": RESULTS_PER_PAGE,
        "start": start_index,
        "dateRestrict": date_restrict,
        "sort": "date",
        "lr": "lang_en|lang_tl",
        "gl": "ph",
    }
    try:
        r = requests.get(GOOGLE_CSE_BASE, params=params, timeout=30)
        time.sleep(REQUEST_DELAY)
        if r.status_code == 200:
            return r.json()
        logger.warning("Google CSE HTTP %d for query '%s'", r.status_code, query[:50])
    except requests.RequestException as exc:
        logger.error("Google CSE request error: %s", exc)
    return {}


def _parse_item(item: dict) -> dict | None:
    link = item.get("link") or item.get("formattedUrl") or ""
    if not link:
        return None
    if not _is_credible(link):
        return None

    title = (item.get("title") or "").strip()
    if not title:
        return None

    snippet = (item.get("snippet") or "").strip()

    # Reject articles with no food insecurity signal in title or snippet.
    combined = (title + " " + snippet).lower()
    if not any(kw in combined for kw in CALABARZON_FOOD_SIGNALS):
        return None
    article_id = hashlib.md5(link.encode()).hexdigest()

    # Try to extract publish date from metatags
    published = ""
    metatags = item.get("pagemap", {}).get("metatags", [{}])
    if metatags:
        mt = metatags[0]
        published = (
            mt.get("article:published_time")
            or mt.get("og:updated_time")
            or mt.get("datePublished")
            or ""
        )

    domain = link.split("/")[2].lstrip("www.")

    return {
        "title": title,
        "link": link,
        "article_id": article_id,
        "published": published,
        "summary": snippet[:500],
        "source_domain": domain,
        "fetcher_source": "google_cse",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_google_cse_articles(
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Fetch CALABARZON food insecurity articles via Google Custom Search API.

    Covers:
      - All 147 municipalities/cities × 6 food insecurity sub-queries
      - All 5 provinces × 10 food insecurity queries
      - 33 thematic/regional queries

    Parameters
    ----------
    start_date : "2020-01-01"
    end_date   : "2025-12-31"

    Returns
    -------
    list[dict]  — standard corpus records
    """
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_CX:
        logger.error(
            "GOOGLE_CSE_API_KEY or GOOGLE_CSE_CX not set in .env — skipping Google CSE fetch"
        )
        return []

    date_restrict = _date_restrict(start_date, end_date)
    records: list[dict] = []
    seen_ids: set[str] = set()

    # ── Per-LGU queries ───────────────────────────────────────────────────
    lgu_queries: list[str] = []
    for province, lgus in CALABARZON_LGUS.items():
        for lgu in lgus:
            for sub in LGU_FOOD_QUERIES:
                lgu_queries.append(f'"{lgu}" {sub}')
        for sub in PROVINCE_EXTRA_QUERIES:
            lgu_queries.append(f'"{province}" {sub}')

    all_queries = lgu_queries + THEMATIC_QUERIES

    logger.info("Google CSE: %d total queries", len(all_queries))

    for query in all_queries:
        for page_num in range(MAX_PAGES):
            start_index = page_num * RESULTS_PER_PAGE + 1
            data = _fetch_page(query, start_index, date_restrict)
            if not data:
                break

            items = data.get("items") or []
            if not items:
                break

            for item in items:
                record = _parse_item(item)
                if record is None:
                    continue
                key = record["article_id"]
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                records.append(record)

            total_results = int(
                data.get("searchInformation", {}).get("totalResults", 0)
            )
            fetched_so_far = start_index + len(items) - 1
            if fetched_so_far >= min(total_results, 100):
                break

    logger.info(
        "fetch_google_cse_articles: %d credible articles (%s to %s)",
        len(records), start_date, end_date,
    )
    return records
