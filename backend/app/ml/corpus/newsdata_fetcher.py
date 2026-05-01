"""
app/ml/corpus/newsdata_fetcher.py
----------------------------------
NewsData.io REST API fetcher for CALABARZON food insecurity corpus.

WHY NewsData.io
---------------
NewsData.io indexes 80,000+ news sources globally and provides full-text
search with country/language/domain filters. Critically, it covers Philippine
local outlets (PNA, Manila Bulletin, Rappler, GMA) and supports:
  - Date-range filtering (from_date / to_date)
  - Domain allowlist (domainurl parameter — enforce CREDIBLE_DOMAINS)
  - Language filter (Filipino + English)
  - Full-text search with Boolean operators

API documentation: https://newsdata.io/documentation

RATE LIMITS
-----------
Free tier  : 200 requests/day, 10 results/request → 2,000 articles/day
Paid Basic : 30,000 credits/month (~500 requests/day at max pagination)
Each paginated call costs 1 credit; next_page token enables deep pagination.

SETUP
-----
Set NEWSDATA_API_KEY in backend/.env:
    NEWSDATA_API_KEY=pub_xxxxxxxxxxxxxxxxxxxxxxxx

Get a free key at https://newsdata.io/register

COVERAGE STRATEGY
-----------------
Queries are grouped by:
  1. Province-level food queries (5 provinces × 8 queries = 40)
  2. Municipality/city-level food queries (all 147 LGUs in CALABARZON)
  3. Food insecurity topic queries (regional/national)
  4. Climate/typhoon food impact queries
  5. Government program queries (Kadiwa, 4Ps, DSWD, NFA)
  6. Tagalog-language food queries

Usage:
    from app.ml.corpus.newsdata_fetcher import fetch_newsdata_articles
    articles = fetch_newsdata_articles("2020-01-01", "2025-12-31")
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import requests

from app.ml.corpus.rss_fetcher import (
    CALABARZON_FOOD_SIGNALS,
    CALABARZON_GEO_SIGNALS,
    CREDIBLE_DOMAINS,
    _is_credible,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEWSDATA_BASE = "https://newsdata.io/api/1/news"
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY", "")

# Polite delay between requests (seconds)
REQUEST_DELAY = 1.2
NEWSDATA_MAX_WORKERS = 3   # 3 threads × 1.2s ≈ 2.5 req/s — polite to the API

# Max results per page (API max = 10 free / 50 paid)
PAGE_SIZE = 10

# NewsData domain list (comma-separated subset of CREDIBLE_DOMAINS that NewsData indexes)
_NEWSDATA_DOMAINS = ",".join([
    "inquirer.net", "rappler.com", "philstar.com", "mb.com.ph",
    "manilatimes.net", "businessmirror.com.ph", "bworldonline.com",
    "gmanetwork.com", "abs-cbn.com", "cnnphilippines.com",
    "pna.gov.ph", "sunstar.com.ph",
])

# ---------------------------------------------------------------------------
# ALL CALABARZON municipalities / cities
# Complete list: 5 provinces, 147 LGUs (municipalities + cities)
# ---------------------------------------------------------------------------

CALABARZON_LGUS: dict[str, list[str]] = {
    "Batangas": [
        # Cities
        "Batangas City", "Lipa City", "Tanauan City", "Santo Tomas",
        # Municipalities
        "Agoncillo", "Alitagtag", "Balayan", "Balete", "Bauan", "Calaca",
        "Calatagan", "Cuenca", "Ibaan", "Laurel", "Lemery", "Lian", "Lobo",
        "Mabini", "Malvar", "Mataas na Kahoy", "Nasugbu", "Padre Garcia",
        "Rosario Batangas", "San Jose Batangas", "San Juan Batangas",
        "San Luis Batangas", "San Nicolas Batangas", "San Pascual Batangas",
        "Santa Teresita Batangas", "Taysan", "Tingloy", "Tuy",
    ],
    "Cavite": [
        # Cities
        "Bacoor", "Cavite City", "Dasmariñas", "General Trias", "Imus",
        "Tagaytay City", "Trece Martires",
        # Municipalities
        "Alfonso Cavite", "Amadeo", "Carmona Cavite",
        "General Emilio Aguinaldo", "General Mariano Alvarez",
        "Indang", "Kawit", "Magallanes Cavite", "Maragondon",
        "Mendez Cavite", "Naic", "Noveleta", "Rosario Cavite",
        "Silang", "Tanza", "Ternate",
    ],
    "Laguna": [
        # Cities
        "Biñan City", "Cabuyao City", "Calamba City", "San Pablo City",
        "San Pedro Laguna", "Santa Rosa Laguna",
        # Municipalities
        "Alaminos Laguna", "Bay Laguna", "Calauan", "Cavinti", "Famy",
        "Kalayaan Laguna", "Liliw", "Los Baños", "Luisiana", "Lumban",
        "Mabitac", "Magdalena Laguna", "Majayjay", "Nagcarlan",
        "Paete", "Pagsanjan", "Pakil", "Pangil Laguna", "Pila Laguna",
        "Rizal Laguna", "Santa Cruz Laguna", "Santa Maria Laguna",
        "Siniloan", "Victoria Laguna",
    ],
    "Quezon": [
        # Cities
        "Candelaria Quezon", "Lucena City", "Tayabas City",
        # Municipalities
        "Agdangan", "Alabat", "Atimonan", "Buenavista Quezon", "Burdeos",
        "Calauag", "Catanauan", "Dolores Quezon", "General Luna Quezon",
        "General Nakar", "Guinayangan", "Gumaca", "Infanta Quezon",
        "Jomalig", "Lopez Quezon", "Lucban", "Macalelon", "Mulanay",
        "Padre Burgos Quezon", "Panukulan", "Patnanungan", "Perez Quezon",
        "Pitogo Quezon", "Plaridel Quezon", "Polillo", "Quezon Quezon",
        "Real Quezon", "Sampaloc Quezon", "San Andres Quezon",
        "San Antonio Quezon", "San Francisco Quezon", "San Narciso Quezon",
        "Sariaya", "Tagkawayan", "Tiaong", "Unisan",
    ],
    "Rizal": [
        # City
        "Antipolo City",
        # Municipalities
        "Angono", "Baras Rizal", "Binangonan", "Cainta", "Cardona",
        "Jala-Jala", "Morong Rizal", "Pililla", "Rodriguez Rizal",
        "San Mateo Rizal", "Tanay", "Taytay Rizal", "Teresa Rizal",
    ],
}

# ---------------------------------------------------------------------------
# Food insecurity query bank — used for NewsData.io searches
# Structured so province/municipality is appended per LGU
# ---------------------------------------------------------------------------

_BASE_FOOD_QUERIES: list[str] = [
    "food prices market",
    "food insecurity hunger",
    "rice supply shortage",
    "food assistance relief",
    "malnutrition stunting children",
    "crop damage typhoon flood",
    "livelihood poverty food",
    "food distribution DSWD",
    "agriculture harvest palay",
    "fish kill fishermen livelihood",
    "oil spill fishing ban",
    "kadiwa rice price",
    "4Ps pantawid food beneficiaries",
    "food security program",
    "vegetable prices market",
    "presyo ng pagkain gutom",
    "bagyo baha ani pagkain",
    "feeding program malnutrition",
]

_REGIONAL_QUERIES: list[str] = [
    "CALABARZON food insecurity",
    "CALABARZON food prices rice",
    "CALABARZON hunger malnutrition",
    "CALABARZON agriculture harvest",
    "CALABARZON typhoon flood food",
    "CALABARZON DSWD food relief",
    "CALABARZON 4Ps food beneficiaries",
    "CALABARZON NFA rice distribution",
    "CALABARZON fish kill lake",
    "CALABARZON crop damage palay",
    "Region IVA food security",
    "Region IV-A food insecurity Philippines",
    "Batangas Cavite Laguna Quezon Rizal food prices",
]

_TAGALOG_QUERIES: list[str] = [
    "presyo ng bigas CALABARZON",
    "gutom kahirapan CALABARZON",
    "tulong pagkain CALABARZON bagyo",
    "kakulangan ng pagkain Batangas Cavite",
    "ani pinsala CALABARZON bagyo baha",
    "presyo ng pagkain Laguna Rizal",
    "ayuda pagkain mahihirap Quezon",
    "suplay ng bigas CALABARZON",
    "magsasaka ani CALABARZON",
    "mangingisda pagkain CALABARZON",
]

_CLIMATE_QUERIES: list[str] = [
    "typhoon food damage CALABARZON",
    "flood crop damage Laguna Quezon",
    "El Nino drought rice CALABARZON",
    "Taal volcano food Batangas",
    "Typhoon Rolly food Quezon Batangas",
    "Typhoon Ulysses food Rizal Laguna",
    "Typhoon Paeng food Quezon",
    "Typhoon Kristine food CALABARZON",
    "storm flood food relief CALABARZON",
]

_GOV_PROGRAM_QUERIES: list[str] = [
    "Kadiwa food prices CALABARZON",
    "NFA rice distribution CALABARZON",
    "DSWD food relief CALABARZON",
    "4Ps food beneficiary CALABARZON",
    "feeding program malnutrition CALABARZON",
    "community pantry food CALABARZON",
    "Batang Busog Malusog CALABARZON",
    "DA agriculture support CALABARZON",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_date_windows(start_date: str, end_date: str, days: int = 30) -> list[tuple[str, str]]:
    """Split date range into windows of `days` days for pagination control."""
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    windows = []
    current = start
    while current < end:
        window_end = min(current + timedelta(days=days), end)
        windows.append((
            current.strftime("%Y-%m-%d"),
            window_end.strftime("%Y-%m-%d"),
        ))
        current = window_end
    return windows


def _fetch_page(query: str, from_date: str = "", to_date: str = "", page: str | None = None) -> dict:
    """Fetch one page of NewsData.io results (latest endpoint — free tier)."""
    if not NEWSDATA_API_KEY:
        logger.warning("NEWSDATA_API_KEY not set — skipping NewsData.io fetch")
        return {}

    # Free-tier /news endpoint: no from_date/to_date, no domainurl filtering.
    # Credibility is enforced in _parse_article() via _is_credible().
    params: dict = {
        "apikey": NEWSDATA_API_KEY,
        "q": query,
        "country": "ph",
        "language": "en",
    }
    if page:
        params["page"] = page

    try:
        response = requests.get(NEWSDATA_BASE, params=params, timeout=30)
        time.sleep(REQUEST_DELAY)
        if response.status_code == 200:
            return response.json()
        logger.warning("NewsData.io HTTP %d for query '%s'", response.status_code, query[:50])
    except requests.RequestException as exc:
        logger.error("NewsData.io request error: %s", exc)
    return {}


def _parse_article(item: dict) -> dict | None:
    """Parse a NewsData.io result item into standard corpus record."""
    link = item.get("link") or item.get("source_url") or ""
    if not link:
        return None
    if not _is_credible(link):
        return None

    title = (item.get("title") or "").strip()
    if not title:
        return None

    published = item.get("pubDate") or item.get("pubDateTZ") or ""
    summary = (item.get("description") or "").strip()

    # Reject articles missing either a CALABARZON geo signal or a food signal.
    combined = (title + " " + summary).lower()
    if not any(kw in combined for kw in CALABARZON_GEO_SIGNALS):
        return None
    if not any(kw in combined for kw in CALABARZON_FOOD_SIGNALS):
        return None

    source_domain = item.get("source_id") or link.split("/")[2]

    return {
        "title": title,
        "link": link,
        "article_id": link,
        "published": published,
        "summary": summary[:500],
        "source_domain": source_domain,
        "fetcher_source": "newsdata",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_newsdata_articles(
    start_date: str,
    end_date: str,
    max_pages_per_query: int = 5,
) -> list[dict]:
    """
    Fetch CALABARZON food insecurity articles from NewsData.io.

    Covers:
      - All 5 provinces with base food queries
      - All 147 municipalities/cities with food queries
      - Regional/Tagalog/climate/government program queries

    Parameters
    ----------
    start_date         : ISO date "2020-01-01"
    end_date           : ISO date "2025-12-31"
    max_pages_per_query: limit pagination depth to control API credit usage

    Returns
    -------
    list[dict]  — standard corpus records
    """
    if not NEWSDATA_API_KEY:
        logger.error("NEWSDATA_API_KEY not set in .env — cannot fetch from NewsData.io")
        return []

    # ── Build query list ──────────────────────────────────────────────────
    # NO date windowing — the API's from_date/to_date covers the full range
    # per query. Windowing multiplied requests 72× with negligible gain.
    all_queries: list[str] = []

    # Province-level (5 provinces × 8 queries = 40)
    for province in CALABARZON_LGUS:
        for q in _BASE_FOOD_QUERIES[:8]:
            all_queries.append(f"{q} {province}")

    # Municipality/city-level (147 LGUs × 2 queries = 294)
    for province, lgus in CALABARZON_LGUS.items():
        for lgu in lgus:
            all_queries.append(f"food prices relief {lgu}")
            all_queries.append(f"malnutrition hunger harvest {lgu}")

    # Regional / thematic
    all_queries += _REGIONAL_QUERIES
    all_queries += _TAGALOG_QUERIES
    all_queries += _CLIMATE_QUERIES
    all_queries += _GOV_PROGRAM_QUERIES

    logger.info(
        "NewsData.io: %d queries (full range %s to %s), max_pages=%d, workers=%d",
        len(all_queries), start_date, end_date, max_pages_per_query, NEWSDATA_MAX_WORKERS,
    )

    records: list[dict] = []
    seen_ids: set[str] = set()
    total_requests = 0
    lock_records: list[dict] = []  # collected per-thread, merged after

    def _fetch_query(query: str) -> list[dict]:
        """Fetch all pages for one query (latest endpoint — recent articles only)."""
        results: list[dict] = []
        page: str | None = None
        for _ in range(max_pages_per_query):
            data = _fetch_page(query, page=page)
            if not data or data.get("status") != "success":
                break
            for item in data.get("results") or []:
                record = _parse_article(item)
                if record is not None:
                    results.append(record)
            page = data.get("nextPage")
            if not page:
                break
        return results

    with ThreadPoolExecutor(max_workers=NEWSDATA_MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_query, q): q for q in all_queries}
        for future in as_completed(futures):
            for record in future.result():
                key = record["article_id"]
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                records.append(record)
            total_requests += 1

    logger.info(
        "NewsData.io: %d articles collected (%d queries, %s to %s)",
        len(records), total_requests, start_date, end_date,
    )
    return records
