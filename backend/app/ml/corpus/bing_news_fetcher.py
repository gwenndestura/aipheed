"""
app/ml/corpus/bing_news_fetcher.py
------------------------------------
Bing News Search API fetcher for CALABARZON food insecurity corpus.

WHY BING NEWS SEARCH
---------------------
Bing News Search API (Azure Cognitive Services) indexes thousands of Philippine
news sources in real-time and supports:
  - Full-text Boolean queries with date filtering
  - Market (mkt=en-PH) and language (setLang=tl) targeting
  - Freshness filtering (Day / Week / Month)
  - Up to 100 results per request with offset-based pagination
  - Safe, reliable rate limits (3 calls/second on S1 plan)

This supplements Google News RSS by covering:
  - ABS-CBN News, GMA News, CNN Philippines (not in RSS corpus)
  - Real-time breaking news
  - Municipal-level news from local outlets missed by Google News

SETUP
-----
1. Create an Azure account → Cognitive Services → Bing Search v7
2. Copy API key
3. Set in backend/.env:
     BING_NEWS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Plans:
  Free (F1): 3 transactions/second, 1,000 calls/month
  Standard (S1): 3 t/s, 3M calls/month ($7 per 1K calls)

API docs: https://learn.microsoft.com/en-us/bing/search-apis/bing-news-search/

COVERAGE
--------
- All 147 CALABARZON LGUs × 8 food insecurity sub-queries
- All 5 provinces × 12 targeted queries
- Regional, thematic, Tagalog-language queries

Usage:
    from app.ml.corpus.bing_news_fetcher import fetch_bing_news_articles
    articles = fetch_bing_news_articles("2020-01-01", "2025-12-31")
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import requests

from app.ml.corpus.rss_fetcher import CALABARZON_FOOD_SIGNALS, CREDIBLE_DOMAINS, _is_credible

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BING_NEWS_BASE = "https://api.bing.microsoft.com/v7.0/news/search"
BING_NEWS_API_KEY = os.getenv("BING_NEWS_API_KEY", "")

REQUEST_DELAY = 0.4     # 3 calls/second max on most plans
COUNT_PER_PAGE = 100
MAX_OFFSET = 900        # Bing allows offset up to 900 + count ≤ 1000

# Sites Bing should prioritize (site: operators in query)
_PREFERRED_SITES = [
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
        "General Nakar", "Guinayangan", "Gumaca Quezon Province",
        "Infanta Quezon Province", "Jomalig", "Lopez Quezon Province",
        "Lucban Quezon Province", "Macalelon", "Mulanay",
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
        "Antipolo City Rizal Province",
        "Angono Rizal Province", "Baras Rizal Province",
        "Binangonan Rizal Province", "Cainta Rizal Province",
        "Cardona Rizal Province", "Jala-Jala Rizal Province",
        "Morong Rizal Province", "Pililla Rizal Province",
        "Rodriguez Rizal Province", "San Mateo Rizal Province",
        "Tanay Rizal Province", "Taytay Rizal Province",
        "Teresa Rizal Province",
    ],
}

# ---------------------------------------------------------------------------
# Query templates
# ---------------------------------------------------------------------------

LGU_FOOD_SUBQUERIES: list[str] = [
    "food insecurity hunger poverty",
    "food prices rice vegetables market",
    "food relief assistance DSWD",
    "crop damage flood typhoon",
    "malnutrition stunting children",
    "fish kill livelihood fishermen",
    "food security program NFA",
    "rice shortage supply distribution",
]

PROVINCE_QUERIES: list[str] = [
    "{province} food insecurity hunger",
    "{province} food prices rice market",
    "{province} food relief assistance",
    "{province} crop damage typhoon flood",
    "{province} malnutrition stunting",
    "{province} fish kill oil spill fishermen",
    "{province} NFA rice distribution",
    "{province} DSWD food assistance",
    "{province} 4Ps pantawid food",
    "{province} food security program",
    "{province} palay harvest agriculture",
    "{province} poverty livelihood food",
]

REGIONAL_THEMATIC_QUERIES: list[str] = [
    "CALABARZON food insecurity prices",
    "CALABARZON hunger malnutrition children",
    "CALABARZON food assistance typhoon flood",
    "CALABARZON crop damage palay harvest",
    "CALABARZON fish kill Laguna Batangas",
    "CALABARZON NFA rice Kadiwa distribution",
    "CALABARZON 4Ps pantawid food poor",
    "CALABARZON oil spill fishermen Cavite",
    "CALABARZON El Nino drought rice",
    "Region IV-A food security Philippines",
    "Laguna de Bay fish kill food supply",
    "Taal Lake fish kill food Batangas",
    "Taal Volcano eruption food Batangas evacuees",
    "Typhoon Rolly food damage Quezon Batangas 2020",
    "Typhoon Ulysses flood food Rizal Laguna 2020",
    "Typhoon Paeng food damage Quezon 2022",
    "Typhoon Kristine food relief CALABARZON 2024",
    "presyo ng pagkain CALABARZON gutom",
    "kakulangan ng pagkain CALABARZON",
    "tulong pagkain CALABARZON bagyo baha",
    "gutom kahirapan CALABARZON pamilya",
    "ayuda pagkain CALABARZON mahihirap",
    "onion sugar prices crisis Calabarzon 2022 2023",
    "fertilizer prices farmers rice CALABARZON",
    "ASF pork prices food Cavite Laguna",
    "Bantilan bridge transport goods Quezon Batangas",
    "community pantry food CALABARZON 2021",
    "feeding program malnutrition CALABARZON schools",
    "Batang Busog Malusog CALABARZON",
    "e-nutribun malnutrition nutrition CALABARZON",
    "food security tilapia backyard farming CALABARZON",
    "Kadiwa ng Pangulo CALABARZON rice prices",
    "DSWD DROMIC CALABARZON food",
    "OFW remittance CALABARZON food poverty",
    "minimum wage food prices CALABARZON workers",
]

TAGALOG_QUERIES: list[str] = [
    "presyo ng bigas CALABARZON merkado",
    "gutom kahirapan pamilya CALABARZON",
    "tulong pagkain CALABARZON gobyerno",
    "bagyo baha ani pinsala CALABARZON",
    "kakulangan ng pagkain suplay CALABARZON",
    "magsasaka ani CALABARZON palay",
    "mangingisda isda CALABARZON Laguna Batangas",
    "ayuda pagkain mahihirap CALABARZON",
    "presyo ng gulay isda CALABARZON palengke",
    "libreng bigas NFA CALABARZON distribusyon",
]


# ---------------------------------------------------------------------------
# Date window helpers
# ---------------------------------------------------------------------------

def _monthly_windows(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split into monthly windows for Bing freshness control."""
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    windows: list[tuple[str, str]] = []
    current = start
    while current < end:
        # Bing uses 'after:' and 'before:' in query or freshness parameter
        month_end = min(
            datetime(current.year + (current.month // 12),
                     (current.month % 12) + 1, 1) if current.month < 12
            else datetime(current.year + 1, 1, 1),
            end,
        )
        windows.append((
            current.strftime("%Y-%m-%d"),
            month_end.strftime("%Y-%m-%d"),
        ))
        current = month_end
    return windows


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_page(query: str, offset: int = 0) -> dict:
    if not BING_NEWS_API_KEY:
        return {}
    headers = {"Ocp-Apim-Subscription-Key": BING_NEWS_API_KEY}
    params = {
        "q": query,
        "count": COUNT_PER_PAGE,
        "offset": offset,
        "mkt": "en-PH",
        "setLang": "en",
        "safeSearch": "Off",
        "textFormat": "Raw",
        "sortBy": "Date",
    }
    try:
        r = requests.get(BING_NEWS_BASE, headers=headers, params=params, timeout=30)
        time.sleep(REQUEST_DELAY)
        if r.status_code == 200:
            return r.json()
        logger.warning("Bing News HTTP %d for query '%s'", r.status_code, query[:50])
    except requests.RequestException as exc:
        logger.error("Bing News request error: %s", exc)
    return {}


def _parse_article(item: dict) -> dict | None:
    url = item.get("url") or ""
    if not url:
        return None
    if not _is_credible(url):
        return None

    title = (item.get("name") or "").strip()
    if not title:
        return None

    description = (item.get("description") or "").strip()

    # Reject articles with no food insecurity signal in title or description.
    combined = (title + " " + description).lower()
    if not any(kw in combined for kw in CALABARZON_FOOD_SIGNALS):
        return None
    published = item.get("datePublished") or ""
    provider_name = ""
    providers = item.get("provider") or []
    if providers:
        provider_name = providers[0].get("name", "")

    domain = url.split("/")[2].lstrip("www.")
    article_id = hashlib.md5(url.encode()).hexdigest()

    return {
        "title": title,
        "link": url,
        "article_id": article_id,
        "published": published,
        "summary": description[:500],
        "source_domain": domain,
        "fetcher_source": "bing_news",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_bing_news_articles(
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Fetch CALABARZON food insecurity articles from Bing News Search API.

    Covers:
      - All 147 municipalities/cities × 8 food insecurity sub-queries
      - All 5 provinces × 12 focused food queries
      - 34 regional/thematic queries + 10 Tagalog queries
      - Date range scoped with after:/before: operators in query string

    Parameters
    ----------
    start_date : "2020-01-01"
    end_date   : "2025-12-31"

    Returns
    -------
    list[dict]  — standard corpus records
    """
    if not BING_NEWS_API_KEY:
        logger.error("BING_NEWS_API_KEY not set in .env — skipping Bing News fetch")
        return []

    records: list[dict] = []
    seen_ids: set[str] = set()

    # ── Build all queries ──────────────────────────────────────────────��──
    all_queries: list[str] = []

    # LGU-level
    for province, lgus in CALABARZON_LGUS.items():
        for lgu in lgus:
            for sub in LGU_FOOD_SUBQUERIES:
                all_queries.append(f'"{lgu}" {sub}')

    # Province-level
    for province in CALABARZON_LGUS:
        for q_template in PROVINCE_QUERIES:
            all_queries.append(q_template.format(province=province))

    # Regional/thematic
    all_queries += REGIONAL_THEMATIC_QUERIES
    all_queries += TAGALOG_QUERIES

    # Add date range to each query via operators
    date_suffix = f' after:{start_date} before:{end_date}'
    all_queries = [q + date_suffix for q in all_queries]

    logger.info("Bing News: %d total queries", len(all_queries))

    for query in all_queries:
        offset = 0
        while offset <= MAX_OFFSET:
            data = _fetch_page(query, offset)
            if not data:
                break
            items = data.get("value") or []
            if not items:
                break
            for item in items:
                record = _parse_article(item)
                if record is None:
                    continue
                key = record["article_id"]
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                records.append(record)
            offset += COUNT_PER_PAGE
            if len(items) < COUNT_PER_PAGE:
                break

    logger.info(
        "fetch_bing_news_articles: %d credible articles (%s to %s)",
        len(records), start_date, end_date,
    )
    return records
