"""
app/ml/corpus/gnews_fetcher.py
-------------------------------
GNews API article fetcher for CALABARZON food insecurity corpus.

Query template: "food prices Philippines {province}"
Results filtered to the same CREDIBLE_DOMAINS allowlist as rss_fetcher.py.
Exponential backoff on rate-limit (HTTP 429) responses.

Usage:
    from app.ml.corpus.gnews_fetcher import fetch_gnews_articles
    articles = fetch_gnews_articles("2024-01-01", "2024-12-31", api_key="...")
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

from app.ml.corpus.rss_fetcher import CREDIBLE_DOMAINS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GNews API configuration
# ---------------------------------------------------------------------------

GNEWS_BASE_URL = "https://gnews.io/api/v4/search"

# CALABARZON provinces for query expansion
CALABARZON_PROVINCES = [
    "Batangas",
    "Cavite",
    "Laguna",
    "Quezon",
    "Rizal",
]

# Number of articles per query (GNews free tier max = 10)
MAX_ARTICLES_PER_QUERY = 10

# Backoff config
MAX_RETRIES = 5
BACKOFF_BASE = 2.0  # seconds; doubles on each retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower().lstrip("www.")


def _is_credible(url: str) -> bool:
    """Return True only if the article URL's domain is in CREDIBLE_DOMAINS."""
    domain = _extract_domain(url)
    if domain in CREDIBLE_DOMAINS:
        return True
    for allowed in CREDIBLE_DOMAINS:
        if domain.endswith("." + allowed):
            return True
    return False


def _gnews_get(params: dict, api_key: str) -> list[dict]:
    """
    Call GNews /search with exponential backoff on 429 responses.
    Returns the 'articles' list from the JSON response.
    """
    params = {**params, "token": api_key}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(GNEWS_BASE_URL, params=params, timeout=20)
        except requests.RequestException as exc:
            logger.error("GNews request error (attempt %d): %s", attempt, exc)
            time.sleep(BACKOFF_BASE ** attempt)
            continue

        if response.status_code == 200:
            data = response.json()
            return data.get("articles", [])

        if response.status_code == 429:
            wait = BACKOFF_BASE ** attempt
            logger.warning(
                "GNews rate-limited (attempt %d/%d). Sleeping %.1fs.",
                attempt, MAX_RETRIES, wait,
            )
            time.sleep(wait)
            continue

        logger.error(
            "GNews returned HTTP %d for params %s", response.status_code, params
        )
        break

    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_gnews_articles(
    start_date: str,
    end_date: str,
    api_key: str,
) -> list[dict]:
    """
    Fetch CALABARZON food-price articles from GNews API.

    Queries one template per CALABARZON province:
        "food prices Philippines {province}"

    All returned articles are validated against CREDIBLE_DOMAINS —
    any article from an unverified or unknown source is discarded.

    Parameters
    ----------
    start_date : str   — ISO date e.g. "2024-01-01"
    end_date   : str   — ISO date e.g. "2024-12-31"
    api_key    : str   — GNEWS_API_KEY from .env

    Returns
    -------
    list[dict]
        Each record:
            title         : str
            link          : str
            published     : str  — ISO UTC datetime string
            summary       : str
            source_domain : str  — domain from CREDIBLE_DOMAINS
    """
    records: list[dict] = []
    seen_links: set[str] = set()

    for province in CALABARZON_PROVINCES:
        query = f"food prices Philippines {province}"
        logger.info("GNews query: '%s'", query)

        params = {
            "q": query,
            "lang": "en",
            "country": "ph",
            "max": MAX_ARTICLES_PER_QUERY,
            "from": start_date,
            "to": end_date,
            "sortby": "publishedAt",
        }

        articles = _gnews_get(params, api_key)
        logger.info("GNews returned %d articles for province '%s'", len(articles), province)

        for article in articles:
            url: str = article.get("url", "")
            if not url:
                continue

            # 1. Domain validation — discard unverified sources
            if not _is_credible(url):
                logger.debug("Non-credible domain from GNews, discarding: %s", url)
                continue

            # 2. Deduplication
            if url in seen_links:
                continue
            seen_links.add(url)

            # 3. Parse published date
            pub_str: str = article.get("publishedAt", "")
            try:
                published_dt = datetime.fromisoformat(
                    pub_str.replace("Z", "+00:00")
                ).astimezone(timezone.utc)
                published_iso = published_dt.isoformat()
            except Exception:
                published_iso = pub_str

            source_domain = _extract_domain(url)

            records.append(
                {
                    "title": (article.get("title") or "").strip(),
                    "link": url.strip(),
                    "published": published_iso,
                    "summary": (article.get("description") or "").strip(),
                    "source_domain": source_domain,
                }
            )

        # Polite delay between province queries
        time.sleep(1.0)

    logger.info(
        "fetch_gnews_articles: %d credible articles collected (%s to %s)",
        len(records), start_date, end_date,
    )
    return records