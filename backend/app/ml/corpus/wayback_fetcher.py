"""
app/ml/corpus/wayback_fetcher.py
---------------------------------
Wayback Machine CDX API fetcher for 2020–2022 historical corpus gaps.

Queries the CDX API restricted to CREDIBLE_DOMAINS only — never crawls
arbitrary archived pages. robots.txt-compliant. URL-hash deduplication.

Uses waybackpy for CDX lookups, then validates each snapshot URL before
fetching article metadata.

Usage:
    from app.ml.corpus.wayback_fetcher import fetch_wayback_articles
    articles = fetch_wayback_articles("2020-01-01", "2022-12-31")
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

from app.ml.corpus.rss_fetcher import (
    CALABARZON_KEYWORDS,
    CREDIBLE_DOMAINS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CDX_API_URL = "https://web.archive.org/cdx/search/cdx"

# Crawl delay for Wayback CDX requests (their robots.txt requests politeness)
WAYBACK_CRAWL_DELAY = 2.0

# Max snapshots to retrieve per domain per keyword query
CDX_LIMIT_PER_QUERY = 500

# Food/insecurity-related path keywords used to filter CDX URLs without
# fetching each snapshot page — much faster than content-based filtering.
FOOD_PATH_KEYWORDS: frozenset[str] = frozenset({
    "food", "rice", "bigas", "presyo", "price", "inflation", "cpi",
    "hunger", "gutom", "poverty", "kahirapan", "supply", "harvest", "ani",
    "agriculture", "crop", "typhoon", "bagyo", "flood", "baha", "drought",
    "tuyot", "ofw", "remittance", "unemployment", "livelihood",
    "nutrition", "malnutrition", "dswd", "4ps", "pantawid",
    "relief", "ayuda", "shortage", "kakulangan",
})

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "aiPHeed-Thesis-Bot/1.0 "
            "(DLSU-D research crawler; contact: thesis@dlsud.edu.ph)"
        )
    }
)

# ---------------------------------------------------------------------------
# Robots.txt compliance — Wayback Machine
# ---------------------------------------------------------------------------

_WAYBACK_ROBOT_URL = "https://web.archive.org/robots.txt"
_wayback_robot: RobotFileParser | None = None


def _get_wayback_robots() -> RobotFileParser:
    global _wayback_robot
    if _wayback_robot is not None:
        return _wayback_robot
    rp = RobotFileParser()
    rp.set_url(_WAYBACK_ROBOT_URL)
    try:
        rp.read()
        logger.info("Loaded Wayback robots.txt")
    except Exception as exc:
        logger.warning("Could not load Wayback robots.txt: %s", exc)
    _wayback_robot = rp
    return rp


def _wayback_allowed(url: str) -> bool:
    rp = _get_wayback_robots()
    return rp.can_fetch(SESSION.headers["User-Agent"], url)


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower().lstrip("www.")


def _is_credible_domain(url: str) -> bool:
    """Only return True for URLs belonging to our CREDIBLE_DOMAINS set."""
    domain = _extract_domain(url)
    if domain in CREDIBLE_DOMAINS:
        return True
    for allowed in CREDIBLE_DOMAINS:
        if domain.endswith("." + allowed):
            return True
    return False


# ---------------------------------------------------------------------------
# URL-hash deduplication
# ---------------------------------------------------------------------------

_seen_hashes: set[str] = set()


def _already_seen(url: str) -> bool:
    h = hashlib.sha256(url.encode()).hexdigest()
    if h in _seen_hashes:
        return True
    _seen_hashes.add(h)
    return False


# ---------------------------------------------------------------------------
# CDX API query
# ---------------------------------------------------------------------------

def _cdx_query_domain_year(
    domain: str,
    year: int,
) -> list[dict]:
    """
    Query Wayback CDX for all 200-status snapshots of `domain` within `year`.
    Returns list of dicts with: original, timestamp.
    Keyword filtering happens in Python after this call.
    """
    params = {
        "url": f"{domain}/*",
        "output": "json",
        "fl": "original,timestamp",
        "filter": "statuscode:200",
        "from": f"{year}0101000000",
        "to": f"{year}1231235959",
        "limit": CDX_LIMIT_PER_QUERY,
        "collapse": "urlkey",
    }

    if not _wayback_allowed(CDX_API_URL):
        logger.warning("Wayback robots.txt disallows CDX. Skipping.")
        return []

    try:
        resp = SESSION.get(CDX_API_URL, params=params, timeout=60)
        time.sleep(WAYBACK_CRAWL_DELAY)
    except requests.RequestException as exc:
        logger.warning("CDX timeout for %s/%d: %s", domain, year, exc)
        return []

    if resp.status_code != 200:
        logger.error("CDX HTTP %d for %s/%d", resp.status_code, domain, year)
        return []

    try:
        rows = resp.json()
    except Exception as exc:
        logger.error("CDX JSON parse error: %s", exc)
        return []

    if not rows or len(rows) < 2:
        return []

    header = rows[0]
    return [dict(zip(header, row)) for row in rows[1:]]


def _url_has_food_keyword(url: str) -> bool:
    """Return True if the URL path/slug contains a food-related keyword."""
    path = urlparse(url).path.lower()
    return any(kw in path for kw in FOOD_PATH_KEYWORDS)


# ---------------------------------------------------------------------------
# Snapshot metadata fetch
# ---------------------------------------------------------------------------

def _wayback_snapshot_url(original_url: str, timestamp: str) -> str:
    """Construct the Wayback Machine snapshot URL."""
    return f"https://web.archive.org/web/{timestamp}/{original_url}"


def _fetch_snapshot_metadata(original_url: str, timestamp: str) -> dict | None:
    """
    Retrieve the snapshot page and extract title + first 500 chars as summary.
    Returns None on any error.
    """
    snapshot_url = _wayback_snapshot_url(original_url, timestamp)

    if not _wayback_allowed(snapshot_url):
        return None

    try:
        resp = SESSION.get(snapshot_url, timeout=20, allow_redirects=True)
        time.sleep(WAYBACK_CRAWL_DELAY)
    except requests.RequestException as exc:
        logger.debug("Snapshot fetch error for %s: %s", snapshot_url, exc)
        return None

    if resp.status_code != 200:
        return None

    # Lightweight HTML title extraction (avoid heavy BS4 parse on every page)
    content = resp.text
    title = ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content[:8000], "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        # Strip Wayback Machine prefix from title if present
        if "| Wayback Machine" in title:
            title = title.split("| Wayback Machine")[0].strip()
        # Short summary from first paragraph text
        first_p = soup.find("p")
        summary = first_p.get_text(strip=True)[:500] if first_p else ""
    except Exception:
        summary = ""

    return {
        "title": title,
        "summary": summary,
        "snapshot_url": snapshot_url,
    }


# ---------------------------------------------------------------------------
# Keyword filter
# ---------------------------------------------------------------------------

def _matches_keywords(title: str, summary: str) -> bool:
    combined = (title + " " + summary).lower()
    return any(kw in combined for kw in CALABARZON_KEYWORDS)


# ---------------------------------------------------------------------------
# Timestamp → ISO datetime
# ---------------------------------------------------------------------------

def _cdx_ts_to_iso(timestamp: str) -> str:
    """Convert CDX timestamp 'YYYYMMDDHHmmss' to ISO datetime string."""
    try:
        dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return timestamp


def _cdx_ts_to_yyyymmdd(timestamp: str) -> str:
    return timestamp[:8] if len(timestamp) >= 8 else timestamp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _title_from_url(url: str) -> str:
    """Derive a human-readable title from a URL path slug."""
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1] if "/" in path else path
    # Strip file extension
    slug = slug.rsplit(".", 1)[0] if "." in slug else slug
    # Convert separators to spaces and title-case
    return slug.replace("-", " ").replace("_", " ").title()


def fetch_wayback_articles(
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Fetch historical articles (2020–2022 gap-fill) via Wayback Machine CDX API.

    Uses URL-path keyword filtering instead of fetching each snapshot page —
    vastly faster and avoids Wayback rate limits. Title is derived from the
    URL slug; summary is left empty for downstream NLP processing.

    Restricted to CREDIBLE_DOMAINS. robots.txt-compliant. URL-hash dedup.

    Parameters
    ----------
    start_date : str   — ISO date e.g. "2020-01-01"
    end_date   : str   — ISO date e.g. "2022-12-31"

    Returns
    -------
    list[dict]
        Each record:
            title         : str   — derived from URL slug
            link          : str   — original (non-Wayback) URL
            published     : str   — ISO UTC datetime from CDX timestamp
            summary       : str   — empty; filled by downstream NLP
            source_domain : str   — domain from CREDIBLE_DOMAINS
    """
    start_year = datetime.fromisoformat(start_date).year
    end_year = min(datetime.fromisoformat(end_date).year, 2022)  # Wayback for historical gap-fill

    records: list[dict] = []

    for domain in sorted(CREDIBLE_DOMAINS):
        domain_count = 0
        for year in range(start_year, end_year + 1):
            snapshots = _cdx_query_domain_year(domain, year)

            for snap in snapshots:
                original_url: str = snap.get("original", "")
                timestamp: str = snap.get("timestamp", "")
                if not original_url or not timestamp:
                    continue
                if not _is_credible_domain(original_url):
                    continue
                # Filter by food-related keyword in URL path
                if not _url_has_food_keyword(original_url):
                    continue
                if _already_seen(original_url):
                    continue

                title = _title_from_url(original_url)
                published_iso = _cdx_ts_to_iso(timestamp)
                source_domain = _extract_domain(original_url)

                records.append({
                    "title": title,
                    "link": original_url.strip(),
                    "published": published_iso,
                    "summary": "",
                    "source_domain": source_domain,
                })
                domain_count += 1

        if domain_count:
            logger.info("Wayback: %d articles from %s (%d-%d)", domain_count, domain, start_year, end_year)

    logger.info(
        "fetch_wayback_articles: %d articles collected (%s to %s)",
        len(records), start_date, end_date,
    )
    return records