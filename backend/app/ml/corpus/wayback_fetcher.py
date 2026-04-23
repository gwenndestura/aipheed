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

CDX_API_URL = "http://web.archive.org/cdx/search/cdx"

# Crawl delay for Wayback CDX requests (their robots.txt requests politeness)
WAYBACK_CRAWL_DELAY = 2.0

# Max snapshots to retrieve per domain query (avoid massive downloads)
CDX_LIMIT_PER_DOMAIN = 200

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

def _cdx_query(
    domain: str,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
) -> list[dict]:
    """
    Query Wayback CDX API for snapshots of `domain` between the given dates.
    Restricted to the domain itself — no wildcard sub-path crawling.

    Returns list of dicts with: original_url, timestamp, statuscode, mimetype.
    """
    params = {
        "url": f"{domain}/*",
        "output": "json",
        "fl": "original,timestamp,statuscode,mimetype",
        "filter": "statuscode:200",
        "from": start_yyyymmdd,
        "to": end_yyyymmdd,
        "limit": CDX_LIMIT_PER_DOMAIN,
        "collapse": "urlkey",  # one snapshot per unique URL
    }

    cdx_url = CDX_API_URL
    if not _wayback_allowed(cdx_url):
        logger.warning("Wayback robots.txt disallows CDX access. Skipping.")
        return []

    try:
        resp = SESSION.get(cdx_url, params=params, timeout=30)
        time.sleep(WAYBACK_CRAWL_DELAY)
    except requests.RequestException as exc:
        logger.error("CDX request error for domain %s: %s", domain, exc)
        return []

    if resp.status_code != 200:
        logger.error("CDX returned HTTP %d for domain %s", resp.status_code, domain)
        return []

    try:
        rows = resp.json()
    except Exception as exc:
        logger.error("CDX JSON parse error for domain %s: %s", domain, exc)
        return []

    if not rows or len(rows) < 2:
        return []

    # First row is the header
    header = rows[0]
    results = []
    for row in rows[1:]:
        record = dict(zip(header, row))
        results.append(record)

    logger.info("CDX: %d snapshots for domain=%s (%s–%s)", len(results), domain, start_yyyymmdd, end_yyyymmdd)
    return results


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

def fetch_wayback_articles(
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Fetch historical articles (2020–2022 gap-fill) via Wayback Machine CDX API.

    Restricted to CREDIBLE_DOMAINS only — never crawls arbitrary archived pages.
    Applies robots.txt compliance and URL-hash deduplication.

    Parameters
    ----------
    start_date : str   — ISO date e.g. "2020-01-01"
    end_date   : str   — ISO date e.g. "2022-12-31"

    Returns
    -------
    list[dict]
        Each record:
            title         : str
            link          : str   — original (non-Wayback) URL
            published     : str   — ISO UTC datetime
            summary       : str
            source_domain : str   — domain from CREDIBLE_DOMAINS
    """
    # Convert dates to CDX format (YYYYMMDDHHmmss)
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    start_cdx = start_dt.strftime("%Y%m%d000000")
    end_cdx = end_dt.strftime("%Y%m%d235959")

    records: list[dict] = []

    for domain in sorted(CREDIBLE_DOMAINS):
        logger.info("CDX query for domain: %s (%s – %s)", domain, start_cdx, end_cdx)
        snapshots = _cdx_query(domain, start_cdx, end_cdx)

        for snap in snapshots:
            original_url: str = snap.get("original", "")
            timestamp: str = snap.get("timestamp", "")

            if not original_url or not timestamp:
                continue

            # 1. Domain validation — only CREDIBLE_DOMAINS (redundant safety check)
            if not _is_credible_domain(original_url):
                logger.debug("Non-credible CDX result discarded: %s", original_url)
                continue

            # 2. URL deduplication
            if _already_seen(original_url):
                continue

            # 3. Fetch snapshot metadata (title + summary)
            meta = _fetch_snapshot_metadata(original_url, timestamp)
            if meta is None:
                # Include stub record without content — downstream parser may fill it
                meta = {"title": "", "summary": "", "snapshot_url": ""}

            # 4. Keyword filter
            if not _matches_keywords(meta["title"], meta["summary"]):
                continue

            published_iso = _cdx_ts_to_iso(timestamp)
            source_domain = _extract_domain(original_url)

            records.append(
                {
                    "title": meta["title"].strip(),
                    "link": original_url.strip(),
                    "published": published_iso,
                    "summary": meta["summary"].strip(),
                    "source_domain": source_domain,
                }
            )

    logger.info(
        "fetch_wayback_articles: %d articles collected (%s to %s)",
        len(records), start_date, end_date,
    )
    return records