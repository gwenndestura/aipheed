"""
app/ml/corpus/psa_fetcher.py
----------------------------
Automated PSA report discovery and download.

Targets two official government statistical sources for CALABARZON:
  - https://psa.gov.ph        (national PSA — main releases)
  - https://rsso04a.psa.gov.ph (RSSO IV-A — regional bulletins)

Report series fetched:
  1. Monthly Price Survey bulletins  → food CPI + rice retail price
  2. Regional Labor Force Survey     → quarterly unemployment rate
  3. Full Year Poverty Statistics    → poverty incidence, CALABARZON

Usage:
    from app.ml.corpus.psa_fetcher import fetch_psa_reports
    records = fetch_psa_reports("2020-01-01", "2025-12-31")
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import pdfplumber
from bs4 import BeautifulSoup

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PSA_DOMAINS = {
    "psa.gov.ph",
    "rsso04a.psa.gov.ph",
}

# Local storage root — gitignored
RAW_DIR = Path("data/raw/psa_reports")

# Target listing pages per report series
LISTING_PAGES: list[dict] = [
    # --- Monthly Price Survey (food CPI + rice retail price) ---
    {
        "report_type": "price_survey",
        "url": "https://psa.gov.ph/price-indices/seasonally-adjusted-cpi/index",
        "domain": "rsso04a.psa.gov.ph",
    },
    {
        "report_type": "price_survey",
        "url": "https://psa.gov.ph/price-indices/cpi-ir",
        "domain": "psa.gov.ph",
    },
    # --- Regional Labor Force Survey (unemployment) ---
    {
        "report_type": "labor_force_survey",
        "url": "https://psa.gov.ph/taxonomy/term/111",
        "domain": "rsso04a.psa.gov.ph",
    },
    {
        "report_type": "labor_force_survey",
        "url": "https://psa.gov.ph/statistics/labor-force-survey",
        "domain": "psa.gov.ph",
    },
    # --- Poverty Statistics CALABARZON ---
    {
        "report_type": "poverty_statistics",
        "url": "https://psa.gov.ph/statistics/poverty",
        "domain": "rsso04a.psa.gov.ph",
    },
    {
        "report_type": "poverty_statistics",
        "url": "https://psa.gov.ph/poverty-press-releases",
        "domain": "psa.gov.ph",
    },
]

# Request throttle (seconds between HTTP calls — be a polite crawler)
CRAWL_DELAY = 1.5

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
# Robots.txt compliance
# ---------------------------------------------------------------------------

_robots_cache: dict[str, RobotFileParser] = {}


def _get_robot_parser(base_url: str) -> RobotFileParser:
    """Return a cached RobotFileParser for the given base URL."""
    if base_url in _robots_cache:
        return _robots_cache[base_url]
    rp = RobotFileParser()
    robots_url = urljoin(base_url, "/robots.txt")
    try:
        rp.set_url(robots_url)
        rp.read()
        logger.info("robots.txt loaded: %s", robots_url)
    except Exception as exc:
        logger.warning("Could not load robots.txt for %s: %s", base_url, exc)
    _robots_cache[base_url] = rp
    return rp


def _is_allowed(url: str) -> bool:
    """Return True if our user-agent is allowed to fetch this URL."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = _get_robot_parser(base)
    allowed = rp.can_fetch(SESSION.headers["User-Agent"], url)
    if not allowed:
        logger.warning("robots.txt disallows: %s", url)
    return allowed


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------

def _is_psa_domain(url: str) -> bool:
    """
    Strict domain check — only accept URLs from PSA_DOMAINS.
    Rejects any redirect or link pointing to non-PSA hosts.
    """
    parsed = urlparse(url)
    hostname = parsed.netloc.lower().lstrip("www.")
    return any(hostname == d or hostname.endswith("." + d) for d in PSA_DOMAINS)


# ---------------------------------------------------------------------------
# URL-hash deduplication
# ---------------------------------------------------------------------------

_seen_urls: set[str] = set()


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


def _already_fetched(url: str) -> bool:
    h = _url_hash(url)
    if h in _seen_urls:
        return True
    _seen_urls.add(h)
    return False


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _safe_get(url: str, timeout: int = 20) -> Optional[requests.Response]:
    """GET with robots.txt compliance, domain validation, and error handling."""
    if not _is_psa_domain(url):
        logger.error("Domain validation failed — rejected: %s", url)
        return None
    if not _is_allowed(url):
        return None
    try:
        response = SESSION.get(url, timeout=timeout, allow_redirects=True)
        # Re-validate after any redirect
        if not _is_psa_domain(response.url):
            logger.error(
                "Redirect to non-PSA domain detected: %s → %s", url, response.url
            )
            return None
        logger.info(
            "[%s] GET %s", response.status_code, url,
            extra={"timestamp": datetime.utcnow().isoformat(), "http_status": response.status_code},
        )
        return response
    except requests.RequestException as exc:
        logger.error("Request failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Link extraction from listing pages
# ---------------------------------------------------------------------------

def _extract_pdf_html_links(
    listing_url: str, base_domain: str
) -> list[str]:
    """
    Scrape a PSA listing page and return all PDF / HTML article links
    that belong to the same PSA domain.
    """
    resp = _safe_get(listing_url)
    if resp is None or resp.status_code != 200:
        return []
    time.sleep(CRAWL_DELAY)

    soup = BeautifulSoup(resp.text, "html.parser")
    links: list[str] = []

    for a_tag in soup.find_all("a", href=True):
        href: str = a_tag["href"].strip()
        # Resolve relative URLs
        full_url = urljoin(listing_url, href)
        # Only keep PSA-domain links
        if not _is_psa_domain(full_url):
            continue
        # Focus on PDF files and article/press-release pages
        lower = full_url.lower()
        if lower.endswith(".pdf") or "/article/" in lower or "/press-release/" in lower:
            links.append(full_url)

    logger.info("Found %d candidate links on %s", len(links), listing_url)
    return links


# ---------------------------------------------------------------------------
# Quarter inference from URL / title text
# ---------------------------------------------------------------------------

import re

_QUARTER_PATTERNS = [
    # "Q1 2024", "Q2-2023"
    re.compile(r"Q([1-4])[\s\-_]*(20\d{2})", re.IGNORECASE),
    # "2024-Q3", "2024Q4"
    re.compile(r"(20\d{2})[\s\-_]*Q([1-4])", re.IGNORECASE),
    # "January 2024", "March-2023"
    re.compile(r"(January|February|March|April|May|June|July|August|"
               r"September|October|November|December)[\s\-,]*(20\d{2})", re.IGNORECASE),
]

_MONTH_TO_Q = {
    "january": "Q1", "february": "Q1", "march": "Q1",
    "april": "Q2", "may": "Q2", "june": "Q2",
    "july": "Q3", "august": "Q3", "september": "Q3",
    "october": "Q4", "november": "Q4", "december": "Q4",
}


def _infer_year_quarter(text: str) -> str:
    """Best-effort year_quarter extraction from a URL or title string."""
    for pat in _QUARTER_PATTERNS:
        m = pat.search(text)
        if m:
            groups = m.groups()
            if len(groups) == 2:
                if groups[0].isdigit() and len(groups[0]) == 1:
                    # Q1 2024
                    return f"{groups[1]}_Q{groups[0]}"
                elif groups[1].isdigit() and len(groups[1]) == 1:
                    # 2024 Q3
                    return f"{groups[0]}_Q{groups[1]}"
                else:
                    # month year
                    month = groups[0].lower()
                    year = groups[1]
                    quarter = _MONTH_TO_Q.get(month, "Q1")
                    return f"{year}_{quarter}"
    return "unknown_quarter"


# ---------------------------------------------------------------------------
# File download + save
# ---------------------------------------------------------------------------

def _download_and_save(
    url: str, report_type: str, year_quarter: str
) -> Optional[Path]:
    """
    Download a PDF or HTML file and save to:
        data/raw/psa_reports/{report_type}/{year_quarter}.pdf|html

    Returns the local path on success, None on failure.
    Skips if local file already exists (URL-hash dedup also in place).
    """
    ext = ".pdf" if url.lower().endswith(".pdf") else ".html"
    out_dir = RAW_DIR / report_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year_quarter}{ext}"

    if out_path.exists():
        logger.info("Already on disk, skipping: %s", out_path)
        return out_path

    resp = _safe_get(url)
    if resp is None or resp.status_code != 200:
        return None
    time.sleep(CRAWL_DELAY)

    out_path.write_bytes(resp.content)
    logger.info(
        "Saved %s → %s (%d bytes)",
        url, out_path, len(resp.content),
    )
    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_psa_reports(
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Discover and download PSA statistical bulletins for CALABARZON.

    Parameters
    ----------
    start_date : str
        ISO date string e.g. "2020-01-01"
    end_date : str
        ISO date string e.g. "2025-12-31"

    Returns
    -------
    list[dict]
        Each record contains:
            report_type   : str   — "price_survey" | "labor_force_survey" | "poverty_statistics"
            year_quarter  : str   — e.g. "2024_Q2"
            source_url    : str   — original PSA URL
            local_path    : str   — saved file path (gitignored)
            fetched_at    : str   — UTC ISO timestamp
            http_status   : int   — HTTP status of final download
    """
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    records: list[dict] = []

    for page_config in LISTING_PAGES:
        report_type = page_config["report_type"]
        listing_url = page_config["url"]

        logger.info("Scanning listing page [%s]: %s", report_type, listing_url)
        links = _extract_pdf_html_links(listing_url, page_config["domain"])

        for link in links:
            if _already_fetched(link):
                logger.debug("Duplicate URL, skipping: %s", link)
                continue

            year_quarter = _infer_year_quarter(link)

            # Best-effort date filtering using year from year_quarter
            year_str = year_quarter.split("_")[0]
            if year_str.isdigit():
                year = int(year_str)
                if not (start_dt.year <= year <= end_dt.year):
                    logger.debug(
                        "Year %d outside range [%d, %d], skipping: %s",
                        year, start_dt.year, end_dt.year, link,
                    )
                    continue

            fetched_at = datetime.utcnow().isoformat()
            resp_check = _safe_get(link)
            http_status = resp_check.status_code if resp_check else 0
            time.sleep(CRAWL_DELAY)

            local_path = _download_and_save(link, report_type, year_quarter)

            record = {
                "report_type": report_type,
                "year_quarter": year_quarter,
                "source_url": link,
                "local_path": str(local_path) if local_path else None,
                "fetched_at": fetched_at,
                "http_status": http_status,
            }
            records.append(record)
            logger.info(
                "Fetched record: type=%s quarter=%s status=%s",
                report_type, year_quarter, http_status,
            )

    logger.info(
        "fetch_psa_reports complete: %d records fetched for %s – %s",
        len(records), start_date, end_date,
    )
    return records


# ---------------------------------------------------------------------------
# Output schema reference (for Member B / DB schema design)
# ---------------------------------------------------------------------------
#
# fetch_psa_reports() returns list[dict] with these fields:
#
#   report_type   : str   — one of "price_survey", "labor_force_survey",
#                           "poverty_statistics"
#   year_quarter  : str   — "{YYYY}_Q{N}" e.g. "2024_Q2"
#   source_url    : str   — the original PSA URL crawled
#   local_path    : str   — path under data/raw/psa_reports/ (gitignored)
#   fetched_at    : str   — UTC ISO-8601 timestamp of download
#   http_status   : int   — HTTP status code of the download response
#
# Downstream parsed fields produced by psa_report_parser.py:
#   province_code : str   — PSGC code (e.g. "PH040100000")
#   quarter       : str   — "{YYYY}-Q{N}" e.g. "2024-Q2"
#   food_cpi      : float — Monthly Price Survey food CPI (price_survey)
#   rice_price    : float — Rice retail price, PHP/kg  (price_survey)
#   unemployment  : float — Unemployment rate, %        (labor_force_survey)
#   poverty_incidence : float — Poverty incidence, %   (poverty_statistics)
#   population    : int   — Municipal/provincial pop    (poverty_statistics)