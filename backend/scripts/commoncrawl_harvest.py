"""
scripts/commoncrawl_harvest.py
-------------------------------
Common Crawl harvester for credible Philippine news domains.

Two things at once:
  1. DISCOVERY — food-slugged article URLs from the 17 credible news
     domains across every CC-MAIN crawl in 2020-2026, including articles
     no other fetcher surfaced.
  2. BUILT-IN ENRICHMENT — each hit's stored HTML is fetched straight
     from Common Crawl's data bucket (byte-range WARC reads), so records
     arrive with real title + lead paragraphs and never touch publisher
     servers or rate limits.

Output: data/raw/commoncrawl.parquet (corpus format, pre-enriched)
Checkpoints: data/raw/checkpoints_cc/ (per crawl-domain index pulls)

Usage:
  venv\\Scripts\\python scripts\\commoncrawl_harvest.py
  venv\\Scripts\\python scripts\\commoncrawl_harvest.py --crawls-limit 10
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import re
import sys
import time
from html import unescape
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cc_harvest")

INDEX_HOST = "https://index.commoncrawl.org"
DATA_HOST = "https://data.commoncrawl.org"
UA = {"User-Agent": "aiPHeed-thesis-research/1.0 (academic food-security study)"}

OUT = Path("data/raw/commoncrawl.parquet")
CKPT_DIR = Path("data/raw/checkpoints_cc")

DOMAINS = [
    "newsinfo.inquirer.net", "business.inquirer.net", "philstar.com",
    "gmanetwork.com/news", "news.abs-cbn.com", "mb.com.ph", "pna.gov.ph",
    "rappler.com", "manilatimes.net", "businessmirror.com.ph",
    "sunstar.com.ph", "tribune.net.ph", "journal.com.ph", "remate.ph",
    "abante.com.ph", "bworldonline.com", "manilastandard.net",
]

SLUG_RE = (
    r".*(food|rice|palay|bigas|hunger|gutom|famine|malnutri|feeding|ayuda|"
    r"relief|pantawid|4ps|kadiwa|nfa|subsid|farm|agri|magsasaka|harvest|"
    r"crop|fisher|isda|tilapia|asf|price|presyo|inflation|bilihin|palengke|"
    r"typhoon|bagyo|flood|baha|drought|el-nino|evacuat|bakwit|displac|"
    r"poverty|kahirapan|unemploy|shortage|kakulangan|onion|sibuyas|sugar|"
    r"asukal|fertilizer|vegetable|gulay|galunggong).*"
)

_META_TITLE = [
    re.compile(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', re.I),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:title', re.I),
    re.compile(r"<title[^>]*>([^<]+)</title>", re.I),
]
_P_TAG = re.compile(r"<p[^>]*>(.*?)</p>", re.I | re.S)
_TAG_STRIP = re.compile(r"<[^>]+>")
_BOILER = re.compile(
    r"cookie|subscribe|sign.?up|newsletter|all rights reserved|advertis|"
    r"follow us|read more|click here|terms of (use|service)|privacy policy|"
    r"copyright|by continuing", re.I)


def _clean(t: str) -> str:
    t = unescape(t).strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s*[|\-–]\s*[A-Z][\w .]{2,40}$", "", t)
    return t


def _title_lead(html: str) -> tuple[str, str]:
    title = ""
    for pat in _META_TITLE:
        m = pat.search(html)
        if m:
            title = _clean(m.group(1))
            if title:
                break
    out, total = [], 0
    for m in _P_TAG.finditer(html):
        text = _clean(_TAG_STRIP.sub(" ", m.group(1)))
        if len(text) < 60 or _BOILER.search(text):
            continue
        out.append(text)
        total += len(text)
        if total >= 500:
            break
    return title, " ".join(out)[:500]


def _get(url: str, params=None, tries: int = 5, timeout: int = 60):
    """GET with exponential backoff on 503 (index host is chronically busy)."""
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 503):
                time.sleep(8 * (attempt + 1))
                continue
            return None
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


_COLLINFO_CACHE = CKPT_DIR / "collinfo.json"


def _crawl_ids() -> list[str]:
    """
    Crawl id list, cached locally — the set changes ~monthly, and the index
    host throttles aggressively, so never let an unreachable host kill a
    run that could proceed from cache. If neither host nor cache is
    available, wait out the throttle in 30-min rests (up to 6 hours).
    """
    for attempt in range(12):
        r = _get(f"{INDEX_HOST}/collinfo.json")
        if r is not None:
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            _COLLINFO_CACHE.write_text(r.text, encoding="utf-8")
            break
        if _COLLINFO_CACHE.exists():
            logger.warning("index host unreachable — using cached collinfo")
            break
        logger.warning("index host unreachable and no cache — resting 30 min "
                       "(attempt %d/12)", attempt + 1)
        time.sleep(1800)
    else:
        raise SystemExit("Cannot reach Common Crawl index host after 6h")

    data = json.loads(_COLLINFO_CACHE.read_text(encoding="utf-8"))
    ids = [c["id"] for c in data]
    keep = [i for i in ids if re.match(r"CC-MAIN-202[0-6]-", i)]
    keep.sort()
    return keep


def _cdx_domain_crawl(crawl_id: str, domain: str) -> list[dict]:
    """
    All food-slugged 200-OK HTML captures for domain in one crawl.

    Only checkpoints a pair when the sweep ENDED CLEANLY (a page returned
    no rows). If the index server 503s out mid-sweep, the pair is left
    un-checkpointed so a later --resume retries it — an exhausted-retries
    "gave up" must never masquerade as "domain has no matches".
    """
    ckpt = CKPT_DIR / f"{crawl_id}_{domain.replace('/', '_')}.parquet"
    if ckpt.exists():
        return pd.read_parquet(ckpt).to_dict("records"), True

    rows: list[dict] = []
    page = 0
    clean_end = False
    while True:
        r = _get(
            f"{INDEX_HOST}/{crawl_id}-index",
            params=[
                ("url", f"{domain}/*"),
                ("output", "json"),
                ("filter", "=status:200"),
                ("filter", "~mime:.*html.*"),
                ("filter", f"~url:{SLUG_RE}"),
                ("collapse", "urlkey"),
                ("page", str(page)),
                ("pageSize", "5"),
            ],
            timeout=90,
        )
        if r is None:
            logger.warning("index gave up (503s) on %s %s page %d — will retry later",
                           crawl_id, domain, page)
            break
        lines = [ln for ln in r.text.strip().splitlines() if ln.startswith("{")]
        if not lines:
            clean_end = True
            break
        for ln in lines:
            try:
                d = json.loads(ln)
                rows.append({
                    "url": d.get("url", ""),
                    "timestamp": d.get("timestamp", ""),
                    "filename": d.get("filename", ""),
                    "offset": int(d.get("offset", 0)),
                    "length": int(d.get("length", 0)),
                })
            except Exception:
                continue
        page += 1
        time.sleep(2.0)
        if page > 300:
            clean_end = True
            break

    if clean_end:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(ckpt, index=False)
    # Politeness gap between pairs — the index host throttles sustained load.
    time.sleep(5.0)
    return rows, clean_end


def _fetch_warc_html(filename: str, offset: int, length: int) -> str:
    """Byte-range fetch one WARC record from the CC data bucket; return HTML."""
    hdrs = dict(UA)
    hdrs["Range"] = f"bytes={offset}-{offset + length - 1}"
    for attempt in range(3):
        try:
            r = requests.get(f"{DATA_HOST}/{filename}", headers=hdrs, timeout=60)
            if r.status_code in (200, 206):
                raw = gzip.GzipFile(fileobj=io.BytesIO(r.content)).read()
                text = raw.decode("utf-8", errors="replace")
                # WARC record = warc headers \r\n\r\n http headers \r\n\r\n body
                parts = text.split("\r\n\r\n", 2)
                return parts[2] if len(parts) == 3 else text
            if r.status_code in (429, 503):
                time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(3 * (attempt + 1))
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawls-limit", type=int, default=0,
                    help="use only the N most recent crawls (0 = all)")
    args = ap.parse_args()

    crawls = _crawl_ids()
    if args.crawls_limit:
        crawls = crawls[-args.crawls_limit:]
    logger.info("Crawls in window: %d | domains: %d", len(crawls), len(DOMAINS))

    # ── Phase 1: index sweep (checkpointed per crawl-domain) ─────────────
    index_rows: list[dict] = []
    total_pairs = len(crawls) * len(DOMAINS)
    done_pairs = 0
    consecutive_gaveups = 0
    for crawl_id in crawls:
        for domain in DOMAINS:
            rows, clean = _cdx_domain_crawl(crawl_id, domain)
            index_rows.extend(rows)
            done_pairs += 1
            if clean:
                consecutive_gaveups = 0
            else:
                consecutive_gaveups += 1
                if consecutive_gaveups >= 5:
                    logger.warning(
                        "index host storm (5 consecutive give-ups) — "
                        "resting 30 min before continuing")
                    time.sleep(1800)
                    consecutive_gaveups = 0
            if done_pairs % 20 == 0:
                logger.info("index sweep %d/%d pairs — %d capture rows",
                            done_pairs, total_pairs, len(index_rows))

    idx = pd.DataFrame(index_rows).drop_duplicates(subset=["url"])
    logger.info("Index sweep complete: %d unique food-slugged URLs", len(idx))

    # ── Phase 2: WARC fetch + extract (resumable via partial output) ─────
    done_ids: set[str] = set()
    records: list[dict] = []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        records = prev.to_dict("records")
        done_ids = set(prev["article_id"])
        logger.info("Resuming WARC phase: %d already extracted", len(done_ids))

    since_save = 0
    for i, row in enumerate(idx.to_dict("records"), 1):
        aid = hashlib.md5(row["url"].encode()).hexdigest()
        if aid in done_ids:
            continue
        html = _fetch_warc_html(row["filename"], row["offset"], row["length"])
        title, lead = _title_lead(html) if html else ("", "")
        if not title or len(title) < 15:
            continue
        ts = row["timestamp"]
        published = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ""
        records.append({
            "title": title[:300],
            "link": row["url"],
            "article_id": aid,
            "published": published,
            "summary": lead,
            "source_domain": urlparse(row["url"]).netloc.lower().lstrip("www."),
            "fetcher_source": "commoncrawl",
        })
        done_ids.add(aid)
        since_save += 1
        if since_save >= 1000:
            pd.DataFrame(records).to_parquet(OUT, index=False)
            since_save = 0
            logger.info("WARC extracted %d/%d — %d article records",
                        i, len(idx), len(records))

    pd.DataFrame(records).to_parquet(OUT, index=False)
    logger.info("Common Crawl harvest done: %d pre-enriched articles -> %s",
                len(records), OUT)


if __name__ == "__main__":
    main()
