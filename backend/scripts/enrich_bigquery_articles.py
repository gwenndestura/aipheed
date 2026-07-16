"""
scripts/enrich_bigquery_articles.py
------------------------------------
Fetch real headlines + descriptions for the BigQuery-harvested articles.

The BigQuery GDELT harvest yields URLs only; slug-derived pseudo-titles
starve the XLM-R NLI scorer (a Taal-eruption evacuation article scored
0.027 from its slug vs ~0.56 with real headline+description). This script
GETs each publisher page and extracts og:title / og:description (falling
back to <title> / meta description), writing them into the title/summary
fields the thesis pipeline scores on.

- Per-domain politeness: >= 1.0s between hits to the same domain
- 6 concurrent workers across domains, 12s timeout, 1 retry
- Checkpoints every 2,000 articles (resumable)
- Articles whose fetch fails keep their slug title (enriched=False)

Usage:
  venv\\Scripts\\python scripts\\enrich_bigquery_articles.py
"""

from __future__ import annotations

import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
logger = logging.getLogger("enrich_bq")

SRC = Path("data/raw/gdelt_bigquery.parquet")
OUT = Path("data/raw/gdelt_bigquery_enriched.parquet")
CKPT = Path("data/raw/checkpoints_enrich/enrich_progress.parquet")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

_domain_lock = threading.Lock()
_domain_last: dict[str, float] = {}

_META_RE = {
    "og_title": re.compile(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', re.I),
    "og_title2": re.compile(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:title', re.I),
    "og_desc": re.compile(
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)', re.I),
    "og_desc2": re.compile(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:description', re.I),
    "meta_desc": re.compile(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)', re.I),
    "title_tag": re.compile(r"<title[^>]*>([^<]+)</title>", re.I),
}


def _polite_wait(domain: str) -> None:
    """Ensure >=1s between requests to the same domain."""
    while True:
        with _domain_lock:
            last = _domain_last.get(domain, 0.0)
            now = time.time()
            if now - last >= 1.0:
                _domain_last[domain] = now
                return
            wait = 1.0 - (now - last)
        time.sleep(wait)


def _clean(text: str) -> str:
    text = unescape(text).strip()
    text = re.sub(r"\s+", " ", text)
    # Strip trailing " - Publisher" / " | Publisher"
    text = re.sub(r"\s*[|\-–]\s*[A-Z][\w .]{2,40}$", "", text)
    return text


def _extract(html: str) -> tuple[str, str]:
    title = ""
    desc = ""
    for key in ("og_title", "og_title2", "title_tag"):
        m = _META_RE[key].search(html)
        if m:
            title = _clean(m.group(1))
            if title:
                break
    for key in ("og_desc", "og_desc2", "meta_desc"):
        m = _META_RE[key].search(html)
        if m:
            desc = _clean(m.group(1))
            if desc:
                break
    return title, desc


def _fetch_one(rec: dict) -> dict:
    url = rec["link"]
    domain = urlparse(url).netloc.lower()
    for attempt in range(2):
        _polite_wait(domain)
        try:
            r = requests.get(
                url,
                headers={"User-Agent": UA, "Accept-Language": "en-PH,en;q=0.9"},
                timeout=12,
                allow_redirects=True,
            )
            if r.status_code != 200:
                continue
            title, desc = _extract(r.text[:120_000])
            if title and len(title) >= 15:
                return {
                    "article_id": rec["article_id"],
                    "real_title": title[:300],
                    "real_summary": desc[:500],
                    "enriched": True,
                }
            return {"article_id": rec["article_id"], "real_title": "",
                    "real_summary": "", "enriched": False}
        except Exception:
            time.sleep(1.0 + attempt)
    return {"article_id": rec["article_id"], "real_title": "",
            "real_summary": "", "enriched": False}


def main() -> None:
    df = pd.read_parquet(SRC)
    logger.info("Articles to enrich: %d", len(df))

    done: dict[str, dict] = {}
    if CKPT.exists():
        prev = pd.read_parquet(CKPT)
        done = {r["article_id"]: r for r in prev.to_dict("records")}
        logger.info("Resuming: %d already processed", len(done))

    todo = [r for r in df.to_dict("records") if r["article_id"] not in done]
    results: list[dict] = list(done.values())
    n_start = len(results)

    CKPT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(_fetch_one, r) for r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 2000 == 0:
                pd.DataFrame(results).to_parquet(CKPT, index=False)
                ok = sum(1 for r in results if r.get("enriched"))
                rate = i / max(time.time() - t0, 1)
                logger.info(
                    "enriched %d/%d (%.0f%%) — %d with real titles — %.1f req/s",
                    n_start + i, len(df), (n_start + i) / len(df) * 100, ok, rate,
                )

    res_df = pd.DataFrame(results)
    res_df.to_parquet(CKPT, index=False)

    merged = df.merge(res_df, on="article_id", how="left")
    use_real = merged["enriched"].fillna(False).astype(bool)
    merged.loc[use_real, "title"] = merged.loc[use_real, "real_title"]
    merged.loc[use_real, "summary"] = merged.loc[use_real, "real_summary"]
    merged = merged.drop(columns=["real_title", "real_summary"])

    merged.to_parquet(OUT, index=False)
    ok = int(use_real.sum())
    logger.info("Saved %s — %d/%d articles enriched with real title/description",
                OUT, ok, len(merged))


if __name__ == "__main__":
    main()
