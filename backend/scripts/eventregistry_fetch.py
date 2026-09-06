"""
scripts/eventregistry_fetch.py
-------------------------------
Fetch CALABARZON food-insecurity articles from Event Registry (NewsAPI.ai) —
a 150k-publisher index that covers outlets GDELT does not, so this captures
the recall gap. API key from .env (EVENTREGISTRY_API_KEY).

Per-province queries (free tier caps 15 keywords/query), English + Tagalog,
full pagination. Returns real article body text (no enrichment needed).
Province is NOT trusted from the query keyword ("Quezon" also matches Quezon
City / NCR) — the downstream geocoder (with NCR masking) decides province.

Output: data/raw/eventregistry_raw.parquet (corpus format)
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.corpus.rss_fetcher import CREDIBLE_DOMAINS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("eventregistry")

OUT = Path("data/raw/eventregistry_raw.parquet")
API = "https://eventregistry.org/api/v1/article/getArticles"

PROVINCES = ["Batangas", "Cavite", "Laguna", "Rizal", "Quezon"]
FOOD = ["food", "rice", "hunger", "farmer", "harvest", "poverty", "fisherfolk",
        "malnutrition", "ayuda", "crop", "palay", "drought", "food price"]
LANGS = ["eng", "tgl"]
PAGE = 100


def _key() -> str:
    for line in open(Path(__file__).resolve().parents[1] / ".env"):
        if line.startswith("EVENTREGISTRY_API_KEY"):
            return line.split("=", 1)[1].strip()
    raise SystemExit("EVENTREGISTRY_API_KEY not in .env")


def _domain(u: str) -> str:
    return urlparse(u or "").netloc.lower().lstrip("www.")


def _credible(u: str) -> bool:
    d = _domain(u)
    if d in CREDIBLE_DOMAINS:
        return True
    p = d.split(".")
    return any(".".join(p[i:]) in CREDIBLE_DOMAINS for i in range(1, len(p)))


def _fetch(key: str, prov: str, lang: str) -> list[dict]:
    q = {"$query": {"$and": [
        {"keyword": prov},
        {"$or": [{"keyword": k} for k in FOOD]},
        {"dateStart": "2020-01-01", "dateEnd": "2026-08-01", "lang": lang},
    ]}}
    out, page = [], 1
    while True:
        try:
            r = requests.post(API, data={
                "query": json.dumps(q), "resultType": "articles",
                "articlesCount": str(PAGE), "articlesPage": str(page),
                "articlesSortBy": "date", "apiKey": key,
            }, timeout=90)
            d = r.json()
        except Exception as exc:
            logger.warning("  %s/%s page %d error: %s", prov, lang, page, exc)
            break
        arts = d.get("articles", {}) if isinstance(d, dict) else {}
        results = arts.get("results", [])
        if not results:
            if isinstance(d, dict) and d.get("error"):
                logger.warning("  %s/%s: %s", prov, lang, d["error"])
            break
        for a in results:
            url = a.get("url", "")
            out.append({
                "title": (a.get("title") or "").strip(),
                "link": url,
                "article_id": hashlib.md5(url.encode()).hexdigest(),
                "published": (a.get("date") or "")[:10],
                "summary": (a.get("body") or "")[:500],
                "source_domain": _domain(url),
                "fetcher_source": "eventregistry",
            })
        total = arts.get("totalResults", 0)
        if page * PAGE >= total or page >= 50:
            break
        page += 1
        time.sleep(1.0)
    return out


def main() -> None:
    key = _key()
    rows: list[dict] = []
    for prov in PROVINCES:
        for lang in LANGS:
            got = _fetch(key, prov, lang)
            logger.info("%s/%s: %d articles", prov, lang, len(got))
            rows.extend(got)
    df = pd.DataFrame(rows).drop_duplicates(subset=["article_id"])
    logger.info("unique articles: %d", len(df))
    df["cred"] = df["link"].map(_credible)
    df = df[df["cred"]].drop(columns=["cred"])
    logger.info("credible-domain: %d", len(df))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    logger.info("saved -> %s (%d articles)", OUT, len(df))


if __name__ == "__main__":
    main()
