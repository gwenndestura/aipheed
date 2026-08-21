"""
scripts/collect_unemployment.py
-------------------------------
ACTIVE collection of unemployment / job-loss news for CALABARZON from the live
APIs (not just a sweep of already-collected pools). Three axes:

  1. Event Registry TOPIC concepts (Unemployment / Layoff / Factory / Labour
     economics) constrained to mention a CALABARZON province.
  2. Event Registry keyword queries (retrenchment / plant closure / job losses ...
     AND a CALABARZON province).
  3. Google News RSS per-LGU x an unemployment/closure/TUPAD query string.

Saves to data/raw/unemployment_pool.parquet (a persistent raw pool). The
economic-shock guard + tagging then runs the normal way:
    python scripts/collect_economic_shock.py            # re-sweeps all pools
    python scripts/collect_economic_shock.py --merge    # guard -> tag -> append
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlparse

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eventregistry_cover_lgus import _key  # noqa: E402

OUT = Path("data/raw/unemployment_pool.parquet")

CONCEPTS = {
    "Unemployment": "http://en.wikipedia.org/wiki/Unemployment",
    "Layoff": "http://en.wikipedia.org/wiki/Layoff",
    "Factory": "http://en.wikipedia.org/wiki/Factory",
    "Labour_economics": "http://en.wikipedia.org/wiki/Labour_economics",
}
KEYWORDS = ["retrenchment", "plant closure", "factory closure", "job losses",
            "layoff", "displaced workers", "mass layoff", "business closure"]
PROVS = ["Batangas", "Cavite", "Laguna", "Rizal", "Quezon"]

GNEWS_Q = ('("job losses" OR "job cuts" OR layoff OR layoffs OR retrenchment OR '
           '"plant closure" OR "factory closure" OR "factory shutdown" OR '
           '"displaced workers" OR "lost their jobs" OR "mass layoff" OR jobless OR '
           'unemployment OR TUPAD OR "nawalan ng trabaho" OR "tanggal sa trabaho" OR '
           '"sarang planta" OR "closed its plant" OR "ceased operations")')


def _rec(title, url, date, body, src):
    return {"title": (title or "").strip(), "link": url,
            "article_id": hashlib.md5((url or "").encode()).hexdigest(),
            "published": (date or "")[:16], "summary": (body or "")[:2000],
            "source_domain": (urlparse(url).netloc.lower().lstrip("www.") if url else ""),
            "fetcher_source": src}


def _er_concepts(key, rows):
    for name, uri in CONCEPTS.items():
        page, got = 1, 0
        while page <= 20:
            q = {"$query": {"$and": [
                {"conceptUri": uri},
                {"$or": [{"keyword": p} for p in PROVS]},
                {"dateStart": "2020-01-01", "dateEnd": "2026-08-20"},
            ]}}
            try:
                r = requests.post("https://eventregistry.org/api/v1/article/getArticles",
                                  data={"query": json.dumps(q), "resultType": "articles",
                                        "articlesCount": "100", "articlesPage": str(page),
                                        "apiKey": key}, timeout=90)
                arts = (r.json().get("articles", {}) or {})
            except Exception:
                break
            res = arts.get("results", [])
            if not res:
                break
            for a in res:
                rows.append(_rec(a.get("title"), a.get("url"), a.get("date"),
                                 a.get("body"), "er_unemployment"))
                got += 1
            if page * 100 >= arts.get("totalResults", 0):
                break
            page += 1
            time.sleep(0.6)
        print(f"  ER concept {name}: {got}", flush=True)


def _er_keywords(key, rows):
    for kw in KEYWORDS:
        q = {"$query": {"$and": [
            {"keyword": kw, "keywordLoc": "body"},
            {"$or": [{"keyword": p} for p in PROVS]},
            {"dateStart": "2020-01-01", "dateEnd": "2026-08-20"},
        ]}}
        try:
            r = requests.post("https://eventregistry.org/api/v1/article/getArticles",
                              data={"query": json.dumps(q), "resultType": "articles",
                                    "articlesCount": "100", "articlesPage": "1",
                                    "apiKey": key}, timeout=90)
            res = (r.json().get("articles", {}) or {}).get("results", [])
        except Exception:
            res = []
        for a in res:
            rows.append(_rec(a.get("title"), a.get("url"), a.get("date"),
                             a.get("body"), "er_unemployment"))
        print(f"  ER kw '{kw}': {len(res)}", flush=True)
        time.sleep(0.6)


def _gnews_one(lgu):
    """Fetch one LGU's unemployment RSS, with per-worker 429 backoff."""
    u = (f"https://news.google.com/rss/search?q={quote(f'\"{lgu}\" {GNEWS_Q}')}"
         f"&hl=en-PH&gl=PH&ceid=PH:en")
    delay = 2.0
    out = []
    for _ in range(5):
        try:
            r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        except Exception:
            return out
        if r.status_code == 200:
            for m in re.findall(r"<item>(.*?)</item>", r.text, re.S):
                t = re.search(r"<title>(.*?)</title>", m, re.S)
                de = re.search(r"<description>(.*?)</description>", m, re.S)
                pu = re.search(r"<pubDate>(.*?)</pubDate>", m, re.S)
                li = re.search(r"<link>(.*?)</link>", m, re.S)
                title = html.unescape(re.sub("<[^>]+>", "", t.group(1))) if t else ""
                desc = html.unescape(re.sub("<[^>]+>", "", de.group(1))) if de else ""
                out.append(_rec(title, li.group(1) if li else "",
                                pu.group(1) if pu else "", desc, "gnews_unemployment"))
            return out
        if r.status_code in (429, 503):
            time.sleep(delay)
            delay = min(delay * 1.6, 25)
    return out


def _gnews(rows, workers=8):
    from app.ml.corpus.gdelt_fetcher import CALABARZON_LGUS
    lgus = [l for ls in CALABARZON_LGUS.values() for l in ls]
    print(f"Google News RSS per-LGU unemployment ({len(lgus)} LGUs, {workers} workers)...", flush=True)
    lock = threading.Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_gnews_one, lgu): lgu for lgu in lgus}
        for fut in as_completed(futs):
            got = fut.result()
            with lock:
                rows.extend(got)
                done += 1
                if done % 30 == 0:
                    print(f"    {done}/{len(lgus)}, {len(rows)} rows", flush=True)


def main():
    rows = []
    try:
        key = _key()
        _er_concepts(key, rows)
        _er_keywords(key, rows)
    except Exception as e:
        print(f"  ! ER skipped: {str(e)[:60]}")
    _gnews(rows)
    df = pd.DataFrame(rows)
    df = df[df["link"].str.len() > 0].drop_duplicates("article_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nsaved {len(df)} unique -> {OUT}")


if __name__ == "__main__":
    main()
