"""
scripts/collect_poverty_prices.py
---------------------------------
COMPREHENSIVE multi-axis collection of CALABARZON news on the two food-insecurity
transmission chains the thesis cares about:

    poverty -> household purchasing power -> food affordability -> food access
    food prices / inflation -> higher food costs -> reduced purchasing power

Axes (all run in one pass, RSS parallelised across 8 worker threads):
  1. Event Registry TOPIC concepts  x CALABARZON province keyword.
  2. Event Registry KEYWORD queries x CALABARZON province keyword (English +
     Filipino), covering poverty, income, wages, cost of living, inflation, and
     per-commodity prices (rice, meat, fish, vegetables, fruit).
  3. Google News RSS, PROVINCE level x 8 themed query groups.
  4. Google News RSS, MUNICIPALITY/CITY level (all 142 LGUs) x a combined
     poverty+prices+hunger query, so small LGUs are searched too.

Output: data/raw/poverty_prices_pool.parquet (raw; relevance is decided later by
scripts/merge_poverty_prices.py, which requires a meaningful food connection).
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

OUT = Path("data/raw/poverty_prices_pool.parquet")
PROVS = ["Batangas", "Cavite", "Laguna", "Rizal", "Quezon"]

CONCEPTS = {
    "Poverty": "http://en.wikipedia.org/wiki/Poverty",
    "Extreme_poverty": "http://en.wikipedia.org/wiki/Extreme_poverty",
    "Food_prices": "http://en.wikipedia.org/wiki/Food_prices",
    "Inflation": "http://en.wikipedia.org/wiki/Inflation",
    "Cost_of_living": "http://en.wikipedia.org/wiki/Cost_of_living",
    "Consumer_price_index": "http://en.wikipedia.org/wiki/Consumer_price_index",
    "Food_security": "http://en.wikipedia.org/wiki/Food_security",
    "Hunger": "http://en.wikipedia.org/wiki/Hunger",
    "Malnutrition": "http://en.wikipedia.org/wiki/Malnutrition",
    "Household_income": "http://en.wikipedia.org/wiki/Household_income",
    "Minimum_wage": "http://en.wikipedia.org/wiki/Minimum_wage",
    "Purchasing_power": "http://en.wikipedia.org/wiki/Purchasing_power",
    "Rice": "http://en.wikipedia.org/wiki/Rice",
    "Meat": "http://en.wikipedia.org/wiki/Meat",
    "Vegetable": "http://en.wikipedia.org/wiki/Vegetable",
}
KEYWORDS = [
    # poverty / income
    "poverty incidence", "poor families", "indigent families", "low income",
    "household income", "minimum wage", "cost of living", "purchasing power",
    "extreme poverty", "poverty threshold",
    # prices / inflation
    "food prices", "rice price", "price increase", "food inflation",
    "price of vegetables", "price of pork", "price of fish", "basic commodities",
    "price hike", "consumer price index", "sugar price", "onion price",
    # hunger / access
    "involuntary hunger", "food shortage", "food assistance", "hunger incidence",
    # Filipino
    "presyo ng bigas", "mahal na bilihin", "taas presyo", "kahirapan",
    "walang makain", "presyo ng gulay", "presyo ng karne", "gutom",
]

# Google News RSS themed groups (province level).
RSS_GROUPS = {
    "poverty": '(poverty OR "poverty incidence" OR "poor families" OR indigent OR kahirapan OR mahihirap)',
    "income": '("household income" OR "low income" OR "minimum wage" OR "cost of living" OR sahod OR sweldo OR "purchasing power")',
    "food_prices": '("food prices" OR "price of rice" OR "rice price" OR "food price increase" OR "presyo ng bigas" OR "mahal na bilihin" OR "taas ng presyo")',
    "inflation": '(inflation OR "food inflation" OR "consumer price index" OR "price hike" OR "basic commodities" OR bilihin)',
    "commodity_prices": '("price of pork" OR "price of chicken" OR "price of fish" OR "price of vegetables" OR "presyo ng gulay" OR "presyo ng karne" OR "presyo ng isda" OR "sugar price" OR "onion price")',
    "hunger": '(hunger OR gutom OR "walang makain" OR "food shortage" OR "kakulangan sa pagkain" OR malnutrition OR malnutrisyon)',
    "food_access": '("food security" OR "food insecurity" OR "food access" OR "cannot afford food" OR "food assistance" OR ayuda OR Kadiwa OR "food pack")',
    "supply_shock": '("supply shortage" OR "crop damage" OR "harvest loss" OR "agricultural damage" OR "supply disruption" OR "fish kill")',
}
# One combined query for every municipality/city.
RSS_LGU = ('(poverty OR kahirapan OR "food prices" OR "presyo ng bigas" OR '
           '"mahal na bilihin" OR inflation OR hunger OR gutom OR malnutrition OR '
           '"food shortage" OR "low income" OR ayuda OR "price increase" OR bilihin)')


def _rec(title, url, date, body, src):
    return {"title": (title or "").strip(), "link": url or "",
            "article_id": hashlib.md5((url or "").encode()).hexdigest(),
            "published": (date or "")[:16], "summary": (body or "")[:2000],
            "source_domain": (urlparse(url).netloc.lower().lstrip("www.") if url else ""),
            "fetcher_source": src}


def _er_post(key, query, page=1):
    try:
        r = requests.post("https://eventregistry.org/api/v1/article/getArticles",
                          data={"query": json.dumps(query), "resultType": "articles",
                                "articlesCount": "100", "articlesPage": str(page),
                                "apiKey": key}, timeout=90)
        return (r.json().get("articles", {}) or {})
    except Exception:
        return {}


def _er_concepts(key, rows):
    for name, uri in CONCEPTS.items():
        got, page = 0, 1
        while page <= 20:
            q = {"$query": {"$and": [
                {"conceptUri": uri},
                {"$or": [{"keyword": p} for p in PROVS]},
                {"dateStart": "2020-01-01", "dateEnd": "2026-08-21"},
            ]}}
            arts = _er_post(key, q, page)
            res = arts.get("results", [])
            if not res:
                break
            for a in res:
                rows.append(_rec(a.get("title"), a.get("url"), a.get("date"),
                                 a.get("body"), "er_poverty_prices"))
                got += 1
            if page * 100 >= arts.get("totalResults", 0):
                break
            page += 1
            time.sleep(0.5)
        print(f"  ER concept {name}: {got}", flush=True)


def _er_keywords(key, rows):
    for kw in KEYWORDS:
        q = {"$query": {"$and": [
            {"keyword": kw, "keywordLoc": "body"},
            {"$or": [{"keyword": p} for p in PROVS]},
            {"dateStart": "2020-01-01", "dateEnd": "2026-08-21"},
        ]}}
        arts = _er_post(key, q)
        res = arts.get("results", [])
        for a in res:
            rows.append(_rec(a.get("title"), a.get("url"), a.get("date"),
                             a.get("body"), "er_poverty_prices"))
        print(f"  ER kw {kw!r}: {len(res)}", flush=True)
        time.sleep(0.5)


def _rss_one(q, src):
    u = f"https://news.google.com/rss/search?q={quote(q)}&hl=en-PH&gl=PH&ceid=PH:en"
    delay, out = 2.0, []
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
                                pu.group(1) if pu else "", desc, src))
            return out
        if r.status_code in (429, 503):
            time.sleep(delay)
            delay = min(delay * 1.6, 25)
    return out


def _rss(rows, workers=8):
    from app.ml.corpus.gdelt_fetcher import CALABARZON_LGUS
    lgus = [l for ls in CALABARZON_LGUS.values() for l in ls]
    jobs = []
    for prov in PROVS:                              # province x themed groups
        for gname, gq in RSS_GROUPS.items():
            jobs.append((f'"{prov}" {gq}', f"gnews_pp_{gname}"))
    for lgu in lgus:                                # every municipality/city
        jobs.append((f'"{lgu}" {RSS_LGU}', "gnews_pp_lgu"))
    print(f"Google News RSS: {len(jobs)} queries "
          f"({len(PROVS)}x{len(RSS_GROUPS)} province + {len(lgus)} LGU), {workers} workers...",
          flush=True)
    lock, done = threading.Lock(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_rss_one, q, s) for q, s in jobs]
        for fut in as_completed(futs):
            got = fut.result()
            with lock:
                rows.extend(got)
                done += 1
                if done % 40 == 0:
                    print(f"    {done}/{len(jobs)}, {len(rows)} rows", flush=True)


def main():
    rows = []
    try:
        key = _key()
        _er_concepts(key, rows)
        _er_keywords(key, rows)
    except Exception as e:
        print(f"  ! ER skipped: {str(e)[:70]}")
    _rss(rows, workers=8)
    df = pd.DataFrame(rows)
    df = df[df["link"].str.len() > 0].drop_duplicates("article_id")

    def nt(s):
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
    df["_nt"] = df["title"].fillna("").map(nt)
    df = df.drop_duplicates("_nt").drop(columns=["_nt"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nsaved {len(df)} unique -> {OUT}")
    print(df["fetcher_source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
