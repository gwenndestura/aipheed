"""
scripts/collect_deep.py
------------------------
Comprehensive multi-source deep collection of CALABARZON food-insecurity news,
beyond the province-level Event Registry sweep already done. Adds:

  1. Event Registry PER-LGU (all 142) — catches articles that name a town but
     not its province (missed by the province-keyword query), paginated.
  2. NewsData.io — CALABARZON + food-insecurity terms.
  3. Google Custom Search (CSE) — food-insecurity site queries per province.

All new articles are pooled (full body where available), de-duplicated, and
written to data/raw/deep_pool.parquet. Run `--merge` to score them on 12 cores,
geocode, keep the relevant CALABARZON ones, and append to corpus_geocoded.

Usage:
  venv\\Scripts\\python scripts\\collect_deep.py            # collect
  venv\\Scripts\\python scripts\\collect_deep.py --merge    # score+geocode+append (12-core)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eventregistry_cover_lgus import _key, FOOD, _COLLIDING  # noqa: E402

OUT = Path("data/raw/deep_pool.parquet")
GEO = Path("data/processed/corpus_geocoded.parquet")
GAZ = Path("data/processed/psgc_gazetteer.parquet")


def _env(name: str) -> str | None:
    for l in open(Path(__file__).resolve().parents[1] / ".env", encoding="utf-8"):
        if l.startswith(name):
            return l.split("=", 1)[1].strip()
    return None


def _domain(u: str) -> str:
    return urlparse(u or "").netloc.lower().lstrip("www.")


def _rec(title, url, date, body, src) -> dict:
    return {"title": (title or "").strip(), "link": url,
            "article_id": hashlib.md5((url or "").encode()).hexdigest(),
            "published": (date or "")[:10], "summary": (body or "")[:2000],
            "source_domain": _domain(url), "fetcher_source": src}


# ── 1. Event Registry per-LGU (all 142), paginated ──────────────────────────
def _er_lgu(key, lgu, prov, pages=3) -> list[dict]:
    place = _COLLIDING.get(lgu, lgu)
    used = len(place.split()) + len(prov.split())
    food = FOOD[:max(3, 15 - used)]
    out, page = [], 1
    while page <= pages:
        q = {"$query": {"$and": [
            {"keyword": place, "keywordLoc": "body,title"},
            {"keyword": prov, "keywordLoc": "body,title"},
            {"$or": [{"keyword": k} for k in food]},
            {"dateStart": "2020-01-01", "dateEnd": "2026-08-20"},
        ]}}
        try:
            r = requests.post("https://eventregistry.org/api/v1/article/getArticles",
                              data={"query": json.dumps(q), "resultType": "articles",
                                    "articlesCount": "100", "articlesPage": str(page),
                                    "apiKey": key}, timeout=90)
            d = r.json()
        except Exception:
            break
        arts = (d.get("articles", {}) or {}) if isinstance(d, dict) else {}
        res = arts.get("results", [])
        if not res:
            break
        for a in res:
            out.append(_rec(a.get("title"), a.get("url", ""), a.get("date"),
                            a.get("body"), "er_lgu"))
        if page * 100 >= arts.get("totalResults", 0):
            break
        page += 1
        time.sleep(0.6)
    return out


# ── 2. NewsData.io ──────────────────────────────────────────────────────────
def _newsdata(key, pages=5) -> list[dict]:
    out, npage = [], None
    terms = "food OR hunger OR rice OR malnutrition OR ayuda OR farmer OR palay OR poverty"
    for prov in ["CALABARZON", "Batangas", "Cavite", "Laguna", "Rizal", "Quezon"]:
        npage = None
        for _ in range(pages):
            p = {"apikey": key, "q": f"{prov} ({terms})", "country": "ph", "language": "en"}
            if npage:
                p["page"] = npage
            try:
                r = requests.get("https://newsdata.io/api/1/news", params=p, timeout=30)
                d = r.json()
            except Exception:
                break
            for a in d.get("results", []) or []:
                out.append(_rec(a.get("title"), a.get("link", ""), a.get("pubDate"),
                                a.get("description") or a.get("content"), "newsdata"))
            npage = d.get("nextPage")
            time.sleep(0.5)
            if not npage:
                break
    return out


# ── 3. Google Custom Search ─────────────────────────────────────────────────
def _cse_valid(key, cx) -> bool:
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1",
                         params={"key": key, "cx": cx, "q": "test", "num": 1}, timeout=20)
        return r.status_code == 200
    except Exception:
        return False


def _google_cse(key, cx) -> list[dict]:
    out = []
    queries = [
        "CALABARZON food insecurity", "Batangas hunger food", "Cavite food shortage",
        "Laguna food crisis rice", "Quezon province hunger farmers", "Rizal food insecurity",
        "CALABARZON malnutrition", "CALABARZON ayuda bigas", "CALABARZON crop damage typhoon",
    ]
    for q in queries:
        for start in (1, 11):  # 2 pages x 10 = 20 per query (free tier: 100 q/day)
            try:
                r = requests.get("https://www.googleapis.com/customsearch/v1",
                                 params={"key": key, "cx": cx, "q": q, "num": 10,
                                         "start": start, "gl": "ph"}, timeout=30)
                d = r.json()
            except Exception:
                break
            items = d.get("items", []) or []
            for a in items:
                out.append(_rec(a.get("title"), a.get("link", ""), "",
                                a.get("snippet"), "google_cse"))
            if len(items) < 10:
                break
            time.sleep(0.4)
    return out


def collect() -> None:
    rows: list[dict] = []
    key = _key()
    gaz = pd.read_parquet(GAZ).drop_duplicates("lgu_psgc")[["province_name", "lgu_name"]]
    print(f"[1] Event Registry per-LGU (all {len(gaz)})...", flush=True)
    for i, t in enumerate(gaz.itertuples(index=False), 1):
        got = _er_lgu(key, t.lgu_name, t.province_name)
        rows.extend(got)
        if i % 20 == 0:
            print(f"    {i}/{len(gaz)} LGUs, {len(rows)} rows", flush=True)

    nd = _env("NEWSDATA_API_KEY")
    if nd:
        print("[2] NewsData.io...", flush=True)
        rows.extend(_newsdata(nd))
    # Google CSE key is invalid (HTTP 400) — skipped. Re-enable if a valid
    # GOOGLE_CSE_API_KEY is provided.
    ck, cx = _env("GOOGLE_CSE_API_KEY"), _env("GOOGLE_CSE_CX")
    if ck and cx and _cse_valid(ck, cx):
        print("[3] Google CSE...", flush=True)
        rows.extend(_google_cse(ck, cx))
    else:
        print("[3] Google CSE skipped (invalid/missing key)", flush=True)

    df = pd.DataFrame(rows)
    df = df[df["link"].str.len() > 0].drop_duplicates("article_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nsaved {len(df)} unique articles -> {OUT}", flush=True)
    print("by source:", df["fetcher_source"].value_counts().to_dict(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        # reuse the 12-core parallel merge from eventregistry_broad
        import scripts.eventregistry_broad as broad
        broad.OUT = OUT
        broad.merge()
    else:
        collect()


if __name__ == "__main__":
    main()
