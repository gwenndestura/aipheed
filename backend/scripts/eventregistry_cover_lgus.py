"""
scripts/eventregistry_cover_lgus.py
------------------------------------
Targeted Event Registry (NewsAPI.ai) fetch to fill Region IV-A LGU coverage
gaps. GDELT's DOC API only indexes a small credible-domain set and throttles
hard; Event Registry indexes ~150k publishers including the local/regional
outlets that actually cover small municipalities — so it is the better source
for the 98 CALABARZON LGUs that have no food-insecurity-relevant article yet.

Token-economical: ONE request per LGU (both languages, first page of 100),
querying the LGU name + province + the bilingual food-insecurity lexicon.
Real article bodies are returned (good for the geocoder + NLI scorer).

Output: data/raw/targeted_lgu_eventregistry.parquet  (corpus record format)

Usage:
  venv\\Scripts\\python scripts\\eventregistry_cover_lgus.py
  venv\\Scripts\\python scripts\\eventregistry_cover_lgus.py --report
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

OUT = Path("data/raw/targeted_lgu_eventregistry.parquet")
TARGETS = Path("data/processed/_uncovered_lgus.parquet")
API = "https://eventregistry.org/api/v1/article/getArticles"

# Event Registry free tier caps a query at 15 keywords. With the LGU name and
# province taking 2 slots, the food lexicon must be <= 13 terms.
FOOD = ["food", "rice", "hunger", "farmer", "poverty", "fisherfolk",
        "malnutrition", "ayuda", "palay", "gutom", "bigas", "crop", "DSWD"]

# Municipalities whose plain name equals a province — use the fuller phrasing.
_COLLIDING = {"Rizal": "municipality of Rizal", "Quezon": "municipality of Quezon"}


def _key() -> str:
    for line in open(Path(__file__).resolve().parents[1] / ".env", encoding="utf-8"):
        if line.startswith("EVENTREGISTRY_API_KEY"):
            return line.split("=", 1)[1].strip()
    raise SystemExit("EVENTREGISTRY_API_KEY not in .env")


def _domain(u: str) -> str:
    return urlparse(u or "").netloc.lower().lstrip("www.")


def _fetch_lgu(key: str, lgu: str, prov: str) -> list[dict]:
    """One request: LGU (or colliding-name phrase) AND province AND any food term.

    Event Registry counts every WORD of a multi-word keyword toward its 15-keyword
    cap, so a 2-3 word LGU name ("San Jose", "General Emilio Aguinaldo") would
    blow the budget. Size the food OR-group to whatever slots remain."""
    place = _COLLIDING.get(lgu, lgu)
    used = len(place.split()) + len(prov.split())
    food = FOOD[:max(3, 15 - used)]          # keep >=3 food terms; trim to fit
    q = {"$query": {"$and": [
        {"keyword": place, "keywordLoc": "body,title"},
        {"keyword": prov, "keywordLoc": "body,title"},
        {"$or": [{"keyword": k} for k in food]},
        {"dateStart": "2020-01-01", "dateEnd": "2026-08-19"},
    ]}}
    try:
        r = requests.post(API, data={
            "query": json.dumps(q), "resultType": "articles",
            "articlesCount": "100", "articlesPage": "1",
            "articlesSortBy": "date", "apiKey": key,
        }, timeout=90)
        d = r.json()
    except Exception as exc:
        print(f"      error {lgu}: {exc}", flush=True)
        return []
    if isinstance(d, dict) and d.get("error"):
        print(f"      API error {lgu}: {d['error']}", flush=True)
        return []
    results = (d.get("articles", {}) or {}).get("results", []) if isinstance(d, dict) else []
    out = []
    for a in results:
        url = a.get("url", "")
        out.append({
            "title": (a.get("title") or "").strip(),
            "link": url,
            "article_id": hashlib.md5(url.encode()).hexdigest(),
            "published": (a.get("date") or "")[:10],
            "summary": (a.get("body") or "")[:500],
            "source_domain": _domain(url),
            "fetcher_source": "targeted_lgu_er",
            "target_lgu": lgu,
        })
    return out


def collect() -> None:
    key = _key()
    targets = pd.read_parquet(TARGETS)
    seen: set[str] = set()
    rows: list[dict] = []
    done_lgus: set[str] = set()
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        rows = prev.to_dict("records")
        seen = set(prev["article_id"].dropna())
        done_lgus = set(prev.get("target_lgu", pd.Series(dtype=str)).dropna())
        print(f"resuming — {len(rows)} articles, {len(done_lgus)} LGUs done", flush=True)

    print(f"targeting {len(targets)} uncovered LGUs via Event Registry", flush=True)
    for i, t in enumerate(targets.itertuples(index=False), 1):
        lgu, prov = t.lgu_name, t.province_name
        if lgu in done_lgus:
            continue
        got = _fetch_lgu(key, lgu, prov)
        new = 0
        for rec in got:
            if rec["article_id"] not in seen:
                seen.add(rec["article_id"])
                rows.append(rec)
                new += 1
        # Mark the LGU queried even at 0 hits so resume doesn't repeat it.
        if new == 0:
            rows.append({"article_id": None, "target_lgu": lgu, "title": None,
                         "link": None, "published": None, "summary": None,
                         "source_domain": None, "fetcher_source": "targeted_lgu_er"})
        print(f"  [{i:3d}/{len(targets)}] {prov}/{lgu:22s} +{new}  (total {sum(1 for r in rows if r['article_id'])})", flush=True)
        time.sleep(1.0)
        if i % 10 == 0:
            pd.DataFrame(rows).to_parquet(OUT, index=False)

    pd.DataFrame(rows).to_parquet(OUT, index=False)
    real = sum(1 for r in rows if r["article_id"])
    print(f"\nsaved {real} articles across {len(targets)} LGUs -> {OUT}", flush=True)


def report() -> None:
    from app.ml.corpus.location_geocoder import geocode_location_batch
    from app.ml.nlp.classifier import load_classifier, score_article

    pool = pd.read_parquet(OUT)
    pool = pool[pool["article_id"].notna()].copy()
    print(f"targeted ER pool: {len(pool)} articles across "
          f"{pool['target_lgu'].nunique()} LGUs queried")

    ck = Path("data/raw/checkpoints/xlmr_scores.parquet")
    done = {r["article_id"]: r for r in pd.read_parquet(ck).to_dict("records")} if ck.exists() else {}
    clf = load_classifier()
    recs = []
    for r in pool.to_dict("records"):
        aid = r["article_id"]
        if aid in done:
            rel = bool(done[aid]["is_relevant"])
        else:
            s = score_article(clf, str(r.get("title") or ""), str(r.get("summary") or ""))
            rel = bool(s["is_relevant"])
        recs.append({**r, "is_relevant": rel})
    df = pd.DataFrame(recs)
    rel = df[df["is_relevant"]]
    tagged = geocode_location_batch(rel)
    hit = tagged[tagged["lgu_name"].notna()]
    print(f"relevant: {len(rel)} | LGU-tagged: {len(hit)} | "
          f"NEW distinct LGUs covered: {hit['lgu_psgc'].nunique()}")
    for r in hit[["province_name", "lgu_name"]].drop_duplicates().sort_values(
            ["province_name", "lgu_name"]).itertuples(index=False):
        print(f"  + {r.province_name}/{r.lgu_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report() if args.report else collect()


if __name__ == "__main__":
    main()
