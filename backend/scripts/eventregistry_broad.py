"""
scripts/eventregistry_broad.py
-------------------------------
Deep Event Registry sweep of ALL of CALABARZON (per province + food-insecurity
lexicon, both languages, full pagination, real article bodies). Wider than the
per-uncovered-LGU sweep — pulls the ~1,100 food-related articles ER indexes for
Region IV-A so genuinely-new stories are captured for the covered LGUs too.

Then scores (XLM-R NLI), geocodes (142-LGU gazetteer), keeps the relevant
CALABARZON ones, and APPENDS the new rows to corpus_geocoded.parquet so
`build_final_dataset.py` folds them in on its next run.

Usage:
  venv\\Scripts\\python scripts\\eventregistry_broad.py            # collect
  venv\\Scripts\\python scripts\\eventregistry_broad.py --merge    # score+geocode+append
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
from scripts.eventregistry_cover_lgus import _key, FOOD  # noqa: E402

OUT = Path("data/raw/eventregistry_broad.parquet")
GEO = Path("data/processed/corpus_geocoded.parquet")
API = "https://eventregistry.org/api/v1/article/getArticles"
PROVINCES = ["Batangas", "Cavite", "Laguna", "Rizal", "Quezon"]
LANGS = ["eng", "tgl"]
PAGE = 100
MAX_PAGES = 40


def _domain(u: str) -> str:
    return urlparse(u or "").netloc.lower().lstrip("www.")


def _fetch(key: str, prov: str, lang: str) -> list[dict]:
    out, page = [], 1
    while page <= MAX_PAGES:
        q = {"$query": {"$and": [
            {"keyword": prov, "keywordLoc": "body,title"},
            {"$or": [{"keyword": k} for k in FOOD[:13]]},
            {"dateStart": "2020-01-01", "dateEnd": "2026-08-20", "lang": lang},
        ]}}
        try:
            r = requests.post(API, data={
                "query": json.dumps(q), "resultType": "articles",
                "articlesCount": str(PAGE), "articlesPage": str(page),
                "articlesSortBy": "date", "apiKey": key,
            }, timeout=90)
            d = r.json()
        except Exception as exc:
            print(f"  {prov}/{lang} p{page} error: {exc}", flush=True)
            break
        arts = (d.get("articles", {}) or {}) if isinstance(d, dict) else {}
        results = arts.get("results", [])
        if not results:
            if isinstance(d, dict) and d.get("error"):
                print(f"  {prov}/{lang}: {d['error'][:60]}", flush=True)
            break
        for a in results:
            url = a.get("url", "")
            out.append({
                "title": (a.get("title") or "").strip(),
                "link": url,
                "article_id": hashlib.md5(url.encode()).hexdigest(),
                "published": (a.get("date") or "")[:10],
                # keep a longer body than the 500-char default so LGU/barangay
                # mentions deeper in the article are geocodable.
                "summary": (a.get("body") or "")[:2000],
                "source_domain": _domain(url),
                "fetcher_source": "eventregistry_broad",
            })
        total = arts.get("totalResults", 0)
        if page * PAGE >= total:
            break
        page += 1
        time.sleep(1.0)
    return out


def collect() -> None:
    key = _key()
    rows: list[dict] = []
    for prov in PROVINCES:
        for lang in LANGS:
            got = _fetch(key, prov, lang)
            print(f"{prov}/{lang}: {len(got)}", flush=True)
            rows.extend(got)
    df = pd.DataFrame(rows).drop_duplicates("article_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nsaved {len(df)} unique articles -> {OUT}", flush=True)


# ── Parallel scoring (12 worker processes; each loads XLM-R once) ────────────
WORKERS = 12
_CLF = None


def _init_worker(threads: int) -> None:
    import torch
    torch.set_num_threads(max(1, threads))
    global _CLF
    from app.ml.nlp.classifier import load_classifier
    _CLF = load_classifier()


def _score_chunk(records: list[dict]) -> list[dict]:
    from app.ml.nlp.classifier import score_article
    out = []
    for r in records:
        s = score_article(_CLF, str(r.get("title") or ""), str(r.get("summary") or ""))
        r = dict(r)
        r.update({"food_insecurity_score": s["food_insecurity_score"],
                  "is_relevant": bool(s["is_relevant"]),
                  "top_hypothesis": s["top_hypothesis"],
                  "top_topic_name": s["top_topic_name"]})
        out.append(r)
    return out


def merge() -> None:
    """Score (12-core parallel) + geocode the broad pool; append relevant
    CALABARZON rows not already present to corpus_geocoded.parquet."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    from app.ml.corpus.location_geocoder import geocode_location_batch

    pool = pd.read_parquet(OUT)
    existing = pd.read_parquet(GEO)
    pool = pool[~pool["article_id"].isin(set(existing["article_id"]))]
    pool = pool[~pool["link"].isin(set(existing["link"].dropna()))].copy()
    print(f"broad pool: {len(pool)} not already in corpus")

    ck = Path("data/raw/checkpoints/xlmr_scores.parquet")
    done = {r["article_id"]: r for r in pd.read_parquet(ck).to_dict("records")} if ck.exists() else {}

    cached, todo = [], []
    for r in pool.to_dict("records"):
        s = done.get(r["article_id"])
        if s is not None:
            r.update({"food_insecurity_score": s.get("food_insecurity_score"),
                      "is_relevant": bool(s["is_relevant"]),
                      "top_hypothesis": s.get("top_hypothesis"),
                      "top_topic_name": s.get("top_topic_name")})
            cached.append(r)
        else:
            todo.append(r)
    print(f"  cached: {len(cached)} | to score: {len(todo)} on {WORKERS} cores", flush=True)

    scored_new: list[dict] = []
    if todo:
        threads = max(1, (os.cpu_count() or 12) // WORKERS)
        chunks = [todo[i:i + 25] for i in range(0, len(todo), 25)]
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker,
                                 initargs=(threads,)) as ex:
            for n, res in enumerate(ex.map(_score_chunk, chunks), 1):
                scored_new.extend(res)
                if n % 4 == 0:
                    print(f"  scored {len(scored_new)}/{len(todo)}", flush=True)

    scored = pd.DataFrame(cached + scored_new)
    rel = scored[scored["is_relevant"]].copy()
    print(f"relevant: {len(rel)}")

    rel = geocode_location_batch(rel)
    rel = rel[rel["province_name"].notna()].copy()
    print(f"relevant + CALABARZON-geocoded: {len(rel)}")

    def _q(p):
        try:
            dt = pd.Timestamp(p); return f"{dt.year}-Q{(dt.month-1)//3+1}"
        except Exception:
            return ""
    rel["quarter"] = rel["published"].map(_q)
    combined = pd.concat([existing, rel[[c for c in rel.columns if c in existing.columns or c in
                                         ("province_name","lgu_name","lgu_psgc","barangay_name",
                                          "barangay_psgc","match_level","geo_specificity","geo_conflict")]]],
                         ignore_index=True)
    combined = combined.drop_duplicates("article_id").drop_duplicates("link")
    combined.to_parquet(GEO, index=False)
    print(f"corpus_geocoded: {len(existing)} -> {len(combined)} (+{len(combined)-len(existing)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    merge() if args.merge else collect()


if __name__ == "__main__":
    main()
