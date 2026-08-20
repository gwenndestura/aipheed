"""
scripts/collect_precise.py
---------------------------
High-precision, comprehensive CALABARZON food-insecurity collection that avoids
the ambiguous-name noise that ruined the per-LGU keyword sweep. Two sources:

  1. Event Registry CONCEPT-based — articles ER semantically tagged as ABOUT a
     CALABARZON province (Wikipedia concept URIs), AND a food-insecurity term.
     Full pagination, real article bodies. Precise: no "Real"/"Laguna" (Spanish)
     foreign matches.
  2. Google News RSS — food-insecurity terms x all 142 LGUs, gentle rate with
     429 backoff (the throttle from earlier runs has since cooled).

Then --merge: gate BEFORE scoring (food-anchor + food-insecurity topic +
CALABARZON geo + negatives), NLI-score only the small survivor set (single-core,
no OOM), 3-way dedup (id + link + title), and append to corpus_geocoded.

Usage:
  venv\\Scripts\\python scripts\\collect_precise.py            # collect
  venv\\Scripts\\python scripts\\collect_precise.py --merge    # gate+score+append
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, quote

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eventregistry_cover_lgus import _key  # noqa: E402

OUT = Path("data/raw/precise_pool.parquet")
GEO = Path("data/processed/corpus_geocoded.parquet")

CONCEPTS = {
    "Batangas": "http://en.wikipedia.org/wiki/Batangas",
    "Cavite": "http://en.wikipedia.org/wiki/Cavite",
    "Laguna": "http://en.wikipedia.org/wiki/Laguna_(province)",
    "Quezon": "http://en.wikipedia.org/wiki/Quezon",
    "Rizal": "http://en.wikipedia.org/wiki/Rizal_(province)",
    "CALABARZON": "http://en.wikipedia.org/wiki/Calabarzon",
}
FOOD = ["food", "hunger", "rice", "malnutrition", "farmer", "fisherfolk",
        "poverty", "ayuda", "harvest", "palay", "gutom", "crop", "fish"]


def _domain(u):
    return urlparse(u or "").netloc.lower().lstrip("www.")


def _rec(title, url, date, body, src):
    return {"title": (title or "").strip(), "link": url,
            "article_id": hashlib.md5((url or "").encode()).hexdigest(),
            "published": (date or "")[:10], "summary": (body or "")[:2000],
            "source_domain": _domain(url), "fetcher_source": src}


def _er_concept(key, uri, pages=40):
    out, page = [], 1
    while page <= pages:
        q = {"$query": {"$and": [
            {"conceptUri": uri},
            {"$or": [{"keyword": k} for k in FOOD]},
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
                            a.get("body"), "er_concept"))
        if page * 100 >= arts.get("totalResults", 0):
            break
        page += 1
        time.sleep(0.6)
    return out


def _gnews(pages_lgus):
    from app.ml.corpus.gdelt_fetcher import CALABARZON_LGUS
    out, delay = [], 2.5
    lgus = [(l, p) for p, ls in CALABARZON_LGUS.items() for l in ls]
    for i, (lgu, prov) in enumerate(lgus, 1):
        q = f'"{lgu}" food OR hunger OR rice OR ayuda OR farmer OR palay'
        u = f"https://news.google.com/rss/search?q={quote(q)}&hl=en-PH&gl=PH&ceid=PH:en"
        for attempt in range(4):
            try:
                r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            except Exception:
                break
            if r.status_code == 200:
                for m in re.findall(r"<item>(.*?)</item>", r.text, re.S):
                    t = re.search(r"<title>(.*?)</title>", m, re.S)
                    de = re.search(r"<description>(.*?)</description>", m, re.S)
                    pu = re.search(r"<pubDate>(.*?)</pubDate>", m, re.S)
                    li = re.search(r"<link>(.*?)</link>", m, re.S)
                    title = html.unescape(re.sub("<[^>]+>", "", t.group(1))) if t else ""
                    desc = html.unescape(re.sub("<[^>]+>", "", de.group(1))) if de else ""
                    out.append(_rec(title, li.group(1) if li else "",
                                    pu.group(1)[:16] if pu else "", desc, "gnews_rss"))
                time.sleep(delay)
                break
            if r.status_code in (429, 503):
                time.sleep(delay * 4)
                delay = min(delay * 1.5, 20)
        if i % 20 == 0:
            print(f"    gnews {i}/{len(lgus)} LGUs, {len(out)} items", flush=True)
    return out


def collect():
    key = _key()
    rows = []
    print("[1] Event Registry concept-based (semantic 'about province')...", flush=True)
    for name, uri in CONCEPTS.items():
        got = _er_concept(key, uri)
        print(f"    {name}: {len(got)}", flush=True)
        rows.extend(got)
    print("[2] Google News RSS (food x 142 LGUs, gentle)...", flush=True)
    rows.extend(_gnews(True))
    df = pd.DataFrame(rows)
    df = df[df["link"].str.len() > 0].drop_duplicates("article_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nsaved {len(df)} unique articles -> {OUT}", flush=True)
    print("by source:", df["fetcher_source"].value_counts().to_dict(), flush=True)


def merge():
    from app.ml.corpus.location_geocoder import geocode_location_batch
    from build_final_dataset import _NEGATIVE, _matched_topics
    from precision_pass import FOOD_ANCHOR
    from app.ml.corpus.location_geocoder import _OTHER_LOC

    pool = pd.read_parquet(OUT)
    c = pd.read_parquet(GEO)

    def nt(s):
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
    hid, hln, htt = set(c["article_id"]), set(c["link"].dropna()), set(c["title"].fillna("").map(nt))
    pool["_nt"] = pool["title"].fillna("").map(nt)
    new = pool[~(pool["article_id"].isin(hid) | pool["link"].isin(hln) | pool["_nt"].isin(htt))]
    new = new.drop_duplicates("article_id").drop_duplicates("link").drop_duplicates("_nt").drop(columns=["_nt"])
    print(f"new (3-way deduped): {len(new)}")

    new = geocode_location_batch(new)
    new = new[new["province_name"].notna()].copy()
    txt = (new["title"].fillna("") + " " + new["summary"].fillna("")).astype(str)
    gate = (txt.map(lambda t: bool(FOOD_ANCHOR.search(t)))
            & txt.map(lambda t: len(_matched_topics(t)) > 0)
            & ~txt.map(lambda t: bool(_OTHER_LOC.search(t.lower())))
            & ~txt.map(lambda t: bool(_NEGATIVE.search(t))))
    new = new[gate].copy()
    print(f"after CALABARZON + food-insecurity gate: {len(new)}")

    # NLI-score the gated survivors on 8 cores (safe on 32 GB RAM; 12 OOMs).
    import os
    from concurrent.futures import ProcessPoolExecutor
    from scripts.eventregistry_broad import _init_worker, _score_chunk
    WORKERS = 8
    ck = Path("data/raw/checkpoints/xlmr_scores.parquet")
    done = {r["article_id"]: r for r in pd.read_parquet(ck).to_dict("records")} if ck.exists() else {}
    cached, todo = [], []
    for r in new.to_dict("records"):
        s = done.get(r["article_id"])
        if s is not None:
            r["is_relevant"] = bool(s["is_relevant"])
            cached.append(r)
        else:
            todo.append(r)
    print(f"scoring {len(todo)} survivors on {WORKERS} cores ({len(cached)} cached)", flush=True)
    scored = []
    if todo:
        threads = max(1, (os.cpu_count() or 8) // WORKERS)
        chunks = [todo[i:i + 25] for i in range(0, len(todo), 25)]
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker,
                                 initargs=(threads,)) as ex:
            for res in ex.map(_score_chunk, chunks):
                scored.extend(res)
    allrows = cached + scored
    kept = pd.DataFrame([r for r in allrows if r.get("is_relevant")])
    print(f"NLI-relevant (final new): {len(kept)}")
    if kept.empty:
        print("nothing to add"); return

    def q(p):
        try:
            dt = pd.Timestamp(p); return f"{dt.year}-Q{(dt.month-1)//3+1}"
        except Exception:
            return ""
    kept["quarter"] = kept["published"].map(q)
    cols = list(c.columns) + [x for x in ("province_name", "lgu_name", "lgu_psgc", "barangay_name",
              "barangay_psgc", "match_level", "geo_specificity", "geo_conflict") if x not in c.columns]
    comb = pd.concat([c, kept[[x for x in cols if x in kept.columns]]], ignore_index=True)
    comb = comb.drop_duplicates("article_id").drop_duplicates("link")
    comb.to_parquet(GEO, index=False)
    print(f"corpus_geocoded: {len(c)} -> {len(comb)} (+{len(comb)-len(c)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else collect()


if __name__ == "__main__":
    main()
