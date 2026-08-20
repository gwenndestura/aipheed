"""
scripts/collect_er_topic.py
----------------------------
Event Registry by food-insecurity TOPIC concept (semantic "this article is about
food security / hunger / malnutrition / rice / poverty / agriculture / fisheries"),
constrained to mention a CALABARZON province. A different axis from the place-
concept and keyword sweeps already done.

--merge: 3-way dedup -> geocode -> loosened validated gate (food-anchor ASF-aware
+ any food-insecurity topic + CALABARZON geo + negative/noise filters) -> append.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eventregistry_cover_lgus import _key  # noqa: E402

OUT = Path("data/raw/er_topic_pool.parquet")
GEO = Path("data/processed/corpus_geocoded.parquet")

TOPICS = {
    "Food_security": "http://en.wikipedia.org/wiki/Food_security",
    "Malnutrition": "http://en.wikipedia.org/wiki/Malnutrition",
    "Poverty": "http://en.wikipedia.org/wiki/Poverty",
    "Rice": "http://en.wikipedia.org/wiki/Rice",
    "Agriculture": "http://en.wikipedia.org/wiki/Agriculture",
    "Fishery": "http://en.wikipedia.org/wiki/Fishery",
    "Drought": "http://en.wikipedia.org/wiki/Drought",
    "Coconut": "http://en.wikipedia.org/wiki/Coconut",
}
PROVS = ["Batangas", "Cavite", "Laguna", "Rizal", "Quezon"]


def _rec(a):
    url = a.get("url", "")
    return {"title": (a.get("title") or "").strip(), "link": url,
            "article_id": hashlib.md5(url.encode()).hexdigest(),
            "published": (a.get("date") or "")[:10],
            "summary": (a.get("body") or "")[:2000],
            "source_domain": urlparse(url).netloc.lower().lstrip("www."),
            "fetcher_source": "er_topic"}


def _fetch(key, uri, pages=30):
    out, page = [], 1
    while page <= pages:
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
            d = r.json()
        except Exception:
            break
        arts = (d.get("articles", {}) or {}) if isinstance(d, dict) else {}
        res = arts.get("results", [])
        if not res:
            break
        out.extend(_rec(a) for a in res)
        if page * 100 >= arts.get("totalResults", 0):
            break
        page += 1
        time.sleep(0.6)
    return out


def collect():
    key = _key()
    rows = []
    for name, uri in TOPICS.items():
        got = _fetch(key, uri)
        print(f"  {name}: {len(got)}", flush=True)
        rows.extend(got)
    df = pd.DataFrame(rows)
    df = df[df["link"].str.len() > 0].drop_duplicates("article_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nsaved {len(df)} unique -> {OUT}", flush=True)


def merge():
    from app.ml.corpus.location_geocoder import geocode_location_batch, _OTHER_LOC
    from build_final_dataset import _NEGATIVE, _matched_topics
    from precision_pass import FOOD_ANCHOR
    from collect_gnews_round3 import _NOISE2
    from collect_commodity import _EXTRA_NOISE
    _ASF = re.compile(r"\b(asf|african swine fever|swine fever)\b", re.I)
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
    keep = ((txt.map(lambda t: bool(FOOD_ANCHOR.search(t)) or bool(_ASF.search(t)))
             & txt.map(lambda t: len(_matched_topics(t)) > 0))
            & ~txt.map(lambda t: bool(_OTHER_LOC.search(t.lower())))
            & ~txt.map(lambda t: bool(_NEGATIVE.search(t)))
            & ~txt.map(lambda t: bool(_NOISE2.search(t)))
            & ~txt.map(lambda t: bool(_EXTRA_NOISE.search(t))))
    new = new[keep].copy()
    print(f"gate survivors: {len(new)}")
    if new.empty:
        print("nothing to add"); return
    new["is_relevant"] = True

    def q(p):
        try:
            dt = pd.Timestamp(p); return f"{dt.year}-Q{(dt.month-1)//3+1}"
        except Exception:
            return ""
    new["quarter"] = new["published"].map(q)
    cols = list(c.columns) + [x for x in ("province_name", "lgu_name", "lgu_psgc", "barangay_name",
              "barangay_psgc", "match_level", "geo_specificity", "geo_conflict") if x not in c.columns]
    comb = pd.concat([c, new[[x for x in cols if x in new.columns]]], ignore_index=True)
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
