"""
scripts/collect_commodity.py
-----------------------------
Commodity-specific Google News RSS pass — the last untapped keyword axis. Rounds
1-3 covered general food / Tagalog-disaster / determinant terms; this targets
STAPLE COMMODITIES x production/price/loss context, per LGU:

  rice/palay, corn/mais, coconut/copra, coffee, pork/hog/baboy, vegetables/gulay,
  bangus/tilapia/isda, sugar, onion  x  (production/harvest/price/shortage/damage/
  farmer/fisherfolk)

--merge uses the LOOSENED, validated gate (learned from manual review):
  food-anchor (now ASF-aware) + ANY food-insecurity topic (incl agricultural
  production, which is thesis-scope "agri production affecting food availability")
  + CALABARZON geo + the negative + noise filters. No manual pass needed to catch
  the agri-production/ASF blocks this time.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("data/raw/gnews_commodity_pool.parquet")
GEO = Path("data/processed/corpus_geocoded.parquet")

COMMODITY = ('rice OR palay OR bigas OR corn OR mais OR coconut OR copra OR coffee OR '
             'pork OR hog OR baboy OR "African Swine Fever" OR vegetable OR gulay OR '
             'bangus OR tilapia OR isda OR "fish kill" OR sugar OR onion OR sibuyas')
CONTEXT = ('production OR harvest OR ani OR presyo OR price OR shortage OR kulang OR '
           'damage OR pinsala OR farmer OR magsasaka OR mangingisda OR crisis OR losses')


def _rec(title, url, date, body):
    return {"title": (title or "").strip(), "link": url,
            "article_id": hashlib.md5((url or "").encode()).hexdigest(),
            "published": (date or "")[:16], "summary": (body or "")[:1500],
            "source_domain": (url.split("/")[2].lstrip("www.") if "//" in (url or "") else ""),
            "fetcher_source": "gnews_commodity"}


def _fetch(q, rows, delay):
    u = f"https://news.google.com/rss/search?q={quote(q)}&hl=en-PH&gl=PH&ceid=PH:en"
    for _ in range(4):
        try:
            r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        except Exception:
            return delay
        if r.status_code == 200:
            for m in re.findall(r"<item>(.*?)</item>", r.text, re.S):
                t = re.search(r"<title>(.*?)</title>", m, re.S)
                de = re.search(r"<description>(.*?)</description>", m, re.S)
                pu = re.search(r"<pubDate>(.*?)</pubDate>", m, re.S)
                li = re.search(r"<link>(.*?)</link>", m, re.S)
                title = html.unescape(re.sub("<[^>]+>", "", t.group(1))) if t else ""
                desc = html.unescape(re.sub("<[^>]+>", "", de.group(1))) if de else ""
                rows.append(_rec(title, li.group(1) if li else "",
                                 pu.group(1) if pu else "", desc))
            time.sleep(delay)
            return delay
        if r.status_code in (429, 503):
            time.sleep(delay * 4); delay = min(delay * 1.5, 20)
    return delay


def collect():
    from app.ml.corpus.gdelt_fetcher import CALABARZON_LGUS
    lgus = [l for ls in CALABARZON_LGUS.values() for l in ls]
    rows, delay = [], 2.5
    print(f"per-LGU commodity queries ({len(lgus)})...", flush=True)
    for i, lgu in enumerate(lgus, 1):
        delay = _fetch(f'"{lgu}" ({COMMODITY}) ({CONTEXT})', rows, delay)
        if i % 20 == 0:
            print(f"    {i}/{len(lgus)}, {len(rows)} items", flush=True)
    df = pd.DataFrame(rows)
    df = df[df["link"].str.len() > 0].drop_duplicates("article_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nsaved {len(df)} unique -> {OUT}", flush=True)


# Reuse round-3's extended noise filter.
from collect_gnews_round3 import _NOISE2  # noqa: E402
_ASF = re.compile(r"\b(asf|african swine fever|swine fever)\b", re.I)
_EXTRA_NOISE = re.compile(
    r"\b(restaurant|food trip|foodie|foodpanda|must-try|dishes|cuisine|hidden gem|food fest|"
    r"museum|heritage|resort|tourism|glamorize|fiesta|festival|pahiyas|pahimis|niyogyugan|"
    r"stray cats?|serial cat|\bzoo\b|monkey|golf|wind farm|solar farm|solar power|hydroponics|"
    r"pag-ibig|housing unit|water supply|water interrupt|water district|holy week|"
    r"pertussis|measles|shabu|firearm|shot dead|farm raid|lego|gateway terminal)\b", re.I)


def merge():
    from app.ml.corpus.location_geocoder import geocode_location_batch, _OTHER_LOC
    from build_final_dataset import _NEGATIVE, _matched_topics
    from precision_pass import FOOD_ANCHOR
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
    print(f"gate survivors (food-anchor+ASF + any topic + geo + noise filters): {len(new)}")
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
