"""
scripts/collect_gnews_tagalog.py
---------------------------------
Untapped Google News RSS depth: Tagalog food terms, disaster+food combos, and
food-program queries across all 142 CALABARZON LGUs (the earlier sweep used one
English query per LGU only). Gentle rate with 429 backoff.

--merge: 3-way dedup (id+link+title) -> geocode -> keep rows that pass the
validated STRICT gate (food-anchor + a CORE food-insecurity topic + CALABARZON
geo + negative/noise filters). The NLI step is intentionally NOT used as a hard
gate here: manual review proved it has a high false-negative rate on short news
leads, so the lexical food-insecurity gate is the relevance signal. Appends to
corpus_geocoded with fetcher_source=gnews_tagalog.
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

OUT = Path("data/raw/gnews_tagalog_pool.parquet")
GEO = Path("data/processed/corpus_geocoded.parquet")

# Three query variants per LGU (the {lgu} placeholder is phrase-quoted).
QUERY_VARIANTS = [
    'gutom OR bigas OR ayuda OR presyo OR magsasaka OR mangingisda OR ani OR pagkain OR palay',
    '(bagyo OR baha OR tagtuyot OR bulkan OR "fish kill") (palay OR ani OR isda OR pananim OR pagkain OR magsasaka)',
    'Kadiwa OR "food pack" OR feeding OR malnutrition OR "rice subsidy" OR "P20 rice" OR "food security"',
]


def _rec(title, url, date, body):
    return {"title": (title or "").strip(), "link": url,
            "article_id": hashlib.md5((url or "").encode()).hexdigest(),
            "published": (date or "")[:16], "summary": (body or "")[:1500],
            "source_domain": (url.split("/")[2].lstrip("www.") if "//" in (url or "") else ""),
            "fetcher_source": "gnews_tagalog"}


def collect():
    from app.ml.corpus.gdelt_fetcher import CALABARZON_LGUS
    lgus = [(l, p) for p, ls in CALABARZON_LGUS.items() for l in ls]
    rows, delay = [], 2.5
    for i, (lgu, prov) in enumerate(lgus, 1):
        for variant in QUERY_VARIANTS:
            q = f'"{lgu}" {variant}'
            u = f"https://news.google.com/rss/search?q={quote(q)}&hl=en-PH&gl=PH&ceid=PH:en"
            for _ in range(4):
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
                        rows.append(_rec(title, li.group(1) if li else "",
                                         pu.group(1) if pu else "", desc))
                    time.sleep(delay)
                    break
                if r.status_code in (429, 503):
                    time.sleep(delay * 4); delay = min(delay * 1.5, 20)
        if i % 20 == 0:
            print(f"  {i}/{len(lgus)} LGUs, {len(rows)} items", flush=True)
    df = pd.DataFrame(rows)
    df = df[df["link"].str.len() > 0].drop_duplicates("article_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nsaved {len(df)} unique -> {OUT}", flush=True)


# CORE food-insecurity topics — a real insecurity signal, not just "a farm".
_CORE = {"hunger_food_deprivation", "malnutrition_undernutrition", "food_prices_affordability",
         "food_assistance_programs", "crop_losses_disaster", "poverty_food_access",
         "food_security_general", "rice_staple_supply", "fisheries_livestock"}
# Noise the earlier read exposed (foreign place collisions, tourism, culture, sports).
_NOISE2 = re.compile(
    r"\b(tinubu|naira|nigeria|vanguard news|guardian niger|nirsal|california|calif\.|merced|"
    r"fresno|cbs news|abc30|farm progress|farm[- ]to[- ]table|agri[- ]?tourism|tourist spot|"
    r"farm resort|farm stay|resto farm|\bmpbl\b|basketball|guinness|restaurant|vegan|samgyup|"
    r"bulalo|underrated|eco-tourism|flower farm|hunger[- ]strike)\b", re.I)


def merge():
    from app.ml.corpus.location_geocoder import geocode_location_batch, _OTHER_LOC
    from build_final_dataset import _NEGATIVE, TOPIC_PATTERNS
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

    def core(t):
        return any(n in _CORE and p.search(t) for n, p in TOPIC_PATTERNS.items())
    keep = (txt.map(lambda t: bool(FOOD_ANCHOR.search(t)))
            & txt.map(core)
            & ~txt.map(lambda t: bool(_OTHER_LOC.search(t.lower())))
            & ~txt.map(lambda t: bool(_NEGATIVE.search(t)))
            & ~txt.map(lambda t: bool(_NOISE2.search(t))))
    new = new[keep].copy()
    print(f"strict food-insecurity gate survivors (trust-gate, NLI bypassed): {len(new)}")
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
