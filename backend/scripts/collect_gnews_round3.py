"""
scripts/collect_gnews_round3.py
--------------------------------
Third Google News RSS pass on untapped angles (rounds 1-2 covered LGU x general
food + Tagalog/disaster):

  1. Per-LGU DETERMINANT query — food-insecurity determinant terms not yet used:
     malnutrition/stunting, fish kill/ASF/red tide, El Nino/drought, NFA/4Ps/
     feeding/food-pack, onion/sugar/coconut prices.
  2. DOMAIN-TARGETED site: queries — mine each credible Philippine outlet directly
     for CALABARZON food coverage the keyword queries missed.

--merge: 3-way dedup -> geocode -> strict food-insecurity gate (food-anchor +
CORE topic + CALABARZON geo + negative/noise) -> append. NLI bypassed as a hard
gate (proven high false-negative rate); lexical gate is the relevance signal.
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

OUT = Path("data/raw/gnews_round3_pool.parquet")
GEO = Path("data/processed/corpus_geocoded.parquet")

DETERMINANTS = ('malnutrition OR stunting OR nutrisyon OR "fish kill" OR ASF OR "red tide" OR '
                '"El Nino" OR drought OR tagtuyot OR NFA OR 4Ps OR feeding OR "food pack" OR '
                'Kadiwa OR onion OR sibuyas OR "sugar price" OR copra OR "coconut price"')
OUTLETS = ["pna.gov.ph", "pia.gov.ph", "rappler.com", "inquirer.net", "mb.com.ph",
           "gmanetwork.com", "philstar.com", "manilatimes.net", "abs-cbn.com",
           "bworldonline.com", "sunstar.com.ph", "businessmirror.com.ph"]
PROVINCES = ["Batangas", "Cavite", "Laguna", "Rizal", "Quezon"]


def _rec(title, url, date, body):
    return {"title": (title or "").strip(), "link": url,
            "article_id": hashlib.md5((url or "").encode()).hexdigest(),
            "published": (date or "")[:16], "summary": (body or "")[:1500],
            "source_domain": (url.split("/")[2].lstrip("www.") if "//" in (url or "") else ""),
            "fetcher_source": "gnews_round3"}


def _fetch(q, rows, delay=2.5):
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
    print(f"[1] per-LGU determinant queries ({len(lgus)})...", flush=True)
    for i, lgu in enumerate(lgus, 1):
        delay = _fetch(f'"{lgu}" {DETERMINANTS}', rows, delay)
        if i % 20 == 0:
            print(f"    {i}/{len(lgus)}, {len(rows)} items", flush=True)
    print("[2] domain-targeted site: queries...", flush=True)
    for outlet in OUTLETS:
        for prov in PROVINCES:
            delay = _fetch(f'site:{outlet} {prov} (rice OR hunger OR malnutrition OR '
                           f'ayuda OR farmer OR fisherfolk OR Kadiwa OR "food security" OR '
                           f'"fish kill" OR ASF OR drought)', rows, delay)
        print(f"    {outlet}: total {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df = df[df["link"].str.len() > 0].drop_duplicates("article_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nsaved {len(df)} unique -> {OUT}", flush=True)


_CORE = {"hunger_food_deprivation", "malnutrition_undernutrition", "food_prices_affordability",
         "food_assistance_programs", "crop_losses_disaster", "poverty_food_access",
         "food_security_general", "rice_staple_supply", "fisheries_livestock"}
_NOISE2 = re.compile(
    r"\b(tinubu|naira|nigeria|vanguard news|guardian niger|nirsal|danfulani|california|calif\.|"
    r"merced|fresno|cbs news|abc30|farm progress|yishun|singapore|farm[- ]to[- ]table|"
    r"agri[- ]?tourism|tourist spot|farm resort|farm stay|resto farm|\bmpbl\b|basketball|"
    r"guinness|restaurant|vegan|samgyup|bulalo|underrated|eco-tourism|flower farm|"
    r"hunger[- ]strike|guide service|fishing report|coconut worms|stray cat|feeding.*monkey)\b", re.I)


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
    keep = (txt.map(lambda t: bool(FOOD_ANCHOR.search(t))) & txt.map(core)
            & ~txt.map(lambda t: bool(_OTHER_LOC.search(t.lower())))
            & ~txt.map(lambda t: bool(_NEGATIVE.search(t)))
            & ~txt.map(lambda t: bool(_NOISE2.search(t))))
    new = new[keep].copy()
    print(f"strict food-insecurity gate survivors: {len(new)}")
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
