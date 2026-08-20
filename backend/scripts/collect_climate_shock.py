"""
scripts/collect_climate_shock.py
--------------------------------
Corpus-wide CLIMATE-SHOCK sweep. A typhoon / flood / landslide / drought that
strikes CALABARZON is a food-security *determinant* (the FAO stability +
availability pillars) even when the individual article does not spell out the
word "food": the shock disrupts harvests, markets, and household access. Earlier
collection rounds all required a food-anchor term, so every round silently
dropped these. This sweep applies the disaster=determinant rule *consistently*
across the entire collected universe (every text-bearing raw pool), then adds the
survivors as a distinct `is_climate_shock` determinant class (hypothesis T7 =
"Food stability / disaster-displacement").

Two guards keep it honest — the same two the food-anchored rounds enforced:
  1. CALABARZON must actually be HIT, not merely listed. Metro-Manila / Quezon
     City / northern-Luzon / Benguet / foreign subjects (frequent mis-geocodes:
     "Quezon City" -> Quezon province, "Los Banos" -> California/Argentina) and
     nationwide death-toll / storm-tracking wires are dropped. A real CALABARZON
     LGU must be named, or a CALABARZON province must sit next to a local-impact
     verb (flooded / evacuated / classes suspended / state of calamity).
  2. Syndication dedup — one event, not the 10-20 outlets that reprint the wire.

--merge appends the guarded, deduped survivors to corpus_geocoded.parquet with
is_climate_shock=True; build_final_dataset then exempts exactly those rows from
the food-anchor gate (disaster is the determinant) while still enforcing geo +
other-region + noise guards.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GEO = Path("data/processed/corpus_geocoded.parquet")
FINAL = Path("data/processed/calabarzon_food_insecurity_dataset.parquet")
OUT = Path("data/raw/_climate_shock_candidates.parquet")

# URL-only GDELT dumps carry no title/body -> nothing to classify.
SKIP = {"gdelt_bq_gov_raw.parquet", "gdelt_bq_national_raw.parquet",
        "gdelt_bq_urls_raw.parquet"}
COLS = ["title", "link", "article_id", "published", "summary", "source_domain"]

DISASTER = re.compile(
    r"\b(typhoon|bagyo|habagat|monsoon|tropical (depression|storm|cyclone)|\bflood|"
    r"baha|landslide|pagguho|storm surge|drought|tagtuyot|el ni[nñ]o|la ni[nñ]a|"
    r"calamity|kalamidad|inclement weather|evacuat|state of calamity|super typhoon|"
    r"cyclone|deluge|inundat|submerged)\b", re.I)
CALZN = re.compile(r"\b(cavite|laguna|batangas|rizal|quezon|calabarzon|region iv-?a)\b", re.I)

# Metro-Manila cities, other-region subjects, and foreign tokens that signal a
# mis-geocode or a non-CALABARZON subject.
BADGEO = re.compile(
    r"\b(quezon city|metro manila|\bNCR\b|kalakhang maynila|para[nñ]aque|taguig|"
    r"marikina|pasig|makati|caloocan|malabon|navotas|valenzuela|muntinlupa|"
    r"las pi[nñ]as|pateros|mandaluyong|pasay|manila city|city of manila|"
    r"northern (philippines|luzon)|benguet|baguio|la trinidad|ilocos|isabela|"
    r"cagayan|mindanao|visayas|pangasinan|zambales|bataan|abra|kalinga|nueva ecija|"
    r"nueva vizcaya|tarlac|bulacan|pampanga|aurora|la union|ifugao|apayao|"
    r"nevada|las le[nñ]as|california|mexic)\b", re.I)
# Nationwide toll / tracking wires (by headline) — not a CALABARZON-local event.
NATIONAL = re.compile(
    r"\b(death toll|reported dead|killed|nationwide|across the country|"
    r"six regions|10 regions|\d+ regions|whip|batter (luzon|the philippines)|"
    r"drench (luzon|the philippines)|affect(ed|s)? \d[\d.,]* ?(million|m\b))\b", re.I)
FOREIGN = re.compile(r"[àâçéèêëîïôûü]|\b(tras la|precios|actividades|nevada|chuva|warga|selatan)\b", re.I)

PROV = r"(Cavite|Laguna|Batangas|Rizal|Quezon)"
IMPACT = (r"(flood|baha|submerged|inundat|suspend|walang pasok|classes|evacuat|"
          r"state of calamity|declared? .{0,20}calamity|displaced|damage|pinsala|"
          r"stranded|rescued?|swept away|landslide)")
LOCAL = re.compile(IMPACT + r".{0,60}?" + PROV + "|" + PROV + r".{0,60}?" + IMPACT, re.I)


def _norm_pool(f: str) -> pd.DataFrame | None:
    df = pd.read_parquet(f)
    if not {"title", "summary"}.issubset(df.columns):
        return None
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    df = df[COLS].copy()
    blob = (df["title"].fillna("") + " " + df["summary"].fillna("")).astype(str)
    df = df[blob.map(lambda t: bool(DISASTER.search(t)) and bool(CALZN.search(t)))]
    return df


def collect() -> None:
    parts = []
    for f in sorted(glob.glob("data/raw/*.parquet")):
        if os.path.basename(f) in SKIP:
            continue
        try:
            d = _norm_pool(f)
        except Exception as e:
            print(f"  ! skip {os.path.basename(f)}: {str(e)[:50]}")
            continue
        if d is not None and len(d):
            parts.append(d)
            print(f"  {os.path.basename(f):44s} disaster+CALABARZON: {len(d)}", flush=True)
    pool = pd.concat(parts, ignore_index=True)
    pool = pool[pool["link"].fillna("").str.len() > 0]

    def nt(s):
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
    pool["_nt"] = pool["title"].fillna("").map(nt)
    pool = (pool.drop_duplicates("article_id").drop_duplicates("link")
            .drop_duplicates("_nt"))
    print(f"\nprefiltered unique disaster+CALABARZON candidates: {len(pool)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pool.drop(columns=["_nt"]).to_parquet(OUT, index=False)
    print(f"saved -> {OUT}")


def _guard(new: pd.DataFrame) -> pd.DataFrame:
    txt = (new["title"].fillna("") + " " + new["summary"].fillna("")).astype(str)
    ttl = new["title"].fillna("").astype(str)
    lgu = new["lgu_name"]
    has_lgu = lgu.notna() & (lgu.astype(str).str.len() > 0)
    calzn_subject = has_lgu | txt.map(lambda t: bool(LOCAL.search(t)))
    keep = (calzn_subject
            & ~txt.map(lambda t: bool(BADGEO.search(t)))
            & ~ttl.map(lambda t: bool(NATIONAL.search(t)))
            & ~txt.map(lambda t: bool(FOREIGN.search(t))))
    return new[keep].copy()


def merge() -> None:
    from app.ml.corpus.location_geocoder import geocode_location_batch
    pool = pd.read_parquet(OUT)
    c = pd.read_parquet(GEO)

    def nt(s):
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
    hid, hln, htt = set(c["article_id"]), set(c["link"].dropna()), set(c["title"].fillna("").map(nt))
    if FINAL.exists():
        fin = pd.read_parquet(FINAL)
        hln |= set(fin["url"].dropna())
        htt |= set(fin["title"].fillna("").map(nt))
    pool["_nt"] = pool["title"].fillna("").map(nt)
    new = pool[~(pool["article_id"].isin(hid) | pool["link"].isin(hln) | pool["_nt"].isin(htt))]
    new = new.drop_duplicates("article_id").drop_duplicates("link").drop_duplicates("_nt").drop(columns=["_nt"])
    print(f"new (not already in corpus/dataset): {len(new)}")

    new = geocode_location_batch(new)
    new = new[new["province_name"].notna()].copy()
    print(f"geocoded to a CALABARZON province: {len(new)}")

    new = _guard(new)
    # syndication dedup: one event, keep the longest-body copy
    new["_nt"] = new["title"].fillna("").map(nt)
    new = new.sort_values("summary", key=lambda s: s.str.len(), ascending=False).drop_duplicates("_nt")
    new = new.drop(columns=["_nt"])
    print(f"after geo-subject guard + syndication dedup: {len(new)}")
    if new.empty:
        print("nothing to add")
        return

    # Tag as the disaster/stability determinant (hypothesis T7) and mark the class.
    new["is_relevant"] = True
    new["is_climate_shock"] = True
    new["top_hypothesis"] = "T7"
    new["top_topic_name"] = "disaster_displacement"
    new["food_insecurity_score"] = 0.5
    new["fetcher_source"] = "climate_shock"

    def q(p):
        try:
            dt = pd.Timestamp(p)
            return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        except Exception:
            return ""
    new["quarter"] = new["published"].map(q)

    if "is_climate_shock" not in c.columns:
        c["is_climate_shock"] = False
    cols = list(c.columns)
    for x in cols:
        if x not in new.columns:
            new[x] = None
    comb = pd.concat([c, new[cols]], ignore_index=True)
    comb = comb.drop_duplicates("article_id").drop_duplicates("link")
    comb["is_climate_shock"] = comb["is_climate_shock"].fillna(False)
    comb.to_parquet(GEO, index=False)
    print(f"corpus_geocoded: {len(c)} -> {len(comb)} (+{len(comb) - len(c)} climate-shock)")
    print("province spread of additions:", new["province_name"].value_counts().to_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else collect()


if __name__ == "__main__":
    main()
