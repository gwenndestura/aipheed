"""
scripts/collect_economic_shock.py
---------------------------------
Corpus-wide ECONOMIC-SHOCK sweep. A job loss / layoff / retrenchment / factory or
business closure in CALABARZON is a food-security *determinant* (FAO access
pillar): lost wages cut a household's ability to buy food, even when the article
does not use the word "food". Like the climate-shock sweep, this applies the rule
consistently across the entire collected universe and adds survivors as a distinct
is_economic_shock class (hypothesis T10 = "Livelihood / employment loss affecting
food access", dimension F).

Scope is deliberately tight: SPECIFIC unemployment / job-loss / closure signals,
not the broad "livelihood / income" topic (which the pipeline keeps as a weak,
review-only signal). A national "PH jobless rate rose" wire is not a CALABARZON
event — the same two guards as the climate sweep apply:
  1. CALABARZON must be where the jobs are lost — Metro-Manila / other-region /
     foreign subjects and nationwide labour-statistics wires are dropped; a real
     CALABARZON LGU must be named, or a province must sit next to a worker/company/
     closure cue.
  2. Syndication dedup — one event, not the many outlets reprinting the wire.

--merge appends the guarded, deduped survivors to corpus_geocoded.parquet with
is_economic_shock=True; build_final_dataset exempts exactly those rows from the
food-anchor gate and the needs_review softie cut.
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
OUT = Path("data/raw/_economic_shock_candidates.parquet")

SKIP = {"gdelt_bq_gov_raw.parquet", "gdelt_bq_national_raw.parquet",
        "gdelt_bq_urls_raw.parquet"}
COLS = ["title", "link", "article_id", "published", "summary", "source_domain"]

# Specific employment-shock signals (NOT the broad livelihood/income topic).
UNEMP = re.compile(
    r"\b(unemployment|jobless|joblessness|job loss(es)?|job cut(s)?|lay ?off(s|ed)?|"
    r"laid off|retrench\w*|displaced workers?|mass (termination|layoff|retrench)|"
    r"plant closure|factory closure|factory shutdown|closed (down )?(its )?(plant|factory|"
    r"mill|operations)|cease(d)? operations|ceased operation|halt(ed)? operations|"
    r"business closure|shut ?down|redundan\w*|workforce reduction|downsiz\w*|furlough\w*|"
    r"terminated? \d+ workers?|lost (their )?jobs|out of work|no work no pay|"
    r"nawalan ng trabaho|tanggal sa trabaho|nawalan ng hanapbuhay|sarang? (planta|pabrika|"
    r"kumpanya)|tanggalan|nagsara(ng)? (planta|pabrika|kumpanya|negosyo)|walang trabaho|"
    r"displaced OFWs?|repatriated (workers|OFWs?)|retrenched)\b", re.I)
CALZN = re.compile(r"\b(cavite|laguna|batangas|rizal|quezon|calabarzon|region iv-?a)\b", re.I)

BADGEO = re.compile(
    r"\b(quezon city|metro manila|\bNCR\b|kalakhang maynila|para[nñ]aque|taguig|marikina|"
    r"pasig|makati|caloocan|malabon|navotas|valenzuela|muntinlupa|las pi[nñ]as|pateros|"
    r"mandaluyong|pasay|manila city|city of manila|northern (philippines|luzon)|benguet|"
    r"baguio|la trinidad|ilocos|isabela|cagayan|mindanao|visayas|pangasinan|zambales|"
    r"bataan|abra|kalinga|nueva ecija|nueva vizcaya|tarlac|bulacan|pampanga|aurora|"
    r"la union|ifugao|apayao|california|mexic|negros|occidental|oriental|pontevedra|"
    r"bacolod|iloilo|capiz|aklan|antique)\b", re.I)
# Off-topic even with a closure/worker token: gambling raids, political visits.
_ECON_NOISE = re.compile(r"\b(internet gaming|\bpogo\b|offshore gaming|e-?sabong|casino raid)\b", re.I)
# The employment shock must be visible in the TITLE (not only a stray body word),
# so a politician-visit or macro story whose body merely mentions "trabaho" is dropped.
_TITLE_SIGNAL = re.compile(
    r"\b(plant|mill|factory|pabrika|planta|worker|manggagawa|jobs?|job loss|tupad|"
    r"displaced|shut ?down|closure|closed|lay ?off|laid off|retrench|livelihood|"
    r"kabuhayan|cash (aid|assistance)|financial assistance|hiring|hire|no work|"
    r"unemploy|jobless|nawalan|tanggal|negosyo|pandemic-affected|lose livelihood|"
    r"lost (their )?jobs|out of work)\b", re.I)
# Nationwide labour-statistics / macro wires (by headline) — not a local event.
NATIONAL = re.compile(
    r"\b(unemployment rate|jobless rate|labor force survey|underemployment rate|"
    r"psa (reports?|data)|nationwide|across the country|\d+ regions|"
    r"national (unemployment|jobless)|philippine(s)? (unemployment|jobless|economy)|"
    r"gdp|inflation rate)\b", re.I)
FOREIGN = re.compile(r"[àâçéèêëîïôûü]|\b(tras la|precios|nevada|chuva|warga|selatan|phk|pemutusan)\b", re.I)

PROV = r"(Cavite|Laguna|Batangas|Rizal|Quezon)"
CUE = (r"(worker|employe|plant|factory|mill|company|firm|ecozone|economic zone|"
       r"industrial (park|estate)|laid off|retrench|closure|closed|displaced|"
       r"manggagawa|trabahador|empleyado|pabrika|planta)")
LOCAL = re.compile(CUE + r".{0,60}?" + PROV + "|" + PROV + r".{0,60}?" + CUE, re.I)


def _norm_pool(f: str) -> pd.DataFrame | None:
    df = pd.read_parquet(f)
    if not {"title", "summary"}.issubset(df.columns):
        return None
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    df = df[COLS].copy()
    blob = (df["title"].fillna("") + " " + df["summary"].fillna("")).astype(str)
    df = df[blob.map(lambda t: bool(UNEMP.search(t)) and bool(CALZN.search(t)))]
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
            print(f"  {os.path.basename(f):44s} unemployment+CALABARZON: {len(d)}", flush=True)
    pool = pd.concat(parts, ignore_index=True)
    pool = pool[pool["link"].fillna("").str.len() > 0]

    def nt(s):
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
    pool["_nt"] = pool["title"].fillna("").map(nt)
    pool = (pool.drop_duplicates("article_id").drop_duplicates("link").drop_duplicates("_nt"))
    print(f"\nprefiltered unique unemployment+CALABARZON candidates: {len(pool)}")
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
            & ttl.map(lambda t: bool(_TITLE_SIGNAL.search(t)))      # employment shock in the headline
            & ~txt.map(lambda t: bool(BADGEO.search(t)))
            & ~txt.map(lambda t: bool(_ECON_NOISE.search(t)))
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
    new["_nt"] = new["title"].fillna("").map(nt)
    new = new.sort_values("summary", key=lambda s: s.str.len(), ascending=False).drop_duplicates("_nt")
    new = new.drop(columns=["_nt"])
    print(f"after geo-subject guard + syndication dedup: {len(new)}")
    if new.empty:
        print("nothing to add")
        return

    new["is_relevant"] = True
    new["is_economic_shock"] = True
    new["top_hypothesis"] = "T10"
    new["top_topic_name"] = "employment_loss"
    new["food_insecurity_score"] = 0.5
    new["fetcher_source"] = "economic_shock"

    def q(p):
        try:
            dt = pd.Timestamp(p)
            return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        except Exception:
            return ""
    new["quarter"] = new["published"].map(q)

    if "is_economic_shock" not in c.columns:
        c["is_economic_shock"] = False
    cols = list(c.columns)
    for x in cols:
        if x not in new.columns:
            new[x] = None
    comb = pd.concat([c, new[cols]], ignore_index=True)
    comb = comb.drop_duplicates("article_id").drop_duplicates("link")
    comb["is_economic_shock"] = comb["is_economic_shock"].fillna(False)
    comb.to_parquet(GEO, index=False)
    print(f"corpus_geocoded: {len(c)} -> {len(comb)} (+{len(comb) - len(c)} economic-shock)")
    print("province spread of additions:", new["province_name"].value_counts().to_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else collect()


if __name__ == "__main__":
    main()
