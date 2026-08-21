"""
scripts/collect_poverty_shock.py
--------------------------------
Corpus-wide POVERTY / INCOME-LOSS sweep. Poverty and low income are the FAO
ECONOMIC-ACCESS pillar of food security: a household below the poverty line
cannot afford enough food even when food is physically available and prices are
normal. Like the climate- and economic-shock sweeps, every earlier round required
a food-anchor term, and the pipeline additionally treats `poverty_food_access` as
a WEAK (review-only) signal — so poverty-driven food-access stories were dropped
twice over. This adds them as a distinct is_poverty_shock class (hypothesis T4 =
"Food accessibility (poverty)", dimension B).

Scope is deliberately tight, because unlike a typhoon or a layoff, poverty is a
chronic condition mentioned in passing constantly. An article qualifies only if it
carries a CONCRETE poverty / affordability / income-deprivation signal (poverty
incidence, below the poverty line, indigent/poorest families, cannot afford,
walang pambili, 4Ps/Pantawid, ayuda for the poor, subsistence/minimum wage
shortfall) — not the bare word "poor".

Two guards, as in the other sweeps:
  1. CALABARZON must be the subject — Metro-Manila / other-region / foreign
     subjects and NATIONWIDE poverty-statistics wires ("PSA: national poverty
     incidence falls to X%") are dropped; a real CALABARZON LGU must be named, a
     province must sit next to a poverty cue, or the lead must carry a CALABARZON
     dateline.
  2. Syndication dedup — one story, not the many outlets reprinting it.

--merge appends survivors with is_poverty_shock=True; build_final_dataset exempts
exactly those rows from the food-anchor gate, the topic gate, and the
needs_review softie cut.
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
OUT = Path("data/raw/_poverty_shock_candidates.parquet")

SKIP = {"gdelt_bq_gov_raw.parquet", "gdelt_bq_national_raw.parquet",
        "gdelt_bq_urls_raw.parquet"}
COLS = ["title", "link", "article_id", "published", "summary", "source_domain"]

# Concrete poverty / affordability / income-deprivation signals.
POVERTY = re.compile(
    r"\b(poverty incidence|below the poverty line|poverty line|poverty threshold|"
    r"poverty rate|poorest (families|households|municipalit|barangay|towns?)|"
    r"indigent (families|households|residents)|informal settlers?|"
    r"cannot afford|can'?t afford|unaffordable|hindi (kayang|makayanan) bilhin|"
    r"walang (pambili|pang-?bili|maipambili|makain|kita)|kapos (sa|ang)|hikahos|"
    r"nagugutom dahil sa kahirapan|kahirapan|mahihirap na pamilya|dukha|"
    r"\b4ps\b|pantawid pamilya|conditional cash transfer|cash aid for the poor|"
    r"ayuda para sa mahihirap|subsistence (level|income)|below subsistence|"
    r"minimum wage (not enough|insufficient|shortfall)|living wage|"
    r"food poverty|food[- ]poor|hand[- ]to[- ]mouth|no income|loss of income|"
    r"income loss|lost income|reduced income|nawalan ng kita)\b", re.I)
CALZN = re.compile(r"\b(cavite|laguna|batangas|rizal|quezon|calabarzon|region iv-?a)\b", re.I)

BADGEO = re.compile(
    r"\b(quezon city|metro manila|\bNCR\b|kalakhang maynila|para[nñ]aque|taguig|marikina|"
    r"pasig|makati|caloocan|malabon|navotas|valenzuela|muntinlupa|las pi[nñ]as|pateros|"
    r"mandaluyong|pasay|manila city|city of manila|northern (philippines|luzon)|benguet|"
    r"baguio|la trinidad|ilocos|isabela|cagayan|mindanao|visayas|pangasinan|zambales|"
    r"bataan|abra|kalinga|nueva ecija|nueva vizcaya|tarlac|bulacan|pampanga|aurora|"
    r"la union|ifugao|apayao|negros|samar|leyte|bohol|cebu|davao|iloilo|bacolod|"
    r"zamboanga|california|mexic|india|bangladesh|dhaka|africa|indonesia)\b", re.I)
# Nationwide poverty-statistics / macro wires — not a CALABARZON-local story.
NATIONAL = re.compile(
    r"\b(national poverty|philippine(s)? poverty|poverty incidence (in the )?(philippines|country)|"
    r"psa (reports?|data|says)|nationwide|across the country|\d+ regions|"
    r"world bank|adb\b|asian development bank|imf\b|global poverty|"
    r"gdp|inflation rate|economic growth)\b", re.I)
FOREIGN = re.compile(r"[àâçéèêëîïôûü]|\b(tras la|precios|pobreza|chuva|warga|selatan|kemiskinan)\b", re.I)

PROV = r"(Cavite|Laguna|Batangas|Rizal|Quezon)"
CUE = (r"(poor|poverty|indigent|mahirap|kahirapan|famil|household|resident|"
       r"beneficiar|barangay|\b4ps\b|pantawid|ayuda|income|kita|wage)")
LOCAL = re.compile(CUE + r".{0,60}?" + PROV + "|" + PROV + r".{0,60}?" + CUE, re.I)
DATELINE = re.compile(r"^.{0,200}?\b[A-Z][A-Za-z ]{0,24}?,?\s*" + PROV +
                      r"\s*(CITY)?\s*,?\s*(Philippines)?\s*[-–—]", re.I)
# The poverty/affordability angle must be visible in the TITLE, so a story whose
# body merely mentions "4Ps" or "mahirap" in passing does not qualify.
TITLE_SIGNAL = re.compile(
    r"\b(poverty|poor|poorest|indigent|mahirap|mahihirap|kahirapan|dukha|"
    r"cannot afford|can'?t afford|unaffordable|afford|walang (pambili|makain|kita)|"
    r"kapos|hikahos|\b4ps\b|pantawid|ayuda|cash (aid|assistance|transfer)|"
    r"subsistence|minimum wage|living wage|income|kita|hungry|gutom|"
    r"food (aid|assistance|pack|pantry)|feeding|libreng|no income|jobless)\b", re.I)


# 4Ps / Pantawid is a poverty program, so its name pulls in a lot of PROGRAM-
# ADMINISTRATION news that says nothing about food access or poverty hardship:
# housing turnovers, SIM-card and national-ID rollouts, drug-bust delistings,
# anti-fraud task forces, bank/fintech tie-ups, job fairs. Excluded unless the
# story also carries a real food/hardship/income-loss angle (see _FOOD_HARDSHIP).
_PROGRAM_NOISE = re.compile(
    r"\b(housing (unit|project)|turn(ed)? over .{0,20}home|4ph\b|pag-?ibig|dhsud|"
    r"\bsim cards?\b|philsys|national id|\bcbms\b|biometric|"
    r"drug (bust|test|den)|gambling|illegal drugs|delisted|erring|fraud|"
    r"task force|watchlist|\batm\b|banking|fintech|e-?wallet|cash card|"
    r"job fair|hired[- ]on[- ]the[- ]spot|job opportunit\w*|graduat\w*|scholarship|"
    r"data sim|connectivity|internet access)\b", re.I)
# A genuine food-access / hardship / income-loss angle that rescues a 4Ps story.
_FOOD_HARDSHIP = re.compile(
    r"\b(food|rice|bigas|gutom|hunger|nutrition|malnutri|feeding|kain|pagkain|"
    r"lost income|lament .{0,15}income|no income|nawalan ng kita|walang kita|"
    r"cannot afford|can'?t afford|walang pambili|kapos|hikahos|"
    r"poverty (incidence|line|threshold|rate)|poorest|indigent|"
    r"validat\w+ .{0,30}(famil\w*|household\w*)|ayuda)\b", re.I)


def _norm_pool(f: str) -> pd.DataFrame | None:
    df = pd.read_parquet(f)
    if not {"title", "summary"}.issubset(df.columns):
        return None
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    df = df[COLS].copy()
    blob = (df["title"].fillna("") + " " + df["summary"].fillna("")).astype(str)
    df = df[blob.map(lambda t: bool(POVERTY.search(t)) and bool(CALZN.search(t)))]
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
            print(f"  {os.path.basename(f):44s} poverty+CALABARZON: {len(d)}", flush=True)
    pool = pd.concat(parts, ignore_index=True)
    pool = pool[pool["link"].fillna("").str.len() > 0]

    def nt(s):
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
    pool["_nt"] = pool["title"].fillna("").map(nt)
    pool = (pool.drop_duplicates("article_id").drop_duplicates("link").drop_duplicates("_nt"))
    print(f"\nprefiltered unique poverty+CALABARZON candidates: {len(pool)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pool.drop(columns=["_nt"]).to_parquet(OUT, index=False)
    print(f"saved -> {OUT}")


def _guard(new: pd.DataFrame) -> pd.DataFrame:
    txt = (new["title"].fillna("") + " " + new["summary"].fillna("")).astype(str)
    ttl = new["title"].fillna("").astype(str)
    lead = new["summary"].fillna("").astype(str)
    lgu = new["lgu_name"]
    has_lgu = lgu.notna() & (lgu.astype(str).str.len() > 0)
    calzn_subject = (has_lgu | txt.map(lambda t: bool(LOCAL.search(t)))
                     | lead.map(lambda t: bool(DATELINE.search(t))))
    # Program-administration noise, judged on the TITLE only (the story's subject
    # lives in the headline, so an incidental "CBMS"/"PhilSys" mention in the body
    # of a genuine poverty story never triggers it). Rescued when the headline also
    # carries a real food / hardship / income-loss angle.
    prog_noise = (ttl.map(lambda t: bool(_PROGRAM_NOISE.search(t)))
                  & ~ttl.map(lambda t: bool(_FOOD_HARDSHIP.search(t))))
    keep = (calzn_subject
            & ttl.map(lambda t: bool(TITLE_SIGNAL.search(t)))
            & ~prog_noise
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
    from build_final_dataset import _is_foreign, _is_offtopic
    if len(new):
        btxt = (new["title"].fillna("") + " " + new["summary"].fillna("")).astype(str)
        drop = [_is_foreign(d, t) or _is_offtopic(str(ti))
                for d, t, ti in zip(new["source_domain"], btxt, new["title"])]
        new = new[~pd.Series(drop, index=new.index)].copy()
    new["_nt"] = new["title"].fillna("").map(nt)
    new = new.sort_values("summary", key=lambda s: s.str.len(), ascending=False).drop_duplicates("_nt")
    new = new.drop(columns=["_nt"])
    print(f"after geo-subject guard + filters + syndication dedup: {len(new)}")
    if new.empty:
        print("nothing to add")
        return

    new["is_relevant"] = True
    new["is_poverty_shock"] = True
    new["top_hypothesis"] = "T4"
    new["top_topic_name"] = "poverty_hardship"
    new["food_insecurity_score"] = 0.5
    new["fetcher_source"] = "poverty_shock"

    def q(p):
        try:
            dt = pd.Timestamp(p)
            return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        except Exception:
            return ""
    new["quarter"] = new["published"].map(q)

    if "is_poverty_shock" not in c.columns:
        c["is_poverty_shock"] = False
    cols = list(c.columns)
    for x in cols:
        if x not in new.columns:
            new[x] = None
    comb = pd.concat([c, new[cols]], ignore_index=True)
    comb = comb.drop_duplicates("article_id").drop_duplicates("link")
    comb["is_poverty_shock"] = comb["is_poverty_shock"].fillna(False)
    comb.to_parquet(GEO, index=False)
    print(f"corpus_geocoded: {len(c)} -> {len(comb)} (+{len(comb) - len(c)} poverty-shock)")
    print("province spread of additions:", new["province_name"].value_counts().to_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else collect()


if __name__ == "__main__":
    main()
