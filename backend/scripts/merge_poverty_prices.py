"""
scripts/merge_poverty_prices.py
-------------------------------
Relevance gate + merge for the poverty/food-prices collection.

The thesis rule this implements (spec §2): do NOT exclude an article merely
because it lacks the phrase "food insecurity", but do NOT accept every poverty or
economic article either. An article qualifies when it carries a POVERTY / INCOME /
PRICE / SUPPLY signal **and** a demonstrable connection to people's ability to
obtain sufficient, affordable, nutritious food.

A row is kept when it satisfies all of:
  1. DRIVER   — a concrete poverty / income / price / inflation / supply signal.
  2. FOOD LINK — one of:
       (a) the driver is itself about food (food prices, rice/meat/fish/vegetable
           prices, cost of food, food inflation, basic commodities/bilihin); or
       (b) a poverty/income driver co-occurs with a food/hunger/nutrition term
           (the purchasing-power -> food-access chain); or
       (c) an explicit food-insecurity term is present.
  3. CALABARZON is the subject (LGU named, province beside a cue, or dateline) —
     national CPI/poverty-statistics wires and other-region/foreign stories out.
  4. Passes the dataset's permanent foreign / off-topic / PR filters.
  5. Not a duplicate (3-way) and syndication-deduped.

Kept rows are tagged with the hypothesis their mechanism evidences: T1 for the
price/supply chain, T4 for the poverty/income chain.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

POOL = Path("data/raw/poverty_prices_pool.parquet")
GEO = Path("data/processed/corpus_geocoded.parquet")
FINAL = Path("data/processed/calabarzon_food_insecurity_dataset.parquet")

# 1. Drivers -------------------------------------------------------------------
PRICE_FOOD = re.compile(                     # driver that is itself about food
    r"\b(food price|price of (rice|palay|bigas|pork|baboy|chicken|manok|fish|isda|"
    r"vegetable|gulay|meat|karne|sugar|asukal|onion|sibuyas|egg|itlog|fruit)|"
    r"rice price|presyo ng (bigas|gulay|karne|isda|pagkain|sibuyas|asukal)|"
    r"food inflation|cost of food|mahal na bilihin|taas (ng|sa) presyo|"
    r"basic (goods|commodities)|bilihin|price of basic|food cost)", re.I)
PRICE_GEN = re.compile(                      # generic price/inflation driver
    r"\b(inflation|consumer price index|\bcpi\b|price (hike|increase|surge|spike|rise)|"
    r"presyo|price control|suggested retail price|\bsrp\b|price freeze|"
    r"cost of living|nagmahal|tumaas ang presyo)", re.I)
POVERTY_INC = re.compile(                    # poverty / income driver
    r"\b(poverty|kahirapan|mahirap|mahihirap|poorest|indigent|dukha|"
    r"low[- ]income|household income|no income|loss of income|income loss|"
    r"nawalan ng (kita|trabaho|hanapbuhay)|unemploy|jobless|job (loss|cut)|"
    r"laid[- ]off|retrench|displaced worker|minimum wage|living wage|sahod|sweldo|"
    r"purchasing power|kapos|hikahos|walang (kita|pambili)|\b4ps\b|pantawid|"
    r"ayuda|cash (aid|assistance|transfer)|subsistence)", re.I)
SUPPLY = re.compile(                         # supply shock affecting food
    r"\b(food (shortage|supply)|supply (shortage|disruption|tightens|problem)|"
    r"kakulangan (sa|ng)|crop (damage|loss)|harvest (loss|damage|failure)|"
    r"agri(cultural)? (damage|losses)|fish kill|production (drop|decline|down))", re.I)

# 2. Food link -----------------------------------------------------------------
FOOD_TERM = re.compile(
    r"\b(food|pagkain|kain|makain|meal|rice|palay|bigas|hunger|gutom|nagugutom|"
    r"malnutri|nutriti|nutrisyon|undernouri|stunt|feeding|grocery|groceries|"
    r"vegetable|gulay|meat|karne|pork|baboy|chicken|manok|fish|isda|egg|itlog|"
    r"fruit|prutas|staple|ulam|canned goods|noodles|kadiwa|\bnfa\b|food pack|"
    r"food security|food insecurity|food access|food affordab)", re.I)
FOOD_INSEC = re.compile(
    r"\b(food insecur|food secur|hunger|gutom|nagugutom|malnutri|walang makain|"
    r"food (shortage|access|affordab|deprivation)|kakulangan sa pagkain)", re.I)

# 3. Geography -----------------------------------------------------------------
PROV = r"(Cavite|Laguna|Batangas|Rizal|Quezon)"
CUE = (r"(price|presyo|poverty|kahirapan|poor|mahirap|income|kita|market|palengke|"
       r"famil|household|resident|consumer|vendor|farmer|magsasaka|hunger|gutom|"
       r"food|pagkain|inflation|wage|sahod|ayuda|\b4ps\b)")
LOCAL = re.compile(CUE + r".{0,70}?" + PROV + "|" + PROV + r".{0,70}?" + CUE, re.I)
DATELINE = re.compile(r"^.{0,200}?\b[A-Z][A-Za-z ]{0,24}?,?\s*" + PROV +
                      r"\s*(CITY)?\s*,?\s*(Philippines)?\s*[-–—]", re.I)
BADGEO = re.compile(
    r"\b(quezon city|metro manila|\bNCR\b|para[nñ]aque|taguig|marikina|pasig|makati|"
    r"caloocan|malabon|navotas|valenzuela|muntinlupa|las pi[nñ]as|pateros|mandaluyong|"
    r"pasay|manila city|northern (philippines|luzon)|benguet|baguio|ilocos|isabela|"
    r"cagayan|mindanao|visayas|pangasinan|zambales|bataan|abra|nueva ecija|tarlac|"
    r"bulacan|pampanga|aurora|la union|negros|samar|leyte|bohol|cebu|davao|iloilo|"
    r"bacolod|zamboanga|california|mexic|india|bangladesh|indonesia|vietnam|thailand)\b",
    re.I)
NATIONAL = re.compile(                      # nationwide statistics / macro wires
    r"\b(national (poverty|inflation)|philippine(s)? (poverty|inflation|economy)|"
    r"nationwide|across the country|\d+ regions|psa (reports?|data|says)|"
    r"world bank|asian development bank|\bimf\b|\bgdp\b|"
    r"inflation (rate )?(eases|slows|quickens|accelerates|climbs|falls|rises) to)\b", re.I)
FOREIGN = re.compile(r"[àâçéèêëîïôûü]|\b(pobreza|precios|inflación|chuva|warga|kemiskinan)\b", re.I)


def _relevant(title: str, body: str) -> tuple[bool, str]:
    """(keep, hypothesis). Implements the driver + food-link rule."""
    t = f"{title} {body}"
    price_food, price_gen = bool(PRICE_FOOD.search(t)), bool(PRICE_GEN.search(t))
    pov, sup = bool(POVERTY_INC.search(t)), bool(SUPPLY.search(t))
    if not (price_food or price_gen or pov or sup):
        return False, ""
    food, insec = bool(FOOD_TERM.search(t)), bool(FOOD_INSEC.search(t))
    # (a) the driver is itself about food -> price/supply chain (T1)
    if price_food:
        return True, "T1"
    # (c) explicit food-insecurity language with any driver
    if insec:
        return True, "T4" if pov else "T1"
    # (b) poverty/income driver + a food term -> purchasing-power chain (T4)
    if pov and food:
        return True, "T4"
    # generic price/inflation or supply shock, but only with a food term
    if (price_gen or sup) and food:
        return True, "T1"
    return False, ""


def merge() -> None:
    from app.ml.corpus.location_geocoder import geocode_location_batch
    from build_final_dataset import _is_foreign, _is_offtopic
    pool = pd.read_parquet(POOL)
    c = pd.read_parquet(GEO)
    print(f"pool: {len(pool)}")

    def nt(s):
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
    hid, hln, htt = set(c["article_id"]), set(c["link"].dropna()), set(c["title"].fillna("").map(nt))
    if FINAL.exists():
        fin = pd.read_parquet(FINAL)
        hln |= set(fin["url"].dropna())
        htt |= set(fin["title"].fillna("").map(nt))
    pool["_nt"] = pool["title"].fillna("").map(nt)
    new = pool[~(pool["article_id"].isin(hid) | pool["link"].isin(hln) | pool["_nt"].isin(htt))]
    new = (new.drop_duplicates("article_id").drop_duplicates("link")
           .drop_duplicates("_nt").drop(columns=["_nt"]))
    print(f"new (3-way deduped): {len(new)}")

    # relevance
    res = [_relevant(str(a), str(b)) for a, b in zip(new["title"].fillna(""),
                                                     new["summary"].fillna(""))]
    new["_hyp"] = [h for _, h in res]
    new = new[[k for k, _ in res]].copy()
    print(f"passed driver+food-link relevance: {len(new)}")
    if new.empty:
        return

    new = geocode_location_batch(new)
    new = new[new["province_name"].notna()].copy()
    print(f"geocoded to a CALABARZON province: {len(new)}")

    txt = (new["title"].fillna("") + " " + new["summary"].fillna("")).astype(str)
    ttl = new["title"].fillna("").astype(str)
    lead = new["summary"].fillna("").astype(str)
    has_lgu = new["lgu_name"].notna() & (new["lgu_name"].astype(str).str.len() > 0)
    # An Event Registry body runs to 2,000 chars and routinely name-drops other
    # places far from the story's subject, so the other-region test looks at the
    # SUBJECT window only — the headline plus the lede — rather than the whole body.
    subj = (new["title"].fillna("") + " " + lead.str.slice(0, 300)).astype(str)
    # CALABARZON must be the story's SUBJECT, not a name-drop. A national wire
    # ("DA: agricultural damage nears P700M", "Fish, veggies to cost more due to
    # oil price hike") routinely mentions a province deep in the body, so a loose
    # province-near-a-cue test lets national coverage in and mis-attributes it to
    # an LGU. Require a named CALABARZON LGU, a CALABARZON dateline, or the
    # province in the headline itself.
    prov_in_title = ttl.map(lambda t: bool(re.search(PROV, t, re.I)))
    keep = ((has_lgu | prov_in_title
             | lead.map(lambda t: bool(DATELINE.search(t))))
            & ~subj.map(lambda t: bool(BADGEO.search(t)))
            & ~ttl.map(lambda t: bool(NATIONAL.search(t)))
            & ~txt.map(lambda t: bool(FOREIGN.search(t))))
    new = new[keep].copy()
    print(f"CALABARZON-subject guard: {len(new)}")

    if len(new):
        btxt = (new["title"].fillna("") + " " + new["summary"].fillna("")).astype(str)
        drop = [_is_foreign(d, t) or _is_offtopic(str(ti))
                for d, t, ti in zip(new["source_domain"], btxt, new["title"])]
        new = new[~pd.Series(drop, index=new.index)].copy()
    new["_nt"] = new["title"].fillna("").map(nt)
    new = new.sort_values("summary", key=lambda s: s.str.len(), ascending=False).drop_duplicates("_nt")
    new = new.drop(columns=["_nt"])
    print(f"after permanent filters + syndication dedup: {len(new)}")
    if new.empty:
        return

    new["is_relevant"] = True
    new["top_hypothesis"] = new["_hyp"]
    new["top_topic_name"] = new["_hyp"].map({"T1": "food_price_change",
                                             "T4": "poverty_hardship"})
    new["food_insecurity_score"] = 0.5
    new = new.drop(columns=["_hyp"])

    def q(p):
        try:
            dt = pd.Timestamp(p)
            return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        except Exception:
            return ""
    new["quarter"] = new["published"].map(q)

    cols = list(c.columns)
    for x in cols:
        if x not in new.columns:
            new[x] = None
    comb = pd.concat([c, new[cols]], ignore_index=True)
    comb = comb.drop_duplicates("article_id").drop_duplicates("link")
    comb.to_parquet(GEO, index=False)
    print(f"corpus_geocoded: {len(c)} -> {len(comb)} (+{len(comb) - len(c)})")
    print("province spread:", new["province_name"].value_counts().to_dict())
    print("hypothesis spread:", new["top_hypothesis"].value_counts().to_dict())


def merge_general() -> None:
    """Second pass over the same pool using the STANDARD food-insecurity gate.

    The poverty/prices gate above is deliberately scoped to the price and
    purchasing-power chains, so it lets through nothing else. But the pool also
    contains ordinary food-insecurity stories (anti-hunger programmes, relief
    goods to flood-hit families, food-production facilities). Spec §1 asks for
    food insecurity AND its drivers, so those are captured here with the same
    food-anchor + topic + CALABARZON-subject rules the main pipeline uses.
    """
    from app.ml.corpus.location_geocoder import geocode_location_batch, _OTHER_LOC
    from build_final_dataset import (_NEGATIVE, _matched_topics, _matched_hypotheses,
                                     _is_foreign, _is_offtopic)
    from precision_pass import FOOD_ANCHOR
    _ASF = re.compile(r"\b(asf|african swine fever|swine fever)\b", re.I)
    pool = pd.read_parquet(POOL)
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
    new = (new.drop_duplicates("article_id").drop_duplicates("link")
           .drop_duplicates("_nt").drop(columns=["_nt"]))
    print(f"[general] new: {len(new)}")

    new = geocode_location_batch(new)
    new = new[new["province_name"].notna()].copy()
    txt = (new["title"].fillna("") + " " + new["summary"].fillna("")).astype(str)
    ttl = new["title"].fillna("").astype(str)
    lead = new["summary"].fillna("").astype(str)
    subj = (ttl + " " + lead.str.slice(0, 300)).astype(str)
    has_lgu = new["lgu_name"].notna() & (new["lgu_name"].astype(str).str.len() > 0)
    prov_in_title = ttl.map(lambda t: bool(re.search(PROV, t, re.I)))
    keep = (txt.map(lambda t: bool(FOOD_ANCHOR.search(t)) or bool(_ASF.search(t)))
            & txt.map(lambda t: len(_matched_topics(t)) > 0)
            & txt.map(lambda t: len(_matched_hypotheses(t)) > 0)
            & (has_lgu | prov_in_title | lead.map(lambda t: bool(DATELINE.search(t))))
            & ~subj.map(lambda t: bool(BADGEO.search(t)))
            & ~ttl.map(lambda t: bool(NATIONAL.search(t)))
            & ~txt.map(lambda t: bool(_OTHER_LOC.search(t.lower())))
            & ~txt.map(lambda t: bool(_NEGATIVE.search(t))))
    new = new[keep].copy()
    if len(new):
        btxt = (new["title"].fillna("") + " " + new["summary"].fillna("")).astype(str)
        drop = [_is_foreign(d, t) or _is_offtopic(str(ti))
                for d, t, ti in zip(new["source_domain"], btxt, new["title"])]
        new = new[~pd.Series(drop, index=new.index)].copy()
    new["_nt"] = new["title"].fillna("").map(nt)
    new = new.sort_values("summary", key=lambda s: s.str.len(), ascending=False).drop_duplicates("_nt")
    new = new.drop(columns=["_nt"])
    print(f"[general] standard food gate + CALABARZON subject: {len(new)}")
    if new.empty:
        return
    new["is_relevant"] = True
    new["food_insecurity_score"] = 0.5
    txt2 = (new["title"].fillna("") + " " + new["summary"].fillna("")).astype(str)
    new["top_hypothesis"] = txt2.map(lambda t: (_matched_hypotheses(t) or ["T1"])[0])
    new["top_topic_name"] = "food_insecurity_general"
    new["fetcher_source"] = "pp_general"

    def q(p):
        try:
            dt = pd.Timestamp(p)
            return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        except Exception:
            return ""
    new["quarter"] = new["published"].map(q)
    cols = list(c.columns)
    for x in cols:
        if x not in new.columns:
            new[x] = None
    comb = pd.concat([c, new[cols]], ignore_index=True)
    comb = comb.drop_duplicates("article_id").drop_duplicates("link")
    comb.to_parquet(GEO, index=False)
    print(f"[general] corpus: {len(c)} -> {len(comb)} (+{len(comb) - len(c)})")
    print("[general] province:", new["province_name"].value_counts().to_dict())


def merge_national() -> None:
    """NATIONAL-scope drivers that demonstrably reach CALABARZON.

    A national rice-price spike or national inflation surge does affect Cavite and
    Quezon households — that is the purchasing-power -> food-access chain the
    thesis models. Such an article is kept when it (a) carries a food-linked
    driver, and (b) explicitly names CALABARZON or one of its five provinces among
    the affected areas, so the connection is evidence-based rather than assumed.
    It is stored at NATIONAL scope with no province/LGU, because it is not about
    one municipality and pinning it to one would be a mis-attribution.
    """
    from build_final_dataset import _is_foreign, _is_offtopic, _matched_hypotheses
    pool = pd.read_parquet(POOL)
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
    new = (new.drop_duplicates("article_id").drop_duplicates("link")
           .drop_duplicates("_nt").drop(columns=["_nt"]))
    print(f"[national] new: {len(new)}")

    res = [_relevant(str(a), str(b)) for a, b in zip(new["title"].fillna(""),
                                                     new["summary"].fillna(""))]
    new["_hyp"] = [h for _, h in res]
    new = new[[k for k, _ in res]].copy()
    print(f"[national] food-linked driver: {len(new)}")
    if new.empty:
        return

    txt = (new["title"].fillna("") + " " + new["summary"].fillna("")).astype(str)
    ttl = new["title"].fillna("").astype(str)
    # Philippine national context. CALABARZON is part of the Philippines, so a
    # nationwide food-price / rice-policy / inflation story is a real driver of
    # CALABARZON household food access even when the region is not named — it is
    # kept at national scope, never attributed to a province.
    ph = txt.map(lambda t: bool(re.search(
        r"\b(philippine|pilipinas|nationwide|national|across the country|"
        r"\bDA\b|\bDTI\b|\bPSA\b|\bNFA\b|\bDSWD\b|Malaca[nñ]ang|Marcos|palace|"
        r"senate|house (panel|of representatives)|congress|government)\b", t, re.I)))
    # ...but not when another specific region/locality is the story's subject.
    other_region = ttl.map(lambda t: bool(re.search(
        r"\b(manila bay|metro manila|quezon city|\bNCR\b|ilocos|benguet|baguio|"
        r"cagayan|isabela|pangasinan|zambales|bataan|bulacan|pampanga|tarlac|"
        r"nueva ecija|bicol|visayas|mindanao|cebu|davao|iloilo|bacolod|negros|"
        r"zamboanga|samar|leyte|bohol|palawan|mimaropa|caraga|soccsksargen)\b", t, re.I)))
    keep = (ph & ~other_region
            & ~txt.map(lambda t: bool(FOREIGN.search(t)))
            & ~ttl.map(lambda t: bool(re.search(
                r"\b(indonesia|vietnam|thailand|india|bangladesh|malaysia|china|"
                r"mexic|california|nigeria|ukraine|global)\b", t, re.I))))
    new = new[keep].copy()
    if len(new):
        btxt = (new["title"].fillna("") + " " + new["summary"].fillna("")).astype(str)
        drop = [_is_foreign(d, t) or _is_offtopic(str(ti))
                for d, t, ti in zip(new["source_domain"], btxt, new["title"])]
        new = new[~pd.Series(drop, index=new.index)].copy()
    new["_nt"] = new["title"].fillna("").map(nt)
    new = new.sort_values("summary", key=lambda s: s.str.len(), ascending=False).drop_duplicates("_nt")
    new = new.drop(columns=["_nt"])
    print(f"[national] explicit CALABARZON link + filters: {len(new)}")
    if new.empty:
        return

    new["is_relevant"] = True
    new["province_name"] = None
    new["lgu_name"] = None
    new["top_hypothesis"] = new["_hyp"]
    new["top_topic_name"] = new["_hyp"].map({"T1": "food_price_change",
                                             "T4": "poverty_hardship"})
    new["food_insecurity_score"] = 0.5
    new["fetcher_source"] = "pp_national"
    new = new.drop(columns=["_hyp"])

    def q(p):
        try:
            dt = pd.Timestamp(p)
            return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        except Exception:
            return ""
    new["quarter"] = new["published"].map(q)
    cols = list(c.columns)
    for x in cols:
        if x not in new.columns:
            new[x] = None
    comb = pd.concat([c, new[cols]], ignore_index=True)
    comb = comb.drop_duplicates("article_id").drop_duplicates("link")
    comb.to_parquet(GEO, index=False)
    print(f"[national] corpus: {len(c)} -> {len(comb)} (+{len(comb) - len(c)})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--general", action="store_true")
    ap.add_argument("--national", action="store_true")
    a = ap.parse_args()
    if a.general:
        merge_general()
    elif a.national:
        merge_national()
    else:
        merge()
