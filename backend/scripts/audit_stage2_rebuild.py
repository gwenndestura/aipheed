"""Stage 2 of the dataset audit: apply the article-by-article review, then rebuild.

Stage 1 fixed mechanical defects. This stage applies the manual relevance review
(every surviving row was read: title, lead, date, location) and rebuilds the
derived columns so none of them is blank or keyword-guessed:

  * keep/drop        -- per-article verdict from the review file
  * geography        -- corrections where the tagged LGU was not the article's
                        subject (dateline collisions, surname collisions)
  * hypothesis       -- the single thesis hypothesis each article was verified to
                        support, replacing the regex multi-label
  * category / dimension / event_type / topics -- derived from that hypothesis
                        through the thesis's own CATEGORY map, so they can never
                        be empty or inconsistent with each other
  * dedup            -- near-identical retellings of one story collapsed to one row

Inputs : _audit_stage1.parquet, review decisions TSV, geography-fix TSV
Outputs: the cleaned dataset (CSV + parquet), a combined drop audit, and a
         coverage matrix refresh.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "data/processed/_audit_stage1.parquet"
STAGE1_DROPS = ROOT / "data/processed/_audit_stage1_drops.csv"
OUT_CSV = ROOT / "data/processed/calabarzon_food_insecurity_dataset.csv"
OUT_PARQUET = ROOT / "data/processed/calabarzon_food_insecurity_dataset.parquet"
OUT_DROPS = ROOT / "data/processed/calabarzon_dataset_dropped_audit.csv"
GAZETTEER = ROOT / "data/processed/psgc_gazetteer.parquet"

# ── the thesis's 10 hypotheses (app/ml/nlp/classifier.py) ──────────────────────
HYPOTHESIS_TEXT = {
    "T1": "Food supply disruption or food price increases affecting access to food",
    "T1b": "Fish kill or aquaculture collapse reducing fish food supply",
    "T2": "Health services or nutrition programs unavailable or unaffordable",
    "T3": "Government food security programs ineffective or unavailable",
    "T4": "Economic hardship reducing household income and food purchasing power",
    "T5": "Infrastructure failures limiting food transport or storage",
    "T6": "Agricultural land loss or conversion reducing food production",
    "T7": "Civil displacement or evacuation reducing food access",
    "T8": "Social unrest or conflict disrupting food systems",
    "T9": "OFW remittance reduction reducing household food purchasing power",
}
CATEGORY = {
    "T1": ("B", "Food accessibility / affordability (prices)"),
    "T2": ("C", "Food utilization / nutrition (malnutrition)"),
    "T3": ("E", "Hunger / food deprivation (assistance)"),
    "T6": ("A", "Food availability (crop production)"),
    "T1b": ("A", "Food availability (fisheries)"),
    "T4": ("B", "Food accessibility (poverty)"),
    "T5": ("D", "Food stability (supply chain)"),
    "T7": ("D", "Food stability (disaster / displacement)"),
    "T8": ("D", "Food stability (unrest)"),
    "T9": ("F", "Livelihood affecting food access"),
}
EVENT_MAP = {
    "T1": "food_price_change", "T2": "malnutrition_nutrition", "T3": "food_assistance",
    "T6": "crop_production_loss", "T1b": "fishery_loss", "T4": "poverty_hardship",
    "T5": "supply_disruption", "T7": "disaster_displacement", "T8": "unrest_disruption",
    "T9": "remittance_shock",
}
TOPIC = {
    "T1": "food_prices_affordability", "T1b": "fisheries_livestock",
    "T2": "malnutrition_undernutrition", "T3": "food_assistance_programs",
    "T4": "poverty_food_access", "T5": "supply_chain_distribution",
    "T6": "agricultural_production", "T7": "crop_losses_disaster",
    "T8": "livelihood_income", "T9": "livelihood_income",
}
# Hypotheses whose evidence is food/nutrition itself, versus upstream determinants.
DIRECT = {"T1", "T1b", "T2", "T3", "T6"}

STOP = {"the", "a", "an", "in", "of", "to", "for", "on", "at", "and", "as", "by",
        "is", "are", "was", "were", "be", "from", "with", "amid", "after", "due",
        "s", "sa", "ng", "mga", "na", "ang"}


# Spelling variants that would otherwise split one story into two.
SYNONYM = {"grey": "gray", "colour": "color", "metric": "", "worth": "",
           "million": "m", "billion": "b", "tons": "ton", "tonnes": "ton",
           "tonne": "ton", "brgys": "barangays", "brgy": "barangay",
           "govt": "government", "gov": "government", "agri": "agriculture",
           "agricultural": "agriculture", "damages": "damage", "losses": "loss"}

MONEY = re.compile(r"p\s?([\d,]+(?:\.\d+)?)\s*-?\s*(m|b|million|billion)?", re.I)


def money_values(title):
    """Peso amounts in a headline, normalised to millions."""
    out = set()
    for amount, unit in MONEY.findall(str(title)):
        try:
            value = float(amount.replace(",", ""))
        except ValueError:
            continue
        if (unit or "").lower().startswith("b"):
            value *= 1000
        out.add(value)
    return out


LOCAL_WORDS = re.compile(
    r"\b(town|towns|city|cities|municipalit(?:y|ies)|barangays?|villages?|brgys?)\b", re.I)


def admin_scope(title, city):
    """Whether a headline is about a town/city or about the whole province. A
    town-wide state of calamity and a province-wide one are different events even
    when the wording is nearly identical."""
    if pd.notna(city) or LOCAL_WORDS.search(str(title)):
        return "local"
    return "provincial"


def same_figure(a, b, tol=0.2):
    """Do two headlines quote the same damage figure? One bulletin gets rounded
    differently by each outlet ('P577.39M', 'P577M', 'P578 million'), so compare
    with a tolerance rather than for equality."""
    if not a or not b:
        return False
    return any(abs(x - y) <= tol * max(x, y) for x in a for y in b)


def signature(title):
    """Content-word token set, for near-duplicate detection. Peso amounts are
    removed here and compared separately, so differing roundings of one figure
    do not look like different stories."""
    stripped = MONEY.sub(" ", str(title).lower())
    toks = set()
    for w in re.findall(r"[a-z0-9]+", stripped):
        w = SYNONYM.get(w, w)
        if w and w not in STOP and len(w) > 1 and not w.isdigit():
            toks.add(w)
    return toks


def load_reviews(review_dir):
    """decisions.tsv: row_index, K/D, hypothesis, basis."""
    dec = pd.read_csv(review_dir / "decisions.tsv", sep="\t", header=0,
                      names=["article_id", "verdict", "hypothesis", "basis"])
    geo = pd.read_csv(review_dir / "geo_fix.tsv", sep="\t", header=0,
                      names=["article_id", "province", "city_municipality", "reason"])
    scope = pd.read_csv(review_dir / "scope_fix.tsv", sep="\t", header=0,
                        names=["article_id", "geographic_scope", "provinces_covered",
                               "reason"], keep_default_na=False)
    dates_path = review_dir / "date_fix.tsv"
    dates = (pd.read_csv(dates_path, sep="\t", header=0,
                         names=["article_id", "publication_date", "evidence"])
             if dates_path.exists() else
             pd.DataFrame(columns=["article_id", "publication_date", "evidence"]))
    return (dec.set_index("article_id"), geo.set_index("article_id"),
            scope.set_index("article_id"), dates.set_index("article_id"))


def main(review_dir=ROOT / "data/processed/audit_review"):
    df = pd.read_parquet(STAGE1).reset_index(drop=True)
    dec, geo, scope, dates = load_reviews(Path(review_dir))
    # Articles collected since the last review have no verdict yet. Hold them out
    # rather than guessing: the curated set stays valid and they are listed for
    # review. (Silently keeping them would put unreviewed rows in the dataset;
    # silently dropping them would hide new collection.)
    missing = set(df["article_id"]) - set(dec.index)
    if missing:
        pending = df[df["article_id"].isin(missing)]
        path = Path(review_dir) / "_awaiting_review.csv"
        pending[["article_id", "publication_date", "news_source", "province",
                 "city_municipality", "title", "url"]].to_csv(
            path, index=False, encoding="utf-8-sig")
        print(f"!! {len(missing)} newly collected rows have no review decision and "
              f"were HELD OUT of the dataset.\n"
              f"   Listed in {path.relative_to(ROOT)} -- review them, add verdicts to "
              f"decisions.tsv, then re-run this stage.")
        df = df[~df["article_id"].isin(missing)].reset_index(drop=True)

    # Leads re-fetched from the publishers by audit_recover_leads.py, for rows the
    # original Google-News fetchers stored without any body text.
    recovered = ROOT / "data/processed/audit_review/recovered_leads.csv"
    if recovered.exists():
        got = (pd.read_csv(recovered, encoding="utf-8-sig")
               .dropna(subset=["content_lead"]).drop_duplicates("article_id")
               .set_index("article_id"))
        fill = df["article_id"].map(got["content_lead"])
        df["lead_source"] = df["article_id"].map(got["lead_source"])
        df.loc[df["content_lead"].isna(), "content_lead"] = fill
        df["lead_source"] = df["lead_source"].where(
            df["content_lead"].notna()).fillna(
            pd.Series("original_fetch", index=df.index).where(df["content_lead"].notna()))
        print(f"leads recovered from publishers: {fill.notna().sum()}")
    else:
        df["lead_source"] = None

    # Evidence-based date corrections: the article's own text names a date that
    # contradicts the stored (republication / crawl) stamp.
    corrected = df["article_id"].map(dates["publication_date"])
    if corrected.notna().any():
        print(f"dates corrected from article evidence: {corrected.notna().sum()}")
        df.loc[corrected.notna(), "date_provenance"] = "corrected_from_article_text"
        df["publication_date"] = corrected.fillna(df["publication_date"])

    df["verdict"] = df["article_id"].map(dec["verdict"])
    df["hypothesis_topic"] = df["article_id"].map(dec["hypothesis"])
    df["review_basis"] = df["article_id"].map(dec["basis"])

    drops = []
    for i, r in df[df["verdict"] == "D"].iterrows():
        drops.append({"row": r["_row"], "article_id": r["article_id"], "title": r["title"],
                      "publication_date": r["publication_date"], "province": r["province"],
                      "news_source": r["news_source"], "url": r["url"],
                      "stage": "stage2_relevance_review", "drop_reason": r["review_basis"]})
    keep = df[df["verdict"] == "K"].copy()

    # ---- geography corrections (dateline and surname collisions)
    by_id = {a: i for i, a in keep["article_id"].items()}
    for aid, fix in geo.iterrows():
        if aid not in by_id:
            continue
        i = by_id[aid]
        keep.loc[i, "province"] = fix["province"]
        city = fix["city_municipality"]
        keep.loc[i, "city_municipality"] = None if pd.isna(city) else city

    # ---- derived fields, all from the verified hypothesis so none can be blank
    keep["hypothesis_label"] = keep["hypothesis_topic"].map(HYPOTHESIS_TEXT)
    keep["food_security_dimension"] = keep["hypothesis_topic"].map(lambda h: CATEGORY[h][0])
    keep["food_insecurity_category"] = keep["hypothesis_topic"].map(lambda h: CATEGORY[h][1])
    keep["event_type"] = keep["hypothesis_topic"].map(EVENT_MAP)
    keep["food_insecurity_topics"] = keep["hypothesis_topic"].map(TOPIC)
    keep["is_direct_food_insecurity"] = keep["hypothesis_topic"].isin(DIRECT)
    keep["relevance_tier"] = keep["is_direct_food_insecurity"].map(
        {True: "HIGH", False: "MEDIUM"})
    keep["geographic_scope"] = keep["city_municipality"].notna().map(
        {True: "city_municipality", False: "province"})
    keep["provinces_covered"] = keep["province"].fillna("")

    # Region-wide and national rows carry no single province: the scope override
    # records what they actually cover, so province-level analysis can filter them
    # out while they stay available as regional / nationwide drivers.
    for aid, fix in scope.iterrows():
        if aid not in by_id:
            continue
        i = by_id[aid]
        keep.loc[i, "geographic_scope"] = fix["geographic_scope"]
        keep.loc[i, "provinces_covered"] = fix["provinces_covered"]
        keep.loc[i, "province"] = None
        keep.loc[i, "city_municipality"] = None

    keep["match_level"] = keep["geographic_scope"].map(
        {"city_municipality": "lgu", "province": "province", "region": "region"})

    def summary(r):
        if r["geographic_scope"] == "region":
            loc = f"CALABARZON ({r['provinces_covered'].replace('|', ', ')})"
        elif pd.notna(r["city_municipality"]):
            loc = f"{r['city_municipality']}, {r['province']}"
        else:
            loc = f"{r['province']} (province-wide)"
        return (f"{r['review_basis'].capitalize()}. Supports {r['hypothesis_topic']}: "
                f"{r['hypothesis_label']}. Location: {loc}.")

    keep["relevance_summary"] = keep.apply(summary, axis=1)

    # ---- near-duplicate collapse: one story told by several outlets, or twice by one
    keep = keep.sort_values(["publication_date", "article_id"]).reset_index(drop=True)
    keep["_sig"] = keep["title"].map(signature)
    keep["_money"] = keep["title"].map(money_values)
    keep["_admin"] = [admin_scope(t, c) for t, c in
                      zip(keep["title"], keep["city_municipality"])]
    # Counts ("6 areas" vs "8 areas") must still separate two stories; peso amounts
    # are stripped here and compared separately with a rounding tolerance.
    keep["_nums"] = keep["title"].map(
        lambda t: frozenset(re.findall(r"\d+",
                            MONEY.sub(" ", str(t)).replace(",", ""))))
    keep["_day"] = pd.to_datetime(keep["publication_date"])
    dup_of = {}
    for i in range(len(keep)):
        if i in dup_of:
            continue
        a = keep.loc[i]
        for j in range(i + 1, len(keep)):
            if j in dup_of:
                continue
            b = keep.loc[j]
            if abs((b["_day"] - a["_day"]).days) > 5:
                break
            # Same place at the same granularity: a province-wide declaration is
            # not a duplicate of a single town's, and Lobo's is not Calatagan's.
            if (a["geographic_scope"] != b["geographic_scope"]
                    or str(a["province"]) != str(b["province"])):
                continue
            if (pd.isna(a["city_municipality"]) != pd.isna(b["city_municipality"])
                    or (pd.notna(a["city_municipality"])
                        and a["city_municipality"] != b["city_municipality"])):
                continue
            # A town-wide declaration is not the province-wide one.
            if a["_admin"] != b["_admin"]:
                continue
            # Peso amounts are stripped from the signature, so two damage updates
            # can look identical in wording. The figure is the news: if both quote
            # one and they differ, this is a later bulletin, not a duplicate.
            both_priced = bool(a["_money"]) and bool(b["_money"])
            money_match = both_priced and same_figure(a["_money"], b["_money"])
            if both_priced and not money_match:
                continue
            # Otherwise differing counts mean different reporting ("6 areas" vs
            # "8 areas"). Skipped when the same peso figure already identifies the
            # story, since a mangled amount ("P74 5 M") leaves stray digits behind.
            if not money_match and a["_nums"] != b["_nums"]:
                continue
            inter = len(a["_sig"] & b["_sig"])
            union = len(a["_sig"] | b["_sig"]) or 1
            overlap = inter / union
            if overlap >= 0.55:
                dup_of[j] = i
                continue
            # One damage bulletin, reworded by each outlet. Headlines share little
            # wording, so the quoted figure carries the identification. Same day
            # needs no wording overlap at all; across days require a little.
            if (money_match and a["hypothesis_topic"] == b["hypothesis_topic"]
                    and (abs((b["_day"] - a["_day"]).days) <= 1 or overlap >= 0.2)):
                dup_of[j] = i
    # Second pass: identical headlines are the same story even when the dates
    # disagree. Google-feed dates ("rfc") are crawl stamps and drift on
    # re-surfaced articles, so a directly fetched ISO date wins the tie.
    norm = (keep["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())
    rank = keep.assign(
        _norm=norm,
        _has_lead=keep["content_lead"].notna(),
        _iso=keep["date_provenance"].eq("iso"),
        _has_city=keep["city_municipality"].notna(),
    ).sort_values(["_has_lead", "_iso", "_has_city", "publication_date"],
                  ascending=[False, False, False, True])
    for _, grp in rank.groupby("_norm"):
        if len(grp) < 2:
            continue
        best = grp.index[0]
        for j in grp.index[1:]:
            dup_of.setdefault(j, best)

    for j, i in dup_of.items():
        r, keeper = keep.loc[j], keep.loc[i]
        drops.append({"row": r["_row"], "article_id": r["article_id"], "title": r["title"],
                      "publication_date": r["publication_date"], "province": r["province"],
                      "news_source": r["news_source"], "url": r["url"],
                      "stage": "stage2_dedup",
                      "drop_reason": f"near-duplicate of '{keeper['title'][:70]}' "
                                     f"({keeper['news_source']}, {keeper['publication_date']})"})
    keep = keep.drop(index=list(dup_of)).reset_index(drop=True)

    # ---- final column set
    cols = ["title", "publication_date", "date_provenance", "news_source", "url",
            "content_lead", "lead_source", "title_provenance", "province", "city_municipality", "provinces_covered", "geographic_scope",
            "match_level", "hypothesis_topic", "hypothesis_label",
            "food_security_dimension", "food_insecurity_category",
            "food_insecurity_topics", "event_type", "relevance_tier",
            "is_direct_food_insecurity",
            "review_basis", "relevance_summary", "data_source", "article_id"]
    # affected_commodity / affected_population are dropped: the historical fetchers
    # populated them for under 3% of rows, so they carry no usable signal.
    out = keep[cols].sort_values(["province", "city_municipality", "publication_date"],
                                 na_position="last").reset_index(drop=True)

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    out.to_parquet(OUT_PARQUET, index=False)

    # LGU coverage matrix, regenerated so it always matches the published rows.
    gz = (pd.read_parquet(GAZETTEER)[["province_name", "lgu_name", "lgu_psgc"]]
          .drop_duplicates()
          .rename(columns={"province_name": "province", "lgu_name": "city_municipality"}))
    counts = out.groupby(["province", "city_municipality"]).size().rename("articles")
    cov = gz.merge(counts, on=["province", "city_municipality"], how="left")
    cov["articles"] = cov["articles"].fillna(0).astype(int)
    cov["coverage"] = cov["articles"].gt(0).map({True: "covered_relevant", False: "absent"})
    cov.sort_values(["province", "city_municipality"]).to_csv(
        ROOT / "data/processed/calabarzon_lgu_coverage_matrix.csv",
        index=False, encoding="utf-8-sig")
    print(f"coverage matrix: {int(cov['articles'].gt(0).sum())} of {len(cov)} LGUs covered")

    # Restore the audited README. The legacy build_final_dataset.py overwrites the
    # published one with pre-audit text, so the canonical copy lives in audit_review
    # and is re-published on every run of this stage.
    readme = Path(review_dir) / "README_audited.md"
    if readme.exists():
        (ROOT / "data/processed/calabarzon_dataset_README.md").write_text(
            readme.read_text(encoding="utf-8"), encoding="utf-8")
        print("README restored from audit_review/README_audited.md")

    stage1_drops = pd.read_csv(STAGE1_DROPS, encoding="utf-8") if STAGE1_DROPS.exists() \
        else pd.DataFrame()
    all_drops = pd.concat([stage1_drops, pd.DataFrame(drops)], ignore_index=True)
    all_drops.to_csv(OUT_DROPS, index=False, encoding="utf-8-sig")

    print(f"stage-1 in        : {len(df)}")
    print(f"relevance drops   : {sum(1 for d in drops if d['stage'].endswith('review'))}")
    print(f"duplicate drops   : {len(dup_of)}")
    print(f"RETAINED          : {len(out)}")
    print(f"\nprovinces: {out['province'].value_counts().to_dict()}")
    print(f"LGUs covered: {out['city_municipality'].nunique()}")
    print(f"\nhypotheses:\n{out['hypothesis_topic'].value_counts().to_string()}")
    print(f"\ncategories:\n{out['food_insecurity_category'].value_counts().to_string()}")
    print(f"\nblank checks:")
    for c in cols:
        n = out[c].isna().sum()
        if n:
            print(f"   {c}: {n} null")


if __name__ == "__main__":
    main(*sys.argv[1:])
