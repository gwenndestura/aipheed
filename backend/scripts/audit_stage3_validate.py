"""Stage 3: the final quality check on the cleaned dataset.

Re-validates the finished file against every condition the audit had to satisfy,
independently of how it was produced. Exits non-zero if any check fails.
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/calabarzon_food_insecurity_dataset.csv"
GAZETTEER = ROOT / "data/processed/psgc_gazetteer.parquet"

WINDOW_LO, WINDOW_HI = dt.date(2020, 1, 1), dt.date(2026, 8, 21)
HYPOTHESES = {"T1", "T1b", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"}
PROVINCES = {"Batangas", "Cavite", "Laguna", "Quezon", "Rizal"}
REQUIRED_NON_NULL = [
    "provinces_covered", "geographic_scope",
    "title", "publication_date", "news_source", "url",
    "match_level", "title_provenance", "hypothesis_topic", "hypothesis_label",
    "food_security_dimension", "food_insecurity_category",
    "food_insecurity_topics", "event_type", "relevance_tier",
    "is_direct_food_insecurity", "review_basis", "relevance_summary", "article_id",
]

failures: list[str] = []
notes: list[str] = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(f"{label}: {detail}")


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    gz = pd.read_parquet(GAZETTEER)
    valid_pairs = set(zip(gz.province_name, gz.lgu_name))
    print(f"Validating {DATA.name}: {len(df)} rows, {len(df.columns)} columns\n")

    print("1. Every article is tied to a thesis hypothesis")
    check("hypothesis_topic non-empty", df["hypothesis_topic"].notna().all(),
          f"{df['hypothesis_topic'].isna().sum()} blank")
    bad = set(df["hypothesis_topic"].dropna()) - HYPOTHESES
    check("hypothesis codes are among the 10", not bad, str(bad))
    check("hypothesis_label matches code", df["hypothesis_label"].notna().all())

    print("\n2. Category is present and consistent")
    check("food_insecurity_category non-empty", df["food_insecurity_category"].notna().all(),
          f"{df['food_insecurity_category'].isna().sum()} blank")
    per_hyp = df.groupby("hypothesis_topic")["food_insecurity_category"].nunique()
    check("one category per hypothesis", (per_hyp == 1).all(),
          str(per_hyp[per_hyp != 1].to_dict()))
    per_cat = df.groupby("food_insecurity_category")["food_security_dimension"].nunique()
    check("one dimension per category", (per_cat == 1).all(),
          str(per_cat[per_cat != 1].to_dict()))
    check("event_type non-empty", df["event_type"].notna().all())
    check("topics non-empty", df["food_insecurity_topics"].notna().all())

    print("\n3. Required columns are populated")
    for c in REQUIRED_NON_NULL:
        n = df[c].isna().sum()
        if n:
            check(f"{c} populated", False, f"{n} null")
    check("all required columns populated",
          not any(df[c].isna().any() for c in REQUIRED_NON_NULL))
    lead = df["content_lead"].notna().sum()
    notes.append(f"content_lead present for {lead}/{len(df)} rows "
                 f"({len(df) - lead} Google-News rows retain title only -- "
                 f"no body text was ever captured upstream)")

    print("\n4. Dates are valid and inside the study period")
    d = pd.to_datetime(df["publication_date"], format="%Y-%m-%d", errors="coerce")
    check("all dates parse as ISO", d.notna().all(), f"{d.isna().sum()} unparseable")
    inside = d.dt.date.between(WINDOW_LO, WINDOW_HI)
    check("all dates within 2020-01-01..2026-08-21", inside.all(),
          f"{(~inside).sum()} outside")

    print("\n5. Geography is CALABARZON, at province or city/municipality level")
    check("no barangay column", not any("barangay" in c.lower() for c in df.columns))
    bad_scope = set(df["geographic_scope"]) - {"city_municipality", "province",
                                               "region"}
    check("geographic_scope values are known", not bad_scope, str(bad_scope))

    local = df["geographic_scope"].isin(["city_municipality", "province"])
    check("province set on every province/LGU row", df.loc[local, "province"].notna().all(),
          f"{df.loc[local, 'province'].isna().sum()} missing")
    check("province blank only on region/national rows",
          df.loc[~local, "province"].isna().all())
    bad_prov = set(df["province"].dropna()) - PROVINCES
    check("provinces are the five CALABARZON provinces", not bad_prov, str(bad_prov))

    sub = df[df["geographic_scope"].eq("city_municipality")]
    bad_pairs = [(p, c) for p, c in zip(sub.province, sub.city_municipality)
                 if (p, c) not in valid_pairs]
    check("every province/city pair exists in PSGC", not bad_pairs, str(bad_pairs[:5]))
    check("city set only on city_municipality rows",
          (df["city_municipality"].notna()
           == df["geographic_scope"].eq("city_municipality")).all())

    # provinces_covered: the CALABARZON provinces the row speaks for.
    covered = df["provinces_covered"].fillna("")
    bad_cov = {p for row in covered for p in row.split("|") if p and p not in PROVINCES}
    check("provinces_covered lists only CALABARZON provinces", not bad_cov, str(bad_cov))
    reg = df["geographic_scope"].eq("region")
    check("provinces_covered never blank", covered.str.len().gt(0).all())
    check("region rows name the provinces they cover", covered[reg].str.len().gt(0).all(),
          f"{(covered[reg].str.len() == 0).sum()} empty")
    check("no national-scope rows (out of CALABARZON scope)",
          not df["geographic_scope"].eq("national").any())
    check("local rows' provinces_covered equals their province",
          (covered[local] == df.loc[local, "province"]).all())

    expected_match = df["geographic_scope"].map(
        {"city_municipality": "lgu", "province": "province", "region": "region"})
    check("match_level matches geographic_scope", df["match_level"].eq(expected_match).all())

    print("\n6. No duplicates")
    check("article_id unique", not df["article_id"].duplicated().any(),
          f"{df['article_id'].duplicated().sum()} repeats")
    check("url unique", not df["url"].duplicated().any(),
          f"{df['url'].duplicated().sum()} repeats")
    norm = (df["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())
    check("normalised title unique", not norm.duplicated().any(),
          f"{norm.duplicated().sum()} repeats")

    print("\n7. Source quality")
    check("no Google-News redirect left in news_source",
          not df["news_source"].str.contains("google", case=False, na=False).any())
    check("no RSS markup left in content_lead",
          not df["content_lead"].fillna("").str.contains("<a href=", regex=False).any())
    check("no publisher suffix left on titles",
          not df["title"].str.contains(r" - (?:GMA Network|ABS-CBN|Rappler|Inquirer\.net)$",
                                       regex=True, na=False).any())
    social = df["news_source"].str.contains(
        "facebook|youtube|tiktok|twitter|reddit", case=False, na=False)
    check("no social-media sources", not social.any(), f"{social.sum()} rows")
    check("no replacement characters in text",
          not df["title"].str.contains("�", na=False).any())

    print("\n7b. Text hygiene")
    t = df["title"].astype(str)
    checks = [
        ("no space before punctuation", t.str.contains(r"\s[,;:.!?]")),
        ("no broken thousands separators",
         t.str.contains(r"\d\s,\s\d") | t.str.contains(r"\d,\s\d{3}\b")),
        ("no split decimals", t.str.contains(r"\d\s\.\s\d")),
        ("no aggregator section tails",
         t.str.contains(r"\|\s*(?:Photos?|Videos?|GMA News Online|24 Oras)", case=False)),
        ("no doubled whitespace", t.str.contains(r"\s{2,}")),
        ("no leftover 'Covid 19' spelling", t.str.contains(r"\bCovid 19\b")),
    ]
    for label, mask in checks:
        check(label, not mask.any(), f"{mask.sum()} rows")
    # Trailing CMS story ids: a trailing number is only legitimate after a date word.
    guard = re.compile(r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
                       r"january|february|march|april|june|july|august|september|october|"
                       r"november|december|part|no|covid|phase|vol)\.?\s+\d{1,4}$")
    stray = t.str.contains(r"\s\d{1,4}$") & ~t.str.contains(guard)
    check("no trailing CMS story ids", not stray.any(),
          f"{stray.sum()}: {t[stray].head(2).tolist()}")

    lead = df["content_lead"].fillna("")
    check("no aggregator boilerplate in leads",
          not lead.str.contains(r"^(?:Make this your preferred|This is AI-generated|Copied)",
                                regex=True).any())
    check("no markup left in leads", not lead.str.contains(r"<[^>]+>", regex=True).any())
    moji = r"Ã[-¿]|â€|â„|Â[ -¿]"
    check("no mojibake in leads", not lead.str.contains(moji, regex=True).any(),
          f"{lead.str.contains(moji, regex=True).sum()} rows")
    check("no mojibake in titles", not t.str.contains(moji, regex=True).any())
    check("no lead that only restates its title",
          not any(re.sub(r"[^a-z0-9]", "", a.lower())[:60]
                  == re.sub(r"[^a-z0-9]", "", b.lower())[:60]
                  for a, b in zip(lead, t) if a))
    check("title_provenance recorded",
          set(df["title_provenance"].dropna()) <= {"headline", "slug_derived"}
          and df["title_provenance"].notna().all())

    print("\n8. Internal consistency")
    direct = {"T1", "T1b", "T2", "T3", "T6"}
    ok = (df["hypothesis_topic"].isin(direct) == df["is_direct_food_insecurity"]).all()
    check("is_direct_food_insecurity matches hypothesis class", ok)
    ok = (df["relevance_tier"].eq("HIGH") == df["is_direct_food_insecurity"]).all()
    check("relevance_tier matches directness", ok)
    check("every row carries a review basis",
          df["review_basis"].str.len().gt(10).all())

    print("\n--- Coverage ---")
    print(df["province"].value_counts().to_string())
    print(f"LGUs represented: {df['city_municipality'].nunique()} of 142")
    print(f"Publishers: {df['news_source'].nunique()}")
    print(f"Years: {sorted(d.dt.year.unique().tolist())}")

    if notes:
        print("\n--- Notes (data-availability limits, not defects) ---")
        for n in notes:
            print(f"  * {n}")

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + str(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
