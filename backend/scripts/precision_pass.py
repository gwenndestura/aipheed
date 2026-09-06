"""
scripts/precision_pass.py
==============================================================================
Precision post-filter for the CALABARZON food-insecurity re-analysis.

The main pipeline (reanalyze_calabarzon.py) scored every article with the
XLM-R NLI classifier and wrote progress.parquet (all 154k rows) + relevant.parquet
(153 HIGH/MEDIUM retained). A manual audit of that retained set found two
systematic false-positive sources:

  1. NLI over-firing on short leads — max P(entailment) reaches 0.98-1.00 on
     off-topic text ("Pag-IBIG housing loans" -> fishery p=0.99), because the
     model's food judgment had no lexical cross-check.
  2. Incidental geocoding — a province token anywhere in the body pins the
     province even when another region clearly dominates
     ("Lapu-Lapu / Mindanao" -> "Quezon").

This pass adds two AGREEMENT gates on the ALREADY-SCORED data (no model
re-run). An article is retained only if the semantic score AND independent
lexical/geographic evidence agree:

  FOOD-ANCHOR gate  a concrete food / agriculture / hunger / nutrition term
                    must actually appear in title+lead (not just generic
                    price/poverty/supply words the NLI hallucinates from).
  GEO gate          an explicit CALABARZON place-name must appear in
                    title+lead (disambiguated: Quezon City / Jose Rizal are
                    NOT the provinces), OR a trusted prior province with NO
                    competing non-CALABARZON location present.

Reuses reanalyze_lib for geographic disambiguation. Outputs:
  data/processed/reanalysis/relevant_precise.parquet   (high-precision set)
  data/processed/reanalysis/dropped_from_relevant.parquet (audit: what fell out & why)
  data/processed/reanalysis/precision_summary.json
and prints a before/after report.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import reanalyze_lib as R  # geographic disambiguation + normalize

OUT = ROOT / "data" / "processed" / "reanalysis"
PROG = OUT / "progress.parquet"

SOURCES = [
    "gdelt_calabarzon_recovered_enriched", "gdelt_bq_national_enriched",
    "gdelt_bigquery_enriched", "gdelt_bq_gov_enriched",
    "eventregistry_raw", "commoncrawl",
]

# ── FOOD-ANCHOR lexicon ─────────────────────────────────────────────────────
# Concrete food / agriculture / hunger / nutrition / fishery terms. Deliberately
# EXCLUDES bare "price / inflation / supply / poverty / typhoon" — those are the
# determinant words the NLI hallucinates a food angle from. A real anchor is an
# actual foodstuff, farming/fishing activity, hunger, nutrition, or food-program
# term. EN + Filipino.
# Leading \b + suffix-permissive stems (match crops/farming/malnutrition/…);
# short or collision-prone terms are pinned as whole words in the 2nd group.
FOOD_ANCHOR = re.compile(
    r"\b(food|pagkain|rice|palay|gulay|veget|onion|sibuyas|kamatis|fisher|fish|"
    r"crop|harvest|pananim|farm|magsasaka|hunger|hungr|gutom|nagugutom|famine|starv|"
    r"malnutri|undernouri|stunting|stunted|wasting|nutri|feeding|nutribun|ayuda|"
    r"kadiwa|poultry|livestock|coconut|copra|sugar|asukal|coffee|kape|barako|"
    r"agri|aquacultur|fishpond|fishkill|mangingisda|fisherfolk)"
    r"|\b(bigas|isda|bangus|tilapia|galunggong|milkfish|sardine|tuna|ani|manok|"
    r"itlog|egg|hog|swine|baboy|pork|meat|karne|niyog|corn|mais|nfa|red tide|"
    r"fish kill|food pack|relief goods|community pantry|libreng bigas|food insecur|"
    r"food security|kakulangan ng (?:pagkain|bigas))\b", re.I)


def _norm(s: str) -> str:
    return R.normalize(s)


# ── Dimension-match lexicon ─────────────────────────────────────────────────
# The winning hypothesis must carry its OWN keyword in the text, so the model's
# dimension label is lexically corroborated. Kills mislabels like a drag-racing
# story tagged fishery_loss, or a recipe tagged food_price_change.
DIM_LEX = {
    "T1":  re.compile(r"\b(price|presyo|inflation|afford|cost|expensive|mahal|cheaper|"
                      r"suppl|shortage|kakulangan|bilihin|subsid|import|tariff)", re.I),
    "T2":  re.compile(r"\b(malnutri|undernouri|stunt|wasting|nutri|feeding|nutribun|"
                      r"hunger|hungr|gutom|famine|starv|diet|food insecur|food security|"
                      r"food deprivation|walang makain)", re.I),
    "T3":  re.compile(r"\b(ayuda|relief|food pack|kadiwa|subsid|assistance|distribut|"
                      r"donat|rice aid|cash aid|pantry|libreng|dswd|nfa)", re.I),
    "T6":  re.compile(r"\b(crop|harvest|palay|farm|magsasaka|yield|planting|"
                      r"agricultur|livestock|poultry|damage)", re.I),
    "T1b": re.compile(r"\b(fish|isda|bangus|tilapia|milkfish|aquacultur|fishpond|"
                      r"fish ?pen|fishkill|fish kill|mangingisda|red tide)", re.I),
    "T4":  re.compile(r"\b(poverty|kahirapan|jobless|unemploy|layoff|income|hardship|"
                      r"walang trabaho|livelihood|hanapbuhay)\b", re.I),
    "T5":  re.compile(r"\b(transport|road|bridge|logistic|storage|farm-to-market|"
                      r"delivery|supply chain|port|warehouse)\b", re.I),
    "T7":  re.compile(r"\b(evacuat|displaced|bakwit|typhoon|bagyo|flood|baha|calamity|"
                      r"shelter|landslide|eruption|nasalanta)\b", re.I),
    "T8":  re.compile(r"\b(strike|protest|welga|unrest|rally|riot)\b", re.I),
    "T9":  re.compile(r"\b(ofw|remittance|overseas filipino|padala|migrant worker)\b", re.I),
}


# Core food dimensions (establish relevance) and their human labels / events.
CORE_DIMS = ["T2", "T3", "T1", "T6", "T1b"]  # priority order for relabeling
DIM_LABEL = {
    "T1": "Food accessibility / affordability",
    "T2": "Food utilization / nutrition",
    "T3": "Hunger / food deprivation (assistance)",
    "T6": "Food availability (production)",
    "T1b": "Food availability (fisheries)",
}
EVENT_OF = {
    "T1": "food_price_change", "T2": "malnutrition_nutrition",
    "T3": "food_assistance", "T6": "crop_production_loss", "T1b": "fishery_loss",
}


def dimension_evidence(top_hyp: str, raw: str):
    """
    Require lexical evidence for a CORE food dimension, but do not demand the
    exact one the NLI guessed. Returns (ok, effective_hyp):
      - ok = at least one core food dimension keyword is present
      - effective_hyp = the NLI's top if it is core and matches, else the
        highest-priority core dimension whose keyword is present
    Determinant-only articles (no core food dimension in text) -> ok=False.
    """
    matched = [d for d in CORE_DIMS if DIM_LEX[d].search(raw)]
    if not matched:
        return False, top_hyp
    if str(top_hyp) in matched:
        return True, str(top_hyp)
    return True, matched[0]


# ── Authoritative CALABARZON gazetteer (all 142 LGUs) from the PSA LGU census ─
CAL_CODE_NAME = {
    "PH040100000": "Cavite", "PH040200000": "Laguna", "PH040300000": "Quezon",
    "PH040400000": "Rizal", "PH040500000": "Batangas",
}

# Matcher construction, the ambiguous-name list and the alias handling all live
# in reanalyze_lib so that this script and reanalyze_calabarzon.py cannot drift
# apart. See reanalyze_lib.build_lgu_matchers for the demote-never-drop rule
# that keeps all 142 LGUs reachable.
AMBIGUOUS_LGU = R.AMBIGUOUS_LGU

# Expanded non-CALABARZON location gazetteer for conflict detection (adds the
# cities that slipped through the audit: Mandaue, Lapu-Lapu, Banilad = Cebu…).
OTHER_LOC = re.compile(
    r"\b(cebu|mandaue|lapu-?lapu|banilad|talisay city|mindanao|davao|iloilo|"
    r"bacolod|zamboanga|cagayan de oro|cdo|leyte|tacloban|samar|bicol|albay|"
    r"legazpi|naga city|sorsogon|catanduanes|masbate|ilocos|vigan|laoag|"
    r"pangasinan|dagupan|bulacan|malolos|pampanga|angeles city|tarlac|zambales|"
    r"olongapo|nueva ecija|cabanatuan|baguio|benguet|la union|palawan|"
    r"puerto princesa|mindoro|calapan|romblon|marinduque|boracay|aklan|antique|"
    r"capiz|roxas city|guimaras|surigao|butuan|agusan|bukidnon|malaybalay|"
    r"misamis|ozamiz|dipolog|pagadian|cotabato|general santos|gensan|"
    r"sultan kudarat|maguindanao|marawi|lanao|sulu|basilan|tawi-tawi|kalinga|"
    r"apayao|ifugao|mountain province|abra|batanes|isabela|cagayan valley|"
    r"tuguegarao|quirino|aurora)\b", re.I)


def build_summary_map() -> dict:
    frames = []
    for s in SOURCES:
        p = ROOT / "data" / "raw" / f"{s}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            if "summary" in d.columns:
                frames.append(d[["article_id", "summary"]])
    for f in (ROOT / "data" / "raw" / "checkpoints").glob("gdelt_20*.parquet"):
        d = pd.read_parquet(f)
        if "summary" in d.columns:
            frames.append(d[["article_id", "summary"]])
    sm = pd.concat(frames).dropna(subset=["article_id"]).drop_duplicates("article_id")
    return dict(zip(sm["article_id"], sm["summary"].astype(str)))


def cal_place_in_text(norm: str, matchers: dict) -> tuple[bool, str, str]:
    """
    Does the (normalised) title+lead name a CALABARZON place?
    Uses the full 142-LGU gazetteer. Returns (found, province, city).
      - unambiguous LGU name  -> found, its province
      - province-level token (Cavite/Batangas/Laguna, Quezon/Rizal province,
        CALABARZON alias) via reanalyze_lib disambiguation
      - ambiguous LGU name (San Pedro, Rizal town…) only if its province
        name co-occurs in the text
    """
    found, prov, city = R.match_cal_lgu(norm, matchers)
    if found:
        return True, prov, city

    g = R.geo_classify(norm, None)
    if g["geo_level"] in ("strong", "region"):
        return True, g["province"], g["city"]
    return False, "", ""


def evaluate(df: pd.DataFrame, smap: dict, matchers: dict) -> pd.DataFrame:
    """Add gate columns to the HIGH/MEDIUM candidate rows."""
    rows = []
    for r in df.itertuples():
        title = str(getattr(r, "title", "") or "")
        summ = smap.get(getattr(r, "article_id"), "")
        raw = f"{title} {summ}"
        norm = _norm(f"{title} . {summ}")

        food_ok = bool(FOOD_ANCHOR.search(raw))
        dim_ok, eff_hyp = dimension_evidence(getattr(r, "top_hypothesis", None), raw)

        has_cal_token, cal_prov, cal_city = cal_place_in_text(norm, matchers)
        has_other = bool(OTHER_LOC.search(raw))
        prior_prov = getattr(r, "province", None)
        prior_ok = isinstance(prior_prov, str) and prior_prov not in ("", "None")

        # geo verdict
        if has_cal_token and not (has_other and cal_city == ""):
            geo_ok, geo_basis = True, "text_place_name"
            province = cal_prov or prior_prov or "CALABARZON"
        elif prior_ok and not has_other:
            geo_ok, geo_basis = True, "trusted_prior_no_conflict"
            province = prior_prov
        else:
            geo_ok = False
            geo_basis = "conflict_other_region" if has_other else "no_calabarzon_evidence"
            province = prior_prov
        g = {"city": cal_city}

        retained = bool(food_ok and geo_ok and dim_ok)
        drop_reason = None
        if not retained:
            fails = []
            if not geo_ok:
                fails.append(f"geo:{geo_basis}")
            if not food_ok:
                fails.append("no_food_anchor")
            if not dim_ok:
                fails.append("dim_mismatch")
            drop_reason = "+".join(fails)

        rows.append({
            "food_anchor_ok": food_ok,
            "dimension_match_ok": dim_ok,
            "geo_ok": geo_ok,
            "geo_basis": geo_basis,
            "precision_tier": ("A_text_verified" if geo_basis == "text_place_name"
                               else "B_prior") if retained else None,
            "province_verified": province,
            "city_verified": g["city"] or getattr(r, "city_municipality", None),
            "dimension_effective": DIM_LABEL.get(eff_hyp) if retained else None,
            "event_effective": EVENT_OF.get(eff_hyp) if retained else None,
            "precise_retained": retained,
            "drop_reason": drop_reason,
        })
    add = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, add], axis=1)


def vc(s):
    return {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}


def main() -> None:
    prog = pd.read_parquet(PROG)
    smap = build_summary_map()

    # Build the CALABARZON LGU matchers from the authoritative PSA census.
    matchers = R.build_lgu_matchers(str(ROOT / "data/processed/lgu_census.parquet"))
    n_un, n_am = len(matchers["unambig"]), len(matchers["ambig"])
    print(f"gazetteer: {n_un + n_am} CALABARZON LGU names "
          f"({n_un} unambiguous, {n_am} require province cue)\n")

    # BEFORE = the pipeline's retained set (HIGH/MEDIUM, CALABARZON, deduped),
    # reproduced from progress with the same rule the pipeline used.
    cand = prog[
        prog["calabarzon_relevance"].isin(["PROVINCE", "REGIONAL"])
        & prog["food_insecurity_relevance"].isin(["HIGH", "MEDIUM"])
    ].copy()
    # dedup by normalized title (same as pipeline _finalize)
    cand["_nt"] = (cand["title"].fillna("").str.lower()
                   .str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip())
    cand = cand[~(cand["_nt"].duplicated(keep="first") & (cand["_nt"].str.len() > 0))]

    ev = evaluate(cand, smap, matchers)
    precise = ev[ev["precise_retained"]].copy()
    dropped = ev[~ev["precise_retained"]].copy()

    # write outputs
    keep_cols = [c for c in ev.columns if not c.startswith("_")]
    precise[keep_cols].to_parquet(OUT / "relevant_precise.parquet", index=False)
    dropped[keep_cols + ["_nt"]].drop(columns=["_nt"]).to_parquet(
        OUT / "dropped_from_relevant.parquet", index=False)

    summary = {
        "before_retained": int(len(cand)),
        "after_retained": int(len(precise)),
        "dropped": int(len(dropped)),
        "drop_reasons": vc(dropped["drop_reason"]),
        "before_by_province": vc(cand["province"]),
        "after_by_province": vc(precise["province_verified"]),
        "after_by_relevance": vc(precise["food_insecurity_relevance"]),
        "after_by_dimension": vc(precise["dimension_effective"]),
        "after_by_event": vc(precise["event_effective"]),
        "after_by_tier": vc(precise["precision_tier"]),
        "after_geo_basis": vc(precise["geo_basis"]),
    }
    (OUT / "precision_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── before/after report ────────────────────────────────────────────────
    print("=" * 68)
    print("PRECISION PASS — BEFORE / AFTER")
    print("=" * 68)
    print(f"Retained BEFORE : {len(cand):>4}")
    print(f"Retained AFTER  : {len(precise):>4}")
    print(f"Dropped         : {len(dropped):>4}")
    print("\nDrop reasons:")
    for k, v in sorted(summary["drop_reasons"].items(), key=lambda x: -x[1]):
        print(f"  {k:<34} {v:>4}")
    print("\nProvince   before -> after")
    provs = ["Batangas", "Quezon", "Laguna", "Rizal", "Cavite", "CALABARZON", "None"]
    b, a = summary["before_by_province"], summary["after_by_province"]
    for p in provs:
        if b.get(p) or a.get(p):
            print(f"  {p:<11} {b.get(p,0):>4} -> {a.get(p,0):>4}")
    print("\nAFTER by dimension:")
    for k, v in sorted(summary["after_by_dimension"].items(), key=lambda x: -x[1]):
        print(f"  {k:<40} {v:>3}")
    print("\nAFTER geo basis:", summary["after_geo_basis"])

    print("\n--- SAMPLE of dropped false positives (audit) ---")
    for r in dropped.sort_values("core_score", ascending=False).head(10).itertuples():
        print(f"  [{r.drop_reason:<28}] {str(r.title)[:58]}")
    print("\nAFTER by tier:", summary["after_by_tier"])
    print("\n--- SAMPLE of surviving high-precision retained ---")
    for r in precise.sort_values("core_score", ascending=False).head(20).itertuples():
        print(f"  [{str(r.province_verified):<10} {str(r.precision_tier):<15} "
              f"{str(r.event_effective):<20}] {str(r.title)[:48]}")


if __name__ == "__main__":
    main()
