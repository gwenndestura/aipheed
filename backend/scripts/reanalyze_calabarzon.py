"""
scripts/reanalyze_calabarzon.py
--------------------------------
Deep, stricter re-analysis of the ENTIRE collected corpus to produce a
high-precision CALABARZON food-insecurity news dataset for the aiPHeed thesis.

NOT a new collection. Input = all previously collected articles (union of the
enriched pools + GDELT REST checkpoints), deduplicated, best text preferred.

Multi-stage pipeline per article (content-based, not title keyword matching):
  Stage 1  geographic gate      geocoder on title+content (NCR-masked); assign
                                province + city; flag incidental region-only.
  Stage 2  food-candidate gate  broad bilingual food/agri/disaster lexicon
                                (high recall; only skips clearly-unrelated).
  Stage 3  deep semantic (NLI)  XLM-R entailment over the 10 HungerGist
                                hypotheses; CORE (T1,T2,T3,T6,T1b) establish
                                relevance, determinants (T4,T5,T7,T8,T9) add
                                context only.
  Stage 4  classification       HIGH/MEDIUM/LOW/IRRELEVANT + food-security
                                dimension (A-F) + event_type + severity +
                                commodity + population + reason + direct flag.
  Stage 5  dedup + QC           article_id + normalized-title near-dup;
                                keep best-text copy.

Parallelism: ProcessPoolExecutor; each worker loads XLM-R once and processes a
chunk with per-process torch-thread capping to avoid oversubscription.
Resumable: completed article_ids checkpointed each chunk.

Outputs (data/processed/reanalysis/):
  relevant.parquet    HIGH+MEDIUM, CALABARZON, deduped   (final dataset)
  rejected.parquet    LOW/IRRELEVANT/duplicate
  summary.json        counts by relevance/province/dimension/year/event
  quality_report.json completeness + QC counts

Usage:
  venv\\Scripts\\python scripts\\reanalyze_calabarzon.py [--workers N] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("app").setLevel(logging.WARNING)
logger = logging.getLogger("reanalyze")

OUT = Path("data/processed/reanalysis")
PROGRESS = OUT / "progress.parquet"

ENRICHED_SOURCES = [
    "gdelt_calabarzon_recovered_enriched", "gdelt_bq_national_enriched",
    "gdelt_bigquery_enriched", "gdelt_bq_gov_enriched",
    "eventregistry_raw", "commoncrawl",
]

# ── Classification maps (thesis food-security dimensions A–F) ───────────────
CORE_HYP = ("T1", "T2", "T3", "T6", "T1b")          # establish relevance
DET_HYP = ("T4", "T5", "T7", "T8", "T9")            # context/determinant only
DIMENSION = {  # hypothesis -> (dimension_code, dimension_label)
    "T1": ("B", "Food accessibility / affordability"),
    "T2": ("C", "Food utilization / nutrition"),
    "T3": ("E", "Hunger / food deprivation (assistance)"),
    "T6": ("A", "Food availability (production)"),
    "T1b": ("A", "Food availability (fisheries)"),
    "T4": ("B", "Food accessibility (poverty)"),
    "T5": ("D", "Food stability (supply chain)"),
    "T7": ("D", "Food stability (displacement)"),
    "T8": ("D", "Food stability (unrest)"),
    "T9": ("F", "Livelihood affecting food access"),
}
EVENT_TYPE = {
    "T1": "food_price_change", "T2": "malnutrition_nutrition",
    "T3": "food_assistance", "T6": "crop_production_loss",
    "T1b": "fishery_loss", "T4": "poverty_hardship",
    "T5": "supply_disruption", "T7": "disaster_displacement",
    "T8": "unrest_disruption", "T9": "remittance_shock",
}

# Stage-2 broad food-candidate lexicon (recall-oriented; EN + Filipino)
FOOD_LEXICON = re.compile(
    r"\b(food|rice|palay|bigas|hunger|gutom|famine|malnutri|stunt|nutrition|"
    r"feeding|ayuda|relief|pantawid|4ps|kadiwa|nfa|subsid|farm|agri|"
    r"magsasaka|harvest|ani|crop|fisher|mangingisda|fish|isda|aquacultur|"
    r"tilapia|bangus|poultry|hog|swine|asf|livestock|price|presyo|inflation|"
    r"bilihin|palengke|market|supply|shortage|kakulangan|typhoon|bagyo|flood|"
    r"baha|drought|tagtuyot|nino|nina|calamity|evacuat|bakwit|poverty|"
    r"kahirapan|onion|sibuyas|sugar|asukal|vegetable|gulay|coconut|niyog|corn|mais)\b",
    re.I)

SEVERITY_HIGH = re.compile(
    r"\b(crisis|emergency|state of calamity|famine|starv|death|died|dead|"
    r"thousands|millions|severe|devastat|destroy)\b", re.I)
SEVERITY_MED = re.compile(
    r"\b(damage|loss|shortage|hit|affected|decline|drop|surge|spike|displaced|hungry)\b", re.I)

COMMODITY = {
    "rice": r"\b(rice|palay|bigas)\b", "vegetable": r"\b(vegetable|gulay)\b",
    "fish": r"\b(fish|isda|bangus|tilapia|galunggong)\b",
    "pork": r"\b(pork|hog|swine|baboy)\b", "poultry": r"\b(chicken|poultry|manok|egg|itlog)\b",
    "onion": r"\b(onion|sibuyas)\b", "sugar": r"\b(sugar|asukal)\b",
    "coconut": r"\b(coconut|copra|niyog)\b", "corn": r"\b(corn|mais)\b",
}
POPULATION = {
    "farmers": r"\b(farmer|magsasaka|planter)\b", "fisherfolk": r"\b(fisher|mangingisda)\b",
    "families": r"\b(famil|household|pamilya)\b", "children": r"\b(child|bata|infant|toddler)\b",
    "residents": r"\b(resident|community|barangay|village)\b",
    "workers": r"\b(worker|laborer|manggagawa|employee)\b",
}

_clf = None  # per-process model


def _init_worker(threads: int):
    import torch
    torch.set_num_threads(max(1, threads))
    global _clf
    from app.ml.nlp.classifier import load_classifier
    _clf = load_classifier()


def _first(patterns: dict, text: str) -> str | None:
    for name, pat in patterns.items():
        if re.search(pat, text, re.I):
            return name
    return None


def _all(patterns: dict, text: str) -> list[str]:
    return [n for n, p in patterns.items() if re.search(p, text, re.I)]


def classify_batch(records: list[dict]) -> list[dict]:
    from app.ml.corpus.geocoder import geocode_to_province, _alias_match, _mask_negative_context
    from app.ml.nlp.classifier import HYPOTHESES
    prov_names = {"PH040100000": "Cavite", "PH040200000": "Laguna",
                  "PH040300000": "Quezon", "PH040400000": "Rizal", "PH040500000": "Batangas"}
    out = []
    for r in records:
        title = str(r.get("title") or "")
        body = str(r.get("summary") or "")
        text = f"{title}. {body}".strip()
        low = _mask_negative_context(text.lower())

        # Stage 1 — geographic gate (province + incidental detection)
        pc = r.get("province_code")
        if not pc or (isinstance(pc, float)):
            pc = geocode_to_province(text)
        province = prov_names.get(pc)
        region_only = bool(re.search(r"\bcalabarzon\b|\bregion iv-?a\b|\bregion 4-?a\b", low)) and not province
        if province:
            geo = "PROVINCE"
        elif region_only:
            geo = "REGIONAL"
        else:
            geo = "NONE"

        # Stage 2 — food candidate
        food_candidate = bool(FOOD_LEXICON.search(text))

        # Stage 3 — deep NLI (only if geo-plausible and food-candidate; else cheap skip)
        rel = "IRRELEVANT"; dim = None; dim_label = None; top = None
        core_max = det_max = 0.0
        if geo != "NONE" and food_candidate and len(text) > 20:
            try:
                raw = _clf.classify(text[:512])
                sc = {x["topic_id"]: float(x["score"]) for x in raw}
            except Exception:
                sc = {}
            if sc:
                core = {k: v for k, v in sc.items() if k in CORE_HYP}
                det = {k: v for k, v in sc.items() if k in DET_HYP}
                core_max = max(core.values()) if core else 0.0
                det_max = max(det.values()) if det else 0.0
                top_core = max(core, key=core.get) if core else None
                top_det = max(det, key=det.get) if det else None
                if core_max >= 0.60:
                    rel, top = "HIGH", top_core
                elif core_max >= 0.30:
                    rel, top = "MEDIUM", top_core
                elif det_max >= 0.60 and core_max >= 0.15:
                    rel, top = "MEDIUM", top_det
                elif core_max >= 0.15 or det_max >= 0.40:
                    rel, top = "LOW", (top_core or top_det)
                else:
                    rel, top = "IRRELEVANT", (top_core or top_det)
        if top:
            dim, dim_label = DIMENSION.get(top, (None, None))

        is_direct = bool(core_max >= 0.30)
        sev = "high" if SEVERITY_HIGH.search(text) else ("medium" if SEVERITY_MED.search(text) else "low")
        commodity = _all(COMMODITY, text)
        population = _all(POPULATION, text)
        city = None
        am = _alias_match(low)  # alias may resolve to a city-bearing province; keep province only

        reason = None
        if rel in ("HIGH", "MEDIUM") and top:
            reason = f"{EVENT_TYPE.get(top, top)} -> {dim_label} in {province or 'CALABARZON'}"

        out.append({
            **{k: r.get(k) for k in ("article_id", "title", "link", "published", "source_domain")},
            "content_len": len(body),
            "province": province, "province_code": pc, "city_municipality": city,
            "calabarzon_relevance": geo,
            "food_insecurity_relevance": rel,
            "food_security_dimension": dim, "food_security_dimension_label": dim_label,
            "top_hypothesis": top,
            "top_hypothesis_text": HYPOTHESES.get(top) if top else None,
            "core_score": round(core_max, 4), "determinant_score": round(det_max, 4),
            "is_direct_food_insecurity": is_direct,
            "event_type": EVENT_TYPE.get(top) if top else None,
            "severity": sev if rel in ("HIGH", "MEDIUM") else None,
            "affected_commodity": ",".join(commodity) or None,
            "affected_population": ",".join(population) or None,
            "relevance_reason": reason,
        })
    return out


def assemble_corpus() -> pd.DataFrame:
    frames = []
    for s in ENRICHED_SOURCES:
        p = Path(f"data/raw/{s}.parquet")
        if p.exists():
            d = pd.read_parquet(p)
            keep = [c for c in ["article_id", "title", "link", "published",
                                "summary", "source_domain", "province_code"] if c in d.columns]
            frames.append(d[keep])
    import glob
    for f in glob.glob("data/raw/checkpoints/gdelt_20*.parquet"):
        d = pd.read_parquet(f)
        keep = [c for c in ["article_id", "title", "link", "published",
                            "summary", "source_domain"] if c in d.columns]
        frames.append(d[keep])
    alldf = pd.concat(frames, ignore_index=True)
    alldf["_slen"] = alldf["summary"].fillna("").str.len() if "summary" in alldf.columns else 0
    # prefer the copy with the most body text per article_id
    alldf = alldf.sort_values("_slen", ascending=False).drop_duplicates("article_id", keep="first")
    return alldf.drop(columns=["_slen"]).reset_index(drop=True)


def main() -> None:
    import psutil
    ap = argparse.ArgumentParser()
    avail_gb = psutil.virtual_memory().available / 1e9
    default_workers = max(2, min(4, int(avail_gb / 3.0)))
    ap.add_argument("--workers", type=int, default=default_workers)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=200)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = assemble_corpus()
    if args.limit:
        df = df.head(args.limit)
    logger.info("input corpus: %d unique articles | workers=%d", len(df), args.workers)

    done: dict[str, dict] = {}
    if PROGRESS.exists():
        for r in pd.read_parquet(PROGRESS).to_dict("records"):
            done[r["article_id"]] = r
        logger.info("resume: %d already classified", len(done))

    todo = df[~df["article_id"].isin(done)].to_dict("records")
    chunks = [todo[i:i + args.chunk] for i in range(0, len(todo), args.chunk)]
    logger.info("to classify: %d (%d chunks)", len(todo), len(chunks))

    threads = max(1, os.cpu_count() // args.workers)
    results = list(done.values())
    since = 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker, initargs=(threads,)) as ex:
        futs = {ex.submit(classify_batch, c): i for i, c in enumerate(chunks)}
        for n, fut in enumerate(as_completed(futs), 1):
            results.extend(fut.result())
            since += 1
            if since >= 10:
                pd.DataFrame(results).to_parquet(PROGRESS, index=False)
                since = 0
                rel = sum(1 for x in results if x.get("food_insecurity_relevance") in ("HIGH", "MEDIUM")
                          and x.get("calabarzon_relevance") in ("PROVINCE", "REGIONAL"))
                logger.info("chunk %d/%d — %d classified — %d retained so far",
                            n, len(chunks), len(results), rel)

    res = pd.DataFrame(results)
    pd.DataFrame(results).to_parquet(PROGRESS, index=False)
    _finalize(res)


def _finalize(res: pd.DataFrame) -> None:
    # Stage 5 — dedup: article_id done; near-dup by normalized title
    res["_nt"] = res["title"].fillna("").str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    res["is_duplicate"] = res.duplicated(subset=["_nt"], keep="first") & (res["_nt"].str.len() > 0)

    retained_mask = (
        res["calabarzon_relevance"].isin(["PROVINCE", "REGIONAL"])
        & res["food_insecurity_relevance"].isin(["HIGH", "MEDIUM"])
        & (~res["is_duplicate"])
    )
    relevant = res[retained_mask].drop(columns=["_nt"]).copy()
    rejected = res[~retained_mask].drop(columns=["_nt"]).copy()
    relevant.to_parquet(OUT / "relevant.parquet", index=False)
    rejected.to_parquet(OUT / "rejected.parquet", index=False)

    def _year(s):
        try: return str(s)[:4]
        except Exception: return "?"
    relevant["_year"] = relevant["published"].map(_year)

    summary = {
        "total_analyzed": int(len(res)),
        "by_relevance": res["food_insecurity_relevance"].value_counts().to_dict(),
        "by_geo": res["calabarzon_relevance"].value_counts().to_dict(),
        "duplicates": int(res["is_duplicate"].sum()),
        "final_retained": int(len(relevant)),
        "retained_by_province": relevant["province"].value_counts().to_dict(),
        "retained_by_dimension": relevant["food_security_dimension_label"].value_counts().to_dict(),
        "retained_by_event": relevant["event_type"].value_counts().to_dict(),
        "retained_by_year": relevant["_year"].value_counts().sort_index().to_dict(),
        "retained_direct_vs_indirect": relevant["is_direct_food_insecurity"].value_counts().to_dict(),
    }
    quality = {
        "articles_analyzed": int(len(res)),
        "missing_content": int((res["content_len"] == 0).sum()),
        "missing_dates": int(res["published"].isna().sum() + (res["published"].astype(str).str.len() < 4).sum()),
        "missing_location": int((res["calabarzon_relevance"] == "NONE").sum()),
        "duplicates": int(res["is_duplicate"].sum()),
        "retained": int(len(relevant)),
        "rejected": int(len(rejected)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "quality_report.json").write_text(json.dumps(quality, indent=2, default=str))
    logger.info("DONE. retained=%d rejected=%d", len(relevant), len(rejected))
    logger.info("summary:\n%s", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
