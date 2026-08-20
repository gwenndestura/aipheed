"""Merge the targeted Event Registry LGU pool into the corpus.

Scores each targeted article, keeps the food-insecurity-relevant ones, geocodes
them (province + sub-province) with the fixed geocoder, and appends the genuinely
new rows to corpus_raw.parquet and corpus_geocoded.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

POOL = Path("data/raw/targeted_lgu_eventregistry.parquet")
RAW = Path("data/raw/corpus_raw.parquet")
GEO = Path("data/processed/corpus_geocoded.parquet")


def main() -> None:
    from app.ml.corpus.location_geocoder import geocode_location_batch
    from app.ml.nlp.classifier import load_classifier, score_article

    pool = pd.read_parquet(POOL)
    pool = pool[pool["article_id"].notna()].drop(columns=["target_lgu"], errors="ignore")
    print(f"targeted pool: {len(pool)} articles")

    ck = Path("data/raw/checkpoints/xlmr_scores.parquet")
    done = {r["article_id"]: r for r in pd.read_parquet(ck).to_dict("records")} if ck.exists() else {}
    clf = load_classifier()
    keep = []
    for r in pool.to_dict("records"):
        aid = r["article_id"]
        rel = bool(done[aid]["is_relevant"]) if aid in done else \
            bool(score_article(clf, str(r.get("title") or ""), str(r.get("summary") or ""))["is_relevant"])
        if rel:
            keep.append(r)
    new = pd.DataFrame(keep)
    print(f"relevant: {len(new)}")

    existing = pd.read_parquet(GEO)
    new = new[~new["article_id"].isin(set(existing["article_id"].dropna()))]
    new = new[~new["link"].isin(set(existing["link"].dropna()))]
    print(f"genuinely new (not already in corpus): {len(new)}")
    if new.empty:
        print("nothing to merge")
        return

    new = geocode_location_batch(new)
    combined = pd.concat([existing, new], ignore_index=True)
    combined.to_parquet(GEO, index=False)
    print(f"corpus_geocoded: {len(existing)} -> {len(combined)}")

    # keep corpus_raw in sync (best-effort union on article_id)
    raw = pd.read_parquet(RAW)
    raw_cols = [c for c in raw.columns if c in new.columns]
    raw2 = pd.concat([raw, new[raw_cols]], ignore_index=True).drop_duplicates("article_id")
    raw2.to_parquet(RAW, index=False)
    print(f"corpus_raw: {len(raw)} -> {len(raw2)}")

    tagged = combined[combined["province_name"].notna()]
    print(f"\nCALABARZON-tagged now: {len(tagged)} | distinct LGUs: {tagged['lgu_psgc'].nunique()}")


if __name__ == "__main__":
    main()
