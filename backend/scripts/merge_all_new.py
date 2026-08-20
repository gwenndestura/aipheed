"""
scripts/merge_all_new.py
-------------------------
Fold BOTH new pools into corpus_geocoded with strict 3-way de-duplication
(article_id + link + normalized title) so no article is ever duplicated:

  1. data/raw/deep_pool.parquet         — ER per-LGU + NewsData (needs scoring)
  2. an external CSV (--csv PATH)        — already relevance-scored (e.g. a
                                           Colab/RunPod export the user provides)

Deep-pool rows are scored on 12 cores; external-CSV rows are trusted as-is where
is_relevant is True. All genuinely-new relevant rows are geocoded (142-LGU
gazetteer) and appended; non-CALABARZON rows are dropped.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GEO = Path("data/processed/corpus_geocoded.parquet")
DEEP = Path("data/raw/deep_pool.parquet")


def _nt(s) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def _lexical_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the non-food-insecurity noise BEFORE any NLI scoring: require a
    food-anchor + a specific food-insecurity topic, and no other-region /
    energy / foreign / food-culture cue."""
    from build_final_dataset import _NEGATIVE, _matched_topics  # noqa
    from precision_pass import FOOD_ANCHOR
    from app.ml.corpus.location_geocoder import _OTHER_LOC
    txt = (df["title"].fillna("") + " " + df["summary"].fillna("")).astype(str)
    keep = (txt.map(lambda t: bool(FOOD_ANCHOR.search(t)))
            & txt.map(lambda t: len(_matched_topics(t)) > 0)
            & ~txt.map(lambda t: bool(_OTHER_LOC.search(t.lower())))
            & ~txt.map(lambda t: bool(_NEGATIVE.search(t))))
    return df[keep].copy()


def _nli_relevant(df: pd.DataFrame) -> pd.DataFrame:
    """NLI-score the (already small, gated) set single-core and keep is_relevant.
    Rows carrying a truthy is_relevant already (external pre-scored CSV) are
    trusted as-is; only the unscored remainder is run through the classifier."""
    ck = Path("data/raw/checkpoints/xlmr_scores.parquet")
    done = {r["article_id"]: r for r in pd.read_parquet(ck).to_dict("records")} if ck.exists() else {}
    pre = df["is_relevant"] if "is_relevant" in df.columns else pd.Series([None] * len(df), index=df.index)
    todo = df[~pre.fillna(False).astype(bool) & ~df["article_id"].isin(done)]
    out = df.copy()
    if len(todo):
        from app.ml.nlp.classifier import load_classifier, score_article
        clf = load_classifier()
        print(f"  NLI-scoring {len(todo)} gated survivors (single-core)")
        for idx, r in todo.iterrows():
            s = score_article(clf, str(r.get("title") or ""), str(r.get("summary") or ""))
            out.at[idx, "is_relevant"] = bool(s["is_relevant"])
            out.at[idx, "food_insecurity_score"] = s["food_insecurity_score"]
            out.at[idx, "top_hypothesis"] = s["top_hypothesis"]
            out.at[idx, "top_topic_name"] = s["top_topic_name"]
    # fill cached scores
    for idx, r in out.iterrows():
        if r["article_id"] in done and not bool(pre.get(idx, False)):
            out.at[idx, "is_relevant"] = bool(done[r["article_id"]]["is_relevant"])
    return out[out["is_relevant"] == True].copy()  # noqa: E712


def main() -> None:
    from app.ml.corpus.location_geocoder import geocode_location_batch
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="external pre-scored corpus CSV")
    args = ap.parse_args()

    existing = pd.read_parquet(GEO)
    have_id = set(existing["article_id"].dropna())
    have_link = set(existing["link"].dropna())
    have_title = set(existing["title"].fillna("").map(_nt))
    print(f"existing corpus: {len(existing)}")

    candidates = []
    if DEEP.exists():
        deep = pd.read_parquet(DEEP)
        candidates.append(deep[[c for c in ("title", "link", "article_id", "published",
                                            "summary", "source_domain", "fetcher_source")
                                if c in deep.columns]])
        print(f"  deep pool raw: {len(deep)}")
    if args.csv:
        ext = pd.read_csv(args.csv)
        candidates.append(ext)
        print(f"  external CSV raw: {len(ext)}")
    if not candidates:
        print("no candidates"); return
    allc = pd.concat(candidates, ignore_index=True)

    # ── strict 3-way dedup: against existing corpus AND within candidates ──
    allc["_nt"] = allc["title"].fillna("").map(_nt)
    before = len(allc)
    allc = allc[~allc["article_id"].isin(have_id)]
    allc = allc[~allc["link"].isin(have_link)]
    allc = allc[~allc["_nt"].isin(have_title)]
    allc = allc.drop_duplicates("article_id").drop_duplicates("link").drop_duplicates("_nt")
    allc = allc.drop(columns=["_nt"])
    print(f"  after 3-way dedup vs corpus + self: {len(allc)} new (of {before})")

    # ── geocode + keep CALABARZON ──
    allc = geocode_location_batch(allc)
    allc = allc[allc["province_name"].notna()].copy()
    print(f"  new + CALABARZON-geocoded: {len(allc)}")

    # ── lexical food-insecurity gate BEFORE scoring (drops the disaster-aid /
    #    relief / evacuation / infrastructure noise) ──
    allc = _lexical_gate(allc)
    print(f"  after food-insecurity gate: {len(allc)}")

    # ── NLI-score only the small gated survivor set, keep is_relevant ──
    allc = _nli_relevant(allc)
    print(f"  NLI-relevant (final new): {len(allc)}")

    def _q(p):
        try:
            dt = pd.Timestamp(p); return f"{dt.year}-Q{(dt.month-1)//3+1}"
        except Exception:
            return ""
    allc["quarter"] = allc["published"].map(_q)

    keep_cols = [c for c in existing.columns] + [c for c in
                ("province_name", "lgu_name", "lgu_psgc", "barangay_name",
                 "barangay_psgc", "match_level", "geo_specificity", "geo_conflict")
                if c not in existing.columns]
    combined = pd.concat([existing, allc[[c for c in keep_cols if c in allc.columns]]],
                         ignore_index=True)
    # final safety dedup
    combined = combined.drop_duplicates("article_id").drop_duplicates("link")
    combined.to_parquet(GEO, index=False)
    print(f"corpus_geocoded: {len(existing)} -> {len(combined)} (+{len(combined)-len(existing)})")


if __name__ == "__main__":
    main()
