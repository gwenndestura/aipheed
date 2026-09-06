"""
scripts/rescore_core.py
------------------------
Re-score the current relevant corpus under the CORE-hypothesis relevance gate
(classifier.CORE_TOPIC_IDS) and drop articles that no longer qualify.

Shortcut: core-score <= max-over-10, so any article relevant under the core
gate is already in the current relevant set (corpus_geocoded.parquet). We
therefore only re-score those ~8.7k rows, not the whole 130k pool.

Updates in place:
  • data/raw/checkpoints/xlmr_scores.parquet   (new score/is_relevant/topic)
  • data/processed/corpus_geocoded.parquet     (relevant survivors, province kept)
  • data/processed/corpus_calabarzon.parquet   (geocoded survivors)
Checkpoints every 300 to data/raw/checkpoints/rescore_core_progress.parquet.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("app").setLevel(logging.WARNING)
logger = logging.getLogger("rescore_core")

GEO = Path("data/processed/corpus_geocoded.parquet")
CAL = Path("data/processed/corpus_calabarzon.parquet")
RAW = Path("data/raw/corpus_raw.parquet")
CACHE = Path("data/raw/checkpoints/xlmr_scores.parquet")
CKPT = Path("data/raw/checkpoints/rescore_core_progress.parquet")


def main() -> None:
    from app.ml.nlp.classifier import load_classifier, score_article, CORE_TOPIC_IDS

    clf = load_classifier()
    if clf.mode != "xlm-roberta":
        raise SystemExit("XLM-R not available — cannot re-score.")
    logger.info("core gate: %s", CORE_TOPIC_IDS)

    df = pd.read_parquet(GEO)
    logger.info("current relevant corpus: %d", len(df))

    done: dict[str, dict] = {}
    if CKPT.exists():
        for r in pd.read_parquet(CKPT).to_dict("records"):
            done[r["article_id"]] = r
        logger.info("resume: %d already re-scored", len(done))

    rows = list(done.values())
    recs = df.to_dict("records")
    since = 0
    for i, r in enumerate(recs, 1):
        aid = r["article_id"]
        if aid in done:
            continue
        s = score_article(clf, str(r.get("title") or ""), str(r.get("summary") or ""))
        rows.append({
            "article_id": aid,
            "food_insecurity_score": s["food_insecurity_score"],
            "is_relevant": bool(s["is_relevant"]),
            "top_hypothesis": s["top_hypothesis"],
            "top_topic_name": s["top_topic_name"],
        })
        since += 1
        if since >= 300:
            pd.DataFrame(rows).to_parquet(CKPT, index=False)
            since = 0
            kept = sum(1 for x in rows if x["is_relevant"])
            logger.info("re-scored %d/%d — %d still relevant (%.0f%%)",
                        len(rows), len(df), kept, 100 * kept / len(rows))

    res = pd.DataFrame(rows)
    res.to_parquet(CKPT, index=False)
    kept_ids = set(res[res["is_relevant"]]["article_id"])
    logger.info("re-score done: %d/%d survive core gate (%.0f%%)",
                len(kept_ids), len(df), 100 * len(kept_ids) / len(df))

    # ── Update the score cache (core scores overwrite by article_id) ──────
    cache = pd.read_parquet(CACHE)
    cache = cache[~cache["article_id"].isin(set(res["article_id"]))]
    cache = pd.concat([cache, res], ignore_index=True)
    cache.to_parquet(CACHE, index=False)
    logger.info("score cache updated: %d rows", len(cache))

    # ── Filter corpus to survivors; province_code already attached ────────
    surv = df[df["article_id"].isin(kept_ids)].copy()
    smap = dict(zip(res["article_id"], res["food_insecurity_score"]))
    tmap = dict(zip(res["article_id"], res["top_hypothesis"]))
    surv["food_insecurity_score"] = surv["article_id"].map(smap)
    surv["top_hypothesis"] = surv["article_id"].map(tmap)
    surv.to_parquet(GEO, index=False)
    # Keep corpus_raw in sync too — the earlier omission let a later merge
    # rebuild corpus_geocoded from a stale raw and reintroduce the proxy
    # false positives. All three artifacts must reflect the same survivors.
    surv.to_parquet(RAW, index=False)
    cal = surv[surv["province_code"].notna()]
    cal.to_parquet(CAL, index=False)

    names = {"PH040100000": "Cavite", "PH040200000": "Laguna",
             "PH040300000": "Quezon", "PH040400000": "Rizal", "PH040500000": "Batangas"}
    logger.info("SURVIVING relevant corpus: %d | CALABARZON: %d", len(surv), len(cal))
    logger.info("province breakdown:\n%s",
                cal["province_code"].map(names).value_counts().to_string())


if __name__ == "__main__":
    main()
