"""
scripts/score_audited_articles.py
---------------------------------
Score the audited CALABARZON dataset rows that carry no XLM-RoBERTa NLI score.

The audit (audit_stage1..3) dropped the `relevance_score` column and admitted
239 `corpus_recall` rows that the NLI scorer had never seen. FSSI is defined as
the bias-weighted mean of max_h P(entailment) over the 10 thesis hypotheses, so
every row needs a real score before the FSSI chain can be rebuilt.

Scores already present in corpus_geocoded.parquet are reused as-is; only the
gaps are computed. Output: data/processed/_audited_nli_scores.parquet
  article_id, food_insecurity_score, top_hypothesis, score_provenance
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.nlp.classifier import load_classifier, score_article  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("score_audited")

DATASET = Path("data/processed/calabarzon_food_insecurity_dataset.parquet")
CORPUS = Path("data/processed/corpus_geocoded.parquet")
OUT = Path("data/processed/_audited_nli_scores.parquet")


def main() -> None:
    df = pd.read_parquet(DATASET)[["article_id", "title", "content_lead"]]
    log.info("audited rows: %d", len(df))

    corpus = (
        pd.read_parquet(CORPUS)[["article_id", "food_insecurity_score", "top_hypothesis"]]
        .dropna(subset=["food_insecurity_score"])
        .drop_duplicates("article_id")
    )
    have = df.merge(corpus, on="article_id", how="left")
    known = have[have["food_insecurity_score"].notna()].copy()
    known["score_provenance"] = "corpus_geocoded"
    todo = have[have["food_insecurity_score"].isna()].copy()
    log.info("reused scores: %d | to score: %d", len(known), len(todo))

    if not todo.empty:
        clf = load_classifier()
        rows = []
        for n, (_, r) in enumerate(todo.iterrows(), 1):
            res = score_article(clf, str(r["title"] or ""), str(r["content_lead"] or ""))
            rows.append(
                {
                    "article_id": r["article_id"],
                    "food_insecurity_score": res["food_insecurity_score"],
                    "top_hypothesis": res["top_hypothesis"],
                    "score_provenance": "rescored_" + type(clf).__name__.lstrip("_"),
                }
            )
            if n % 25 == 0 or n == len(todo):
                log.info("scored %d/%d", n, len(todo))
        scored = pd.DataFrame(rows)
    else:
        scored = pd.DataFrame(columns=known.columns)

    cols = ["article_id", "food_insecurity_score", "top_hypothesis", "score_provenance"]
    out = pd.concat([known[cols], scored[cols]], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    log.info("saved %d scores -> %s", len(out), OUT)
    log.info("provenance: %s", dict(out["score_provenance"].value_counts()))
    log.info(
        "score range [%.4f, %.4f] mean %.4f",
        out["food_insecurity_score"].min(),
        out["food_insecurity_score"].max(),
        out["food_insecurity_score"].mean(),
    )


if __name__ == "__main__":
    main()
