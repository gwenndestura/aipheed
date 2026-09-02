"""
scripts/rescore_all_articles.py
-------------------------------
Score ALL audited articles on one consistent input: title + recovered lead.

Why all of them rather than only the gaps: a paired probe over the 93 rows that
already had lead text showed title-only and title+lead scores correlate at only
0.387, with the top-scoring hypothesis flipping on 43% of articles. They are not
the same measurement, so a column mixing both would average two scales inside
every province-quarter FSSI mean. The previously stored scores were worse still:
69 of 132 sat at exactly 0.5000, the neutral fallback rather than model output.

Output: data/processed/_audited_nli_scores.parquet
    article_id, food_insecurity_score, top_hypothesis, score_input, lead_chars
where score_input is "title+lead" or "title_only" (articles whose text could not
be recovered), so the sensitivity of any result to that split can be checked.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.nlp.classifier import load_classifier, score_article  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rescore_all")

DATASET = Path("data/processed/calabarzon_food_insecurity_dataset.parquet")
OUT = Path("data/processed/_audited_nli_scores.parquet")


def main() -> None:
    df = pd.read_parquet(DATASET)
    df["lead"] = df["content_lead"].fillna("").astype(str)
    log.info("scoring %d articles | %d have lead text, %d title-only",
             len(df), int((df["lead"].str.len() > 0).sum()),
             int((df["lead"].str.len() == 0).sum()))

    clf = load_classifier()
    if type(clf).__name__ != "_XLMRobertaClassifier":
        raise RuntimeError(
            f"expected the XLM-RoBERTa classifier, got {type(clf).__name__}. "
            "Keyword fallback scores are not comparable - aborting rather than "
            "writing placeholder values."
        )

    rows = []
    for n, (_, r) in enumerate(df.iterrows(), 1):
        title, lead = str(r["title"] or ""), r["lead"]
        res = score_article(clf, title, lead)
        rows.append({
            "article_id": r["article_id"],
            "food_insecurity_score": res["food_insecurity_score"],
            "top_hypothesis": res["top_hypothesis"],
            "score_input": "title+lead" if lead else "title_only",
            "lead_chars": len(lead),
        })
        if n % 25 == 0 or n == len(df):
            log.info("scored %d/%d", n, len(df))

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    log.info("saved %d scores -> %s", len(out), OUT)

    log.info("score_input mix: %s", dict(out["score_input"].value_counts()))
    print("\n===== SCORE DISTRIBUTION BY INPUT =====")
    print(out.groupby("score_input")["food_insecurity_score"].describe().round(4).to_string())
    print(f"\nexactly 0.5000 (fallback smell): {int((out['food_insecurity_score'] == 0.5).sum())}")
    print(f"above 0.5: {int((out['food_insecurity_score'] > 0.5).sum())} of {len(out)}")
    print("\ntop hypothesis distribution:")
    print(out["top_hypothesis"].value_counts().to_string())


if __name__ == "__main__":
    main()
