"""
scripts/rebuild_fssi_chain.py
-----------------------------
Rebuild the NLP half of the feature matrix from the AUDITED news dataset.

Before this script, fssi_quarterly / trigger_proportions / bias_weights were
built (2026-08-12) from corpus_calabarzon.parquet — a 560-article pre-audit pool
whose geography was assigned lexically. The audit replaced that pool with 371
hand-read articles, so the FSSI features in features_fused.parquet no longer
matched the dataset they claim to summarise.

Chain: audited dataset (+ NLI scores) -> bias_weights -> triggers -> FSSI
       -> features_fused

Province-level scope only. `geographic_scope == 'region'` rows carry no single
province (README: filter to city_municipality|province for province-level work),
so they cannot enter a province-quarter aggregate and are excluded here.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.corpus.bias_weighter import compute_bias_weights  # noqa: E402
from app.ml.features.feature_matrix import (  # noqa: E402
    MODEL_QUARTERS,
    build_feature_matrix,
)
from app.ml.features.fssi_builder import compute_fssi  # noqa: E402
from app.ml.nlp.trigger_classifier import (  # noqa: E402
    classify_triggers_df,
    compute_trigger_proportions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rebuild_fssi")

DATASET = Path("data/processed/calabarzon_food_insecurity_dataset.parquet")
SCORES = Path("data/processed/_audited_nli_scores.parquet")
CENSUS = Path("data/processed/lgu_census.parquet")

# Tracks the feature matrix rather than restating it. Was a hardcoded
# "2020-Q1".."2025-Q4", which discarded every 2026 article in the corpus.
WINDOW_START, WINDOW_END = MODEL_QUARTERS[0], MODEL_QUARTERS[-1]
PROVINCE_SCOPES = ("city_municipality", "province")


def _to_quarter(ts: pd.Series) -> pd.Series:
    d = pd.to_datetime(ts, errors="coerce")
    return d.dt.year.astype("Int64").astype(str) + "-Q" + d.dt.quarter.astype("Int64").astype(str)


def load_articles() -> pd.DataFrame:
    """Audited dataset shaped into the column contract the builders expect."""
    df = pd.read_parquet(DATASET)
    log.info("audited dataset: %d rows", len(df))

    scores = pd.read_parquet(SCORES)[["article_id", "food_insecurity_score", "score_input"]]
    df = df.merge(scores.drop_duplicates("article_id"), on="article_id", how="left")
    missing = int(df["food_insecurity_score"].isna().sum())
    if missing:
        raise RuntimeError(
            f"{missing} audited rows still have no NLI score. "
            "Run scripts/rescore_all_articles.py first."
        )
    log.info("score input mix: %s", dict(df["score_input"].value_counts()))

    # province name -> PSGC code
    prov = (
        pd.read_parquet(CENSUS)[["province_code", "province_name"]]
        .drop_duplicates()
        .rename(columns={"province_name": "province"})
    )
    df = df.merge(prov, on="province", how="left")

    # Province-level scope only
    before = len(df)
    df = df[df["geographic_scope"].isin(PROVINCE_SCOPES)].copy()
    log.info("province-level scope: %d rows (dropped %d region-wide)", len(df), before - len(df))

    unmapped = int(df["province_code"].isna().sum())
    if unmapped:
        raise RuntimeError(f"{unmapped} rows have a province that did not map to a PSGC code")

    # Column contract: bias_weighter needs `published`; trigger_classifier needs
    # `title` + `summary`; fssi_builder needs `quarter` + `food_insecurity_score`.
    df["published"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["summary"] = df["content_lead"].fillna("")
    df["quarter"] = _to_quarter(df["publication_date"])

    before = len(df)
    df = df[df["quarter"].between(WINDOW_START, WINDOW_END)].copy()
    log.info(
        "model window %s..%s: %d rows (dropped %d outside)",
        WINDOW_START, WINDOW_END, len(df), before - len(df),
    )
    return df


def main() -> None:
    articles = load_articles()

    log.info("--- bias weights ---")
    weights = compute_bias_weights(articles)
    log.info("bias_weights: %d province-quarters", len(weights))

    log.info("--- trigger proportions ---")
    tagged = classify_triggers_df(articles)
    triggers = compute_trigger_proportions(tagged)
    log.info("trigger_proportions: %d province-quarters", len(triggers))

    log.info("--- FSSI ---")
    fssi = compute_fssi(articles, weights)
    log.info("fssi_quarterly: %d province-quarters", len(fssi))

    log.info("--- feature matrix ---")
    feats = build_feature_matrix()
    log.info("features_fused: %s", feats.shape)


if __name__ == "__main__":
    main()
