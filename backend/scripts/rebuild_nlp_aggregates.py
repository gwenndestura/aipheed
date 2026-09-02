"""
scripts/rebuild_nlp_aggregates.py
---------------------------------
Rebuild the SCORE-INDEPENDENT half of the NLP chain from the audited dataset:
bias_weights (article counts) and trigger_proportions (keyword categories).

Neither depends on food_insecurity_score, so both are correct regardless of how
the FSSI scoring question is settled. fssi_quarterly is left alone here.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.corpus.bias_weighter import compute_bias_weights  # noqa: E402
from app.ml.nlp.trigger_classifier import (  # noqa: E402
    classify_triggers_df,
    compute_trigger_proportions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rebuild_nlp_agg")

DATASET = Path("data/processed/calabarzon_food_insecurity_dataset.parquet")
CENSUS = Path("data/processed/lgu_census.parquet")
WINDOW_START, WINDOW_END = "2020-Q1", "2025-Q4"
PROVINCE_SCOPES = ("city_municipality", "province")


def load_articles() -> pd.DataFrame:
    df = pd.read_parquet(DATASET)
    log.info("audited dataset: %d rows", len(df))

    prov = (
        pd.read_parquet(CENSUS)[["province_code", "province_name"]]
        .drop_duplicates()
        .rename(columns={"province_name": "province"})
    )
    df = df.merge(prov, on="province", how="left")

    before = len(df)
    df = df[df["geographic_scope"].isin(PROVINCE_SCOPES)].copy()
    log.info("province-level scope: %d rows (dropped %d region-wide)", len(df), before - len(df))

    if int(df["province_code"].isna().sum()):
        raise RuntimeError("some rows have a province that did not map to a PSGC code")

    df["published"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["summary"] = df["content_lead"].fillna("")
    d = pd.to_datetime(df["publication_date"], errors="coerce")
    df["quarter"] = (
        d.dt.year.astype("Int64").astype(str) + "-Q" + d.dt.quarter.astype("Int64").astype(str)
    )

    before = len(df)
    df = df[df["quarter"].between(WINDOW_START, WINDOW_END)].copy()
    log.info("model window: %d rows (dropped %d outside)", len(df), before - len(df))
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
    for c in [c for c in triggers.columns if c.startswith("trigger_")]:
        log.info("  %-24s mean %.4f  nonzero cells %d", c, triggers[c].mean(), int((triggers[c] > 0).sum()))


if __name__ == "__main__":
    main()
