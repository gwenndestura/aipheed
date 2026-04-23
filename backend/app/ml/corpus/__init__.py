"""
app/ml/corpus/__init__.py
--------------------------
Corpus sub-package init.

Provides a convenience function run_corpus_ingestion() that
calls all three fetchers (RSS, GNews, Wayback) and the PSA fetcher,
merges results, and saves data/raw/corpus_raw.parquet.

The parquet file is GITIGNORED — never commit raw corpus data.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_CORPUS_PATH = Path("data/raw/corpus_raw.parquet")

# Common schema fields shared across all news fetchers
CORPUS_COLUMNS = [
    "title",
    "link",
    "published",
    "summary",
    "source_domain",
]


def run_corpus_ingestion(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    gnews_api_key: str | None = None,
    include_wayback: bool = True,
) -> pd.DataFrame:
    """
    Run full corpus ingestion pipeline:
        1. RSS feeds from CREDIBLE_DOMAINS
        2. GNews API (if api_key provided)
        3. Wayback CDX historical gap-fill (2020–2022)

    Merges results, deduplicates by 'link', and saves to
    data/raw/corpus_raw.parquet (gitignored).

    Returns the merged DataFrame.
    """
    from app.ml.corpus.rss_fetcher import fetch_rss_articles, FEED_URLS
    from app.ml.corpus.gnews_fetcher import fetch_gnews_articles
    from app.ml.corpus.wayback_fetcher import fetch_wayback_articles

    all_records: list[dict] = []

    # --- RSS ---
    logger.info("Starting RSS ingestion…")
    rss_records = fetch_rss_articles(FEED_URLS, start_date, end_date)
    all_records.extend(rss_records)
    logger.info("RSS: %d articles", len(rss_records))

    # --- GNews ---
    api_key = gnews_api_key or os.getenv("GNEWS_API_KEY", "")
    if api_key:
        logger.info("Starting GNews ingestion…")
        gnews_records = fetch_gnews_articles(start_date, end_date, api_key)
        all_records.extend(gnews_records)
        logger.info("GNews: %d articles", len(gnews_records))
    else:
        logger.warning("GNEWS_API_KEY not set — skipping GNews ingestion")

    # --- Wayback ---
    if include_wayback:
        # Wayback is used for 2020–2022 historical gap-fill only
        wayback_end = min(end_date, "2022-12-31")
        logger.info("Starting Wayback ingestion (%s – %s)…", start_date, wayback_end)
        wb_records = fetch_wayback_articles(start_date, wayback_end)
        all_records.extend(wb_records)
        logger.info("Wayback: %d articles", len(wb_records))

    # --- Merge and deduplicate ---
    df = pd.DataFrame(all_records, columns=CORPUS_COLUMNS + [
        col for col in (all_records[0].keys() if all_records else [])
        if col not in CORPUS_COLUMNS
    ])

    # Enforce canonical column order
    for col in CORPUS_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[CORPUS_COLUMNS + [c for c in df.columns if c not in CORPUS_COLUMNS]]

    before = len(df)
    df = df.drop_duplicates(subset=["link"]).reset_index(drop=True)
    logger.info("Deduplicated: %d → %d unique articles", before, len(df))

    # --- Save ---
    RAW_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_CORPUS_PATH, index=False)
    logger.info("Saved corpus_raw.parquet: %d rows → %s", len(df), RAW_CORPUS_PATH)

    return df