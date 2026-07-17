"""
scripts/gnews_fetch_only.py
----------------------------
Fetch-ONLY Google News RSS collection: writes per-year checkpoints in the
runner's format and never touches the corpus, scores, or model — so it can
run in parallel with a scoring run of run_expanded_corpus_collection.py.

Merge later with:
  python run_expanded_corpus_collection.py --resume --sources gnews_rss ...

Usage:
  venv\\Scripts\\python scripts\\gnews_fetch_only.py --start 2020 --end 2025

Rate is controlled by AIPHEED_GNEWS_DELAY / AIPHEED_GNEWS_WORKERS env vars
(defaults: 2.5s delay x 2 workers ~= 0.7 req/s, Google-tolerated).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gnews_fetch_only")

CHECKPOINT_DIR = Path("data/raw/checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch-only Google News RSS")
    parser.add_argument("--start", type=int, default=2020, help="first year")
    parser.add_argument("--end", type=int, default=2025, help="last year")
    parser.add_argument(
        "--profile", choices=["low", "full"], default="low",
        help="low: ~300 topical queries, quarterly windows, 1 worker @3s, "
             "3-min rest between windows (~7.2k requests total). "
             "full: every query incl. per-LGU, monthly windows.",
    )
    args = parser.parse_args()

    if args.profile == "low":
        # Must be set BEFORE the fetcher module is imported.
        os.environ.setdefault("AIPHEED_GNEWS_WORKERS", "1")
        os.environ.setdefault("AIPHEED_GNEWS_DELAY", "3.0")
        os.environ.setdefault("AIPHEED_GNEWS_WINDOW_REST", "180")

    from app.ml.corpus.gnews_rss_fetcher import (
        GNEWS_RSS_QUERIES,
        _DOMAIN_TARGETED,
        _LGU_QUERIES,
        fetch_gnews_rss_articles,
    )

    if args.profile == "low":
        drop = set(_LGU_QUERIES) | set(_DOMAIN_TARGETED)
        queries = [q for q in GNEWS_RSS_QUERIES if q not in drop]
        window_months = 3
    else:
        queries = None
        window_months = 1

    logger.info("Profile: %s (%s queries, %d-month windows)",
                args.profile, len(queries) if queries else "all", window_months)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for year in range(args.start, args.end + 1):
        ckpt = CHECKPOINT_DIR / f"gnews_rss_{year}.parquet"
        if ckpt.exists():
            logger.info("%d already checkpointed (%s) — skipping", year, ckpt.name)
            continue
        logger.info("=== Fetching Google News RSS for %d ===", year)
        records = fetch_gnews_rss_articles(
            f"{year}-01-01", f"{year}-12-31",
            window_months=window_months,
            queries=queries,
        )
        for r in records:
            r.setdefault("fetcher_source", "gnews_rss")
        pd.DataFrame(records).to_parquet(ckpt, index=False)
        logger.info("Year %d checkpoint saved: %d articles -> %s",
                    year, len(records), ckpt.name)

    logger.info("All requested years fetched.")


if __name__ == "__main__":
    main()
