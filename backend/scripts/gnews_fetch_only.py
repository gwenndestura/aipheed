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
    args = parser.parse_args()

    from app.ml.corpus.gnews_rss_fetcher import fetch_gnews_rss_articles

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for year in range(args.start, args.end + 1):
        ckpt = CHECKPOINT_DIR / f"gnews_rss_{year}.parquet"
        if ckpt.exists():
            logger.info("%d already checkpointed (%s) — skipping", year, ckpt.name)
            continue
        logger.info("=== Fetching Google News RSS for %d ===", year)
        records = fetch_gnews_rss_articles(
            f"{year}-01-01", f"{year}-12-31", window_months=1
        )
        for r in records:
            r.setdefault("fetcher_source", "gnews_rss")
        pd.DataFrame(records).to_parquet(ckpt, index=False)
        logger.info("Year %d checkpoint saved: %d articles -> %s",
                    year, len(records), ckpt.name)

    logger.info("All requested years fetched.")


if __name__ == "__main__":
    main()
