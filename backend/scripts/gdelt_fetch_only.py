"""
scripts/gdelt_fetch_only.py
----------------------------
Fetch-ONLY GDELT REST collection: writes the runner-format checkpoint
(data/raw/checkpoints/gdelt.parquet) and never touches the corpus, scores,
or model — safe to run in parallel with a scoring run.

Merge later with:
  python run_expanded_corpus_collection.py --resume --sources gdelt ...

Usage:
  venv\\Scripts\\python scripts\\gdelt_fetch_only.py --start 2020-01-01 --end 2025-12-31
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
logger = logging.getLogger("gdelt_fetch_only")

CHECKPOINT = Path("data/raw/checkpoints/gdelt.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch-only GDELT REST")
    parser.add_argument("--start", type=int, default=2020, help="first year")
    parser.add_argument("--end", type=int, default=2025, help="last year")
    parser.add_argument("--window-days", type=int, default=30)
    args = parser.parse_args()

    from app.ml.corpus.gdelt_fetcher import fetch_gdelt_articles

    # Per-year checkpoints (gdelt_2020.parquet ...) so an interruption loses
    # at most one year's fetch, never the whole multi-hour run. The combined
    # runner-format checkpoint is rebuilt from year files at the end.
    all_records: list[dict] = []
    for year in range(args.start, args.end + 1):
        year_ckpt = CHECKPOINT.parent / f"gdelt_{year}.parquet"
        if year_ckpt.exists():
            recs = pd.read_parquet(year_ckpt).to_dict("records")
            logger.info("%d already checkpointed (%d articles) — skipping",
                        year, len(recs))
            all_records += recs
            continue
        logger.info("=== Fetching GDELT REST for %d ===", year)
        recs = fetch_gdelt_articles(
            f"{year}-01-01", f"{year}-12-31",
            prefer_bigquery=False,      # BigQuery already harvested separately
            window_days=args.window_days,
        )
        for r in recs:
            r.setdefault("fetcher_source", "gdelt")
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(recs).to_parquet(year_ckpt, index=False)
        logger.info("Year %d checkpoint saved: %d articles", year, len(recs))
        all_records += recs

    pd.DataFrame(all_records).to_parquet(CHECKPOINT, index=False)
    logger.info("GDELT combined checkpoint saved: %d articles -> %s",
                len(all_records), CHECKPOINT)


if __name__ == "__main__":
    main()
