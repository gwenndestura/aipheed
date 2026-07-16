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
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--window-days", type=int, default=30)
    args = parser.parse_args()

    if CHECKPOINT.exists():
        logger.info("Checkpoint already exists (%s) — delete it to refetch.", CHECKPOINT)
        return

    from app.ml.corpus.gdelt_fetcher import fetch_gdelt_articles

    records = fetch_gdelt_articles(
        args.start, args.end,
        prefer_bigquery=False,          # BigQuery already harvested separately
        window_days=args.window_days,
    )
    for r in records:
        r.setdefault("fetcher_source", "gdelt")

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(CHECKPOINT, index=False)
    logger.info("GDELT checkpoint saved: %d articles -> %s", len(records), CHECKPOINT)


if __name__ == "__main__":
    main()
