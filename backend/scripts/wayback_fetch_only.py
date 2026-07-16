"""
scripts/wayback_fetch_only.py
------------------------------
Fetch-ONLY Wayback Machine collection (historical snapshots of the credible
Philippine news domains). Writes a runner-independent checkpoint and never
touches the corpus — safe alongside other fetch/scoring jobs.

Usage:
  venv\\Scripts\\python scripts\\wayback_fetch_only.py --start 2020-01-01 --end 2025-12-31
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
logger = logging.getLogger("wayback_fetch_only")

CHECKPOINT = Path("data/raw/checkpoints/wayback.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch-only Wayback CDX")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    if CHECKPOINT.exists():
        logger.info("Checkpoint already exists (%s) — delete to refetch.", CHECKPOINT)
        return

    from app.ml.corpus.wayback_fetcher import fetch_wayback_articles

    records = fetch_wayback_articles(args.start, args.end)
    for r in records:
        r.setdefault("fetcher_source", "wayback")

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(CHECKPOINT, index=False)
    logger.info("Wayback checkpoint saved: %d articles -> %s", len(records), CHECKPOINT)


if __name__ == "__main__":
    main()
