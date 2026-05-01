"""
run_expanded_corpus_collection.py
-----------------------------------
Master script — collect CALABARZON food insecurity news from ALL sources.

Sources
-------
  1. Google News RSS      (gnews_rss_fetcher)     — existing, no API key
  2. RSS direct feeds     (rss_fetcher)            — existing, no API key
  3. NewsData.io          (newsdata_fetcher)       — NEWSDATA_API_KEY
  4. GDELT Project        (gdelt_fetcher)          — free, no API key
  5. Google Custom Search (google_cse_fetcher)     — GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX
  6. Bing News Search     (bing_news_fetcher)      — BING_NEWS_API_KEY

Anthropic Classifier
--------------------
  After collection, Claude claude-haiku-4-5 scores each article 0–4 for food
  insecurity relevance. Only articles scoring >= 2 are kept.

Coverage guarantee
------------------
  Queries target ALL 147 CALABARZON LGUs (municipalities + cities) individually
  across all 5 provinces — ensuring no municipality is silently uncovered.

Usage
-----
  # Full collection (all sources, 2020–2025)
  cd backend
  python run_expanded_corpus_collection.py

  # Specific sources only
  python run_expanded_corpus_collection.py --sources gdelt newsdata

  # Skip Anthropic classification (keep all fetched articles)
  python run_expanded_corpus_collection.py --no-classify

  # Date range
  python run_expanded_corpus_collection.py --start 2024-01-01 --end 2025-12-31

Output
------
  data/raw/corpus_raw.parquet          — appended with new articles (deduped)
  data/raw/corpus_raw_expanded.parquet — full expanded corpus this run
  data/processed/corpus_geocoded.parquet — re-geocoded after expansion

Required .env keys (add to backend/.env)
-----------------------------------------
  NEWSDATA_API_KEY=pub_xxxx
  GOOGLE_CSE_API_KEY=AIzaSy...
  GOOGLE_CSE_CX=xxxxxxxxxx
  BING_NEWS_API_KEY=xxxxxxxx
  ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Adjust sys.path so script runs from backend/ directory
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("corpus_collection")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_PATH = Path("data/raw/corpus_raw.parquet")
EXPANDED_PATH = Path("data/raw/corpus_raw_expanded.parquet")
GEOCODED_PATH = Path("data/processed/corpus_geocoded.parquet")

# ---------------------------------------------------------------------------
# LGU coverage report
# ---------------------------------------------------------------------------

ALL_LGUS: dict[str, list[str]] = {
    "Batangas": [
        "Batangas City", "Lipa City", "Tanauan City", "Santo Tomas",
        "Agoncillo", "Alitagtag", "Balayan", "Balete", "Bauan", "Calaca",
        "Calatagan", "Cuenca", "Ibaan", "Laurel", "Lemery", "Lian", "Lobo",
        "Mabini", "Malvar", "Mataas na Kahoy", "Nasugbu", "Padre Garcia",
        "Rosario", "San Jose", "San Juan", "San Luis", "San Nicolas",
        "San Pascual", "Santa Teresita", "Taysan", "Tingloy", "Tuy",
    ],
    "Cavite": [
        "Bacoor", "Cavite City", "Dasmariñas", "General Trias", "Imus",
        "Tagaytay", "Trece Martires", "Alfonso", "Amadeo", "Carmona",
        "General Emilio Aguinaldo", "General Mariano Alvarez", "Indang",
        "Kawit", "Magallanes", "Maragondon", "Mendez", "Naic",
        "Noveleta", "Rosario Cavite", "Silang", "Tanza", "Ternate",
    ],
    "Laguna": [
        "Biñan", "Cabuyao", "Calamba", "San Pablo", "San Pedro", "Santa Rosa",
        "Alaminos", "Bay", "Calauan", "Cavinti", "Famy", "Kalayaan", "Liliw",
        "Los Baños", "Luisiana", "Lumban", "Mabitac", "Magdalena", "Majayjay",
        "Nagcarlan", "Paete", "Pagsanjan", "Pakil", "Pangil", "Pila",
        "Rizal Laguna", "Santa Cruz", "Santa Maria Laguna", "Siniloan", "Victoria",
    ],
    "Quezon": [
        "Candelaria", "Lucena", "Tayabas", "Agdangan", "Alabat", "Atimonan",
        "Buenavista", "Burdeos", "Calauag", "Catanauan", "Dolores",
        "General Luna", "General Nakar", "Guinayangan", "Gumaca", "Infanta",
        "Jomalig", "Lopez", "Lucban", "Macalelon", "Mulanay", "Padre Burgos",
        "Panukulan", "Patnanungan", "Perez", "Pitogo", "Plaridel", "Polillo",
        "Real", "Sampaloc", "San Andres", "San Antonio Quezon", "San Francisco",
        "San Narciso", "Sariaya", "Tagkawayan", "Tiaong", "Unisan",
    ],
    "Rizal": [
        "Antipolo", "Angono", "Baras", "Binangonan", "Cainta", "Cardona",
        "Jala-Jala", "Morong", "Pililla", "Rodriguez", "San Mateo", "Tanay",
        "Taytay", "Teresa",
    ],
}


def _lgu_coverage_report(df: pd.DataFrame) -> None:
    """Print which LGUs have at least 1 article vs. 0."""
    titles_lower = df["title"].str.lower().fillna("")
    logger.info("\n%s\nLGU COVERAGE REPORT\n%s", "="*60, "="*60)
    total_covered = 0
    total_missing = 0
    missing_list: list[str] = []
    for province, lgus in ALL_LGUS.items():
        covered = []
        missing = []
        for lgu in lgus:
            if titles_lower.str.contains(lgu.lower(), na=False).any():
                covered.append(lgu)
            else:
                missing.append(lgu)
        total_covered += len(covered)
        total_missing += len(missing)
        missing_list += [f"{province}/{m}" for m in missing]
        logger.info(
            "%s: %d/%d LGUs covered. Missing: %s",
            province, len(covered), len(lgus),
            ", ".join(missing) if missing else "None",
        )
    logger.info(
        "\nTotal: %d/%d LGUs covered (%.0f%%)",
        total_covered, total_covered + total_missing,
        total_covered / (total_covered + total_missing) * 100,
    )
    if missing_list:
        logger.warning("LGUs with zero article coverage: %s", "; ".join(missing_list))


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dedup(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        key = r.get("article_id") or hashlib.md5(r.get("link", "").encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _dedup_against_existing(new_records: list[dict], existing_df: pd.DataFrame) -> list[dict]:
    if existing_df.empty:
        return new_records
    existing_ids: set[str] = set(existing_df["article_id"].dropna().tolist())
    existing_links: set[str] = set(existing_df["link"].dropna().tolist())
    out = []
    for r in new_records:
        if r.get("article_id") in existing_ids:
            continue
        if r.get("link") in existing_links:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Expand CALABARZON corpus")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument(
        "--sources", nargs="+",
        choices=["gnews_rss", "rss", "gdelt", "newsdata", "google_cse", "bing"],
        default=["gnews_rss", "rss", "gdelt", "newsdata", "google_cse", "bing"],
        help="Which fetchers to run (default: all)",
    )
    parser.add_argument("--no-classify", action="store_true",
                        help="Skip Anthropic Claude food insecurity classification")
    parser.add_argument("--classify-only", action="store_true",
                        help="Only run Claude classification on existing corpus")
    args = parser.parse_args()

    start_date = args.start
    end_date = args.end

    logger.info("="*60)
    logger.info("CALABARZON CORPUS EXPANSION: %s to %s", start_date, end_date)
    logger.info("Sources: %s", ", ".join(args.sources))
    logger.info("="*60)

    # ── Load existing corpus ──────────────────────────────────────────────
    existing_df = pd.DataFrame()
    if RAW_PATH.exists():
        existing_df = pd.read_parquet(RAW_PATH)
        logger.info("Existing corpus: %d articles", len(existing_df))

    all_new: list[dict] = []

    if not args.classify_only:
        # ── 1. Google News RSS ────────────────────────────────────────────
        if "gnews_rss" in args.sources:
            logger.info("[1/6] Google News RSS fetcher...")
            from app.ml.corpus.gnews_rss_fetcher import fetch_gnews_rss_articles
            gnews = fetch_gnews_rss_articles(start_date, end_date)
            logger.info("  Google News RSS: %d articles", len(gnews))
            all_new += gnews

        # ── 2. Direct RSS feeds ───────────────────────────────────────────
        if "rss" in args.sources:
            logger.info("[2/6] Direct RSS feeds...")
            from app.ml.corpus.rss_fetcher import fetch_rss_articles, FEED_URLS
            rss = fetch_rss_articles(FEED_URLS, start_date, end_date)
            for r in rss:
                r.setdefault("article_id", hashlib.md5(r["link"].encode()).hexdigest())
                r.setdefault("fetcher_source", "rss")
            logger.info("  RSS: %d articles", len(rss))
            all_new += rss

        # ── 3. GDELT Project ──────────────────────────────────────────────
        if "gdelt" in args.sources:
            logger.info("[3/6] GDELT Project fetcher...")
            from app.ml.corpus.gdelt_fetcher import fetch_gdelt_articles
            gdelt = fetch_gdelt_articles(start_date, end_date)
            logger.info("  GDELT: %d articles", len(gdelt))
            all_new += gdelt

        # ── 4. NewsData.io ────────────────────────────────────────────────
        if "newsdata" in args.sources:
            logger.info("[4/6] NewsData.io fetcher...")
            from app.ml.corpus.newsdata_fetcher import fetch_newsdata_articles
            newsdata = fetch_newsdata_articles(start_date, end_date)
            logger.info("  NewsData.io: %d articles", len(newsdata))
            all_new += newsdata

        # ── 5. Google Custom Search ───────────────────────────────────────
        if "google_cse" in args.sources:
            logger.info("[5/6] Google Custom Search fetcher...")
            from app.ml.corpus.google_cse_fetcher import fetch_google_cse_articles
            gcse = fetch_google_cse_articles(start_date, end_date)
            logger.info("  Google CSE: %d articles", len(gcse))
            all_new += gcse

        # ── 6. Bing News ──────────────────────────────────────────────────
        if "bing" in args.sources:
            logger.info("[6/6] Bing News Search fetcher...")
            from app.ml.corpus.bing_news_fetcher import fetch_bing_news_articles
            bing = fetch_bing_news_articles(start_date, end_date)
            logger.info("  Bing News: %d articles", len(bing))
            all_new += bing

        # ── Dedup within new batch ────────────────────────────────────────
        all_new = _dedup(all_new)
        logger.info("After dedup (new batch): %d articles", len(all_new))

        # ── Dedup against existing corpus ─────────────────────────────────
        all_new = _dedup_against_existing(all_new, existing_df)
        logger.info("After dedup (vs existing): %d truly new articles", len(all_new))

    else:
        logger.info("--classify-only: loading existing corpus for re-classification")
        all_new = existing_df.to_dict(orient="records") if not existing_df.empty else []

    if not all_new:
        logger.warning("No new articles collected — exiting.")
        return

    # ── Build new DataFrame ───────────────────────────────────────────────
    new_df = pd.DataFrame(all_new)
    for col in ["title", "link", "article_id", "published", "summary",
                "source_domain", "fetcher_source"]:
        if col not in new_df.columns:
            new_df[col] = None

    # ── Anthropic Claude classification ───────────────────────────────────
    if not args.no_classify:
        logger.info("Running Anthropic Claude food insecurity classification...")
        from app.ml.corpus.anthropic_classifier import classify_food_relevance
        new_df = classify_food_relevance(new_df)

        before = len(new_df)
        # Keep only relevant (score >= 2) articles
        new_df = new_df[new_df["claude_fi_relevant"] != "N"].copy()
        logger.info(
            "After Claude filter: %d/%d articles kept (%.1f%%)",
            len(new_df), before, len(new_df) / before * 100 if before else 0,
        )
    else:
        logger.info("Skipping Claude classification (--no-classify flag set)")

    # ── Geocode new articles ──────────────────────────────────────────────
    logger.info("Geocoding new articles to CALABARZON provinces...")
    from app.ml.corpus.geocoder import geocode_to_province
    new_df["province_code"] = new_df["title"].apply(
        lambda t: geocode_to_province(str(t or ""))
    )

    # Add quarter column
    def _to_quarter(pub: str) -> str:
        try:
            dt = pd.Timestamp(pub)
            return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        except Exception:
            return ""

    new_df["quarter"] = new_df["published"].apply(_to_quarter)

    # ── Merge with existing corpus ────────────────────────────────────────
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["article_id"], keep="first")
    combined = combined.drop_duplicates(subset=["link"], keep="first")

    logger.info(
        "Combined corpus: %d articles (%d existing + %d new)",
        len(combined), len(existing_df), len(new_df),
    )

    # ── LGU coverage report ───────────────────────────────────────────────
    calabarzon_combined = combined[combined["province_code"].notna()]
    _lgu_coverage_report(calabarzon_combined)

    # ── Save ──────────────────────────────────────────────────────────────
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(RAW_PATH, index=False)
    logger.info("Saved combined corpus → %s (%d articles)", RAW_PATH, len(combined))

    new_df.to_parquet(EXPANDED_PATH, index=False)
    logger.info("Saved new articles only → %s (%d articles)", EXPANDED_PATH, len(new_df))

    # ── Save geocoded ─────────────────────────────────────────────────────
    GEOCODED_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(GEOCODED_PATH, index=False)
    logger.info("Saved geocoded corpus → %s", GEOCODED_PATH)

    # ── Final stats ───────────────────────────────────────────────────────
    logger.info("\n%s\nFINAL CORPUS STATISTICS\n%s", "="*60, "="*60)
    logger.info("Total articles:       %d", len(combined))
    calabarzon_only = combined[combined["province_code"].notna()]
    logger.info("Calabarzon articles:  %d", len(calabarzon_only))
    if "province_code" in combined.columns:
        prov_map = {
            "PH040100000": "Cavite", "PH040200000": "Laguna",
            "PH040300000": "Rizal",  "PH040400000": "Quezon",
            "PH040500000": "Batangas",
        }
        for code, name in prov_map.items():
            count = (combined["province_code"] == code).sum()
            logger.info("  %-12s %d", name, count)
    if "fetcher_source" in combined.columns:
        logger.info("\nBy source:")
        for src, cnt in combined["fetcher_source"].value_counts().items():
            logger.info("  %-20s %d", src, cnt)


if __name__ == "__main__":
    main()
