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

RAW_PATH       = Path("data/raw/corpus_raw.parquet")
EXPANDED_PATH  = Path("data/raw/corpus_raw_expanded.parquet")
GEOCODED_PATH  = Path("data/processed/corpus_geocoded.parquet")
CHECKPOINT_DIR = Path("data/raw/checkpoints")

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

import re as _re
from collections import defaultdict as _defaultdict

# Stop words removed before Jaccard comparison.
# Includes common action verbs that outlets swap when republishing the same
# PNA wire story (distributes → gives → provides → releases).
_TITLE_STOP: frozenset[str] = frozenset({
    # Articles / prepositions / conjunctions
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "by", "with", "as", "its", "it", "this", "that", "their", "from",
    "after", "before", "over", "more", "new", "up", "out", "into", "amid",
    "due", "per", "vs",
    # Common auxiliary verbs
    "is", "are", "was", "were", "be", "been", "has", "have", "had",
    # Action verbs swapped across reposts of the same story
    "says", "say", "said",
    "report", "reports", "reported",
    "hit", "hits",
    "see", "sees", "seen",
    "get", "gets", "got",
    "give", "gives", "gave",
    "distribute", "distributes", "distributed",
    "provide", "provides", "provided",
    "issue", "issues", "issued",
    "release", "releases", "released",
    # Philippine boilerplate
    "ph", "philippines", "philippine",
})

# Jaccard threshold: ≥ 0.75 word overlap → same story.
# High enough to avoid false positives (Batangas vs Cavite articles
# share ~71% words and are correctly kept separate), low enough to
# catch paraphrased reposts of the same PNA wire story (~83%+ overlap).
_JACCARD_THRESHOLD = 0.75


def _title_wordset(title: str) -> frozenset[str]:
    """Return content-word set for a title, excluding stop words."""
    t = _re.sub(r"[^a-z0-9\s]", " ", title.lower())
    return frozenset(w for w in t.split() if w not in _TITLE_STOP and len(w) > 2)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dedup(records: list[dict]) -> list[dict]:
    """
    Two-pass deduplication:

    Pass 1 — Exact (URL / article_id hash)
        Catches the same URL returned by multiple fetchers or queries.

    Pass 2 — Content (Jaccard similarity on title word sets)
        Catches the same news story republished across outlets with slightly
        different wording (e.g. PNA wire stories on Rappler, MB, Inquirer).
        Uses an inverted index for O(n) candidate lookup instead of O(n²)
        brute force — fast even for 35,000+ articles.

        Threshold 0.75: two titles must share ≥ 75% of their content words
        to be considered duplicates. This prevents different-province
        articles with similar templates from being incorrectly merged.
    """
    seen_ids: set[str] = set()
    seen_wordsets: list[frozenset] = []
    # Inverted index: word → indices into seen_wordsets
    word_index: dict[str, list[int]] = _defaultdict(list)
    out: list[dict] = []

    for r in records:
        # ── Pass 1: exact ID / URL dedup ─────────────────────────────────
        key = r.get("article_id") or hashlib.md5(r.get("link", "").encode()).hexdigest()
        if key in seen_ids:
            continue
        seen_ids.add(key)

        # ── Pass 2: Jaccard title dedup ───────────────────────────────────
        words = _title_wordset(r.get("title") or "")
        is_dup = False

        if len(words) >= 4:  # only dedup titles with ≥ 4 content words
            # Find candidate indices via inverted index (O(|words|) lookup)
            candidate_idxs: set[int] = set()
            for w in words:
                candidate_idxs.update(word_index.get(w, []))

            for idx in candidate_idxs:
                if _jaccard(words, seen_wordsets[idx]) >= _JACCARD_THRESHOLD:
                    is_dup = True
                    break

        if is_dup:
            continue

        # Register new article in index
        idx = len(seen_wordsets)
        for w in words:
            word_index[w].append(idx)
        seen_wordsets.append(words)
        out.append(r)

    return out


def _save_checkpoint(source: str, records: list[dict]) -> None:
    """Save a fetcher's results immediately so a Ctrl+C cannot lose them."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{source}.parquet"
    pd.DataFrame(records).to_parquet(path, index=False)
    logger.info("Checkpoint saved: %s (%d articles) → %s", source, len(records), path)


def _load_checkpoint(source: str) -> list[dict] | None:
    """Return saved records for a source, or None if no checkpoint exists."""
    path = CHECKPOINT_DIR / f"{source}.parquet"
    if path.exists():
        records = pd.read_parquet(path).to_dict(orient="records")
        logger.info("Resuming from checkpoint: %s (%d articles)", source, len(records))
        return records
    return None


def _clear_checkpoints() -> None:
    """Remove all checkpoint files after a successful full run."""
    if CHECKPOINT_DIR.exists():
        for f in CHECKPOINT_DIR.glob("*.parquet"):
            f.unlink()
        logger.info("Checkpoints cleared.")


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
        choices=["gnews_rss", "rss", "gdelt", "newsdata", "google_cse"],
        default=["gnews_rss", "rss", "gdelt", "newsdata", "google_cse"],
        help="Which fetchers to run (default: all)",
    )
    parser.add_argument("--no-classify", action="store_true",
                        help="Skip Anthropic Claude food insecurity classification")
    parser.add_argument("--classify-only", action="store_true",
                        help="Only run Claude classification on existing corpus")
    parser.add_argument("--resume", action="store_true",
                        help="Skip fetchers that already have a saved checkpoint")
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
            cached = _load_checkpoint("gnews_rss") if args.resume else None
            if cached is not None:
                all_new += cached
            else:
                logger.info("[1/5] Google News RSS fetcher...")
                from app.ml.corpus.gnews_rss_fetcher import fetch_gnews_rss_articles
                gnews = fetch_gnews_rss_articles(start_date, end_date)
                for r in gnews:
                    r.setdefault("fetcher_source", "gnews_rss")
                _save_checkpoint("gnews_rss", gnews)
                logger.info("  Google News RSS: %d articles", len(gnews))
                all_new += gnews

        # ── 2. Direct RSS feeds ───────────────────────────────────────────
        # NOTE: RSS feeds are a live window (last 10-50 articles only).
        # The fetcher automatically extends the upper date bound to today so
        # current articles are captured even when end_date is in the past.
        # For the 2020-2025 historical window, gnews_rss and gdelt are the
        # primary sources; RSS contributes near-real-time top-up only.
        if "rss" in args.sources:
            cached = _load_checkpoint("rss") if args.resume else None
            if cached is not None:
                all_new += cached
            else:
                logger.info("[2/5] Direct RSS feeds (near-real-time top-up)...")
                from app.ml.corpus.rss_fetcher import fetch_rss_articles, FEED_URLS
                rss = fetch_rss_articles(FEED_URLS, start_date, end_date)
                for r in rss:
                    r.setdefault("article_id", hashlib.md5(r["link"].encode()).hexdigest())
                    r.setdefault("fetcher_source", "rss")
                _save_checkpoint("rss", rss)
                logger.info("  RSS: %d articles", len(rss))
                all_new += rss

        # ── 3. GDELT Project ──────────────────────────────────────────────
        if "gdelt" in args.sources:
            cached = _load_checkpoint("gdelt") if args.resume else None
            if cached is not None:
                all_new += cached
            else:
                logger.info("[3/5] GDELT Project fetcher...")
                from app.ml.corpus.gdelt_fetcher import fetch_gdelt_articles
                gdelt = fetch_gdelt_articles(start_date, end_date)
                for r in gdelt:
                    r.setdefault("fetcher_source", "gdelt")
                _save_checkpoint("gdelt", gdelt)
                logger.info("  GDELT: %d articles", len(gdelt))
                all_new += gdelt

        # ── 4. NewsData.io ────────────────────────────────────────────────
        if "newsdata" in args.sources:
            cached = _load_checkpoint("newsdata") if args.resume else None
            if cached is not None:
                all_new += cached
            else:
                logger.info("[4/5] NewsData.io fetcher...")
                from app.ml.corpus.newsdata_fetcher import fetch_newsdata_articles
                newsdata = fetch_newsdata_articles(start_date, end_date)
                for r in newsdata:
                    r.setdefault("fetcher_source", "newsdata")
                _save_checkpoint("newsdata", newsdata)
                logger.info("  NewsData.io: %d articles", len(newsdata))
                all_new += newsdata

        # ── 5. Google Custom Search ───────────────────────────────────────
        if "google_cse" in args.sources:
            cached = _load_checkpoint("google_cse") if args.resume else None
            if cached is not None:
                all_new += cached
            else:
                logger.info("[5/5] Google Custom Search fetcher...")
                from app.ml.corpus.google_cse_fetcher import fetch_google_cse_articles
                gcse = fetch_google_cse_articles(start_date, end_date)
                for r in gcse:
                    r.setdefault("fetcher_source", "google_cse")
                _save_checkpoint("google_cse", gcse)
                logger.info("  Google CSE: %d articles", len(gcse))
                all_new += gcse
       
        
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
    _clear_checkpoints()

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
