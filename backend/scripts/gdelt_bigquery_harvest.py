"""
scripts/gdelt_bigquery_harvest.py
----------------------------------
One-shot GDELT BigQuery harvest: every article GDELT geolocated to
CALABARZON (2020-01-01 .. 2026-07-01), pulled locally in one query.

Strategy
--------
The GKG V2Themes column alone costs ~840 GB to scan, blowing the free
1 TB/month budget. So we scan only V2Locations + DocumentIdentifier
(~400 GB), save EVERY CALABARZON-located URL locally, and do all topical
narrowing locally and for free:

  1. broad URL-slug pre-filter (wide net, EN+Tagalog stems) — volume
     control only, so NLI scoring stays computationally feasible
  2. credible Philippine domain check (same list as all other fetchers)
  3. title derived from the URL slug
  4. XLM-RoBERTa zero-shot NLI (downstream, in the main pipeline) makes
     the actual relevance decision from context — never keywords alone

Outputs
-------
  data/raw/checkpoints/gdelt_bq_urls_raw.parquet   — everything (audit)
  data/raw/checkpoints/gdelt_bigquery.parquet      — corpus-format records

Usage
-----
  set GOOGLE_APPLICATION_CREDENTIALS=path\\to\\key.json
  venv\\Scripts\\python scripts\\gdelt_bigquery_harvest.py
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.corpus.rss_fetcher import CREDIBLE_DOMAINS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gdelt_bq_harvest")

LOC_REGEX = (
    r"batangas|cavite|laguna|rizal|quezon|calabarzon|lucena|antipolo|calamba|"
    r"dasmari|bacoor|imus|tagaytay|lipa|tanauan|bi.an|santa rosa|san pablo|"
    r"cabuyao|san pedro|lucban|sariaya|taal"
)

CALABARZON_QUERY = f"""
SELECT
  DocumentIdentifier AS url,
  DATE(_PARTITIONTIME) AS seen_date,
  V2Locations AS locations
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP("2020-01-01")
  AND _PARTITIONTIME <  TIMESTAMP("2026-07-01")
  AND REGEXP_CONTAINS(LOWER(V2Locations), r"{LOC_REGEX}")
"""

# Nationwide sweep: credible PH news domains x food-topic URL slugs.
# Scans only DocumentIdentifier (~81 GB for 6.5y) — location tagging not
# required, so it catches national food-crisis coverage that mentions
# CALABARZON without a GDELT geo tag. Geo relevance is decided downstream
# (geocoder on enriched title+lead).
_NATL_DOMAINS = (
    r"inquirer\.net|philstar\.com|gmanetwork\.com|abs-cbn\.com|mb\.com\.ph|"
    r"pna\.gov\.ph|rappler\.com|manilatimes\.net|businessmirror\.com\.ph|"
    r"sunstar\.com\.ph|tribune\.net\.ph|journal\.com\.ph|remate\.ph|"
    r"abante\.com\.ph|bworldonline\.com|malaya\.com\.ph|manilastandard\.net"
)
_NATL_SLUGS = (
    r"food|rice|palay|bigas|hunger|gutom|famine|malnutri|feeding|ayuda|"
    r"relief|pantawid|4ps|kadiwa|nfa|subsid|farm|agri|magsasaka|harvest|"
    r"crop|fisher|isda|tilapia|asf|price|presyo|inflation|bilihin|palengke|"
    r"typhoon|bagyo|flood|baha|drought|el-nino|evacuat|bakwit|displac|"
    r"poverty|kahirapan|unemploy|shortage|kakulangan|onion|sibuyas|sugar|"
    r"asukal|fertilizer|vegetable|gulay|galunggong"
)

NATIONAL_QUERY = f"""
SELECT
  DocumentIdentifier AS url,
  DATE(_PARTITIONTIME) AS seen_date,
  '' AS locations
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP("2020-01-01")
  AND _PARTITIONTIME <  TIMESTAMP("2026-07-01")
  AND REGEXP_CONTAINS(DocumentIdentifier, r"(?i)({_NATL_DOMAINS})")
  AND REGEXP_CONTAINS(DocumentIdentifier, r"(?i)({_NATL_SLUGS})")
"""

MODES = {
    "calabarzon": {
        "query": CALABARZON_QUERY,
        "raw_out": Path("data/raw/gdelt_bq_urls_raw.parquet"),
        "corpus_out": Path("data/raw/gdelt_bigquery.parquet"),
        "slug_filter": True,   # location-first pool needs topical narrowing
    },
    "national": {
        "query": NATIONAL_QUERY,
        "raw_out": Path("data/raw/gdelt_bq_national_raw.parquet"),
        "corpus_out": Path("data/raw/gdelt_bq_national.parquet"),
        "slug_filter": False,  # already slug-filtered in SQL
    },
}

# Wide topical net for the LOCAL slug pre-filter (volume control before NLI;
# English + Filipino stems across all 10 HungerGist hypothesis domains).
SLUG_TOPICS = re.compile(
    r"food|rice|palay|bigas|hunger|gutom|famine|malnutri|stunt|feeding|"
    r"ayuda|relief|pantawid|4ps|dswd|kadiwa|nfa|subsid|voucher|pension|"
    r"farm|agri|magsasaka|harvest|ani|crop|fisher|mangingisda|fish|isda|"
    r"aquacultur|tilapia|bangus|poultry|hog|swine|asf|livestock|"
    r"price|presyo|inflation|cpi|cost.of.living|bilihin|market|palengke|"
    r"typhoon|bagyo|flood|baha|drought|tagtuyot|el.nino|la.nina|calamity|"
    r"evacuat|bakwit|displac|landslide|eruption|quake|"
    r"poverty|kahirapan|unemploy|trabaho|layoff|wage|sahod|income|"
    r"remittance|ofw|supply|shortage|kakulangan|smuggl|import|onion|sibuyas|"
    r"sugar|asukal|oil|lpg|fertilizer|abono|galunggong|vegetable|gulay"
)


def _slug_title(url: str) -> str:
    """Derive a human-readable pseudo-title from the URL slug."""
    path = unquote(urlparse(url).path)
    seg = max(path.split("/"), key=len, default="")
    seg = re.sub(r"\.(html?|php|aspx?)$", "", seg)
    seg = re.sub(r"^\d+[-_]?", "", seg)          # leading article ids
    words = re.sub(r"[-_]+", " ", seg).strip()
    return words.title() if len(words) >= 15 else ""


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().lstrip("www.")


def _is_credible(url: str) -> bool:
    d = _domain(url)
    if d in CREDIBLE_DOMAINS:
        return True
    parts = d.split(".")
    return any(".".join(parts[i:]) in CREDIBLE_DOMAINS for i in range(1, len(parts)))


def main() -> None:
    from google.cloud import bigquery
    from google.oauth2 import service_account

    parser = argparse.ArgumentParser(description="GDELT BigQuery harvest")
    parser.add_argument("--mode", choices=list(MODES), default="calabarzon")
    args = parser.parse_args()
    mode = MODES[args.mode]
    QUERY = mode["query"]
    raw_out: Path = mode["raw_out"]
    corpus_out: Path = mode["corpus_out"]

    key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not key or not os.path.exists(key):
        raise SystemExit("GOOGLE_APPLICATION_CREDENTIALS not set or file missing")

    creds = service_account.Credentials.from_service_account_file(key)
    client = bigquery.Client(credentials=creds, project=creds.project_id)

    # Cost guard: dry-run first, refuse anything near the free-tier ceiling.
    dry = client.query(QUERY, job_config=bigquery.QueryJobConfig(dry_run=True))
    gb = dry.total_bytes_processed / 1e9
    logger.info("Dry-run estimate: %.1f GB", gb)
    if gb > 700:
        raise SystemExit(f"Query would scan {gb:.0f} GB — refusing (>700 GB guard)")

    logger.info("Running BigQuery harvest (single scan, ~%.0f GB of 1 TB budget)...", gb)
    rows = client.query(QUERY).result(page_size=50_000)

    records = []
    for r in rows:
        records.append({
            "url": r.url,
            "seen_date": r.seen_date.isoformat() if r.seen_date else "",
            "locations": r.locations or "",
        })
        if len(records) % 200_000 == 0:
            logger.info("  downloaded %d rows...", len(records))

    df = pd.DataFrame(records)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_out, index=False)
    logger.info("Saved raw URL set: %d rows -> %s", len(df), raw_out)

    # ── Local narrowing (free, re-runnable) ──────────────────────────────
    df = df.drop_duplicates(subset=["url"])
    logger.info("Unique URLs: %d", len(df))

    df["credible"] = df["url"].map(_is_credible)
    df = df[df["credible"]]
    logger.info("After credible-domain filter: %d", len(df))

    if mode["slug_filter"]:
        df["slug_lower"] = df["url"].str.lower()
        df = df[df["slug_lower"].str.contains(SLUG_TOPICS, regex=True, na=False)]
        logger.info("After broad topical slug pre-filter: %d", len(df))

    df["title"] = df["url"].map(_slug_title)
    df = df[df["title"] != ""]
    logger.info("With recoverable slug titles: %d", len(df))

    out = pd.DataFrame({
        "title": df["title"],
        "link": df["url"],
        "article_id": df["url"].map(lambda u: hashlib.md5(u.encode()).hexdigest()),
        "published": df["seen_date"],
        "summary": "",
        "source_domain": df["url"].map(_domain),
        "fetcher_source": f"gdelt_bigquery_{args.mode}",
    })
    out.to_parquet(corpus_out, index=False)
    logger.info("Saved corpus-format checkpoint: %d articles -> %s", len(out), corpus_out)


if __name__ == "__main__":
    main()
