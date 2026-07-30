"""
scripts/recover_calabarzon_from_gdelt.py
-----------------------------------------
Recover CALABARZON articles that the pipeline discarded before scoring.

The BigQuery CALABARZON sweep saved 1M raw rows WITH GDELT's V2Locations
(geocoded from the full article body). Two pipeline steps then dropped most
of them before they were ever scored:
  1. a topical URL-slug pre-filter (added for scoring-speed control), and
  2. a title-only geocoder that ignored GDELT's own province tags.

This script rebuilds the pool the right way:
  • parse V2Locations, keep rows with a clean CALABARZON ADM1 tag
    (fullname '..., <Province>, Philippines') — GDELT's full-body geocode
  • restrict to the credible-domain allowlist
  • attach province_code directly from GDELT (no title geocoding needed)
  • emit corpus-format rows for enrichment → scoring

Output: data/raw/gdelt_calabarzon_recovered.parquet
The province_code column here is authoritative; downstream geocoding must
COALESCE onto it rather than overwrite it.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.corpus.rss_fetcher import CREDIBLE_DOMAINS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("recover")

RAW = Path("data/raw/gdelt_bq_urls_raw.parquet")
OUT = Path("data/raw/gdelt_calabarzon_recovered.parquet")

PROV_PSGC = {
    "Cavite": "PH040100000", "Laguna": "PH040200000", "Quezon": "PH040300000",
    "Rizal": "PH040400000", "Batangas": "PH040500000",
}
# Match GDELT location fullnames whose ADM1 is a CALABARZON province.
_PROV_PAT = re.compile(
    r",\s*(Batangas|Cavite|Laguna|Rizal|Quezon)\s*,\s*Philippines", re.I)


def _gdelt_province(loc: str) -> str | None:
    """Return the PSGC of the first clean CALABARZON ADM1 tag, else None.

    When multiple CALABARZON provinces appear, the first is used (the runner's
    single-province assignment convention). Quezon City / Rizal Park style
    false positives are excluded because they never render as
    '..., Quezon, Philippines' / '..., Rizal, Philippines' ADM1 fullnames.
    """
    for entry in str(loc).split(";"):
        parts = entry.split("#")
        if len(parts) > 1:
            m = _PROV_PAT.search(parts[1])
            if m:
                return PROV_PSGC[m.group(1).title()]
    return None


def _domain(u: str) -> str:
    return urlparse(u).netloc.lower().lstrip("www.")


def _credible(u: str) -> bool:
    d = _domain(u)
    if d in CREDIBLE_DOMAINS:
        return True
    p = d.split(".")
    return any(".".join(p[i:]) in CREDIBLE_DOMAINS for i in range(1, len(p)))


def main() -> None:
    raw = pd.read_parquet(RAW).dropna(subset=["locations"]).drop_duplicates("url")
    logger.info("raw rows with locations: %d", len(raw))

    raw["province_code"] = raw["locations"].map(_gdelt_province)
    cal = raw[raw["province_code"].notna()].copy()
    logger.info("clean CALABARZON ADM1 tag: %d", len(cal))

    cal = cal[cal["url"].map(_credible)].copy()
    logger.info("credible-domain: %d", len(cal))

    cal["article_id"] = cal["url"].map(lambda u: hashlib.md5(u.encode()).hexdigest())
    out = pd.DataFrame({
        "title": "",                    # filled by enrichment
        "link": cal["url"].values,
        "article_id": cal["article_id"].values,
        "published": cal["seen_date"].astype(str).values,
        "summary": "",
        "source_domain": cal["url"].map(_domain).values,
        "fetcher_source": "gdelt_bq_calabarzon_recovered",
        "province_code": cal["province_code"].values,   # authoritative
    })
    out = out.drop_duplicates(subset=["article_id"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    logger.info("Saved recovery pool: %d articles -> %s", len(out), OUT)
    logger.info("province distribution:\n%s",
                out["province_code"].map({v: k for k, v in PROV_PSGC.items()})
                .value_counts().to_string())


if __name__ == "__main__":
    main()
