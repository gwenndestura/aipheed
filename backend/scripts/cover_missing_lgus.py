"""
scripts/cover_missing_lgus.py
------------------------------
Targeted collection to fill Region IV-A LGU coverage gaps.

The general corpus covers only 44/142 CALABARZON LGUs with a food-insecurity-
relevant article — 98 municipalities/cities have none. This runs a FOCUSED
GDELT DOC-API sweep aimed only at those uncovered LGUs, each paired with the
bilingual food-insecurity lexicon, across yearly windows (2020–2025). GDELT is
used because it is free/no-key and far more throttle-tolerant than the
rate-limited Google News RSS.

Output: data/raw/targeted_lgu_pool.parquet  (corpus record format)
        appended into the collection pipeline as a normal source.

Usage:
  venv\\Scripts\\python scripts\\cover_missing_lgus.py [--start 2020 --end 2025]
  # then score + geocode:
  venv\\Scripts\\python scripts\\cover_missing_lgus.py --report
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.corpus.gdelt_fetcher import (  # noqa: E402
    _build_url, _parse_gdelt_article,
)


def _fetch_with_backoff(url: str, max_tries: int = 6) -> list[dict]:
    """GET the GDELT DOC URL, backing off on HTTP 429 (rate limit) so a throttle
    is distinguished from a genuinely empty result. Returns the raw articles
    list, or [] only after real 200-empty or exhausted retries."""
    delay = 6.0
    for attempt in range(max_tries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                ct = r.headers.get("content-type", "")
                time.sleep(6.0)                       # honour 1-req/5s cap
                if ct.startswith("application/json"):
                    return r.json().get("articles") or []
                return []                             # 200 but non-JSON → treat empty
            if r.status_code == 429:
                print(f"      429 throttled — backing off {delay:.0f}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
        except Exception as exc:
            print(f"      request error: {exc}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)
    return []

OUT = Path("data/raw/targeted_lgu_pool.parquet")
TARGETS = Path("data/processed/_uncovered_lgus.parquet")

# Bilingual food-insecurity OR-group (mirrors gdelt_fetcher.FOOD_COMBINED_TERMS,
# widened slightly for recall on small LGUs).
FOOD_OR = ("food OR hunger OR rice OR relief OR crop OR harvest OR poverty OR "
           "fish OR palay OR ayuda OR gutom OR bigas OR farmer OR fisherfolk OR "
           "malnutrition OR DSWD OR NFA OR pagkain")

# LGU names that collide with a province name: the plain name can't be queried
# unambiguously, so use explicit municipality phrasings.
_COLLIDING = {
    "Rizal": '("municipality of Rizal" OR "bayan ng Rizal")',
    "Quezon": '("municipality of Quezon" OR "Quezon town")',
}


def _query_for(lgu: str, province: str) -> str:
    if lgu in _COLLIDING:
        return f'{_COLLIDING[lgu]} {province} ({FOOD_OR})'
    return f'"{lgu}" {province} ({FOOD_OR})'


# GDELT DOC API enforces 1 request / 5 seconds (HTTP 429 otherwise). _fetch_gdelt
# already sleeps REQUEST_DELAY (1.5s); top it up to a safe 5.5s total.
_EXTRA_SLEEP = 4.0


def collect(start_year: int, end_year: int) -> None:
    targets = pd.read_parquet(TARGETS)
    print(f"targeting {len(targets)} uncovered LGUs, {start_year}-{end_year}", flush=True)

    seen: set[str] = set()
    rows: list[dict] = []
    # Resume: fold any prior partial output back in, and skip LGUs already done.
    done_lgus: set[str] = set()
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        rows = prev.to_dict("records")
        seen = set(prev["article_id"].dropna())
        done_lgus = set(prev.get("target_lgu", pd.Series(dtype=str)).dropna())
        print(f"resuming — {len(rows)} articles, {len(done_lgus)} LGUs already queried", flush=True)

    # One full-range query per LGU: small LGUs return far under the 250 cap, so a
    # single DateDesc window captures everything without extra requests.
    s_dt, e_dt = f"{start_year}0101000000", f"{end_year}1231235959"
    for i, t in enumerate(targets.itertuples(index=False), 1):
        lgu, prov = t.lgu_name, t.province_name
        if lgu in done_lgus:
            continue
        url = _build_url(_query_for(lgu, prov), s_dt, e_dt)
        got = 0
        for item in _fetch_with_backoff(url):       # 429-aware, sleeps 6s on 200
            rec = _parse_gdelt_article(item)
            if rec and rec["article_id"] not in seen:
                seen.add(rec["article_id"])
                rec["fetcher_source"] = "targeted_lgu"
                rec["target_lgu"] = lgu
                rows.append(rec)
                got += 1
        print(f"  [{i:3d}/{len(targets)}] {prov}/{lgu:22s} +{got}  (total {len(rows)})", flush=True)
        if i % 10 == 0:
            pd.DataFrame(rows).to_parquet(OUT, index=False)

    pd.DataFrame(rows).to_parquet(OUT, index=False)
    print(f"\nsaved {len(rows)} articles -> {OUT}", flush=True)


def report() -> None:
    """Score + geocode the targeted pool; show which uncovered LGUs are now hit."""
    from app.ml.corpus.location_geocoder import geocode_location_batch
    from app.ml.nlp.classifier import load_classifier, score_article

    pool = pd.read_parquet(OUT)
    print(f"targeted pool: {len(pool)} articles")

    # Score (reuse cache where possible)
    ck = Path("data/raw/checkpoints/xlmr_scores.parquet")
    done = {r["article_id"]: r for r in pd.read_parquet(ck).to_dict("records")} if ck.exists() else {}
    clf = load_classifier()
    recs = []
    for r in pool.to_dict("records"):
        aid = r["article_id"]
        if aid in done:
            recs.append({**r, "is_relevant": done[aid]["is_relevant"]})
        else:
            s = score_article(clf, str(r.get("title") or ""), str(r.get("summary") or ""))
            recs.append({**r, "is_relevant": bool(s["is_relevant"])})
    df = pd.DataFrame(recs)
    rel = df[df["is_relevant"]]
    tagged = geocode_location_batch(rel)
    new_lgus = tagged[tagged["lgu_name"].notna()]
    print(f"relevant: {len(rel)} | newly LGU-tagged: {len(new_lgus)} "
          f"| distinct new LGUs: {new_lgus['lgu_psgc'].nunique()}")
    for r in new_lgus[["province_name", "lgu_name"]].drop_duplicates().itertuples(index=False):
        print(f"  + {r.province_name}/{r.lgu_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2020)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
    else:
        collect(args.start, args.end)


if __name__ == "__main__":
    main()
