"""
validate_corpus.py
------------------
Deep validation of the collected CALABARZON food insecurity corpus.

Checks performed:
  1.  Schema       — required columns present, no critical nulls
  2.  Source       — all domains in CREDIBLE_DOMAINS allowlist
  3.  Geo signal   — every article has a CALABARZON geo anchor
  4.  Food signal  — every article has a food insecurity signal
  5.  Both signals — every article satisfies the two-signal rule
  6.  Date range   — articles within 2020-01-01 to today
  7.  Duplicates   — URL-level and title-level duplicate check
  8.  Coverage     — articles per province per quarter (flags sparse cells)
  9.  Source mix   — breakdown by fetcher_source
 10.  Random sample — 10 random articles printed for manual spot-check

Usage:
    python validate_corpus.py
    python validate_corpus.py --path data/raw/corpus_raw_expanded.parquet
    python validate_corpus.py --sample 20
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Colour helpers ────────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_RESET  = "\033[0m"

def ok(msg):    print(f"  {_GREEN}[PASS]{_RESET} {msg}")
def warn(msg):  print(f"  {_YELLOW}[WARN]{_RESET} {msg}")
def fail(msg):  print(f"  {_RED}[FAIL]{_RESET} {msg}")

# ── Signals (imported from rss_fetcher) ───────────────────────────────────────
try:
    from app.ml.corpus.rss_fetcher import (
        CALABARZON_GEO_SIGNALS,
        CALABARZON_FOOD_SIGNALS,
        CREDIBLE_DOMAINS,
    )
except ImportError:
    fail("Cannot import from app.ml.corpus.rss_fetcher — run from backend/ directory.")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_PATH   = Path("data/raw/corpus_raw.parquet")
DATE_MIN       = datetime(2020, 1, 1, tzinfo=timezone.utc)
DATE_MAX       = datetime.now(tz=timezone.utc)
MIN_ARTICLES_PER_CELL = 5   # province-quarter cells with fewer articles get a warning
REQUIRED_COLS  = ["title", "link", "article_id", "published", "source_domain", "fetcher_source"]

TITLE_STOP: frozenset[str] = frozenset({
    "a","an","the","and","or","of","in","on","at","to","for","by","with","as",
    "its","it","this","that","their","from","after","before","over","more",
    "new","up","out","into","amid","due","per","vs","is","are","was","were",
    "be","been","has","have","had","says","say","said","report","reports",
    "hit","hits","see","sees","give","gives","distribute","distributes",
    "provide","provides","issue","issues","release","releases",
    "ph","philippines","philippine",
})

PROVINCES = {
    "PH040100000": "Batangas",
    "PH040200000": "Cavite",
    "PH040300000": "Laguna",
    "PH040400000": "Quezon",
    "PH040500000": "Rizal",
}

MODEL_QUARTERS = [f"{yr}-Q{q}" for yr in range(2020, 2026) for q in range(1, 5)]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _combined(row: pd.Series) -> str:
    return ((row.get("title") or "") + " " + (row.get("summary") or "")).lower()

def _has_geo(row: pd.Series) -> bool:
    return any(kw in _combined(row) for kw in CALABARZON_GEO_SIGNALS)

def _has_food(row: pd.Series) -> bool:
    return any(kw in _combined(row) for kw in CALABARZON_FOOD_SIGNALS)

def _is_credible(domain: str) -> bool:
    d = str(domain).lower().lstrip("www.")
    if d in CREDIBLE_DOMAINS:
        return True
    parts = d.split(".")
    for i in range(1, len(parts)):
        if ".".join(parts[i:]) in CREDIBLE_DOMAINS:
            return True
    return False

def _wordset(title: str) -> frozenset:
    t = re.sub(r"[^a-z0-9\s]", " ", title.lower())
    return frozenset(w for w in t.split() if w not in TITLE_STOP and len(w) > 2)

def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Validation checks ─────────────────────────────────────────────────────────

def check_schema(df: pd.DataFrame) -> bool:
    print("\n[1] Schema check")
    passed = True
    for col in REQUIRED_COLS:
        null_n = df[col].isna().sum()
        if col not in df.columns:
            fail(f"Column '{col}' is missing entirely")
            passed = False
        elif null_n > 0:
            warn(f"Column '{col}' has {null_n} nulls ({null_n/len(df)*100:.1f}%)")
        else:
            ok(f"Column '{col}' — {len(df)} rows, 0 nulls")
    return passed


def check_sources(df: pd.DataFrame) -> bool:
    print("\n[2] Credible domain check")
    bad = df[~df["source_domain"].apply(_is_credible)]
    if bad.empty:
        ok(f"All {len(df)} articles from credible domains")
        return True
    fail(f"{len(bad)} articles from non-credible domains:")
    for domain, cnt in bad["source_domain"].value_counts().head(10).items():
        print(f"       {domain}: {cnt} articles")
    return False


def check_geo_signal(df: pd.DataFrame) -> bool:
    print("\n[3] Geo signal check (CALABARZON anchor in title+summary)")
    no_geo = df[~df.apply(_has_geo, axis=1)]
    if no_geo.empty:
        ok(f"All {len(df)} articles have a CALABARZON geo signal")
        return True
    pct = len(no_geo) / len(df) * 100
    warn(f"{len(no_geo)} articles ({pct:.1f}%) missing geo signal — sample:")
    for _, row in no_geo.head(5).iterrows():
        print(f"       [{row.get('fetcher_source','')}] {row['title'][:80]}")
    return pct < 5.0  # pass if < 5% missing


def check_food_signal(df: pd.DataFrame) -> bool:
    print("\n[4] Food signal check (food insecurity keyword in title+summary)")
    no_food = df[~df.apply(_has_food, axis=1)]
    if no_food.empty:
        ok(f"All {len(df)} articles have a food insecurity signal")
        return True
    pct = len(no_food) / len(df) * 100
    warn(f"{len(no_food)} articles ({pct:.1f}%) missing food signal — sample:")
    for _, row in no_food.head(5).iterrows():
        print(f"       [{row.get('fetcher_source','')}] {row['title'][:80]}")
    return pct < 5.0


def check_both_signals(df: pd.DataFrame) -> bool:
    print("\n[5] Two-signal rule (geo AND food)")
    no_both = df[~(df.apply(_has_geo, axis=1) & df.apply(_has_food, axis=1))]
    if no_both.empty:
        ok(f"All {len(df)} articles satisfy the two-signal rule")
        return True
    pct = len(no_both) / len(df) * 100
    # gnews_rss / gdelt only enforce food signal (geo is in the query)
    gnews_gdelt = no_both[no_both["fetcher_source"].isin(["gnews_rss", "gdelt"])]
    other       = no_both[~no_both["fetcher_source"].isin(["gnews_rss", "gdelt"])]
    if not gnews_gdelt.empty:
        ok(f"  {len(gnews_gdelt)} articles from gnews_rss/gdelt missing title-level geo "
           f"(expected — geo is in the query, not the title)")
    if not other.empty:
        fail(f"  {len(other)} articles from other fetchers missing both signals:")
        for _, row in other.head(5).iterrows():
            print(f"       [{row.get('fetcher_source','')}] {row['title'][:80]}")
        return False
    return True


def check_date_range(df: pd.DataFrame) -> bool:
    print("\n[6] Date range check (2020-01-01 → today)")
    parsed = pd.to_datetime(df["published"], errors="coerce", utc=True)
    unparsed = parsed.isna().sum()
    if unparsed:
        warn(f"{unparsed} articles have unparseable dates")

    valid = parsed.dropna()
    too_old  = (valid < DATE_MIN).sum()
    too_new  = (valid > DATE_MAX).sum()
    in_range = len(valid) - too_old - too_new

    ok(f"  {in_range} articles in range (2020–today)")
    if too_old:
        warn(f"  {too_old} articles before 2020-01-01")
    if too_new:
        warn(f"  {too_new} articles after today")
    return too_old == 0


def check_duplicates(df: pd.DataFrame) -> bool:
    print("\n[7] Duplicate check (URL + content)")
    passed = True

    # URL duplicates
    url_dups = df["link"].duplicated().sum()
    id_dups  = df["article_id"].dropna().duplicated().sum()
    if url_dups == 0 and id_dups == 0:
        ok("No URL/ID duplicates found")
    else:
        fail(f"{url_dups} duplicate URLs, {id_dups} duplicate article_ids remaining")
        passed = False

    # Content duplicates (Jaccard ≥ 0.75)
    print("     Running content dedup scan (Jaccard ≥ 0.75) …", end="", flush=True)
    wordsets: list[frozenset] = []
    word_index: dict[str, list[int]] = defaultdict(list)
    content_dups = 0
    for title in df["title"].fillna(""):
        words = _wordset(title)
        if len(words) >= 4:
            candidates = set()
            for w in words:
                candidates.update(word_index.get(w, []))
            dup = any(_jaccard(words, wordsets[i]) >= 0.75 for i in candidates)
            if dup:
                content_dups += 1
            else:
                idx = len(wordsets)
                for w in words:
                    word_index[w].append(idx)
                wordsets.append(words)
        else:
            wordsets.append(words)

    print(f"\r     Content duplicate scan complete.              ")
    if content_dups == 0:
        ok("No near-duplicate titles found (Jaccard < 0.75 for all pairs)")
    else:
        warn(f"{content_dups} near-duplicate titles detected (same story, different wording)")
        passed = False
    return passed


def check_coverage(df: pd.DataFrame) -> bool:
    print("\n[8] Province-quarter coverage")
    if "province_code" not in df.columns or "quarter" not in df.columns:
        warn("province_code or quarter column missing — skipping coverage check")
        return True

    geo_df = df[df["province_code"].isin(PROVINCES.keys()) & df["quarter"].isin(MODEL_QUARTERS)]
    counts = geo_df.groupby(["province_code", "quarter"]).size().reset_index(name="n")

    total_cells = len(PROVINCES) * len(MODEL_QUARTERS)
    filled_cells = len(counts)
    sparse_cells = counts[counts["n"] < MIN_ARTICLES_PER_CELL]
    zero_cells   = total_cells - filled_cells

    ok(f"{filled_cells}/{total_cells} province-quarter cells have articles")
    if zero_cells:
        warn(f"{zero_cells} cells have ZERO articles")
    if not sparse_cells.empty:
        warn(f"{len(sparse_cells)} cells have fewer than {MIN_ARTICLES_PER_CELL} articles:")
        for _, row in sparse_cells.iterrows():
            pname = PROVINCES.get(row["province_code"], row["province_code"])
            print(f"       {pname} {row['quarter']}: {row['n']} articles")

    # Per-province summary
    print()
    print(f"     {'Province':<15} {'Total':>7}  {'Min/Qtr':>8}  {'Avg/Qtr':>8}  {'Max/Qtr':>8}")
    print(f"     {'-'*15} {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")
    for code, name in PROVINCES.items():
        pdata = counts[counts["province_code"] == code]["n"]
        if pdata.empty:
            print(f"     {name:<15} {'0':>7}  {'–':>8}  {'–':>8}  {'–':>8}")
        else:
            print(f"     {name:<15} {pdata.sum():>7}  {pdata.min():>8}  {pdata.mean():>8.1f}  {pdata.max():>8}")

    return zero_cells == 0


def check_source_mix(df: pd.DataFrame) -> bool:
    print("\n[9] Fetcher source breakdown")
    counts = df["fetcher_source"].value_counts()
    total  = len(df)
    for src, cnt in counts.items():
        bar = "█" * int(cnt / total * 40)
        print(f"     {src:<18} {cnt:>6}  {cnt/total*100:>5.1f}%  {bar}")
    ok(f"Total: {total} articles across {len(counts)} sources")
    return True


def random_sample(df: pd.DataFrame, n: int) -> None:
    print(f"\n[10] Random sample ({n} articles for manual spot-check)")
    sample = df.sample(min(n, len(df)), random_state=42)
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        src  = row.get("fetcher_source", "?")
        dom  = row.get("source_domain", "?")
        pub  = str(row.get("published", ""))[:10]
        title = str(row.get("title", ""))[:90]
        geo  = "✓geo"  if _has_geo(row)  else "✗geo"
        food = "✓food" if _has_food(row) else "✗food"
        print(f"     {i:>2}. [{src}] [{dom}] [{pub}] {geo} {food}")
        print(f"         {title}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Deep validation of collected corpus")
    parser.add_argument("--path",   default=str(DEFAULT_PATH), help="Path to corpus parquet file")
    parser.add_argument("--sample", type=int, default=10, help="Number of random articles to print")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        fail(f"File not found: {path}")
        sys.exit(1)

    df = pd.read_parquet(path)
    print(f"\n{'='*60}")
    print(f"  aiPHeed Corpus Validator")
    print(f"  File   : {path}")
    print(f"  Rows   : {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"{'='*60}")

    results = {
        "Schema"      : check_schema(df),
        "Sources"     : check_sources(df),
        "Geo signal"  : check_geo_signal(df),
        "Food signal" : check_food_signal(df),
        "Both signals": check_both_signals(df),
        "Date range"  : check_date_range(df),
        "Duplicates"  : check_duplicates(df),
        "Coverage"    : check_coverage(df),
        "Source mix"  : check_source_mix(df),
    }
    random_sample(df, args.sample)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results.items():
        status = f"{_GREEN}PASS{_RESET}" if passed else f"{_RED}FAIL{_RESET}"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print(f"{'='*60}")
    if all_pass:
        print(f"  {_GREEN}All checks passed. Corpus is ready for NLP processing.{_RESET}")
    else:
        print(f"  {_RED}Some checks failed. Review warnings above before proceeding.{_RESET}")
    print(f"{'='*60}\n")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
