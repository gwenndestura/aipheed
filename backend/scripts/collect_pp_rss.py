"""
scripts/collect_pp_rss.py
-------------------------
Google News RSS half of the poverty/food-prices collection, re-run on its own.

The first pass ran 182 queries across 8 worker threads and Google News answered
every one with a throttle, so the RSS axis contributed nothing (the queries
themselves are fine — verified by hand afterwards). This runs the same job list
with LOW concurrency (3 workers), a slower pace and a much longer backoff, and
appends into the existing pool file rather than replacing it.
"""
from __future__ import annotations

import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_poverty_prices import OUT, PROVS, RSS_GROUPS, RSS_LGU, _rss_one  # noqa: E402


def _fetch(q, src, pace):
    """One query with a patient backoff; returns [] only after real exhaustion."""
    time.sleep(random.uniform(0, pace))
    for attempt in range(6):
        got = _rss_one(q, src)
        if got:
            return got
        time.sleep(pace * (attempt + 1))          # 4s, 8s, 12s, ... on emptiness
    return []


def main(workers=3, pace=4.0):
    from app.ml.corpus.gdelt_fetcher import CALABARZON_LGUS
    lgus = [l for ls in CALABARZON_LGUS.values() for l in ls]
    jobs = [(f'"{p}" {gq}', f"gnews_pp_{g}") for p in PROVS for g, gq in RSS_GROUPS.items()]
    jobs += [(f'"{l}" {RSS_LGU}', "gnews_pp_lgu") for l in lgus]
    print(f"RSS re-run: {len(jobs)} queries, {workers} workers, pace {pace}s", flush=True)

    rows, lock, done = [], threading.Lock(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch, q, s, pace) for q, s in jobs]
        for fut in as_completed(futs):
            got = fut.result()
            with lock:
                rows.extend(got)
                done += 1
                if done % 25 == 0:
                    print(f"    {done}/{len(jobs)}, {len(rows)} rows", flush=True)

    new = pd.DataFrame(rows)
    if new.empty:
        print("RSS still returning nothing — leaving pool unchanged")
        return
    new = new[new["link"].str.len() > 0]
    old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=new.columns)
    comb = pd.concat([old, new], ignore_index=True).drop_duplicates("article_id")

    def nt(s):
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
    comb["_nt"] = comb["title"].fillna("").map(nt)
    comb = comb.drop_duplicates("_nt").drop(columns=["_nt"])
    comb.to_parquet(OUT, index=False)
    print(f"\npool: {len(old)} -> {len(comb)} (+{len(comb) - len(old)})")
    print(comb["fetcher_source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
