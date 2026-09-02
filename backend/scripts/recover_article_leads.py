"""
scripts/recover_article_leads.py
--------------------------------
Recover real lead text for the audited CALABARZON articles that have none.

278 of 371 audited rows carry no body text. 266 of those point at Google News
RSS redirect tokens rather than publisher URLs, and the token no longer decodes
to a URL -- it has to be resolved through the news.google.com batchexecute
endpoint. This script:

  1. resolves Google News tokens -> publisher URL (batchexecute)
  2. fetches the publisher page and extracts real lead paragraphs
  3. falls back to the Wayback Machine for dead/blocked links (2020-21 URLs)

Checkpointed: rerunning skips rows already recovered. Output carries provenance
so the thesis can state exactly where each lead came from.

    data/processed/_recovered_leads.parquet
      article_id, resolved_url, lead, lead_chars, recovery_path, error
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("recover_leads")
logging.getLogger("urllib3").setLevel(logging.WARNING)

DATASET = Path("data/processed/calabarzon_food_insecurity_dataset.parquet")
OUT = Path("data/processed/_recovered_leads.parquet")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
BATCH_URL = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
MIN_LEAD = 80          # chars below which a lead is not worth keeping
DOMAIN_DELAY = 1.0     # default seconds between hits on the same domain
# archive.org throttles hard (429) at the default cadence; give it real room.
DOMAIN_DELAY_OVERRIDE = {"archive.org": 4.0, "web.archive.org": 4.0}

_domain_last: dict[str, float] = {}
_domain_lock = threading.Lock()
_session = requests.Session()
_session.headers.update({"User-Agent": UA, "Accept-Language": "en-PH,en;q=0.9"})


def _polite(domain: str) -> None:
    """Ensure the per-domain delay has elapsed before the next request."""
    delay = DOMAIN_DELAY_OVERRIDE.get(domain, DOMAIN_DELAY)
    while True:
        with _domain_lock:
            now = time.time()
            last = _domain_last.get(domain, 0.0)
            if now - last >= delay:
                _domain_last[domain] = now
                return
            wait = delay - (now - last)
        time.sleep(wait)


def _get_retrying(url: str, domain: str, timeout: int, tries: int = 3):
    """GET with backoff on 429/503, which archive.org returns under load."""
    for attempt in range(tries):
        _polite(domain)
        r = _session.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code not in (429, 503):
            return r
        time.sleep(4.0 * (attempt + 1))
    return r


# ---------------------------------------------------------------------------
# Lead extraction -- same rules as scripts/enrich_bigquery_articles.py, copied
# here rather than imported because that module parses argv at import time.
# ---------------------------------------------------------------------------

_META_RE = {
    "og_desc": re.compile(
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)', re.I),
    "og_desc2": re.compile(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:description', re.I),
    "meta_desc": re.compile(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)', re.I),
}
_P_TAG = re.compile(r"<p[^>]*>(.*?)</p>", re.I | re.S)
_TAG_STRIP = re.compile(r"<[^>]+>")
_BOILERPLATE = re.compile(
    r"cookie|subscribe|sign.?up|newsletter|all rights reserved|advertis|"
    r"follow us|read more|click here|terms of (use|service)|privacy policy|"
    r"inquirer\.net|copyright|by continuing", re.I)


def _clean(text: str) -> str:
    text = unescape(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*[|\-–]\s*[A-Z][\w .]{2,40}$", "", text)
    return text


def _lead_paragraphs(html: str, budget: int = 500) -> str:
    """First real paragraphs of the article body, joined up to `budget` chars."""
    out: list[str] = []
    total = 0
    for m in _P_TAG.finditer(html):
        text = _clean(_TAG_STRIP.sub(" ", m.group(1)))
        if len(text) < 60 or _BOILERPLATE.search(text):
            continue
        out.append(text)
        total += len(text)
        if total >= budget:
            break
    return " ".join(out)[:budget]


def _extract(html: str) -> tuple[str, str]:
    """Return (unused_title, lead). Prefers body paragraphs over meta description."""
    desc = _lead_paragraphs(html)
    if len(desc) < 80:
        for key in ("og_desc", "og_desc2", "meta_desc"):
            m = _META_RE[key].search(html)
            if m:
                meta = _clean(m.group(1))
                if len(meta) > len(desc):
                    desc = meta
                if desc:
                    break
    return "", desc


# ---------------------------------------------------------------------------
# Step 1 -- Google News token -> publisher URL
# ---------------------------------------------------------------------------

_SG = re.compile(r'data-n-a-sg="([^"]+)"')
_TS = re.compile(r'data-n-a-ts="([^"]+)"')
_AID = re.compile(r"/articles/([A-Za-z0-9_\-]+)")
_GARTURL = re.compile(r'garturlres[\\"]+,[\\"]+(https?://[^\\"]+)')


def resolve_gnews(url: str) -> str:
    """Resolve a news.google.com/rss/articles/<token> URL to the publisher URL."""
    _polite("news.google.com")
    r = _session.get(url, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"gnews page http {r.status_code}")
    sg, ts = _SG.search(r.text), _TS.search(r.text)
    aid = _AID.search(url)
    if not (sg and ts and aid):
        raise RuntimeError("no signature/timestamp on gnews page")

    inner = json.dumps([
        "garturlreq",
        [["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1,
          None, None, None, None, None, 0, 1],
         "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0],
        aid.group(1), int(ts.group(1)), sg.group(1),
    ])
    payload = [[["Fbv4je", inner, None, "generic"]]]

    _polite("news.google.com")
    resp = _session.post(BATCH_URL, data={"f.req": json.dumps(payload)}, timeout=25)
    if resp.status_code != 200:
        raise RuntimeError(f"batchexecute http {resp.status_code}")
    m = _GARTURL.search(resp.text)
    if not m:
        raise RuntimeError("batchexecute returned no url")
    return m.group(1)


# ---------------------------------------------------------------------------
# Steps 2/3 -- publisher fetch, Wayback fallback
# ---------------------------------------------------------------------------

def fetch_lead(url: str) -> str:
    _polite(urlparse(url).netloc.lower())
    r = _session.get(url, timeout=20, allow_redirects=True)
    if r.status_code != 200:
        raise RuntimeError(f"http {r.status_code}")
    _, lead = _extract(r.text[:200_000])
    return lead or ""


def wayback_lead(url: str) -> str:
    """
    Recover the lead from a Wayback snapshot.

    Uses the CDX index rather than the /wayback/available endpoint: available
    returns only a single "closest" snapshot and reports none at all for many
    URLs that CDX does hold. Tries successive snapshots, newest first, because
    the earliest capture is often a paywall or consent interstitial.
    """
    cdx = _get_retrying(
        "http://web.archive.org/cdx/search/cdx"
        f"?url={requests.utils.quote(url, safe='')}"
        "&output=json&limit=6&filter=statuscode:200&collapse=digest",
        "web.archive.org", timeout=40,
    )
    if cdx.status_code != 200:
        raise RuntimeError(f"cdx http {cdx.status_code}")
    if not cdx.text.strip():
        raise RuntimeError("no wayback snapshot")
    try:
        rows = cdx.json()
    except Exception:
        raise RuntimeError("cdx returned non-json")
    if len(rows) < 2:
        raise RuntimeError("no wayback snapshot")

    stamps = [r[1] for r in rows[1:]][::-1]  # newest first
    last = ""
    for stamp in stamps[:3]:
        snap = f"https://web.archive.org/web/{stamp}id_/{url}"
        r = _get_retrying(snap, "web.archive.org", timeout=40)
        if r.status_code != 200:
            last = f"snapshot http {r.status_code}"
            continue
        _, lead = _extract(r.text[:200_000])
        if len(lead) >= MIN_LEAD:
            return lead
        last = f"short({len(lead)})"
    raise RuntimeError(last or "no usable snapshot")


def recover_one(rec: dict) -> dict:
    aid, url = rec["article_id"], rec["url"]
    out = {"article_id": aid, "resolved_url": url, "lead": "",
           "lead_chars": 0, "recovery_path": "", "error": ""}
    errs: list[str] = []

    # 1. resolve google news token
    target = url
    if "news.google.com" in urlparse(url).netloc:
        try:
            target = resolve_gnews(url)
            out["resolved_url"] = target
        except Exception as exc:
            out["error"] = f"resolve:{exc}"[:300]
            return out

    # 2. publisher fetch
    try:
        lead = fetch_lead(target)
        if len(lead) >= MIN_LEAD:
            out.update(lead=lead, lead_chars=len(lead), recovery_path="publisher")
            return out
        errs.append(f"publisher:short({len(lead)})")
    except Exception as exc:
        errs.append(f"publisher:{exc}")

    # 3. wayback fallback
    try:
        lead = wayback_lead(target)
        if len(lead) >= MIN_LEAD:
            out.update(lead=lead, lead_chars=len(lead), recovery_path="wayback")
            return out
        errs.append(f"wayback:short({len(lead)})")
    except Exception as exc:
        errs.append(f"wayback:{exc}")

    out["error"] = "; ".join(errs)[:300]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only process N rows (smoke test)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retry-failed", action="store_true",
                    help="retry rows that failed before, reusing the publisher URL "
                         "already resolved so no Google News call is repeated")
    args = ap.parse_args()

    df = pd.read_parquet(DATASET)
    df["lead_len"] = df["content_lead"].fillna("").astype(str).str.len()
    todo = df[df["lead_len"] == 0][["article_id", "url"]].copy()
    log.info("rows missing lead: %d of %d", len(todo), len(df))

    done = pd.DataFrame()
    if OUT.exists():
        done = pd.read_parquet(OUT)
        keep = set(done[done["lead_chars"] >= MIN_LEAD]["article_id"])
        todo = todo[~todo["article_id"].isin(keep)]
        log.info("checkpoint: %d already recovered, %d remaining", len(keep), len(todo))

        if args.retry_failed:
            # Token resolution already succeeded for every failure; go straight
            # to the resolved publisher URL instead of paying for it again.
            resolved = done[
                (done["lead_chars"] < MIN_LEAD)
                & (~done["resolved_url"].fillna("").str.contains("news.google"))
            ][["article_id", "resolved_url"]]
            todo = todo.merge(resolved, on="article_id", how="left")
            todo["url"] = todo["resolved_url"].fillna(todo["url"])
            todo = todo.drop(columns=["resolved_url"])
            log.info("retry mode: %d rows using already-resolved publisher URLs", len(todo))

    if args.limit:
        todo = todo.head(args.limit)
        log.info("limit: processing %d", len(todo))
    if todo.empty:
        log.info("nothing to do")
        return

    recs = todo.to_dict("records")
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(recover_one, r): r for r in recs}
        for n, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if n % 10 == 0 or n == len(recs):
                ok = sum(1 for x in results if x["lead_chars"] >= MIN_LEAD)
                log.info("processed %d/%d | recovered %d", n, len(recs), ok)

    new = pd.DataFrame(results)
    if not done.empty:
        new = pd.concat(
            [done[~done["article_id"].isin(set(new["article_id"]))], new],
            ignore_index=True,
        )
    new.to_parquet(OUT, index=False)

    ok = new[new["lead_chars"] >= MIN_LEAD]
    log.info("saved %d rows -> %s", len(new), OUT)
    log.info("recovered %d (%.1f%%)", len(ok), 100 * len(ok) / max(len(new), 1))
    if not ok.empty:
        log.info("by path: %s", dict(ok["recovery_path"].value_counts()))
        log.info("median lead chars: %d", int(ok["lead_chars"].median()))
    fails = new[new["lead_chars"] < MIN_LEAD]
    if not fails.empty:
        kinds = fails["error"].str.split(":").str[0].value_counts().head(5)
        log.info("failure kinds: %s", dict(kinds))


if __name__ == "__main__":
    main()
