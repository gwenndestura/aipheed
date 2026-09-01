"""Recover the missing `content_lead` text for rows that kept only a title.

269 of the retained rows carry a Google-News redirect URL whose destination
Google no longer encodes in the link, so the original fetchers never stored any
body text. This script finds each article again on its publisher's own site and
takes the lead from there.

Nothing is ever synthesised. A recovered lead is accepted only when the
candidate's headline matches the stored title closely AND its publication date
is within a few days, so a near-miss leaves the field empty rather than
attaching the wrong article's text. Every accepted lead records where it came
from in `lead_source`.

Resumable: results are cached per article_id, so re-running only retries misses.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/calabarzon_food_insecurity_dataset.csv"
CACHE = ROOT / "data/processed/audit_review/recovered_leads.csv"

UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
TITLE_MATCH = 0.72      # SequenceMatcher ratio on normalised headlines
DATE_SLACK = 4          # days between stored date and candidate date
MIN_LEAD = 60           # shorter than this is a stub, not a lead
MAX_LEAD = 400


def norm(s):
    s = html.unescape(str(s)).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def similar(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def get(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def clean(text):
    """Strip tags/entities and squeeze to a single lead paragraph."""
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace(" ", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # Drop boilerplate the aggregators prepend.
    text = re.sub(r"^(Make this your preferred source.*?Google\.\s*"
                  r"|This is AI-generated\..*?errors\.\s*|Copied\s*)", "", text, flags=re.I)
    # Embedded video/player URLs sometimes lead the description; they are not text.
    text = re.sub(r"^(https?://\S+\s*)+", "", text).strip()
    return text[:MAX_LEAD].strip()


def lead_from_page(page):
    """Prefer the article's own description meta, else its first real paragraph."""
    for pat in (r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']{40,})["\']',
                r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']{40,})["\']'):
        m = re.search(pat, page, re.I)
        if m:
            got = clean(m.group(1))
            if len(got) >= MIN_LEAD:
                return got
    for m in re.finditer(r"(?is)<p[^>]*>(.*?)</p>", page):
        got = clean(m.group(1))
        if len(got) >= MIN_LEAD and not re.match(r"(?i)(share|follow|subscribe|advert)", got):
            return got
    return None


# ── per-publisher search: return [(candidate_title, candidate_date, url), ...] ──
def wp_rest(base):
    def search(title, date):
        q = urllib.parse.quote(" ".join(norm(title).split()[:12]))
        url = f"{base}/wp-json/wp/v2/posts?search={q}&per_page=5"
        out = []
        for post in json.loads(get(url)):
            out.append((clean(post["title"]["rendered"]), post["date"][:10],
                        post["link"], clean(post.get("excerpt", {}).get("rendered", ""))))
        return out
    return search


def html_search(url_tmpl, link_pat):
    def search(title, date):
        q = urllib.parse.quote(" ".join(norm(title).split()[:12]))
        page = get(url_tmpl.format(q=q))
        seen, out = set(), []
        for m in re.finditer(link_pat, page, re.I):
            link = html.unescape(m.group(1))
            if link.startswith("/"):
                link = urllib.parse.urljoin(url_tmpl, link)
            if link in seen or not link.startswith("http"):
                continue
            seen.add(link)
            out.append((None, None, link, None))
            if len(out) >= 6:
                break
        return out
    return search


SEARCHERS = {
    "Rappler": wp_rest("https://www.rappler.com"),
    "BusinessWorld": wp_rest("https://www.bworldonline.com"),
    "Philippine Daily Inquirer": html_search(
        "https://newsinfo.inquirer.net/?s={q}",
        r'<h2[^>]*>\s*<a[^>]+href="(https?://[^"]*inquirer\.net/[^"]+)"'),
    "GMA Network": html_search(
        "https://www.gmanetwork.com/news/search/?q={q}",
        r'href="(https?://www\.gmanetwork\.com/news/[a-z]+/[^"]+/\d+/[^"]+)"'),
    "PhilStar": html_search(
        "https://www.philstar.com/search?q={q}",
        r'href="(https?://www\.philstar\.com/[a-z-]+/\d{4}/\d{2}/\d{2}/[^"]+)"'),
}


def wayback(url):
    """Archived copy, for publishers that refuse direct requests."""
    api = f"https://archive.org/wayback/available?url={urllib.parse.quote(url, safe='')}"
    snap = json.loads(get(api)).get("archived_snapshots", {}).get("closest")
    if not snap or not snap.get("available"):
        return None
    return get(snap["url"].replace("http://", "https://"), timeout=40)


def fetch_page(url):
    try:
        return get(url)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        try:
            return wayback(url)
        except Exception:
            return None


def recover(row):
    """Return (lead, source) or (None, reason)."""
    stored_date = pd.Timestamp(row["publication_date"])

    # A real publisher URL: fetch it (or its archived copy) directly.
    if "news.google.com" not in str(row["url"]):
        page = fetch_page(row["url"])
        if page:
            got = lead_from_page(page)
            if got:
                return got, "publisher_url"
        return None, "direct fetch failed"

    search = SEARCHERS.get(row["news_source"])
    if not search:
        return None, f"no searcher for {row['news_source']}"
    try:
        cands = search(row["title"], stored_date)
    except Exception as e:
        return None, f"search failed: {type(e).__name__}"

    best = None
    for cand_title, cand_date, link, excerpt in cands:
        page = None
        if cand_title is None:                      # HTML search: verify on the page
            page = fetch_page(link)
            if not page:
                continue
            m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)',
                          page, re.I) or re.search(r"(?is)<title[^>]*>(.*?)</title>", page)
            cand_title = clean(m.group(1)) if m else ""
            m = re.search(r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})', page) or \
                re.search(r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\'](\d{4}-\d{2}-\d{2})',
                          page, re.I)
            cand_date = m.group(1) if m else None
        score = similar(row["title"], cand_title)
        if score < TITLE_MATCH:
            continue
        if cand_date:
            gap = abs((pd.Timestamp(cand_date) - stored_date).days)
            if gap > DATE_SLACK:
                continue
        if excerpt and len(excerpt) >= MIN_LEAD:
            return excerpt, "publisher_search"
        page = page or fetch_page(link)
        if page:
            got = lead_from_page(page)
            if got:
                return got, "publisher_search"
        best = best or (None, "matched but no lead text on page")
    return best or (None, "no candidate matched title+date")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process at most N rows")
    ap.add_argument("--only", default="", help="restrict to one publisher")
    ap.add_argument("--direct", action="store_true", help="only rows with a real publisher URL")
    args = ap.parse_args()

    df = pd.read_csv(DATA, encoding="utf-8-sig")
    cache = (pd.read_csv(CACHE, encoding="utf-8-sig").set_index("article_id")
             if CACHE.exists() else
             pd.DataFrame(columns=["article_id", "content_lead", "lead_source", "note"])
             .set_index("article_id"))

    todo = df[df["content_lead"].isna()]
    if args.only:
        todo = todo[todo["news_source"] == args.only]
    if args.direct:
        todo = todo[~todo["url"].str.contains("news.google.com", na=False)]
    todo = todo[~todo["article_id"].isin(cache.index[cache["content_lead"].notna()])]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(todo)} rows to attempt")

    hits = 0
    for n, (_, row) in enumerate(todo.iterrows(), 1):
        try:
            lead, source = recover(row)
        except Exception as e:
            lead, source = None, f"error: {type(e).__name__}"
        cache.loc[row["article_id"]] = {
            "content_lead": lead,
            "lead_source": source if lead else None,
            "note": None if lead else source,
        }
        hits += bool(lead)
        flag = "OK " if lead else "-- "
        print(f"  {flag}{n:3}/{len(todo)} [{row['news_source'][:18]:18}] "
              f"{row['title'][:52]}" + (f" :: {source}" if not lead else ""))
        if n % 10 == 0:
            cache.reset_index().to_csv(CACHE, index=False, encoding="utf-8-sig")
        time.sleep(random.uniform(1.0, 2.2))

    cache.reset_index().to_csv(CACHE, index=False, encoding="utf-8-sig")
    print(f"\nrecovered {hits}/{len(todo)} this run; "
          f"cache now holds {cache['content_lead'].notna().sum()} leads")


if __name__ == "__main__":
    main()
