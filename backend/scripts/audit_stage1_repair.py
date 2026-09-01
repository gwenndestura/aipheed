"""Stage 1 of the dataset audit: structural repair + hard-rule drops.

Fixes the mechanical defects in calabarzon_food_insecurity_dataset.csv before
the article-by-article relevance review (stage 2):

  * publication_date  -> ISO yyyy-mm-dd (recovers truncated RFC-822 stamps by
    solving weekday+day+month against the 2020-01-01..2026-08-21 study window)
  * news_source       -> real publisher (764 rows said "news.google.com")
  * title             -> publisher suffix stripped, replacement chars repaired
  * content_lead      -> Google-RSS HTML wrapper blanked (it is not article text)
  * drops             -> out-of-window dates, unverifiable dates, non-news
                         sources (Facebook et al.), national-scope rows

Writes the repaired frame + a per-row drop log for stage 2 to consume.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The pinned pre-audit snapshot, NOT the pipeline's own output file -- stage 2
# overwrites calabarzon_food_insecurity_dataset.csv, so reading it here would
# make a second run audit the already-audited result.
SRC = ROOT / "data/processed/audit_review/dataset_pre_audit.csv"
OUT = ROOT / "data/processed/_audit_stage1.parquet"
DROPS = ROOT / "data/processed/_audit_stage1_drops.csv"

WINDOW_LO = dt.date(2020, 1, 1)
WINDOW_HI = dt.date(2026, 8, 21)

DOW = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
MON_PREFIX = {"Ja": 1, "Fe": 2, "Mar": 3, "Ap": 4, "May": 5, "Jun": 6,
              "Jul": 7, "Au": 8, "Se": 9, "Oc": 10, "No": 11, "De": 12}
MON_FULL = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

# Aggregator/social domains that are not citable news publishers.
NON_NEWS = {"facebook.com", "youtube.com", "tiktok.com", "x.com", "twitter.com",
            "reddit.com", "scribd.com", "pinterest.com", "linkedin.com",
            "instagram.com"}

# Publisher-name normalisation: title suffix / bare domain -> canonical outlet.
PUBLISHER = {
    "inquirer.net": "Philippine Daily Inquirer",
    "newsinfo.inquirer.net": "Philippine Daily Inquirer",
    "business.inquirer.net": "Philippine Daily Inquirer",
    "opinion.inquirer.net": "Philippine Daily Inquirer",
    "globalnation.inquirer.net": "Philippine Daily Inquirer",
    "bandera.inquirer.net": "Philippine Daily Inquirer",
    "lifestyle.inq": "Philippine Daily Inquirer",
    "cebudailynews.inquirer.net": "Cebu Daily News",
    "philstar.com": "PhilStar",
    "interaksyon.philstar.com": "Interaksyon",
    "mb.com.ph": "Manila Bulletin",
    "manila bulletin": "Manila Bulletin",
    "manilatimes.net": "The Manila Times",
    "the manila times": "The Manila Times",
    "bworldonline.com": "BusinessWorld",
    "businessworld online": "BusinessWorld",
    "businessworld": "BusinessWorld",
    "rappler.com": "Rappler",
    "rappler": "Rappler",
    "gmanetwork.com": "GMA Network",
    "gma network": "GMA Network",
    "gma news online": "GMA Network",
    "news.abs-cbn.com": "ABS-CBN News",
    "abs-cbn news": "ABS-CBN News",
    "abs-cbn": "ABS-CBN News",
    "pna.gov.ph": "Philippine News Agency",
    "philippine news agency": "Philippine News Agency",
    "pia.gov.ph": "Philippine Information Agency",
    "philippine information agency": "Philippine Information Agency",
    "sunstar.com.ph": "SunStar",
    "sunstar publishing inc.": "SunStar",
    "sunstar": "SunStar",
    "tribune.net.ph": "Daily Tribune",
    "daily tribune": "Daily Tribune",
    "journal.com.ph": "Journal News Online",
    "journal news online": "Journal News Online",
    "cnnphilippines.com": "CNN Philippines",
    "ptvnews.ph": "PTV News",
    "businessmirror": "BusinessMirror",
    "businessmirror.com.ph": "BusinessMirror",
    "manila standard": "Manila Standard",
    "manilastandard.net": "Manila Standard",
    "bulatlat": "Bulatlat",
    "bulatlat.com": "Bulatlat",
    "pcij.org": "PCIJ",
    "opinyon news": "OpinYon News",
    "presidential communications office": "Presidential Communications Office",
    "dzrh": "DZRH", "manilastandard.net": "Manila Standard",
    "daily guardian": "Daily Guardian", "pco": "Presidential Communications Office",
    "interaksyon": "Interaksyon", "abs-cbn": "ABS-CBN News",
    "inquirer.net": "Philippine Daily Inquirer", "journal.com.ph": "Journal News Online",
}


MOJIBAKE = re.compile(r"Ã[\x80-\xbf]|â€|â„|Â[\xa0-\xbf]")


def demojibake(s):
    """Undo UTF-8 bytes that were decoded as CP1252 ("â€“" -> "–")."""
    if not isinstance(s, str) or not MOJIBAKE.search(s):
        return s
    try:
        fixed = s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s
    return fixed if not MOJIBAKE.search(fixed) else s


def fix_text(s):
    """Repair the U+FFFD replacement characters left by the historical fetchers."""
    if not isinstance(s, str):
        return s
    s = demojibake(s)
    for bad, good in (("Ba�os", "Baños"), ("Los Ba�", "Los Bañ"),
                      ("Pe�a", "Peña"), ("Nu�e", "Nuñe"),
                      ("Ni�o", "Niño"), ("Ni�a", "Niña"),
                      ("Se�or", "Señor")):
        s = s.replace(bad, good)
    if "�" in s:
        # Remaining stray marks stand in for punctuation (dashes, curly quotes).
        s = re.sub(r"\s*�\s*", " - ", s)
    return re.sub(r"\s+", " ", s).strip()


# Agency and programme acronyms that a URL slug flattens to Title Case.
ACRONYMS = {
    "nfa", "da", "dswd", "asf", "bfar", "pcg", "ndrrmc", "dost", "dole", "dti",
    "lgu", "lgus", "ecq", "pbbm", "ofw", "ofws", "nia", "dar", "pia", "sona",
    "dpwh", "denr", "llda", "psa", "pna", "mia", "gsis", "sss", "iirr", "pca",
    "coa", "wps", "irr", "eccd", "tupad", "aics", "cloa", "cloas", "ph", "phl",
    "bir", "boi", "boc", "smc", "owwa", "ppan", "calax", "uplb", "irri",
}
# Function words a real headline leaves lowercase; a slug-derived one capitalises.
LOWER = {"a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to",
         "for", "from", "by", "with", "as", "amid", "after", "over", "into",
         "vs", "via", "due", "up", "out", "off", "per", "sa", "ng", "na"}
MONTHS = {"january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december",
          "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
          "oct", "nov", "dec"}
# Words after which a trailing number is meaningful, not a CMS id.
DATE_GUARD = {"part", "no", "covid", "covid-19", "phase", "vol", "chapter",
              "level", "grade", "top", "batch", "round", "cycle", "sona"}
# Aggregator/section tails that are not part of the headline.
AGG_TAIL = re.compile(
    r"\s*\|\s*(?:Photos?|Videos?|GMA News Online|24 Oras|Journal Online|"
    r"News|Latest|Balitanghali)(?:\s*\|\s*[\w .&-]+)?\s*$", re.I)


def looks_slug_derived(title):
    """A headline recovered from a URL slug: every word capitalised, no
    lowercase function words, and long enough that that is not chance."""
    words = title.split()
    if len(words) < 5:
        return False
    if any(c in title for c in ",:;?'\"“”‘’"):
        return False
    alpha = [w for w in words if w[:1].isalpha()]
    return len(alpha) >= 5 and all(w[:1].isupper() for w in alpha)


def tidy_title(title):
    """Repair the mechanical damage in stored headlines."""
    s = AGG_TAIL.sub("", str(title)).strip()

    # Tokeniser artefacts: " ," -> ",", "P3 . 06" -> "P3.06", "13 , 500" -> "13,500".
    s = re.sub(r"(\d)\s*([.,])\s*(\d)", r"\1\2\3", s)
    s = re.sub(r"\s+([,;:.!?])", r"\1", s)
    # Space after a comma/colon, but never inside a number ("1,000", "12:30").
    s = re.sub(r"([,;:])(?=[^\s\d])", r"\1 ", s)

    s = re.sub(r"\bCovid[ -]?19\b", "COVID-19", s, flags=re.I)

    # Trailing CMS story id ("... Typhoon Leon 2059", "... State of Calamity 1"),
    # but never a real date, an ordinal part, or COVID-19.
    m = re.search(r"\s(\d{1,4})$", s)
    if m:
        head = s[:m.start()].split()
        prev = head[-1].lower().strip(".") if head else ""
        if prev not in MONTHS and prev not in DATE_GUARD:
            s = s[:m.start()]

    # Word truncated by the historical fetcher.
    s = re.sub(r"\bbangu\b", "bangus", s)
    s = re.sub(r",\s*Nego$", "", s)

    if looks_slug_derived(s):
        out = []
        for i, w in enumerate(s.split()):
            low = w.lower().strip(".,")
            if low in ACRONYMS:
                out.append(w.upper())
            elif i and low in LOWER:
                out.append(low)
            else:
                out.append(w)
        s = " ".join(out)
    return re.sub(r"\s+", " ", s).strip()


def _key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# Aggregator and CMS boilerplate that the fetchers captured as if it were text.
LEAD_BOILER = re.compile(
    r"^(?:Make this your preferred source.*?Google\.\s*"
    r"|This is AI-generated\..*?errors?\.\s*"
    r"|Copied\s*|Advertisement\s*|ADVERTISEMENT\s*)+", re.I)


def tidy_lead(text):
    """Strip boilerplate, stray markup and embedded media URLs from a lead."""
    if not isinstance(text, str) or not text.strip():
        return ""
    s = LEAD_BOILER.sub("", demojibake(text)).strip()
    s = re.sub(r"<[^>]+>", " ", s)
    # Embedded player/share links sometimes lead the stored description.
    s = re.sub(r"^(?:https?://\S+\s*)+", "", s)
    # Inline cross-promotion the CMS injects into the body.
    s = re.sub(r"\s*(?:READ(?:\s+MORE)?\s*:|Read more:).*$", "", s, flags=re.I)
    # Sub-editor sign-off initials ("... action. /jpv").
    s = re.sub(r"\s*/[A-Za-z]{2,4}\s*$", "", s)
    s = re.sub(r"\s+([,;:.!?])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()

    # The fetchers cut leads at a fixed length, leaving them mid-sentence. Prefer
    # trimming back to the last complete sentence; if that would throw away most
    # of the text, keep it but mark the cut so it does not read as complete.
    if s and s[-1] not in ".!?\"'”’)":
        cut = max(s.rfind(". "), s.rfind("! "), s.rfind("? "))
        if cut >= 60 and cut >= 0.5 * len(s):
            s = s[:cut + 1]
        else:
            s = s.rstrip(",;:-— ") + "…"
    return "" if len(s) < 25 else s


def split_suffix(title):
    """Google-RSS titles end in ' - <Publisher>'. Return the publisher or None.

    Split on the LAST ' - ' so hyphenated outlets ('ABS-CBN', 'CNN Philippines -
    Regions') survive; reject tails that look like sentence text rather than a
    masthead.
    """
    if not isinstance(title, str):
        return None
    sep = next((d for d in (" - ", " – ", " — ") if d in title), None)
    if sep is None:
        return None
    head, tail = title.rsplit(sep, 1)
    tail = tail.strip()
    if not head.strip() or not 2 <= len(tail) <= 45:
        return None
    # A masthead is a few words with no sentence punctuation. Dots are allowed so
    # bare domains ('Inquirer.net', 'facebook.com') still register, but a dot
    # followed by a space marks a sentence break, not a domain.
    if re.search(r"[,;:!?\"]|\. ", tail) or len(tail.split()) > 6:
        return None
    return tail


def strip_suffix(title):
    tail = split_suffix(title)
    if not tail:
        return title
    sep = next(d for d in (" - ", " – ", " — ") if d in title)
    return title.rsplit(sep, 1)[0]


def parse_date(raw):
    """Return (date, how); how in {iso, rfc, recovered, ambiguous, unparseable}."""
    s = str(raw).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        try:
            return dt.date.fromisoformat(s[:10]), "iso"
        except ValueError:
            return None, "unparseable"
    m = re.match(r"^[A-Z][a-z]{2}, (\d{1,2}) ([A-Z][a-z]{2}) (\d{4})$", s)
    if m:
        day, mon, year = int(m.group(1)), MON_FULL.get(m.group(2)), int(m.group(3))
        if mon:
            try:
                return dt.date(year, mon, day), "rfc"
            except ValueError:
                return None, "unparseable"
    # Truncated stamp, e.g. "Sun, 27 Oc" -- solve for the year inside the window.
    m = re.match(r"^([A-Z][a-z]{2}), (\d{1,2}) ([A-Z][a-z]?)$", s)
    if m:
        dow, day, prefix = m.group(1), int(m.group(2)), m.group(3)
        months = [v for k, v in MON_PREFIX.items() if k.startswith(prefix)]
        hits = set()
        for mon in months:
            for year in range(WINDOW_LO.year, WINDOW_HI.year + 1):
                try:
                    d = dt.date(year, mon, day)
                except ValueError:
                    continue
                if WINDOW_LO <= d <= WINDOW_HI and d.weekday() == DOW[dow]:
                    hits.add(d)
        if len(hits) == 1:
            return hits.pop(), "recovered"
        return None, "ambiguous" if hits else "unparseable"
    return None, "unparseable"


def main():
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    df["_row"] = range(len(df))
    df["_keep"] = True
    drops = []

    def drop(mask, reason):
        hit = mask & df["_keep"]
        for _, r in df[hit].iterrows():
            drops.append({"row": r["_row"], "article_id": r["article_id"],
                          "title": r["title"], "publication_date": r["publication_date"],
                          "province": r["province"], "news_source": r["news_source"],
                          "url": r["url"], "stage": "stage1", "drop_reason": reason})
        df.loc[hit, "_keep"] = False

    # ---- publisher: recover the real outlet, then strip the suffix off the title
    df["_suffix"] = df["title"].astype(str).map(split_suffix)
    df["_domain"] = (df["url"].astype(str)
                     .str.extract(r"https?://(?:www\.)?([^/]+)")[0].str.lower())
    df["title"] = df["title"].astype(str).map(strip_suffix).map(fix_text)
    # Flag before tidying, so the flag reflects how the headline was captured.
    df["title_provenance"] = df["title"].map(
        lambda t: "slug_derived" if looks_slug_derived(t) else "headline")
    df["title"] = df["title"].map(tidy_title)

    # Google-RSS rows carry no usable domain; the upstream corpus kept the real one.
    corpus = ROOT / "data/processed/corpus_geocoded.parquet"
    if corpus.exists():
        up = (pd.read_parquet(corpus)[["article_id", "source_domain"]]
              .dropna().drop_duplicates("article_id")
              .set_index("article_id")["source_domain"].str.lower())
        fallback = df["article_id"].map(up)
        is_google = df["_domain"].fillna("").str.contains("news.google.com")
        df.loc[is_google, "_domain"] = fallback[is_google].fillna(df.loc[is_google, "_domain"])

    # Rows collected after the corpus snapshot are not in that lookup, but their
    # own news_source column already names the outlet. Use it as a last resort.
    stored = df["news_source"].fillna("").astype(str).str.lower()
    still_google = df["_domain"].fillna("").str.contains("news.google.com")
    usable = still_google & ~stored.str.contains("google") & stored.str.len().gt(2)
    df.loc[usable, "_domain"] = stored[usable]

    def publisher(row):
        for cand in (row["_suffix"], row["_domain"]):
            if isinstance(cand, str) and cand.strip().lower() in PUBLISHER:
                return PUBLISHER[cand.strip().lower()]
        if isinstance(row["_suffix"], str) and row["_suffix"].strip():
            return fix_text(row["_suffix"].strip())
        return fix_text(str(row["_domain"]))

    df["news_source"] = df.apply(publisher, axis=1)

    # ---- drop non-news / social sources
    src_key = df["_suffix"].fillna("").str.lower().str.strip()
    drop(src_key.isin(NON_NEWS) | df["_domain"].isin(NON_NEWS),
         "non_news_source (social media post, not a news publisher)")

    # ---- dates
    parsed = df["publication_date"].map(parse_date)
    df["_date"] = [p[0] for p in parsed]
    df["date_provenance"] = [p[1] for p in parsed]
    drop(df["date_provenance"].isin(["ambiguous", "unparseable"]),
         "unverifiable_publication_date (truncated stamp, year not recoverable)")
    drop(df["_date"].map(lambda d: d is not None and not (WINDOW_LO <= d <= WINDOW_HI)),
         "outside_study_period (2020-01-01..2026-08-21)")
    df["publication_date"] = df["_date"].map(lambda d: d.isoformat() if d else None)

    # ---- content_lead: the Google-RSS wrapper is markup, not article text
    lead = df["content_lead"].fillna("").astype(str)
    df["content_lead"] = lead.where(~lead.str.contains("<a href=", regex=False), "").map(fix_text)
    df["content_lead"] = df["content_lead"].map(tidy_lead)
    # A "lead" that only restates the headline carries no extra information.
    def restates_title(lead, title):
        a, b = _key(lead), _key(title)
        return bool(a) and bool(b) and (a.startswith(b[:40]) or b.startswith(a[:40]))

    same = [i for i, r in df.iterrows()
            if isinstance(r["content_lead"], str) and r["content_lead"]
            and restates_title(r["content_lead"], r["title"])]
    df.loc[same, "content_lead"] = ""
    df.loc[df["content_lead"].eq(""), "content_lead"] = None

    # ---- geography. National-scope rows are kept deliberately: they are the
    # nationwide drivers (fisheries policy, monsoon damage, ECCD) that act on
    # CALABARZON, and geographic_scope keeps them separable from the
    # province/LGU rows for province-level aggregation. Any OTHER row without a
    # province is a geocoding failure and is dropped.
    drop(df["province"].isna() & df["geographic_scope"].ne("national"),
         "missing_province (no CALABARZON location resolved)")

    pd.DataFrame(drops).to_csv(DROPS, index=False, encoding="utf-8")
    keep = df[df["_keep"]].drop(columns=["_keep", "_suffix", "_domain", "_date"])
    keep.to_parquet(OUT, index=False)

    print(f"in      : {len(df)}")
    print(f"dropped : {len(drops)}")
    if drops:
        for reason, n in pd.DataFrame(drops)["drop_reason"].value_counts().items():
            print(f"   {n:5d}  {reason}")
    print(f"retained: {len(keep)}")
    print("\ndate provenance:", keep["date_provenance"].value_counts().to_dict())
    print(f"publishers: {keep['news_source'].nunique()}")
    print(keep["news_source"].value_counts().head(15).to_string())
    print(f"\nrows with real content_lead: {keep['content_lead'].notna().sum()}")


if __name__ == "__main__":
    main()
