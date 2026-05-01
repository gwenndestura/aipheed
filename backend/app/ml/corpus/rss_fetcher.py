"""
app/ml/corpus/rss_fetcher.py
-----------------------------
RSS corpus ingestion from verified credible Philippine news outlets.

Fetches articles from a hardcoded CREDIBLE_DOMAINS allowlist only.
Filters to CALABARZON food-related keywords at ingest time.
Returns: title, link, published, summary, source_domain.

Usage:
    from app.ml.corpus.rss_fetcher import fetch_rss_articles, FEED_URLS
    articles = fetch_rss_articles(FEED_URLS, "2020-01-01", "2025-12-31")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import feedparser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CREDIBLE_DOMAINS allowlist — hardcoded, enforced at every ingest step.
# Only verified Philippine news outlets with established editorial standards.
# ---------------------------------------------------------------------------

CREDIBLE_DOMAINS: frozenset[str] = frozenset(
    {
        # ── National broadsheets ──────────────────────────────────────────
        "inquirer.net",
        "newsinfo.inquirer.net",
        "business.inquirer.net",
        "globalnation.inquirer.net",
        "lifestyle.inquirer.net",
        "cebudailynews.inquirer.net",
        "mb.com.ph",
        "philstar.com",
        "manilatimes.net",
        # ── Digital-native / broadcast ────────────────────────────────────
        "rappler.com",
        "sunstar.com.ph",
        "gmanetwork.com",
        "news.gmanetwork.com",
        "news.abs-cbn.com",
        "abs-cbn.com",
        "cnnphilippines.com",
        "onenews.ph",
        "interaksyon.com",
        "one.com.ph",
        # ── State broadcaster & wire ──────────────────────────────────────
        "ptvnews.ph",
        "pna.gov.ph",           # Philippine News Agency (state wire)
        "pia.gov.ph",           # Philippine Information Agency (regional gov news)
        # ── Business press ────────────────────────────────────────────────
        "businessmirror.com.ph",
        "bworldonline.com",
        # ── Government agencies — authoritative food/agri data sources ────
        "da.gov.ph",            # Department of Agriculture
        "dswd.gov.ph",          # Dept. of Social Welfare and Development
        "nfa.gov.ph",           # National Food Authority
        "psa.gov.ph",           # Philippine Statistics Authority
        "neda.gov.ph",          # National Economic Development Authority
        "doh.gov.ph",           # Dept. of Health (nutrition/malnutrition)
        "bfar.da.gov.ph",       # Bureau of Fisheries and Aquatic Resources
        "nmis.gov.ph",          # National Meat Inspection Service
        "bar.gov.ph",           # Bureau of Agricultural Research
        "openstat.psa.gov.ph",
        # ── Regional CALABARZON government news ──────────────────────────
        "calabarzon.da.gov.ph",
        "ro4a.dswd.gov.ph",
        # ── Academic / research ───────────────────────────────────────────
        "uplb.edu.ph",
        "fnri.dost.gov.ph",     # Food & Nutrition Research Institute
        "dost.gov.ph",
    }
)

FEED_URLS: list[str] = [
    # Inquirer subdomains
    "https://newsinfo.inquirer.net/feed",
    "https://business.inquirer.net/feed",
    "https://globalnation.inquirer.net/feed",
    "https://lifestyle.inquirer.net/feed",
    "https://newsinfo.inquirer.net/category/nation/feed",
    "https://cebudailynews.inquirer.net/feed",
    # Philippine Star
    "https://www.philstar.com/rss/headlines",
    "https://www.philstar.com/rss/business",
    # Manila Bulletin
    "https://mb.com.ph/rss/nation",
    "https://mb.com.ph/rss/business",
    # Manila Times
    "https://www.manilatimes.net/news/feed",
    # Rappler
    "https://www.rappler.com/feed/",
    # SunStar
    "https://www.sunstar.com.ph/feed",
    # PTV News
    "https://ptvnews.ph/feed/",
    # Business press
    "https://businessmirror.com.ph/feed",
    "https://www.bworldonline.com/feed",
]

# ---------------------------------------------------------------------------
# CALABARZON food-related keyword filter — TWO-SIGNAL DESIGN
#
# An article must match BOTH a geo signal AND a food signal.
# Requiring only one signal was the root cause of false positives:
# "philippines" alone matches every PH article; "climate" alone matches
# weather news; "trabaho" alone matches any employment story.
# ---------------------------------------------------------------------------

# Signal A — CALABARZON geographic anchor
# An article must reference the region or a specific province/city within it.
CALABARZON_GEO_SIGNALS: frozenset[str] = frozenset({
    # Region identifiers
    "calabarzon", "region iv-a", "region iva", "region 4a",
    # Province names (batangas/cavite/laguna are unambiguous in PH news)
    "batangas", "cavite", "laguna",
    # Quezon and Rizal are ambiguous (Quezon City / José Rizal) so require
    # the "province" qualifier OR a specific LGU name from those provinces
    "quezon province", "province of quezon",
    "rizal province", "province of rizal",
    # Regional government identifiers
    "da-calabarzon", "dswd calabarzon", "dswd-calabarzon",
    "ro iv-a", "nfa iv-a", "psa-calabarzon",
    # Batangas LGUs
    "batangas city", "lipa city", "tanauan", "santo tomas batangas",
    "nasugbu", "lemery", "balayan", "bauan batangas", "mabini batangas",
    "tingloy",
    "agoncillo", "alitagtag", "balete", "calaca", "calatagan",
    "cuenca", "ibaan", "laurel", "lian", "lobo", "malvar",
    "mataas na kahoy", "padre garcia", "taysan",
    # Cavite LGUs
    "bacoor", "dasmariñas", "dasmarinas", "general trias", "tagaytay",
    "imus", "imus city", "trece martires", "silang cavite", "kawit", "tanza",
    "indang", "naic", "noveleta", "amadeo", "carmona",
    "alfonso", "maragondon", "mendez", "ternate", "magallanes",
    # Laguna LGUs
    "calamba city", "santa rosa laguna", "biñan city", "binan city",
    "san pablo city laguna", "cabuyao city", "san pedro laguna",
    "los baños", "los banos", "pagsanjan", "bay laguna",
    "alaminos laguna", "calauan", "cavinti", "famy", "kalayaan",
    "liliw", "luisiana", "lumban", "mabitac", "magdalena",
    "majayjay", "nagcarlan", "paete", "pakil", "pangil",
    "pila laguna", "siniloan", "victoria laguna",
    # Quezon Province LGUs (explicitly disambiguate from Quezon City)
    "lucena city", "tayabas city", "sariaya", "gumaca quezon",
    "infanta quezon", "real quezon", "lopez quezon", "lucban quezon",
    "candelaria quezon", "agdangan", "alabat", "atimonan",
    "buenavista quezon", "burdeos", "calauag", "catanauan",
    "dolores quezon", "general nakar", "guinayangan", "jomalig",
    "macalelon", "mulanay", "panukulan", "patnanungan", "perez quezon",
    "pitogo quezon", "plaridel quezon", "polillo", "sampaloc quezon",
    "san andres quezon", "tagkawayan", "tiaong", "unisan",
    # Rizal Province LGUs (explicitly disambiguate from José Rizal)
    "antipolo city", "cainta rizal", "taytay rizal", "angono rizal",
    "binangonan rizal", "tanay rizal", "morong rizal", "rodriguez rizal",
    "baras rizal", "cardona rizal", "jala-jala", "pililla",
    "san mateo rizal", "teresa rizal",
    # CALABARZON water bodies (food supply proxies: fish kill, irrigation)
    "taal lake", "taal volcano",
    "laguna lake", "laguna de bay",
})

# Signal B — food insecurity topic anchor.
# These terms are safe as sole food signals because they are ALWAYS combined
# with a geo signal (Signal A).  A term like "rice" or "inflation" alone
# would be too broad globally, but "Batangas + rice" or "Laguna + inflation"
# is almost always a food-relevant story in Philippine news.
CALABARZON_FOOD_SIGNALS: frozenset[str] = frozenset({
    # ── Core food insecurity ─────────────────────────────────────────────
    "food insecurity", "food insecure",
    "food security",
    "food shortage", "food crisis", "food scarcity",
    "food prices", "food price",
    "food inflation",
    "food relief", "food assistance", "food aid",
    "food distribution", "food pack", "food packs",
    "food supply", "food production",
    # ── Rice / staples ───────────────────────────────────────────────────
    "rice",         # staple — safe with geo gate (Batangas + rice = rice supply/price story)
    "rice price", "rice prices", "rice supply", "rice shortage",
    "rice crisis", "rice distribution", "nfa rice", "kadiwa",
    "palay",        # unhusked rice — strongly agricultural
    "bigas",        # Filipino: rice (grain)
    "presyo ng bigas", "presyo ng pagkain", "presyo ng gulay",
    "presyo ng isda",
    "presyo",       # Filipino: price — safe with geo gate or credible domain
    "pagkain",      # Filipino: food
    "kakulangan ng pagkain",
    # ── Hunger and malnutrition ──────────────────────────────────────────
    "hunger", "hungry", "gutom",
    "malnutrition", "malnutrisyon",
    "stunting", "wasting", "underweight",
    "malnourished", "undernourished", "undernutrition",
    "feeding program", "supplementary feeding", "therapeutic feeding",
    "batang gutom", "nagugutom", "pagkagutom",
    "gutom na gutom", "walang makain", "wala nang makain",
    # ── Agricultural production and damage ───────────────────────────────
    "crop",         # safe with geo gate
    "crops",
    "harvest",      # safe with geo gate
    "ani",          # Filipino: harvest
    "crop damage", "crop loss", "crop failure",
    "harvest damage", "harvest loss", "damaged crops",
    "pinsala sa ani", "nasira ang ani", "nawasak na pananim",
    "agricultural damage", "agri damage",
    "farm", "farming",
    # ── Farmer / fisherfolk livelihoods (food production actors) ─────────
    "farmer", "farmers",
    "fisherfolk", "fisherfolks",
    "fishermen", "fisherman",
    "magsasaka",    # Filipino: farmer
    "mangingisda",  # Filipino: fisherman
    "farmgate price", "farmgate",
    "livelihood",   # fisherfolk/farmer livelihood loss = food insecurity
    "livelihoods",
    "kabuhayan",    # Filipino: livelihood
    "hanapbuhay",   # Filipino: livelihood / means of living
    # ── Aquatic / fisheries disruption ───────────────────────────────────
    "fish kill", "fish kills", "red tide", "algal bloom",
    "fishing ban",
    "bangus",       # milkfish — primary Laguna de Bay aquaculture product
    "tilapia",      # primary Laguna de Bay aquaculture product
    "aquaculture",  # fish farming (Laguna Lake, Taal Lake, Batangas coast)
    "isda",         # Filipino: fish
    # ── Government food programs (Philippines-specific) ───────────────────
    "dswd",         # Dept. of Social Welfare — central to food aid / 4Ps
    "nfa",          # National Food Authority — rice supply and distribution
    "4ps", "pantawid",
    "relief goods", "relief",
    "conditional cash transfer",
    "rice subsidy", "subsidized rice",
    "food voucher", "community pantry",
    "kadiwa ng pangulo",
    "ayuda",        # Filipino: government aid / cash transfer
    "ayuda pagkain", "libreng bigas", "libreng pagkain",
    "ayuda sa pagkain",
    "calamity",     # calamity declaration triggers food aid distribution
    "evacuees", "evacuation",
    "hatid kalinga", "ahon lahat",   # DA food security programs
    "tupad",        # DOLE emergency livelihood — affects food access
    "slp",          # Sustainable Livelihood Program
    # ── Commodity prices ─────────────────────────────────────────────────
    "inflation",    # safe with geo gate (CALABARZON inflation = CPI/food prices)
    "presyo ng",    # Filipino: "price of"
    "price hike", "price surge", "price spike", "price increase",
    "suggested retail price", "srp",
    "price monitoring",
    "vegetable prices", "fish prices", "pork prices", "chicken prices",
    "egg prices", "onion prices", "onion shortage", "onion crisis",
    "sugar shortage", "sugar prices", "sugar crisis",
    "cooking oil prices",
    "fertilizer prices", "fertilizer",
    "gulay",        # Filipino: vegetables
    "sibuyas",      # Filipino: onion
    "kamatis",      # Filipino: tomato
    "baboy",        # Filipino: pork
    "manok",        # Filipino: chicken
    "karne",        # Filipino: meat
    "asf",          # African Swine Fever — pork supply/price disruption
    # ── Food expenditure / access ─────────────────────────────────────────
    "food expenditure", "food access", "food affordability",
    "poverty incidence",
    "poverty",      # safe with geo gate (Batangas poverty = welfare/food access)
    "palengke",     # Filipino: wet market — where food prices are monitored
    "supply chain", # disruptions directly affect food availability
    # ── Climate × agriculture ─────────────────────────────────────────────
    "typhoon crop", "typhoon harvest", "typhoon damage crop",
    "flood crop", "baha ani", "baha pagkain",
    "flood",        # floods damage crops and cut food supply routes
    "bagyo",        # Filipino: typhoon
    "baha",         # Filipino: flood
    "drought rice", "drought crop",
    "drought",      # safe with geo gate (Cavite drought = water/crop shortage)
    "el nino crop", "el nino harvest", "el nino rice",
    "el niño",      # alternate spelling with ñ
    "la niña",      # alternate spelling with ñ
    "tagtuyot",     # Filipino: dry season / drought
    "tagtuyot ani",
    # ── Agricultural infrastructure ───────────────────────────────────────
    "irrigation",   # water for rice/vegetable farming
    "impoundment",  # irrigation reservoir
    "water level",  # affects irrigation and lake fishing
    # ── OFW food impact ───────────────────────────────────────────────────
    "ofw remittance food", "ofw food",
    "remittance",   # OFW remittances fund household food spending
})

# Backwards-compatible alias used by wayback_fetcher.py
CALABARZON_KEYWORDS: frozenset[str] = CALABARZON_GEO_SIGNALS | CALABARZON_FOOD_SIGNALS


def _extract_domain(url: str) -> str:
    """Return the bare domain from a URL, stripping 'www.'."""
    parsed = urlparse(url)
    return parsed.netloc.lower().lstrip("www.")


def _is_credible(url: str) -> bool:
    """Return True only if the article URL's domain is in CREDIBLE_DOMAINS."""
    domain = _extract_domain(url)
    if domain in CREDIBLE_DOMAINS:
        return True
    # Accept subdomains of any credible root domain
    # e.g. newsinfo.inquirer.net -> root is inquirer.net
    parts = domain.split(".")
    for i in range(1, len(parts)):
        root = ".".join(parts[i:])
        if root in CREDIBLE_DOMAINS:
            return True
    return False


def _parse_published(entry: feedparser.FeedParserDict) -> datetime | None:
    """Parse the published date from a feedparser entry."""
    # Try published_parsed (struct_time)
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    # Fall back to published string
    if hasattr(entry, "published") and entry.published:
        try:
            return parsedate_to_datetime(entry.published).astimezone(timezone.utc)
        except Exception:
            pass
    return None


def _matches_keywords(entry: feedparser.FeedParserDict) -> bool:
    """
    Return True only when the article has BOTH a CALABARZON geo signal AND
    a food insecurity signal.  Requiring both prevents false positives from
    generic terms like 'philippines', 'climate', or 'trabaho' that previously
    allowed basketball, politics, and tourism articles through.
    """
    title = getattr(entry, "title", "") or ""
    summary = getattr(entry, "summary", "") or ""
    combined = (title + " " + summary).lower()
    has_geo = any(kw in combined for kw in CALABARZON_GEO_SIGNALS)
    has_food = any(kw in combined for kw in CALABARZON_FOOD_SIGNALS)
    return has_geo and has_food


def fetch_rss_articles(
    feed_urls: list[str],
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Fetch articles from RSS feeds and filter to credible CALABARZON food sources.

    RSS feeds are a LIVE window — they serve the most recent 10-50 articles only.
    They cannot retrieve historical articles from the past.  When end_date is in
    the past (e.g. running a historical corpus collection today), the feeds will
    return current articles that all fail the upper date bound, producing 0 results.

    This function handles that case by:
      1. Warning the caller when end_date is already in the past.
      2. Extending the effective upper bound to today so that whatever the feed
         currently serves (near-real-time articles) is still collected.
         These articles will be tagged with their actual publish date and will be
         filtered to the model training window downstream (feature_matrix.py only
         keeps 2020-Q1 → 2025-Q4).

    For historical backfill (2020-2025), use gnews_rss_fetcher or gdelt_fetcher
    instead — they support date-parameterized queries against archived indexes.

    Parameters
    ----------
    feed_urls : list[str]
        RSS feed URLs to ingest (only CREDIBLE_DOMAINS pass validation).
    start_date : str
        ISO date string e.g. "2020-01-01"
    end_date : str
        ISO date string e.g. "2025-12-31"

    Returns
    -------
    list[dict]
        Each record:
            title         : str
            link          : str
            published     : str  — ISO UTC datetime
            summary       : str
            source_domain : str  — from CREDIBLE_DOMAINS
    """
    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end_dt   = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    now      = datetime.now(tz=timezone.utc)

    # Live RSS feeds serve current content. If the caller's end_date is already
    # in the past, the current feed content would be silently discarded because
    # every article's publish date is after end_dt. Extend the upper bound to
    # today so that near-real-time articles still pass through.
    effective_end_dt = end_dt
    if end_dt < now:
        logger.warning(
            "fetch_rss_articles: end_date '%s' is in the past (today is %s). "
            "Live RSS feeds only serve current articles — extending upper date "
            "bound to today so current feed content is not silently discarded. "
            "For historical articles (2020-2025), use gnews_rss_fetcher or "
            "gdelt_fetcher, which support date-parameterised historical queries.",
            end_date,
            now.strftime("%Y-%m-%d"),
        )
        effective_end_dt = now

    records: list[dict] = []
    seen_links: set[str] = set()

    for feed_url in feed_urls:
        logger.info("Parsing RSS feed: %s", feed_url)
        try:
            feed = feedparser.parse(feed_url)
        except Exception as exc:
            logger.error("feedparser error for %s: %s", feed_url, exc)
            continue

        for entry in feed.entries:
            link: str = getattr(entry, "link", "") or ""
            if not link:
                continue

            # 1. Domain allowlist validation — discard any non-credible source
            if not _is_credible(link):
                logger.debug("Non-credible domain, discarding: %s", link)
                continue

            # 2. URL deduplication
            if link in seen_links:
                continue
            seen_links.add(link)

            # 3. Date filtering — use effective_end_dt (today if end_date was past)
            published_dt = _parse_published(entry)
            if published_dt:
                if not (start_dt <= published_dt <= effective_end_dt):
                    continue
            # If no date, include (date may be filled by downstream parser)

            # 4. CALABARZON food keyword filter
            if not _matches_keywords(entry):
                continue

            title         = getattr(entry, "title", "") or ""
            summary       = getattr(entry, "summary", "") or ""
            source_domain = _extract_domain(link)

            records.append(
                {
                    "title":         title.strip(),
                    "link":          link.strip(),
                    "published":     published_dt.isoformat() if published_dt else None,
                    "summary":       summary.strip(),
                    "source_domain": source_domain,
                }
            )

    logger.info(
        "fetch_rss_articles: %d articles collected (%s to %s)",
        len(records), start_date,
        effective_end_dt.strftime("%Y-%m-%d"),
    )
    return records