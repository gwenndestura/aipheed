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
        # National broadsheets
        "inquirer.net",
        "newsinfo.inquirer.net",
        "business.inquirer.net",
        "globalnation.inquirer.net",
        "lifestyle.inquirer.net",
        "cebudailynews.inquirer.net",
        "mb.com.ph",
        "philstar.com",
        "manilatimes.net",
        # Digital-native
        "rappler.com",
        "sunstar.com.ph",
        # State broadcaster
        "ptvnews.ph",
        # Business press
        "businessmirror.com.ph",
        "bworldonline.com",
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
# CALABARZON food-related keyword filter
# ---------------------------------------------------------------------------

CALABARZON_KEYWORDS: frozenset[str] = frozenset(
    {
        # Region / province names
        "calabarzon", "batangas", "cavite", "laguna", "quezon", "rizal",
        "region iv-a", "region 4a", "region iva",
        "batangueño", "caviteño", "laguneño",

        # Food security / prices (core)
        "food price", "food insecurity", "food security", "food shortage",
        "presyo", "pagkain", "bigas", "rice price", "rice supply",
        "food inflation", "food cpi", "cpi", "inflation",
        "palengke", "grocery", "market price", "price hike", "price increase",
        "mahal", "mura", "taas ng presyo", "pagmamahal ng bilihin",
        "bilihin", "produkto", "suplay", "kakulangan ng pagkain",
        "seguridad sa pagkain", "kakulangan", "kakapusan",
        "pagkukulang", "hindi sapat na pagkain", "walang makain",

        # Agriculture / supply chain
        "harvest", "ani", "crop", "pananim", "palay", "mais",
        "vegetable", "gulay", "isda", "fish", "meat", "agriculture",
        "supply chain", "import", "smuggling", "hoarding", "food",
        "farmgate", "postharvest", "post-harvest", "crop damage", "crop loss",
        "animal feed", "fertilizer", "farm input", "abono",
        "karne", "baboy", "manok", "itlog", "gatas", "prutas",
        "kamatis", "sibuyas", "bawang", "luya", "sili",
        "mangga", "saging", "niyog", "kamoteng kahoy",
        "pagaani", "magsasaka", "bukid", "taniman", "sakahan",
        "patubig", "irigasyon", "punla", "binhi", "pesticide",
        "insecticide", "herbicide", "kemikal", "suot",
        "nagtatanim", "nag-aani", "pagkasira ng ani",
        "smuggled rice", "smuggled goods", "nakaw na bigas",

        # Market shocks — Carneiro et al. (2025)
        "market shock", "commodity price", "price spike", "price surge",
        "price volatility", "fuel price", "transport cost", "logistics cost",
        "supply disruption", "trade restriction", "export ban",
        "price control", "price ceiling", "price monitoring",
        "presyo ng gasolina", "presyo ng langis", "singil sa kuryente",
        "kuryente", "tubig", "bayarin", "gastos", "pagtaas ng gastos",
        "dagdag na bayad", "karagdagang singil",
        "kakulangan ng suplay", "walang suplay",

        # Economic / poverty
        "poverty", "hunger", "gutom", "malnutrition",
        "unemployment", "livelihood", "trabaho", "economy",
        "purchasing power", "household income", "subsistence",
        "food expenditure", "food access", "food affordability",
        "kahirapan", "hirap", "mahirap", "salat", "destitute",
        "walang trabaho", "naghihirap", "nagugutom", "pagkagutom",
        "malnutrisyon", "malnutrition", "undernutrition",
        "gutom na gutom", "pagkain ng isang beses",
        "iisang kain", "preskwela", "batang gutom",
        "pababa ng kita", "nawalan ng trabaho", "tanggal sa trabaho",
        "sahod", "sweldo", "minimum wage", "sahod na buwanang",

        # Climate triggers — Valentin et al. (2024)
        "bagyo", "baha", "tagtuyot", "typhoon", "flood", "drought",
        "el niño", "el nino", "la niña", "la nina",
        "storm surge", "landslide", "erosion", "crop failure",
        "rainfall", "dry spell", "water shortage", "irrigation",
        "climate", "weather", "calamity", "sakuna", "kalamidad",
        "pagbabago ng klima", "klimang nagbabago", "tag-ulan",
        "tag-init", "init", "ulan", "bagyo", "lindol", "pagguho",
        "pagbaha", "pagpapalakas ng ulan", "malakas na hangin",
        "signal number", "signal no", "lumikas", "evacuation",
        "naapektuhan", "nasalanta", "binaha", "naanod",
        "pagkawasak", "nasira ang ani", "nawasak na pananim",

        # Employment / remittance triggers
        "ofw", "remittance", "layoff", "retrenchment", "job loss",
        "overseas worker", "migrant worker", "minimum wage",
        "padala", "pera mula abroad", "overseas filipino",
        "nawalan ng trabaho", "tanggal sa trabaho", "retrenchment",
        "manggagawa", "empleyado", "kawani", "contractual",
        "endo", "no work no pay", "walang pasok",

        # Conflict / governance — Balashankar et al. (2023)
        "conflict", "unrest", "displacement", "evacuation", "relief",
        "humanitarian", "aid", "ayuda", "relief goods", "food aid",
        "food distribution", "food assistance", "feeding program",
        "tulong", "libreng pagkain", "libreng bigas", "ayuda sa pagkain",
        "food pack", "food packs", "relief pack", "relief packs",
        "pamimigay ng pagkain", "distribusyon ng pagkain",
        "libreng almusal", "libreng tanghalian",

        # Latent risk signals — HungerGist (2023)
        "stunting", "wasting", "underweight", "iodine", "anemia",
        "fnri", "nutrition", "nutrisyon", "supplementary feeding",
        "4ps", "pantawid", "conditional cash transfer", "cct",
        "dswd", "da ", "department of agriculture", "nfa",
        "national food authority", "da-calabarzon",
        "pagstunting", "malnourished na bata", "batang hindi lumalaki",
        "kulang sa sustansya", "sustansya", "bitamina", "mineral",
        "supplemento", "lugaw", "arroz caldo", "lugaw program",

        # Spatial bias / rural coverage — Valentin et al. (2024)
        "rural", "municipality", "barangay", "lgu", "town",
        "provincial", "farmers", "magsasaka", "fisherfolk", "mangingisda",
        "probinsya", "munisipyo", "bayan", "nayon", "baryo",
        "lalawigan", "lungsod", "kabayanan", "komunidad",
        "liblib na lugar", "malalayong lugar", "mahirap abutin",

        # Real-time crisis signals — FAO Data Lab (2025)
        "food crisis", "crisis", "krisis", "shortage", "scarcity",
        "kakulangan", "presyon", "sitwasyon",
        "lumalala", "lumala", "lumalalang sitwasyon",
        "pabigat", "nagpapalala", "patuloy na pagtaas",
        "hindi na kayang bilhin", "hindi na afford",
        "wala nang makain", "hinahanap ang pagkain",

        # General Philippine relevance
        "philippines", "pilipinas",
    }
)


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
    Return True if the article title or summary contains at least one
    CALABARZON food-related keyword (case-insensitive).
    """
    title = getattr(entry, "title", "") or ""
    summary = getattr(entry, "summary", "") or ""
    combined = (title + " " + summary).lower()
    return any(kw in combined for kw in CALABARZON_KEYWORDS)


def fetch_rss_articles(
    feed_urls: list[str],
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Fetch articles from RSS feeds and filter to credible CALABARZON food sources.

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
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

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

            # 3. Date filtering
            published_dt = _parse_published(entry)
            if published_dt:
                if not (start_dt <= published_dt <= end_dt):
                    continue
            # If no date, include (date may be filled by downstream parser)

            # 4. CALABARZON food keyword filter
            if not _matches_keywords(entry):
                continue

            title = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or ""
            source_domain = _extract_domain(link)

            records.append(
                {
                    "title": title.strip(),
                    "link": link.strip(),
                    "published": published_dt.isoformat() if published_dt else None,
                    "summary": summary.strip(),
                    "source_domain": source_domain,
                }
            )

    logger.info(
        "fetch_rss_articles: %d articles collected (%s to %s)",
        len(records), start_date, end_date,
    )
    return records