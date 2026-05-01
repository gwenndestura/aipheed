"""
app/ml/corpus/gnews_rss_fetcher.py
------------------------------------
Google News RSS historical fetcher for CALABARZON food insecurity corpus.

WHY QUARTERLY WINDOWS
---------------------
Google News RSS caps results at ~100 per query per date window. With yearly
windows (6 years × 79 queries = 474 requests) we retrieved 4,384 articles.
Switching to quarterly windows (24 quarters × 165 queries = 3,960 requests)
surfaces the long tail — articles that were "pushed out" by more recent ones
in the yearly window. PNA alone publishes ~5-10 food articles/day (18,000+
over 6 years); we were getting only the top-100 per year per query.

Expected yield: 20,000–35,000 unique credible articles after quarterly expansion
+ expanded query bank (165 queries vs previous 79).

QUERY BANK STRUCTURE
--------------------
1. National food/price queries (English)
2. CALABARZON-specific queries
3. Climate / typhoon / El Niño
4. Filipino/Tagalog language queries (surfaces non-English articles)
5. COVID-19 food impact (2020–2021)
6. Specific typhoon events (Rolly, Ulysses, Odette, Karding, Paeng, etc.)
7. Specific commodity crises (onion, sugar, cooking oil, fertilizer)
8. Government programs (Kadiwa, Ayuda, 4Ps, SLP, DSWD)
9. Poverty/welfare/malnutrition
10. Agriculture/supply chain
11. Official news sources
12. HungerGist 8-topic aligned queries (T1–T8 + T1b + T9)
13. CALABARZON city/municipality level

Usage:
    from app.ml.corpus.gnews_rss_fetcher import fetch_gnews_rss_articles
    articles = fetch_gnews_rss_articles("2020-01-01", "2025-12-31")
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from urllib.parse import quote, urlparse

import feedparser

from app.ml.corpus.rss_fetcher import CALABARZON_FOOD_SIGNALS, CREDIBLE_DOMAINS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GNEWS_RSS_BASE = "https://news.google.com/rss/search"

# Polite delay between Google News requests (seconds)
GNEWS_CRAWL_DELAY = 1.0

# ---------------------------------------------------------------------------
# 1. National food/price queries (English)
# ---------------------------------------------------------------------------

_NATIONAL_FOOD: list[str] = [
    "rice price Philippines",
    "food prices Philippines market",
    "food inflation Philippines CPI",
    "Philippines NFA rice supply",
    "rice smuggling Philippines import",
    "food shortage Philippines supply",
    "Philippines food security crisis",
    "food inflation Philippines PSA",
    "Philippines farmgate price palay",
    "food price monitoring Philippines DTI",
    "Philippines rice retail price",
    "fish price Philippines market",
    "vegetable prices Philippines market",
    "Philippines pork price increase",
    "Philippines chicken price market",
    "Philippines egg price inflation",
]

# ---------------------------------------------------------------------------
# 2. CALABARZON regional queries
# ---------------------------------------------------------------------------

_CALABARZON_FOOD: list[str] = [
    "food prices CALABARZON Philippines",
    "CALABARZON food insecurity",
    "CALABARZON food security rice",
    "CALABARZON agriculture harvest",
    "Batangas food prices rice",
    "Cavite food prices market",
    "Laguna food prices rice",
    "Quezon Province food prices",
    "Rizal food prices market",
    "CALABARZON malnutrition stunting",
    "CALABARZON NFA rice distribution",
    "CALABARZON DSWD food relief",
    "CALABARZON 4Ps food beneficiaries",
    "CALABARZON hunger poverty relief",
    "Region IVA food Philippines",
    "CALABARZON food assistance calamity",
]

# ---------------------------------------------------------------------------
# 3. Climate / typhoon / El Niño (generic)
# ---------------------------------------------------------------------------

_CLIMATE_AGRI: list[str] = [
    "typhoon flood crop damage Philippines harvest",
    "El Nino Philippines drought agriculture",
    "La Nina Philippines flood harvest",
    "Philippines typhoon agriculture damage",
    "flood damage crops Philippines province",
    "Philippines calamity food relief",
    "crop loss Philippines typhoon",
    "drought rice production Philippines",
    "PAGASA El Nino Philippines drought rice",
    "La Nina flood Philippines palay harvest",
    "Philippines storm surge food damage",
]

# ---------------------------------------------------------------------------
# 4. Filipino/Tagalog language queries
# Surfaces articles written in Filipino not captured by English queries.
# PNA, regional outlets, and municipal government releases often use Filipino.
# ---------------------------------------------------------------------------

_TAGALOG_QUERIES: list[str] = [
    "presyo ng bigas Pilipinas",
    "tagtuyot ani Pilipinas",
    "bagyo pinsala ani CALABARZON",
    "gutom kahirapan Pilipinas",
    "presyo ng pagkain merkado Pilipinas",
    "kakulangan ng pagkain Pilipinas",
    "pagbaha pinsala sa palay",
    "El Nino Pilipinas ani",
    "kalamidad tulong pagkain",
    "palayan presyo magsasaka",
    "tulong pagkain mahihirap",
    "suplay ng bigas Pilipinas",
    "presyo ng gulay isda",
    "sakahan CALABARZON ani",
    "baha CALABARZON pagkain",
    "presyo ng isda Laguna Batangas",
    "gutom Pilipinas pamilya",
    "pagtulong ng gobyerno pagkain",
    "kalamidad pagkain tulong gobyerno",
    "ayuda pagkain mahihirap",
]

# ---------------------------------------------------------------------------
# 5. COVID-19 food impact (2020–2021)
# COVID lockdowns caused severe food access disruption in CALABARZON.
# ECQ/MECQ/GCQ periods cut supply chains and left millions food-insecure.
# ---------------------------------------------------------------------------

_COVID_FOOD: list[str] = [
    "COVID Philippines food poverty hunger",
    "COVID lockdown food access Philippines",
    "COVID CALABARZON food supply shortage",
    "pandemic food assistance Philippines DSWD",
    "ECQ food relief Philippines 2020",
    "COVID unemployment food poverty Philippines",
    "lockdown food supply chain Philippines",
    "quarantine food prices Philippines",
    "COVID ayuda food poor Philippines",
    "pandemic hunger Philippines families",
    "ECQ MECQ food assistance CALABARZON",
    "COVID food bank community pantry Philippines",
    "community pantry Philippines 2021 food",
]

# ---------------------------------------------------------------------------
# 6. Specific major typhoon events affecting CALABARZON
# Each typhoon generated hundreds of news articles; querying by name
# surfaces the full coverage not captured by generic typhoon queries.
# ---------------------------------------------------------------------------

_TYPHOON_EVENTS: list[str] = [
    # 2020 — Super Typhoon Rolly (Category 5, hit Quezon/Batangas)
    "Typhoon Rolly Philippines food damage Quezon",
    "Super Typhoon Rolly CALABARZON agriculture",
    # 2020 — Typhoon Ulysses (flooded Rizal/Laguna/CALABARZON severely)
    "Typhoon Ulysses Philippines flood food",
    "Typhoon Ulysses CALABARZON Rizal Laguna food",
    # 2021 — Typhoon Odette
    "Typhoon Odette Philippines food damage 2021",
    # 2022 — Typhoon Karding
    "Typhoon Karding Philippines food agriculture 2022",
    # 2022 — Typhoon Paeng
    "Typhoon Paeng Philippines food relief 2022",
    # 2023 — Typhoon Egay / Falcon
    "Typhoon Egay Philippines food agriculture 2023",
    # 2024 — Typhoon Carina
    "Typhoon Carina Philippines flood food 2024",
    "Typhoon Carina CALABARZON food relief",
    # 2024 — Typhoon Kristine
    "Typhoon Kristine Philippines food 2024",
    # Generic CALABARZON typhoon food
    "typhoon CALABARZON food relief evacuation",
    "typhoon Quezon Batangas Laguna food damage",
]

# ---------------------------------------------------------------------------
# 7. Specific commodity crises (Philippines-specific price spikes)
# ---------------------------------------------------------------------------

_COMMODITY_CRISES: list[str] = [
    # Onion crisis 2022–2023 (onions cost more than meat in PH)
    "onion prices Philippines crisis 2022",
    "onion shortage Philippines expensive",
    "sibuyas presyo Pilipinas",
    # Sugar shortage 2022
    "sugar shortage Philippines 2022",
    "sugar prices Philippines crisis SRA",
    # Cooking oil
    "cooking oil prices Philippines inflation",
    # Fertilizer prices post-Ukraine war
    "fertilizer prices Philippines farmers rice",
    "urea fertilizer Philippines agriculture",
    # Salt
    "salt shortage Philippines asin",
    # LPG / energy affecting food cooking
    "LPG prices Philippines food cooking poor",
    # Pork industry ASF (African Swine Fever)
    "ASF African swine fever Philippines pork food",
    "pork shortage Philippines ASF prices",
    # Egg and poultry
    "poultry prices Philippines egg chicken",
    # Imported food items
    "rice import Philippines prices tariff",
]

# ---------------------------------------------------------------------------
# 8. Government programs and food assistance
# ---------------------------------------------------------------------------

_GOV_PROGRAMS: list[str] = [
    # Kadiwa price monitoring centers
    "Kadiwa Philippines food prices NFA",
    "Kadiwa ng Pangulo Philippines rice",
    # Ayuda / cash transfers
    "Ayuda Philippines food poor DSWD",
    "cash ayuda distribution Philippines food",
    # 4Ps/Pantawid
    "4Ps Pantawid Philippines food beneficiaries",
    "Pantawid pamilya food poor Philippines",
    # Sustainable Livelihood Program
    "SLP Philippines livelihood food DSWD",
    # Hatid Kalinga
    "Hatid Kalinga Philippines food relief",
    # DOLE Tupad
    "TUPAD Philippines livelihood food workers",
    # Rice subsidy
    "rice subsidy Philippines poor NFA",
    "subsidized rice Philippines distribution",
    # Food stamps / voucher
    "food voucher Philippines poor",
    # Cash for work
    "cash for work Philippines calamity food",
    # Sustainable agriculture program
    "Philippines DA Ahon Lahat Pagkaing Sapat",
]

# ---------------------------------------------------------------------------
# 9. Poverty / welfare / malnutrition (expanded)
# ---------------------------------------------------------------------------

_POVERTY_WELFARE: list[str] = [
    "hunger poverty Philippines DSWD",
    "Philippines malnutrition children stunting",
    "4Ps Pantawid food Philippines beneficiaries",
    "Philippines food poverty household income",
    "Philippines feeding program nutrition schools",
    "Philippines community pantry food bank",
    "OFW remittance Philippines food economy",
    "unemployment livelihood Philippines food",
    "minimum wage food prices Philippines",
    "Philippines subsistence poverty food",
    "CALABARZON malnutrition children 2020 2021",
    "Philippines FNRI nutrition survey food",
    "Philippines SWS hunger survey",
    "self-rated hunger Philippines SWS",
    "Philippines involuntary hunger survey",
]

# ---------------------------------------------------------------------------
# 10. Agriculture / supply chain (expanded)
# ---------------------------------------------------------------------------

_AGRICULTURE_SUPPLY: list[str] = [
    "Philippines DA Department Agriculture crop",
    "palay harvest Philippines production yield",
    "vegetables fish prices Philippines market",
    "Philippines agriculture calamity damage report",
    "Philippines food supply chain logistics",
    "Philippines pork chicken egg prices",
    "Philippines rice tariffication law RA 11203",
    "rice import quota Philippines DA",
    "Philippines corn production harvest",
    "Philippines coconut production Quezon",
    "Philippines banana export food security",
    "post-harvest loss Philippines agriculture",
    "Philippines cold chain food distribution",
    "Philippines irrigation rice water shortage",
    "Philippines DA crop damage assessment",
]

# ---------------------------------------------------------------------------
# 11. Official news sources (targeted queries)
# ---------------------------------------------------------------------------

_OFFICIAL_NEWS: list[str] = [
    "Philippine News Agency food prices",
    "PNA rice supply Philippines province",
    "Philippines agriculture department food report",
    "PSA Philippines food poverty statistics",
    "NEDA Philippines food inflation economic",
    "Philippine Information Agency food prices",
    "PIA CALABARZON food relief",
    "DSWD Philippines food assistance report",
    "DA Philippines rice supply report",
    "NFA Philippines rice distribution 2024",
]

# ---------------------------------------------------------------------------
# 12. HungerGist 8-topic aligned queries (Ahn et al., 2023)
# ---------------------------------------------------------------------------

_HUNGERGIST_HEALTH_CIVIL: list[str] = [
    # T2 — Healthcare as high-crisis predictor
    "Philippines delayed social assistance DSWD poor families",
    "Philippines malnutrition hospitalized children CALABARZON",
    "Philippines social welfare cut hunger families",
    "Philippines therapeutic feeding malnutrition program",
    "delayed 4Ps Pantawid Philippines poor",
    # T7 — Civil Life / Displacement
    "Philippines displaced evacuees food calamity",
    "Philippines families hunger misery poverty crisis",
    "CALABARZON informal settlers hunger food",
    "Philippines Taal eruption evacuees food Batangas",
]

_HUNGERGIST_GOVERNANCE_FINANCE: list[str] = [
    # T3 — Governance / Policy
    "Philippines rice price control executive order NFA",
    "Philippines food security policy reform agriculture",
    "Philippines rice import smuggling policy crackdown",
    # T4 — Finance / Economic Development
    "Philippines agriculture budget subsidy DA farmers",
    "Philippines crop insurance palay rice PCIC",
    "Philippines DSWD social protection budget allocation",
    "Philippines livelihood program DOLE food poverty",
    # T5 — Regional Development
    "Philippines farm to market road CALABARZON DA",
    "Philippines cold storage post-harvest rice CALABARZON",
    "Philippines irrigation project CALABARZON DA rice",
]

_HUNGERGIST_LAND_FISH: list[str] = [
    # T6 — Land Use / Agricultural Land Loss
    "Philippines agricultural land conversion CALABARZON",
    "Philippines farmland lost urban development food",
    "CALABARZON farmland housing subdivision food",
    # T1b — Fish Kill (coastal food supply disruption)
    "Philippines fish kill red tide food supply",
    "Laguna Lake fish kill food fishermen CALABARZON",
    "Taal Lake fish kill Batangas food supply",
    "Philippines algal bloom fishing ban livelihood",
    "Laguna de Bay fish kill 2020 2021 2022",
]

# T9 — OFW Remittance Shock (Philippines-specific extension)
_PH_OFW_REMITTANCE: list[str] = [
    "OFW remittance Philippines food poverty",
    "overseas Filipino workers job loss food",
    "OFW deployment ban Philippines economy food",
    "remittance decline Philippines household food",
    "CALABARZON OFW families food poverty",
    "OFW repatriation Philippines food 2020",
]

# ---------------------------------------------------------------------------
# 13. CALABARZON city / municipality level (expanded)
# ---------------------------------------------------------------------------

_CALABARZON_CITIES: list[str] = [
    # Batangas
    "Batangas City food prices market",
    "Lipa City food prices Batangas",
    "Tanauan food prices Batangas",
    "Santo Tomas Batangas food market",
    "Batangas Taal Volcano food evacuees",
    # Cavite
    "Bacoor food prices market Cavite",
    "Dasmarinas food prices Cavite",
    "General Trias food Cavite",
    "Imus Cavite food prices",
    "Tagaytay food market prices Cavite",
    "Cavite City food poverty",
    "Trece Martires Cavite food prices",
    # Laguna
    "Calamba food prices market Laguna",
    "Santa Rosa Laguna food prices",
    "Binan food market Laguna",
    "San Pablo food prices Laguna",
    "Cabuyao food market Laguna",
    "San Pedro Laguna food prices",
    "Los Banos Laguna food prices",
    "Pagsanjan Laguna food poverty",
    # Quezon Province
    "Lucena City food prices market",
    "Tayabas food prices Quezon",
    "Sariaya Quezon food poverty",
    "Infanta Quezon food fishing",
    "Gumaca Quezon food typhoon",
    "Real Quezon Province food fishing",
    "Lopez Quezon food poverty",
    # Rizal
    "Antipolo food prices market Rizal",
    "Cainta food prices Rizal",
    "Taytay Rizal food market",
    "Angono food Rizal poverty",
    "Binangonan food fishing Rizal",
    "Tanay Rizal food poverty agriculture",
    "Morong Rizal food prices",
]

# ---------------------------------------------------------------------------
# T8 — Social Instability (HungerGist)
# ---------------------------------------------------------------------------

_HUNGERGIST_SOCIAL: list[str] = [
    "Philippines protest food prices rally",
    "Philippines strike workers food inflation",
    "Philippines social unrest food prices poor",
    "rally vs food prices Philippines CALABARZON",
]

# ---------------------------------------------------------------------------
# 14. Domain-targeted site: queries — mines each credible domain directly.
#
# Google News RSS supports the site: operator: "site:pna.gov.ph food prices"
# returns only articles from pna.gov.ph that match "food prices".  This
# exhaustively covers domains that generic keyword queries miss — especially
# government agency sites (da.gov.ph, dswd.gov.ph, nfa.gov.ph, psa.gov.ph,
# neda.gov.ph) that contributed 0 articles in previous runs.
# ---------------------------------------------------------------------------

# Philippine News Agency — state wire, 5–10 food articles/day
_SITE_PNA: list[str] = [
    "site:pna.gov.ph food prices Philippines",
    "site:pna.gov.ph rice supply CALABARZON",
    "site:pna.gov.ph agriculture harvest Philippines",
    "site:pna.gov.ph typhoon flood food",
    "site:pna.gov.ph food insecurity poverty",
    "site:pna.gov.ph DSWD food assistance",
    "site:pna.gov.ph NFA rice distribution",
    "site:pna.gov.ph malnutrition hunger Philippines",
    "site:pna.gov.ph calamity food relief",
    "site:pna.gov.ph CALABARZON food",
]

# Philippine Information Agency — regional government news
_SITE_PIA: list[str] = [
    "site:pia.gov.ph food prices CALABARZON",
    "site:pia.gov.ph rice supply Philippines",
    "site:pia.gov.ph agriculture CALABARZON harvest",
    "site:pia.gov.ph food assistance typhoon",
    "site:pia.gov.ph DSWD relief goods",
    "site:pia.gov.ph malnutrition poverty region",
]

# Department of Agriculture — crop damage, price reports, supply bulletins
_SITE_DA: list[str] = [
    "site:da.gov.ph rice production Philippines",
    "site:da.gov.ph food supply Philippines",
    "site:da.gov.ph CALABARZON agriculture",
    "site:da.gov.ph crop damage typhoon",
    "site:da.gov.ph palay harvest Philippines",
    "site:da.gov.ph food prices farmgate",
    "site:da.gov.ph agriculture calamity damage",
]

# DSWD — social welfare, food assistance, 4Ps reports
_SITE_DSWD: list[str] = [
    "site:dswd.gov.ph food assistance Philippines",
    "site:dswd.gov.ph CALABARZON relief goods",
    "site:dswd.gov.ph 4Ps beneficiaries food",
    "site:dswd.gov.ph calamity response food",
    "site:dswd.gov.ph hunger poverty Philippines",
    "site:dswd.gov.ph typhoon food relief",
]

# NFA — rice supply, distribution, price monitoring
_SITE_NFA: list[str] = [
    "site:nfa.gov.ph rice supply Philippines",
    "site:nfa.gov.ph rice distribution CALABARZON",
    "site:nfa.gov.ph rice price monitoring",
    "site:nfa.gov.ph palay procurement Philippines",
]

# PSA — food CPI, poverty statistics, LFS
_SITE_PSA: list[str] = [
    "site:psa.gov.ph food inflation Philippines",
    "site:psa.gov.ph poverty statistics CALABARZON",
    "site:psa.gov.ph rice price survey",
    "site:psa.gov.ph food CPI Philippines",
    "site:psa.gov.ph hunger poverty incidence",
]

# NEDA — food security outlook, economic food impact
_SITE_NEDA: list[str] = [
    "site:neda.gov.ph food security Philippines",
    "site:neda.gov.ph food inflation economic",
    "site:neda.gov.ph agriculture development Philippines",
    "site:neda.gov.ph poverty food Philippines",
]

# Philippine Daily Inquirer — largest broadsheet
_SITE_INQUIRER: list[str] = [
    "site:inquirer.net food prices Philippines",
    "site:inquirer.net CALABARZON food insecurity",
    "site:inquirer.net rice prices Philippines",
    "site:inquirer.net typhoon food damage Philippines",
    "site:inquirer.net hunger poverty Philippines",
    "site:inquirer.net food inflation Philippines",
    "site:inquirer.net agriculture Philippines harvest",
]

# Philippine Star
_SITE_PHILSTAR: list[str] = [
    "site:philstar.com food prices Philippines",
    "site:philstar.com CALABARZON food",
    "site:philstar.com rice prices Philippines",
    "site:philstar.com typhoon food damage",
    "site:philstar.com food inflation Philippines",
    "site:philstar.com poverty hunger Philippines",
]

# GMA Network News — largest PH TV network
_SITE_GMA: list[str] = [
    "site:gmanetwork.com food prices Philippines",
    "site:gmanetwork.com CALABARZON food",
    "site:gmanetwork.com rice prices Philippines",
    "site:gmanetwork.com typhoon food damage",
    "site:gmanetwork.com food insecurity Philippines",
    "site:gmanetwork.com hunger poverty Philippines",
]

# Rappler — digital investigative news
_SITE_RAPPLER: list[str] = [
    "site:rappler.com food insecurity Philippines",
    "site:rappler.com food prices Philippines",
    "site:rappler.com typhoon food Philippines",
    "site:rappler.com poverty hunger Philippines",
    "site:rappler.com CALABARZON food",
    "site:rappler.com agriculture Philippines",
]

# Manila Bulletin
_SITE_MB: list[str] = [
    "site:mb.com.ph food prices Philippines",
    "site:mb.com.ph CALABARZON food",
    "site:mb.com.ph rice prices Philippines",
    "site:mb.com.ph agriculture Philippines",
    "site:mb.com.ph food inflation Philippines",
]

# Manila Times
_SITE_MANILATIMES: list[str] = [
    "site:manilatimes.net food prices Philippines",
    "site:manilatimes.net rice prices Philippines",
    "site:manilatimes.net food inflation Philippines",
    "site:manilatimes.net CALABARZON agriculture",
]

# ABS-CBN News
_SITE_ABSCBN: list[str] = [
    "site:news.abs-cbn.com food prices Philippines",
    "site:news.abs-cbn.com CALABARZON food",
    "site:news.abs-cbn.com typhoon food Philippines",
    "site:news.abs-cbn.com hunger poverty Philippines",
]

# PTV News (state broadcaster)
_SITE_PTV: list[str] = [
    "site:ptvnews.ph food prices Philippines",
    "site:ptvnews.ph rice supply Philippines",
    "site:ptvnews.ph CALABARZON food assistance",
]

# BusinessWorld — agribusiness, economic food coverage
_SITE_BWO: list[str] = [
    "site:bworldonline.com food prices Philippines",
    "site:bworldonline.com agriculture Philippines",
    "site:bworldonline.com food inflation Philippines",
    "site:bworldonline.com rice Philippines",
]

# BusinessMirror
_SITE_BM: list[str] = [
    "site:businessmirror.com.ph food prices Philippines",
    "site:businessmirror.com.ph agriculture Philippines",
    "site:businessmirror.com.ph rice Philippines",
]

# SunStar (multi-city, covers CALABARZON cities)
_SITE_SUNSTAR: list[str] = [
    "site:sunstar.com.ph food prices CALABARZON",
    "site:sunstar.com.ph agriculture Philippines",
    "site:sunstar.com.ph food poverty Philippines",
]

# Daily Tribune
_SITE_TRIBUNE: list[str] = [
    "site:tribune.net.ph food prices Philippines",
    "site:tribune.net.ph rice prices Philippines",
    "site:tribune.net.ph food inflation Philippines",
]

# Interaksyon (TV5)
_SITE_INTERAKSYON: list[str] = [
    "site:interaksyon.philstar.com food Philippines",
    "site:interaksyon.com food prices Philippines",
]

# VERA Files (investigative / fact-check)
_SITE_VERAFILES: list[str] = [
    "site:verafiles.org food Philippines",
    "site:verafiles.org agriculture poverty Philippines",
]

# PSA RSSO IV-A (CALABARZON regional statistical office — press releases)
# rsso04a.psa.gov.ph publishes province-specific inflation reports,
# poverty statistics, and LFS releases for CALABARZON. These are the
# primary statistical source for food CPI, rice prices, and poverty data.
# Direct access is Cloudflare-blocked; these site: queries capture what
# Google has indexed (~90 press release pages).
_SITE_RSSO4A: list[str] = [
    "site:rsso04a.psa.gov.ph inflation CALABARZON",
    "site:rsso04a.psa.gov.ph poverty statistics",
    "site:rsso04a.psa.gov.ph food price situation",
    "site:rsso04a.psa.gov.ph labor force survey",
    "site:rsso04a.psa.gov.ph price situation Batangas",
    "site:rsso04a.psa.gov.ph price situation Cavite",
    "site:rsso04a.psa.gov.ph price situation Laguna",
    "site:rsso04a.psa.gov.ph price situation Quezon",
    "site:rsso04a.psa.gov.ph price situation Rizal",
    "site:rsso04a.psa.gov.ph summary inflation",
    "site:rsso04a.psa.gov.ph CALABARZON statistics",
]

# Consolidated domain-targeted list
_DOMAIN_TARGETED: list[str] = (
    _SITE_PNA
    + _SITE_PIA
    + _SITE_DA
    + _SITE_DSWD
    + _SITE_NFA
    + _SITE_PSA
    + _SITE_NEDA
    + _SITE_INQUIRER
    + _SITE_PHILSTAR
    + _SITE_GMA
    + _SITE_RAPPLER
    + _SITE_MB
    + _SITE_MANILATIMES
    + _SITE_ABSCBN
    + _SITE_PTV
    + _SITE_BWO
    + _SITE_BM
    + _SITE_SUNSTAR
    + _SITE_TRIBUNE
    + _SITE_INTERAKSYON
    + _SITE_VERAFILES
    + _SITE_RSSO4A
)

# ---------------------------------------------------------------------------
# Master query list — all groups combined
# ---------------------------------------------------------------------------

GNEWS_RSS_QUERIES: list[str] = (
    _NATIONAL_FOOD
    + _CALABARZON_FOOD
    + _CLIMATE_AGRI
    + _TAGALOG_QUERIES
    + _COVID_FOOD
    + _TYPHOON_EVENTS
    + _COMMODITY_CRISES
    + _GOV_PROGRAMS
    + _POVERTY_WELFARE
    + _AGRICULTURE_SUPPLY
    + _OFFICIAL_NEWS
    + _HUNGERGIST_HEALTH_CIVIL
    + _HUNGERGIST_GOVERNANCE_FINANCE
    + _HUNGERGIST_LAND_FISH
    + _PH_OFW_REMITTANCE
    + _CALABARZON_CITIES
    + _HUNGERGIST_SOCIAL
    + _DOMAIN_TARGETED        # site:-specific queries for all 26 credible domains
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_domain(href: str) -> str:
    """Extract bare domain from a URL (strips www.)."""
    parsed = urlparse(href)
    return parsed.netloc.lower().lstrip("www.")


def _is_credible(source_href: str) -> bool:
    """Return True if source_href domain is in or a subdomain of CREDIBLE_DOMAINS."""
    domain = _extract_domain(source_href)
    if domain in CREDIBLE_DOMAINS:
        return True
    parts = domain.split(".")
    for i in range(1, len(parts)):
        root = ".".join(parts[i:])
        if root in CREDIBLE_DOMAINS:
            return True
    return False


def _quarter_windows(start_date: str, end_date: str) -> list[tuple[str, str, int, int]]:
    """
    Return list of (after_date, before_date, year, quarter) tuples,
    one per calendar quarter between start_date and end_date.

    Using quarterly windows instead of yearly means each query fetches
    ~100 articles from a 3-month window rather than a 12-month window,
    surfacing the long tail of articles that yearly windows miss.
    """
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    windows: list[tuple[str, str, int, int]] = []

    year = start.year
    # Start at the quarter containing start_date
    q = (start.month - 1) // 3 + 1

    while True:
        # Quarter boundaries
        q_month_start = (q - 1) * 3 + 1
        q_month_end = q * 3

        after = f"{year}-{q_month_start:02d}-01"
        if q < 4:
            before = f"{year}-{q_month_end + 1:02d}-01"
        else:
            before = f"{year + 1}-01-01"

        # Stop if the quarter starts after end_date
        after_dt = datetime.fromisoformat(after)
        if after_dt > end:
            break

        windows.append((after, before, year, q))

        q += 1
        if q > 4:
            q = 1
            year += 1

    return windows


def _build_url(query: str, after: str, before: str) -> str:
    """Build a date-windowed Google News RSS search URL for a quarter."""
    date_filter = f"+before:{before}+after:{after}"
    params = f"q={quote(query)}{date_filter}&hl=en-PH&gl=PH&ceid=PH:en"
    return f"{GNEWS_RSS_BASE}?{params}"


def _parse_entry(entry) -> dict | None:
    """
    Extract a clean record from a feedparser entry.
    Returns None if source domain is not credible or title lacks a food signal.
    """
    source_href: str = entry.get("source", {}).get("href", "")
    if not source_href or not _is_credible(source_href):
        return None

    title: str = (entry.get("title") or "").strip()
    # Strip publisher suffix added by Google News: "Title - Publisher"
    if " - " in title:
        title = title.rsplit(" - ", 1)[0].strip()

    # Reject articles whose title contains no food insecurity signal.
    # Google News sometimes returns tangentially related results for targeted
    # food queries (e.g. a query for "rice prices Philippines" returns a
    # restaurant review that mentions rice).  Checking the title here is
    # cheap and eliminates the most obvious false positives before storage.
    title_lower = title.lower()
    if not any(kw in title_lower for kw in CALABARZON_FOOD_SIGNALS):
        return None

    link: str = entry.get("link", "") or ""
    article_id: str = entry.get("id", link)  # stable Google News article ID

    # Published date
    published_iso: str = ""
    if entry.get("published_parsed"):
        try:
            dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            published_iso = dt.isoformat()
        except Exception:
            pass
    if not published_iso and entry.get("published"):
        published_iso = entry["published"]

    source_domain = _extract_domain(source_href)

    return {
        "title": title,
        "link": link,
        "article_id": article_id,   # dedup key — stable Google News ID
        "published": published_iso,
        "summary": "",
        "source_domain": source_domain,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_gnews_rss_articles(
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Fetch CALABARZON food-insecurity articles from Google News RSS (historical).

    Uses QUARTERLY date windows (not yearly) so each query fetches ~100 articles
    from a 3-month window — surfacing the long tail that yearly windows miss.
    Google News RSS supports `before:/after:` date operators for full historical
    coverage back to 2015+.

    Parameters
    ----------
    start_date : str   — ISO date e.g. "2020-01-01"
    end_date   : str   — ISO date e.g. "2025-12-31"

    Returns
    -------
    list[dict]
        Each record:
            title         : str
            link          : str   — Google News URL (stable, unique per article)
            article_id    : str   — Google News article ID (dedup key)
            published     : str   — ISO UTC datetime
            summary       : str   — empty (not available in RSS)
            source_domain : str   — original publisher domain from CREDIBLE_DOMAINS
    """
    windows = _quarter_windows(start_date, end_date)
    total_requests = len(GNEWS_RSS_QUERIES) * len(windows)

    logger.info(
        "Google News RSS: %d queries × %d quarters = %d requests",
        len(GNEWS_RSS_QUERIES), len(windows), total_requests,
    )

    records: list[dict] = []
    seen_ids: set[str] = set()

    for after, before, year, quarter in windows:
        quarter_count = 0
        for query in GNEWS_RSS_QUERIES:
            feed_url = _build_url(query, after, before)
            try:
                feed = feedparser.parse(feed_url)
                time.sleep(GNEWS_CRAWL_DELAY)
            except Exception as exc:
                logger.debug(
                    "Google News RSS error (%s %d-Q%d): %s",
                    query[:30], year, quarter, exc,
                )
                continue

            for entry in feed.entries:
                record = _parse_entry(entry)
                if record is None:
                    continue

                dedup_key = record["article_id"] or record["link"]
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)

                records.append(record)
                quarter_count += 1

        logger.info(
            "Google News RSS %d-Q%d: +%d articles (total %d)",
            year, quarter, quarter_count, len(records),
        )

    logger.info(
        "fetch_gnews_rss_articles: %d credible articles (%s to %s)",
        len(records), start_date, end_date,
    )
    return records
