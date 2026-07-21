"""
app/ml/corpus/geocoder.py
--------------------------
PSGC fuzzy matching — maps article text to CALABARZON province codes.

Uses thefuzz (RapidFuzz backend) to match location mentions in article
text against official PSGC province names. Returns PSGC province code
if the best match score >= 85, else None.

CALABARZON provinces and PSGC codes:
  Batangas  → PH040500000
  Cavite    → PH040100000
  Laguna    → PH040200000
  Quezon    → PH040300000
  Rizal     → PH040400000

Usage:
    from app.ml.corpus.geocoder import geocode_to_province
    code = geocode_to_province("Bigas prices rise in Cavite municipalities")
    # → "PH040100000"
"""

from __future__ import annotations

import logging
import re

from thefuzz import fuzz, process

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PSGC province registry for CALABARZON (Region IV-A)
# ---------------------------------------------------------------------------

PROVINCE_PSGC: dict[str, str] = {
    "Batangas": "PH040500000",
    "Cavite":   "PH040100000",
    "Laguna":   "PH040200000",
    "Quezon":   "PH040300000",
    "Rizal":    "PH040400000",
}

# Alias lookup table: demonyms, alternate spellings, and CALABARZON city /
# municipality names mapped to their canonical province.
#
# Only names that are UNAMBIGUOUS nationally are listed. Philippine LGU names
# repeat across regions (San Jose, Santa Cruz, Rosario, Victoria, Sampaloc,
# General Luna, Real, Bay ...), so those are deliberately excluded — matching
# them standalone would pull in non-CALABARZON articles and violate the
# region-specificity requirement. Ambiguous LGUs are still reachable via the
# province-name match in stage 2 (e.g. "San Jose, Batangas").
#
# Matching is word-boundary anchored (see _alias_match) so "taal" cannot fire
# on "Bataan" and "lipa" cannot fire on "Lipa" inside another word.
_ALIASES: dict[str, str] = {
    # ── Demonyms ────────────────────────────────────────────────────────
    "batangueño": "Batangas", "batangueña": "Batangas",
    "batangueno": "Batangas", "batanguena": "Batangas",
    "caviteño": "Cavite", "caviteno": "Cavite",
    "lagunense": "Laguna", "quezonian": "Quezon", "rizaleño": "Rizal",
    "rizaleno": "Rizal",

    # ── Batangas ────────────────────────────────────────────────────────
    "batangas city": "Batangas", "lipa city": "Batangas", "lipa": "Batangas",
    "tanauan": "Batangas", "taal": "Batangas", "taal lake": "Batangas",
    "taal volcano": "Batangas", "nasugbu": "Batangas", "balayan": "Batangas",
    "calaca": "Batangas", "lemery": "Batangas", "bauan": "Batangas",
    "mabini batangas": "Batangas", "agoncillo": "Batangas",
    "laurel batangas": "Batangas", "alitagtag": "Batangas",
    "cuenca": "Batangas", "ibaan": "Batangas", "lobo": "Batangas",
    "malvar": "Batangas", "mataas na kahoy": "Batangas",
    "padre garcia": "Batangas", "taysan": "Batangas", "tingloy": "Batangas",
    "tuy batangas": "Batangas", "lian": "Batangas", "balete batangas": "Batangas",
    "santo tomas batangas": "Batangas", "san pascual batangas": "Batangas",

    # ── Cavite ──────────────────────────────────────────────────────────
    "imus": "Cavite", "bacoor": "Cavite", "dasmarinas": "Cavite",
    "dasmariñas": "Cavite", "tagaytay": "Cavite", "general trias": "Cavite",
    "trece martires": "Cavite", "cavite city": "Cavite", "kawit": "Cavite",
    "noveleta": "Cavite", "silang cavite": "Cavite", "amadeo": "Cavite",
    "indang": "Cavite", "naic": "Cavite", "ternate cavite": "Cavite",
    "maragondon": "Cavite", "alfonso cavite": "Cavite", "mendez": "Cavite",
    "carmona": "Cavite", "tanza": "Cavite", "magallanes cavite": "Cavite",
    "gen. mariano alvarez": "Cavite", "general mariano alvarez": "Cavite",
    "general emilio aguinaldo": "Cavite",

    # ── Laguna ──────────────────────────────────────────────────────────
    "san pablo city": "Laguna", "calamba": "Laguna", "los baños": "Laguna",
    "los banos": "Laguna", "cabuyao": "Laguna", "biñan": "Laguna",
    "binan": "Laguna", "santa rosa laguna": "Laguna", "sta. rosa laguna": "Laguna",
    "san pedro laguna": "Laguna", "pagsanjan": "Laguna", "paete": "Laguna",
    "pakil": "Laguna", "pangil": "Laguna", "siniloan": "Laguna",
    "famy": "Laguna", "mabitac": "Laguna", "luisiana": "Laguna",
    "majayjay": "Laguna", "nagcarlan": "Laguna", "liliw": "Laguna",
    "cavinti": "Laguna", "lumban": "Laguna", "kalayaan laguna": "Laguna",
    "pila laguna": "Laguna", "calauan": "Laguna", "magdalena laguna": "Laguna",
    "laguna de bay": "Laguna", "laguna lake": "Laguna",

    # ── Quezon ──────────────────────────────────────────────────────────
    "lucena": "Quezon", "tayabas": "Quezon", "infanta quezon": "Quezon",
    "lucban": "Quezon", "sariaya": "Quezon", "candelaria quezon": "Quezon",
    "atimonan": "Quezon", "gumaca": "Quezon", "calauag": "Quezon",
    "tagkawayan": "Quezon", "guinayangan": "Quezon", "mulanay": "Quezon",
    "catanauan": "Quezon", "macalelon": "Quezon", "pitogo quezon": "Quezon",
    "unisan": "Quezon", "agdangan": "Quezon", "alabat": "Quezon",
    "polillo": "Quezon", "burdeos": "Quezon", "panukulan": "Quezon",
    "patnanungan": "Quezon", "jomalig": "Quezon", "general nakar": "Quezon",
    "tiaong": "Quezon", "padre burgos quezon": "Quezon",
    "san narciso quezon": "Quezon", "san andres quezon": "Quezon",

    # ── Rizal ───────────────────────────────────────────────────────────
    "antipolo": "Rizal", "cainta": "Rizal", "rodriguez rizal": "Rizal",
    "montalban": "Rizal", "tanay": "Rizal", "angono": "Rizal",
    "binangonan": "Rizal", "cardona": "Rizal", "jala-jala": "Rizal",
    "morong rizal": "Rizal", "pililla": "Rizal", "taytay rizal": "Rizal",
    "teresa rizal": "Rizal", "baras rizal": "Rizal", "san mateo rizal": "Rizal",
}

# Names that must NEVER trigger a CALABARZON match on their own: they are
# NCR/other-region places frequently co-mentioned in national coverage.
_NEGATIVE_CONTEXT: tuple[str, ...] = (
    "quezon city", "quezon ave", "quezon memorial", "quezon blvd",
    "quezon boulevard", "rizal park", "rizal ave", "rizal monument",
    "jose rizal", "rizal day", "rizal stadium", "quezon institute",
)

# Minimum fuzzy match score (0–100) to accept a province mapping
MATCH_THRESHOLD = 85

_PROVINCE_NAMES: list[str] = list(PROVINCE_PSGC.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize_text(text: str) -> list[str]:
    """
    Split text into lowercase alpha-space tokens of 3+ chars.
    Strips punctuation while preserving multi-word location spans.
    """
    # Keep letters, spaces, hyphens; lowercase
    cleaned = re.sub(r"[^a-zA-Z\s\-]", " ", text).lower()
    # Yield individual words and 2-word bigrams for multi-word provinces
    words = cleaned.split()
    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]} {words[i + 1]}")
    return tokens


def _mask_negative_context(text_lower: str) -> str:
    """
    Blank out NCR/other-region phrases that contain a CALABARZON province
    word ("Quezon City", "Rizal Park", "Jose Rizal") so they cannot produce
    a false province assignment. Returns the masked text.
    """
    for phrase in _NEGATIVE_CONTEXT:
        if phrase in text_lower:
            text_lower = text_lower.replace(phrase, " ")
    return text_lower


# Longest aliases first: "santa rosa laguna" must win over "laguna", and
# multi-word disambiguators ("silang cavite") must be tried before any
# shorter overlapping entry.
_ALIASES_SORTED: list[tuple[str, str]] = sorted(
    _ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True
)


def _alias_match(text_lower: str) -> str | None:
    """
    Return canonical province name if any alias appears in text as a whole
    word/phrase. Word-boundary anchored so "taal" does not fire inside
    "Bataan" and "lipa" does not fire inside "Filipang".
    """
    for alias, province in _ALIASES_SORTED:
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text_lower):
            return province
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def geocode_to_province(text: str) -> str | None:
    """
    Map article text to a CALABARZON PSGC province code.

    Strategy:
      1. Fast alias lookup for known demonyms and city names.
      2. Exact substring match against province names.
      3. Fuzzy match (thefuzz token_set_ratio) against province names
         on tokenized 1–2 word spans.

    Parameters
    ----------
    text : str
        Article title + summary (combined).

    Returns
    -------
    str | None
        PSGC province code (e.g. "PH040100000") if score >= 85, else None.
    """
    if not text or not isinstance(text, str):
        return None

    # Mask NCR/other-region phrases ("Quezon City", "Rizal Park") first so
    # they cannot be read as CALABARZON province mentions.
    text_lower = _mask_negative_context(text.lower())

    # 1. Alias lookup (demonyms + unambiguous CALABARZON city/municipality names)
    alias_province = _alias_match(text_lower)
    if alias_province:
        logger.debug("Alias match → %s", alias_province)
        return PROVINCE_PSGC[alias_province]

    # 2. Exact word-boundary match against province names
    for province in _PROVINCE_NAMES:
        if re.search(rf"(?<![a-z]){province.lower()}(?![a-z])", text_lower):
            logger.debug("Exact match → %s", province)
            return PROVINCE_PSGC[province]

    # 3. Fuzzy match on 1–2 word tokens from the MASKED text — using the raw
    #    text here would let "Quezon City" / "Jose Rizal" fuzzy-match back to
    #    the province names the mask just removed.
    tokens = _tokenize_text(text_lower)
    best_province: str | None = None
    best_score: int = 0

    for token in tokens:
        if len(token) < 3:
            continue
        result = process.extractOne(
            token,
            _PROVINCE_NAMES,
            scorer=fuzz.token_set_ratio,
        )
        if result is None:
            continue
        matched_province, score = result[0], result[1]
        if score > best_score:
            best_score = score
            best_province = matched_province

    if best_score >= MATCH_THRESHOLD and best_province is not None:
        logger.debug(
            "Fuzzy match → %s (score=%d, token=%s)",
            best_province, best_score, token,
        )
        return PROVINCE_PSGC[best_province]

    logger.debug("No CALABARZON province match in text (best_score=%d)", best_score)
    return None


def geocode_batch(articles: list[dict]) -> list[dict]:
    """
    Geocode a list of article dicts, adding 'province_code' field to each.

    Articles with no CALABARZON match retain province_code = None and are
    NOT dropped — the caller decides whether to filter them.

    Parameters
    ----------
    articles : list[dict]
        Each dict must have 'title' and 'summary' keys.

    Returns
    -------
    list[dict]
        Same list with 'province_code' added to every record.
    """
    for article in articles:
        combined = f"{article.get('title', '')} {article.get('summary', '')}"
        article["province_code"] = geocode_to_province(combined)

    matched = sum(1 for a in articles if a.get("province_code"))
    logger.info(
        "geocode_batch: %d/%d articles matched to CALABARZON province",
        matched, len(articles),
    )
    return articles
