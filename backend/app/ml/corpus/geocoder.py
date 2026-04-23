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

# Common alternate spellings / demonyms mapped to canonical province names
_ALIASES: dict[str, str] = {
    "batangueño":   "Batangas",
    "batangueña":   "Batangas",
    "batangueno":   "Batangas",
    "batanguena":   "Batangas",
    "imus":         "Cavite",
    "bacoor":       "Cavite",
    "dasmarinas":   "Cavite",
    "dasmariñas":   "Cavite",
    "san pablo":    "Laguna",
    "calamba":      "Laguna",
    "los baños":    "Laguna",
    "los banos":    "Laguna",
    "quezon city":  "Rizal",   # Distinct from Quezon province; treat as Rizal/NCR border
    "antipolo":     "Rizal",
    "cainta":       "Rizal",
    "rodriguez":    "Rizal",
    "tanay":        "Rizal",
    "lucena":       "Quezon",
    "tayabas":      "Quezon",
    "infanta":      "Quezon",
}

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


def _alias_match(text_lower: str) -> str | None:
    """Return canonical province name if any alias is found in text."""
    for alias, province in _ALIASES.items():
        if alias in text_lower:
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

    text_lower = text.lower()

    # 1. Alias lookup (fast path for common demonyms and cities)
    alias_province = _alias_match(text_lower)
    if alias_province:
        logger.debug("Alias match → %s", alias_province)
        return PROVINCE_PSGC[alias_province]

    # 2. Exact substring match (case-insensitive)
    for province in _PROVINCE_NAMES:
        if province.lower() in text_lower:
            logger.debug("Exact match → %s", province)
            return PROVINCE_PSGC[province]

    # 3. Fuzzy match on 1–2 word tokens from the text
    tokens = _tokenize_text(text)
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
