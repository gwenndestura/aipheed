"""
app/ml/corpus/location_geocoder.py
-----------------------------------
Sub-province (city / municipality / barangay) geocoding for Region IV-A.

Where ``geocoder.geocode_to_province`` answers *which province*, this module
answers *which specific local area* — pinning an article to the most specific
CALABARZON place its text actually names, down to the barangay. It is the
tagging layer that makes the corpus geographically specific rather than
province-only.

MATCHING STRATEGY (specific → general, with disambiguation)
----------------------------------------------------------
Built from ``data/processed/psgc_gazetteer.parquet`` (province → LGU →
barangay). Philippine place names repeat heavily across provinces and regions,
so every tier is gated to avoid false pins:

  1. BARANGAY  — never matches on its own. A barangay name must appear together
     with its parent LGU or parent province in the text (barangay names like
     "San Isidro" or "Poblacion" exist in hundreds of towns). Generic
     poblacion/numbered-district names are excluded entirely.
  2. LGU       — an LGU name that is unique within Region IV-A and not on the
     national-ambiguity list matches directly. An ambiguous LGU name
     ("San Jose", "Santa Cruz", "Rosario", "San Pedro"...) needs its province
     to be corroborated (by a trusted prior tag or a province mention).
  3. PROVINCE  — delegated to ``geocoder.geocode_to_province`` (alias + fuzzy).
  4. REGION    — bare "CALABARZON" / "Region IV-A" with no province resolved.

A trusted ``prior_province_code`` (e.g. GDELT V2Locations full-body geocode) is
never overridden by weaker text matches; it seeds province and constrains which
LGU/barangay candidates are admissible.

PUBLIC API
----------
    from app.ml.corpus.location_geocoder import geocode_location
    loc = geocode_location("Palay farmers in Molino, Bacoor, Cavite lose crops")
    # loc == {
    #   "province_code": "PH040100000", "province_name": "Cavite",
    #   "lgu_name": "Bacoor", "lgu_psgc": "...",
    #   "barangay_name": "Molino III", "barangay_psgc": "...",
    #   "match_level": "barangay", "geo_specificity": 3,
    # }
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from app.ml.corpus.geocoder import (
    PROVINCE_PSGC,
    _mask_negative_context,
    geocode_to_province,
)

logger = logging.getLogger(__name__)

GAZETTEER_PATH = Path("data/processed/psgc_gazetteer.parquet")

_INTERNAL_TO_NAME = {v: k for k, v in PROVINCE_PSGC.items()}

# LGU names that repeat across Philippine provinces/regions (or collide with
# common words / NCR places). Matching these standalone would mis-pin an
# article, so they require province corroboration. Superset of the list used
# by scripts/reanalyze_calabarzon.py, kept here as the single source of truth.
_NATIONALLY_AMBIGUOUS_LGU: frozenset[str] = frozenset({
    "rizal", "san juan", "san pedro", "san pablo", "santa cruz", "sta. cruz",
    "san jose", "san antonio", "san francisco", "santa maria", "santa rosa",
    "rosario", "victoria", "san andres", "san narciso", "general luna",
    "san isidro", "santo tomas", "real", "mabini", "quezon", "candelaria",
    "dolores", "san nicolas", "magsaysay", "bay", "pila", "silang", "taal",
    "alaminos", "gumaca", "lucban", "malvar", "tanay", "tiaong",
})

# LGU names that also read as a person's surname, a common word, or a natural
# feature — so a mere province match is NOT enough (the article can be squarely
# about that province yet the token be a person). e.g. DA Secretary Francisco
# Tiu "Laurel" tagged as Laurel, Batangas; "Kalayaan" (freedom / the Palawan
# islands) tagged as Kalayaan, Laguna; "Taal" the volcano/lake vs the town.
# These require a locality cue (adjacent province name or a locality keyword),
# not just province presence somewhere in the article.
_LGU_SURNAME_OR_WORD: frozenset[str] = frozenset({
    "laurel", "kalayaan", "taal", "talisay", "sampaloc", "mabini", "real",
    "bay", "pila", "silang", "victoria", "dolores", "candelaria", "general luna",
    "magsaysay", "rosario", "unisan", "polillo",
})

# Risky = needs a locality cue before it may pin an LGU. Union of the two sets.
_LGU_RISKY: frozenset[str] = _NATIONALLY_AMBIGUOUS_LGU | _LGU_SURNAME_OR_WORD

# Words that, appearing right next to a place name, mark it as a locality
# reference rather than a person/common word ("Laurel town", "mayor of Taal",
# "municipality of Real", "Barangay ... Sampaloc").
_LOCALITY_KW = re.compile(
    r"\b(town|towns|municipalit\w*|bayan|lungsod|city|cities|mayor|vice[- ]?mayor|"
    r"barangay|brgy|bgy|village|province of|resident\w*|lgu|governor|councilor)\b"
)

# Minimum characters for a place name to be matchable (guards against short
# tokens fuzzing into unrelated words).
_MIN_LGU_LEN = 4
_MIN_BRGY_LEN = 5

# Strong non-CALABARZON location cues. Many CALABARZON barangay names
# (Salvacion, Nueva, San Miguel, Poblacion...) recur nationwide, so an article
# clearly set in another region must not be pinned to a CALABARZON locality on
# an incidental same-named token. Mirrors scripts/reanalyze_calabarzon.OTHER_LOC.
_OTHER_LOC = re.compile(
    r"\b(cebu|mandaue|lapu-?lapu|banilad|mindanao|davao|iloilo|bacolod|zamboanga|"
    r"cagayan de oro|leyte|tacloban|samar|bicol|albay|legazpi|naga city|sorsogon|"
    r"catanduanes|masbate|ilocos|vigan|pangasinan|dagupan|bulacan|malolos|pampanga|"
    r"angeles city|tarlac|zambales|olongapo|nueva ecija|cabanatuan|baguio|benguet|"
    r"la union|palawan|puerto princesa|mindoro|romblon|marinduque|boracay|aklan|"
    r"antique|capiz|guimaras|surigao|butuan|agusan|bukidnon|misamis|ozamiz|cotabato|"
    r"general santos|gensan|sultan kudarat|maguindanao|marawi|lanao|sulu|basilan|"
    r"tawi-tawi|apayao|ifugao|abra|batanes|isabela|tuguegarao|quirino|"
    r"aurora|barotac|western visayas|eastern visayas|northern samar|cordillera|"
    r"caraga|central luzon|ilocos region|bicol region)\b", re.I)

# ---------------------------------------------------------------------------
# Lazily-built indices (loaded once from the gazetteer parquet)
# ---------------------------------------------------------------------------

_LOADED = False
_LGU_RE: Optional[re.Pattern] = None
_BRGY_RE: Optional[re.Pattern] = None
# lgu_name_lower -> list of dicts {province_code, province_name, lgu_name, lgu_psgc}
_LGU_MAP: dict[str, list[dict]] = {}
# set of lgu_name_lower that resolve to exactly one Region IV-A province
_LGU_UNIQUE_IN_REGION: set[str] = set()
# brgy_name_lower -> list of dicts {province_code, province_name, lgu_name, lgu_psgc, barangay_name, barangay_psgc}
_BRGY_MAP: dict[str, list[dict]] = {}


def _norm(s: str) -> str:
    """Lowercase and collapse the ñ/n and whitespace so text and gazetteer
    names compare on the same footing."""
    s = str(s).lower().strip()
    s = s.replace("ñ", "n").replace("`", "").replace("'", "")
    return re.sub(r"\s+", " ", s)


def _load() -> None:
    global _LOADED, _LGU_RE, _BRGY_RE
    if _LOADED:
        return
    if not GAZETTEER_PATH.exists():
        raise FileNotFoundError(
            f"PSGC gazetteer not found at {GAZETTEER_PATH}. Run "
            "`python -m app.ml.corpus.psgc_gazetteer_fetcher` first."
        )
    gz = pd.read_parquet(GAZETTEER_PATH)

    province_names_lc = {_norm(p) for p in _INTERNAL_TO_NAME.values()}

    for row in gz.itertuples(index=False):
        ln = _norm(row.lgu_name)
        # Municipalities named exactly like a province ("Rizal" in Laguna,
        # "Quezon" in Quezon) can never be told apart from a province mention
        # by text — the province word always co-occurs. Exclude them from LGU
        # matching; they remain reachable only via one of their named barangays.
        if ln in province_names_lc:
            continue
        if len(ln) >= _MIN_LGU_LEN:
            entry = {
                "province_code": row.province_code,
                "province_name": row.province_name,
                "lgu_name": row.lgu_name,
                "lgu_psgc": row.lgu_psgc,
            }
            if ln not in _LGU_MAP:
                _LGU_MAP[ln] = [entry]
            elif not any(e["lgu_psgc"] == row.lgu_psgc for e in _LGU_MAP[ln]):
                _LGU_MAP[ln].append(entry)

        if not row.barangay_generic:
            bn = _norm(row.barangay_name)
            if len(bn) >= _MIN_BRGY_LEN:
                _BRGY_MAP.setdefault(bn, []).append({
                    "province_code": row.province_code,
                    "province_name": row.province_name,
                    "lgu_name": row.lgu_name,
                    "lgu_psgc": row.lgu_psgc,
                    "barangay_name": row.barangay_name,
                    "barangay_psgc": row.barangay_psgc,
                })

    # An LGU name is "unique in region" if all its gazetteer entries share one
    # province (no intra-region collision like Cavite/Batangas both having it).
    for ln, entries in _LGU_MAP.items():
        if len({e["province_code"] for e in entries}) == 1:
            _LGU_UNIQUE_IN_REGION.add(ln)

    # Word-boundary alternation regexes, longest names first so "general trias"
    # wins over "general" and "santa rosa" over "rosa".
    lgu_parts = sorted((re.escape(n) for n in _LGU_MAP), key=len, reverse=True)
    brgy_parts = sorted((re.escape(n) for n in _BRGY_MAP), key=len, reverse=True)
    if lgu_parts:
        _LGU_RE = re.compile(r"(?<![a-z])(" + "|".join(lgu_parts) + r")(?![a-z])")
    if brgy_parts:
        _BRGY_RE = re.compile(r"(?<![a-z])(" + "|".join(brgy_parts) + r")(?![a-z])")

    _LOADED = True
    logger.info(
        "location_geocoder: %d LGU names (%d unique-in-region), %d barangay names",
        len(_LGU_MAP), len(_LGU_UNIQUE_IN_REGION), len(_BRGY_MAP),
    )


def _find_lgus(low: str) -> list[str]:
    """Return distinct normalized LGU names present in text (ordered by length,
    most specific first)."""
    if _LGU_RE is None:
        return []
    seen: list[str] = []
    for m in _LGU_RE.finditer(low):
        n = m.group(0)
        if n not in seen:
            seen.append(n)
    return sorted(seen, key=len, reverse=True)


def _cue_province(low: str, n: str) -> Optional[str]:
    """Province code if a locality cue corroborates that the token `n` is really
    a place reference — either the LGU's own province name sits next to it
    ("Laurel, Batangas"), or a locality keyword does ("Laurel town", "mayor of
    Laurel"). Returns None when `n` reads as a bare surname/word/feature.
    """
    entries = _LGU_MAP.get(n, [])
    prov_by_name = {_norm(e["province_name"]): e["province_code"] for e in entries}
    for m in re.finditer(rf"(?<![a-z]){re.escape(n)}(?![a-z])", low):
        s, e = m.start(), m.end()
        window = low[max(0, s - 30):min(len(low), e + 30)]
        for pn, pc in prov_by_name.items():
            if pn in window:
                return pc
        # A locality keyword near the token corroborates it — but only picks a
        # province when the name is unambiguous within the region.
        near = low[max(0, s - 22):e + 18]
        if _LOCALITY_KW.search(near) and len(prov_by_name) == 1:
            return entries[0]["province_code"]
    return None


def _resolve_lgu(low: str, province_code: Optional[str]) -> Optional[dict]:
    """Pick the best LGU entry given (optional) known province.

    Priority:
      1. A confident, non-risky, region-unique LGU → matches directly (and
         fixes the province).
      2. A risky LGU (ambiguous name, surname, common word, natural feature) →
         only when a locality cue corroborates it (adjacent province or locality
         keyword). This is what stops "Laurel" (the DA Secretary) or "Kalayaan"
         (freedom / Palawan) from pinning a CALABARZON town.
      3. A non-risky LGU that collides *within* the region → disambiguated by
         the known province.
    """
    names = _find_lgus(low)
    if not names:
        return None

    # 1. Confident, non-risky, region-unique name.
    for n in names:
        if n in _LGU_RISKY or n not in _LGU_UNIQUE_IN_REGION:
            continue
        e = _LGU_MAP[n][0]
        if not province_code or e["province_code"] == province_code:
            return e

    # 2. Risky name — require a locality cue.
    for n in names:
        if n not in _LGU_RISKY:
            continue
        pc = _cue_province(low, n)
        if pc is None or (province_code and pc != province_code):
            continue
        for e in _LGU_MAP[n]:
            if e["province_code"] == pc:
                return e

    # 3. Non-risky intra-region collision — disambiguate by known province.
    if province_code:
        for n in names:
            if n in _LGU_RISKY:
                continue
            for e in _LGU_MAP[n]:
                if e["province_code"] == province_code:
                    return e
    return None


# Explicit barangay marker immediately preceding a name ("Barangay San Isidro",
# "Brgy. Molino", "Sitio ...")  — the strongest evidence a token is a barangay
# reference and not a surname, company, or province of the same spelling.
_BRGY_MARKER = r"(?:barangay|brgy\.?|bgy\.?|sitio|purok)\s+"


def _resolve_barangay(low: str, province_code: Optional[str],
                      lgu_psgc: Optional[str]) -> Optional[dict]:
    """Pin a barangay only under a STRONG locality cue, because most barangay
    names are also common words, surnames, or province names ("Quezon", "Rizal",
    "San Miguel", "Leviste"). Two admissible cues:

      (a) an explicit "Barangay/Brgy./Sitio <Name>" marker, or
      (b) the name directly followed by its own parent LGU ("San Isidro,
          Rodriguez") — the comma-adjacency pattern.

    The candidate must also be consistent with any already-pinned LGU/province.
    Bare mentions ("...in Quezon") never qualify.
    """
    if _BRGY_RE is None:
        return None

    def _consistent(c: dict) -> bool:
        if lgu_psgc and c["lgu_psgc"] != lgu_psgc:
            return False
        if province_code and c["province_code"] != province_code:
            return False
        return True

    for m in _BRGY_RE.finditer(low):
        bn = m.group(0)
        candidates = [c for c in _BRGY_MAP.get(bn, []) if _consistent(c)]
        if not candidates:
            continue
        start = m.start()
        # Cue (a): explicit marker directly before the name.
        preceding = low[max(0, start - 12):start]
        if re.search(_BRGY_MARKER + r"$", preceding):
            return candidates[0] if lgu_psgc else _pick_marker_candidate(candidates, low)
        # Cue (b): "<barangay>, <its parent LGU>" adjacency.
        tail = low[m.end():m.end() + 40]
        for c in candidates:
            parent = _norm(c["lgu_name"])
            if re.match(rf"\s*,?\s*(?:{_BRGY_MARKER})?{re.escape(parent)}(?![a-z])", tail):
                return c
    return None


def _pick_marker_candidate(candidates: list[dict], low: str) -> dict:
    """With an explicit marker but no pinned LGU, prefer a candidate whose
    parent LGU or province also appears in the text; else take the first."""
    for c in candidates:
        parent = _norm(c["lgu_name"])
        prov = _norm(c["province_name"])
        if re.search(rf"(?<![a-z]){re.escape(parent)}(?![a-z])", low) or \
           re.search(rf"(?<![a-z]){re.escape(prov)}(?![a-z])", low):
            return c
    return candidates[0]


def geocode_location(text: str,
                     prior_province_code: Optional[str] = None) -> dict:
    """Pin article text to the most specific CALABARZON place it names.

    Parameters
    ----------
    text : str
        Article title + summary (combined).
    prior_province_code : str, optional
        Trusted upstream province (repo-internal code, e.g. GDELT tag). Seeds
        province and constrains LGU/barangay candidates; never overridden.

    Returns
    -------
    dict with keys: province_code, province_name, lgu_name, lgu_psgc,
    barangay_name, barangay_psgc, match_level
    ("barangay"|"lgu"|"province"|"region"|"none"), geo_specificity (0-3).
    """
    _load()
    empty = {
        "province_code": None, "province_name": None,
        "lgu_name": None, "lgu_psgc": None,
        "barangay_name": None, "barangay_psgc": None,
        "match_level": "none", "geo_specificity": 0, "geo_conflict": False,
    }
    if not text or not isinstance(text, str):
        return {**empty, "province_code": prior_province_code,
                "province_name": _INTERNAL_TO_NAME.get(prior_province_code),
                "match_level": "province" if prior_province_code else "none",
                "geo_specificity": 1 if prior_province_code else 0}

    low = _mask_negative_context(text.lower())
    low = low.replace("ñ", "n")

    # Province: trust the prior tag, else geocode from text.
    province_code = prior_province_code or geocode_to_province(text)

    # Conflict guard: an article that names another region must not have a
    # CALABARZON locality pinned from an incidental same-named token. Province
    # geocoding is left untouched (pre-existing behaviour); only the finer tags
    # this module adds are gated:
    #   • barangay — suppressed UNCONDITIONALLY on a competing region. A
    #     specific-barangay claim in a story clearly set elsewhere is almost
    #     always a same-name collision (Salvacion, Nueva, Banilad...), and
    #     barangay precision outweighs the handful of recall lost.
    #   • LGU — suppressed only without a trusted prior; an authoritative
    #     upstream province tag (e.g. GDELT full-body geocode) is allowed to
    #     carry an LGU past an incidental other-region mention.
    has_other = bool(_OTHER_LOC.search(low))
    geo_conflict = has_other and not prior_province_code

    if geo_conflict:
        lgu = None
    else:
        lgu = _resolve_lgu(low, province_code)
        if lgu and not province_code:
            province_code = lgu["province_code"]

    if has_other:
        brgy = None
    else:
        brgy = _resolve_barangay(
            low, province_code, lgu["lgu_psgc"] if lgu else None
        )
    # A corroborated barangay implies (and may supply) its LGU + province.
    if brgy:
        province_code = province_code or brgy["province_code"]
        if not lgu:
            lgu = {"lgu_name": brgy["lgu_name"], "lgu_psgc": brgy["lgu_psgc"],
                   "province_code": brgy["province_code"],
                   "province_name": brgy["province_name"]}

    province_name = _INTERNAL_TO_NAME.get(province_code)
    region_only = bool(re.search(r"\bcalabarzon\b|\bregion iv-?a\b|\bregion 4-?a\b", low))

    if brgy:
        level, spec = "barangay", 3
    elif lgu:
        level, spec = "lgu", 2
    elif province_code:
        level, spec = "province", 1
    elif region_only:
        level, spec = "region", 0
    else:
        level, spec = "none", 0

    return {
        "province_code": province_code,
        "province_name": province_name,
        "lgu_name": lgu["lgu_name"] if lgu else None,
        "lgu_psgc": lgu["lgu_psgc"] if lgu else None,
        "barangay_name": brgy["barangay_name"] if brgy else None,
        "barangay_psgc": brgy["barangay_psgc"] if brgy else None,
        "match_level": level,
        "geo_specificity": spec,
        "geo_conflict": geo_conflict,
    }


def geocode_location_batch(df: pd.DataFrame,
                           text_cols: tuple[str, str] = ("title", "summary"),
                           prior_col: str = "province_code") -> pd.DataFrame:
    """Add province/LGU/barangay columns to a corpus DataFrame.

    Reads a trusted prior province from ``prior_col`` when present. Returns a
    copy with: province_code, province_name, lgu_name, lgu_psgc, barangay_name,
    barangay_psgc, match_level, geo_specificity.
    """
    _load()
    t = df.get(text_cols[0], pd.Series([""] * len(df))).fillna("").astype(str)
    s = df.get(text_cols[1], pd.Series([""] * len(df))).fillna("").astype(str)
    combined = (t + ". " + s).tolist()
    priors = (df[prior_col].tolist() if prior_col in df.columns
              else [None] * len(df))

    results = [
        geocode_location(txt, prc if isinstance(prc, str) else None)
        for txt, prc in zip(combined, priors)
    ]
    res = pd.DataFrame(results, index=df.index)
    out = df.copy()
    for c in res.columns:
        out[c] = res[c]
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tests = [
        "Palay farmers in Molino, Bacoor, Cavite lose crops to flooding",
        "Rice prices climb in Lucena City public markets",
        "DSWD distributes food packs in San Isidro, Rodriguez, Rizal",
        "Fisherfolk in Taal, Batangas hit by fish kill",
        "Hunger rises across CALABARZON amid inflation",
        "San Jose residents seek ayuda",  # ambiguous LGU, no province → should stay unresolved
    ]
    for t in tests:
        print(f"\n{t}\n  -> {geocode_location(t)}")
