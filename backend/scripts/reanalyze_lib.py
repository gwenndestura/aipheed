"""
scripts/reanalyze_lib.py
------------------------------------------------------------------------------
Shared vocabulary + per-article rule logic for the CALABARZON food-insecurity
re-analysis pipeline (see reanalyze_calabarzon.py).

Everything here is module-level and picklable so it can run inside
multiprocessing workers on Windows (spawn start method). No heavy imports
(no torch / transformers) live in this module — the keyword/geo stages must
stay light so they parallelise cleanly.

Two responsibilities:
  1. STAGE 1 — geographic classification (is this article about CALABARZON,
     and which province/municipality), with careful disambiguation of the
     Rizal-the-hero / Quezon-City traps.
  2. STAGE 2 — food-security candidate detection + structured signal
     extraction (event type, severity cues, affected population/commodity,
     food-security dimension hints). These feed Stage 4 classification.

The heavy semantic judgment (HIGH/MEDIUM relevance) is made by the XLM-R NLI
model in Stage 3; this module only provides the fast recall net and the
structured metadata the model score is combined with.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Province codes used by the recovered GDELT pool (kept as a weak prior only)
# ---------------------------------------------------------------------------

PROVINCE_CODE_NAMES: dict[str, str] = {
    "PH040100000": "Cavite",
    "PH040200000": "Laguna",
    "PH040300000": "Quezon",
    "PH040400000": "Rizal",
    "PH040500000": "Batangas",
}

# ---------------------------------------------------------------------------
# CALABARZON municipalities / cities (curated, disambiguated where a bare
# name collides with a town of the same name in another region — those are
# stored with a province qualifier so they only match in an unambiguous form).
# A hit on any of these => STRONG geographic evidence, and pins the province.
# ---------------------------------------------------------------------------

CITY_TO_PROVINCE: dict[str, str] = {}


def _reg(prov: str, names: list[str]) -> None:
    for n in names:
        CITY_TO_PROVINCE[n.lower()] = prov


_reg("Cavite", [
    "bacoor", "dasmarinas", "dasmariñas", "imus", "general trias", "gen. trias",
    "trece martires", "tagaytay", "kawit", "noveleta", "rosario cavite",
    "silang", "amadeo", "alfonso cavite", "carmona", "cavite city", "naic",
    "gen. mariano alvarez", "general mariano alvarez", "gma cavite", "indang",
    "magallanes cavite", "maragondon", "mendez", "tanza", "ternate cavite",
    "cavite export", "caviteño", "caviteno",
    # Completing the 23 Cavite LGUs.
    "general emilio aguinaldo", "gen. emilio aguinaldo",
])
_reg("Laguna", [
    "calamba", "santa rosa laguna", "sta. rosa laguna", "biñan", "binan",
    "san pedro laguna", "cabuyao", "los baños", "los banos", "bay laguna",
    "calauan", "famy", "kalayaan laguna", "luisiana", "lumban", "mabitac",
    "magdalena laguna", "majayjay", "nagcarlan", "paete", "pagsanjan", "pakil",
    "pangil", "pila laguna", "san pablo laguna", "siniloan",
    "santa cruz laguna", "sta. cruz laguna", "victoria laguna", "alaminos laguna",
    "cavinti", "liliw", "rizal laguna", "santa maria laguna", "laguna de bay",
    "sampaloc lake", "laguneño", "laguneno",
])
_reg("Quezon", [
    "lucena", "tayabas", "candelaria quezon", "gumaca", "infanta quezon",
    "real quezon", "pagbilao", "padre burgos quezon", "mauban", "sariaya",
    "tiaong", "unisan", "agdangan", "alabat", "atimonan", "buenavista quezon",
    "calauag", "catanauan", "dolores quezon", "general luna quezon",
    "general nakar", "guinayangan", "jomalig", "lopez quezon", "lucban",
    "macalelon", "mulanay", "panukulan", "patnanungan", "perez quezon",
    "pitogo", "plaridel quezon", "polillo", "san andres quezon",
    "san antonio quezon", "san narciso quezon", "san francisco quezon",
    "sampaloc quezon", "tagkawayan", "quezon province", "bondoc peninsula",
    # Completing the 41 Quezon LGUs. "Burdeos" is distinctive enough to stand
    # alone; the municipality of Quezon never is — a bare "quezon" is far more
    # often Quezon City / the province / the president — so it is admitted only
    # in an explicitly qualified form.
    "burdeos", "quezon, quezon", "municipality of quezon", "bayan ng quezon",
])
_reg("Rizal", [
    "antipolo", "cainta", "taytay rizal", "binangonan", "angono", "baras rizal",
    "cardona", "jala-jala", "jalajala", "morong rizal", "pililla", "pililia",
    "rodriguez rizal", "montalban", "san mateo rizal", "tanay", "teresa rizal",
    "rizal province",
])
_reg("Batangas", [
    "batangas city", "batangas port", "lipa", "lipa city", "tanauan batangas",
    "tanauan city", "santo tomas batangas", "sto. tomas batangas", "taal",
    "taal lake", "taal volcano", "nasugbu", "san jose batangas",
    "mabini batangas", "lemery", "bauan", "laurel batangas", "malvar",
    "agoncillo", "alitagtag", "balayan", "cuenca", "ibaan", "lobo batangas",
    "mataas na kahoy", "padre garcia", "san juan batangas", "san luis batangas",
    "san nicolas batangas", "san pascual batangas", "santa teresita batangas",
    "talisay batangas", "tingloy", "calatagan", "calaca", "verde island",
    "batangueño", "batangueno",
    # Completing the 34 Batangas LGUs. "Taysan" is distinctive; Balete, Lian,
    # Rosario and Tuy are not (balete is also a common tree and a town in
    # Aklan; Rosario exists in Cavite and La Union; "lian"/"tuy" are short
    # strings that collide with ordinary words), so those stay province-tied.
    "taysan", "balete batangas", "lian batangas", "rosario batangas",
    "tuy batangas",
])

# Bare province tokens. Cavite / Batangas / Laguna are safe as bare words in a
# Philippine-news corpus. Quezon and Rizal are NOT (Quezon City, President
# Quezon, Jose Rizal, Rizal Park), so they only pin the province when they
# appear as an explicit "<x> province" form (already in CITY_TO_PROVINCE) or
# alongside a disambiguating cue handled below.
SAFE_PROVINCE_TOKENS: dict[str, str] = {
    "cavite": "Cavite",
    "batangas": "Batangas",
    "laguna": "Laguna",
}

CALABARZON_ALIASES = [
    "calabarzon", "region iv-a", "region iva", "region 4-a", "region 4a",
    "region iv a", "southern tagalog", "region 4-a", "rev. iv-a",
]

# Contexts that look like Quezon/Rizal but are NOT the CALABARZON provinces.
QUEZON_FALSE = [
    "quezon city", "quezon institute", "quezon memorial", "quezon avenue",
    "quezon ave", "quezon boulevard", "quezon blvd", "quezon hall",
    "president quezon", "manuel quezon", "manuel l. quezon", "quezon bridge",
    "quezon circle", "elliptical road", "quezon heights", "mlqu",
    "quezon, bukidnon", "quezon, palawan", "quezon, nueva",  # towns named Quezon
]
RIZAL_FALSE = [
    "jose rizal", "josé rizal", "rizal park", "rizal day", "rizal avenue",
    "rizal ave", "rizal monument", "rizal shrine", "rizal high",
    "rizal memorial", "rizal stadium", "fort santiago", "rizal, kalinga",
    "rizal, nueva ecija", "rizal, occidental", "rizal, cagayan",
    "rizal, laguna", "rizal, palawan", "rizal, zamboanga", "dr. jose rizal",
    "rizal technological", "rizal library", "rizal law", "rizal boulevard",
]

# Conflicting non-CALABARZON provinces / cities. If one of these clearly
# dominates and there is no CALABARZON city hit, geo is downgraded to none.
NON_CALABARZON = [
    "ilocos", "cagayan valley", "nueva ecija", "pangasinan", "pampanga",
    "bulacan", "tarlac", "zambales", "nueva vizcaya", "aurora province",
    "metro manila", "makati", "mandaluyong", "pasig", "taguig", "pasay",
    "paranaque", "parañaque", "muntinlupa", "las piñas", "las pinas",
    "marikina", "valenzuela", "malabon", "navotas", "caloocan", "manila city",
    "bicol", "albay", "camarines", "sorsogon", "masbate", "catanduanes",
    "legazpi", "naga city", "western visayas", "cebu", "iloilo", "bacolod",
    "negros", "eastern visayas", "leyte", "samar", "tacloban",
    "mindanao", "davao", "cagayan de oro", "zamboanga", "general santos",
    "lanao", "cotabato", "maguindanao", "sultan kudarat", "sulu", "basilan",
    "tawi-tawi", "marawi", "palawan", "puerto princesa", "occidental mindoro",
    "oriental mindoro", "romblon", "marinduque", "aklan", "antique", "capiz",
    "guimaras", "boracay", "surigao", "agusan", "bukidnon", "misamis",
    "benguet", "baguio", "ifugao", "kalinga", "mountain province", "abra",
    "batanes", "isabela", "quirino", "apayao",
]

# ---------------------------------------------------------------------------
# FOOD-SECURITY vocabulary. Broad recall net for Stage 2 candidate detection,
# organised by the food-security dimension it points to. Includes English +
# Filipino terms. Ordering matters only for the "first-hit" dimension guess;
# the authoritative dimension comes from the model in Stage 4.
# ---------------------------------------------------------------------------

# dimension code -> (label, keywords)
FOOD_DIMENSIONS: dict[str, tuple[str, list[str]]] = {
    "availability": ("Food availability (production/supply)", [
        "rice supply", "food supply", "food shortage", "food scarcity",
        "supply shortage", "supply disruption", "kakulangan ng bigas",
        "kakulangan ng pagkain", "rice shortage", "rice stock", "buffer stock",
        "palay", "harvest", "ani", "crop", "pananim", "crop damage",
        "crop loss", "crop failure", "harvest loss", "nasira ang ani",
        "damaged crops", "damaged rice", "palay damage", "farm output",
        "agricultural production", "food production", "rice production",
        "vegetable production", "poultry", "livestock", "hog", "swine",
        "african swine fever", "asf", "bird flu", "avian flu", "fish kill",
        "fishkill", "patay na isda", "red tide", "algal bloom", "fishing ban",
        "shellfish ban", "aquaculture", "fishpond", "fishcage", "milkfish",
        "bangus", "tilapia", "smuggled rice", "rice smuggling", "nfa rice",
        "grain", "cold storage", "post-harvest", "farmgate",
    ]),
    "affordability": ("Food accessibility / affordability", [
        "food price", "rice price", "presyo ng bigas", "presyo ng pagkain",
        "price hike", "price spike", "price surge", "price increase",
        "price freeze", "price ceiling", "price cap", "price control",
        "taas ng presyo", "mahal na bilihin", "mahal na pagkain", "bilihin",
        "food inflation", "inflation", "cpi", "consumer price",
        "vegetable price", "fish price", "presyo ng isda", "presyo ng gulay",
        "meat price", "pork price", "chicken price", "egg price",
        "presyo ng manok", "presyo ng baboy", "presyo ng itlog", "onion price",
        "sibuyas", "sugar price", "presyo ng asukal", "cooking oil price",
        "basic necessities", "purchasing power", "cost of living",
        "farmgate price", "market price", "palengke", "commodity price",
    ]),
    "utilization": ("Food utilization / nutrition", [
        "malnutrition", "malnutrisyon", "malnourish", "undernourish",
        "undernutrition", "stunting", "wasting", "underweight", "nutrition",
        "nutrisyon", "feeding program", "supplementary feeding", "nutribun",
        "e-nutribun", "batang busog", "malusog na bata", "dietary",
        "food intake", "nutrient", "micronutrient", "vitamin deficiency",
        "anemia", "first 1000 days", "breastfeeding", "complementary feeding",
    ]),
    "stability": ("Food stability (shocks)", [
        "typhoon", "bagyo", "flood", "baha", "flooding", "drought", "tagtuyot",
        "el nino", "el niño", "la nina", "la niña", "dry spell", "storm surge",
        "landslide", "calamity", "kalamidad", "sakuna", "nasalanta", "binaha",
        "state of calamity", "evacuation", "evacuees", "displaced", "bakwit",
        "lumikas", "ashfall", "eruption", "oil spill", "supply chain",
        "farm-to-market", "farm to market", "bridge collapse", "road closed",
        "rainfall deficit", "monsoon", "habagat", "flash flood",
    ]),
    "hunger": ("Hunger / food deprivation", [
        "hunger", "gutom", "nagugutom", "nagutom", "pagkagutom", "food crisis",
        "food insecurity", "food insecure", "food security", "walang makain",
        "wala nang makain", "hindi makakain", "skipping meals", "skip meals",
        "involuntary hunger", "sws hunger", "hunger incidence", "food deprivation",
        "starvation", "starving", "famine",
    ]),
    "assistance": ("Food assistance / relief", [
        "ayuda", "food aid", "relief goods", "food pack", "food packs",
        "food relief", "relief operation", "feeding", "libreng bigas",
        "free rice", "food assistance", "food distribution", "food program",
        "kadiwa", "bigasang bayan", "4ps", "pantawid", "conditional cash",
        "rice subsidy", "rice assistance", "dswd", "social amelioration",
        "cash for work", "mobile kitchen", "community pantry", "aid distribution",
    ]),
    "livelihood": ("Livelihood factors affecting food access", [
        "farmer", "magsasaka", "fisherfolk", "fishermen", "fishers",
        "mangingisda", "agricultural worker", "farm worker", "livelihood",
        "hanapbuhay", "kabuhayan", "loss of income", "lost income",
        "nawalan ng kita", "nawalan ng hanapbuhay", "poverty", "kahirapan",
        "unemployment", "walang trabaho", "job loss", "layoff", "jobless",
        "remittance", "ofw", "overseas filipino", "padala", "displaced workers",
    ]),
}

# Flat set of all food keywords for the Stage-2 candidate net.
ALL_FOOD_KEYWORDS: list[str] = sorted(
    {kw for _, kws in FOOD_DIMENSIONS.values() for kw in kws},
    key=len, reverse=True,
)

# A stricter "core food" subset: presence strongly implies the article is
# actually about food/hunger (not just weather or generic farming). Used to
# separate proximate-driver-only candidates from direct food candidates.
CORE_FOOD_MARKERS = [
    "food", "pagkain", "hunger", "gutom", "malnutrition", "malnutrisyon",
    "rice", "bigas", "palay", "nutrition", "nutrisyon", "feeding", "ayuda",
    "crop", "harvest", "ani", "vegetable", "gulay", "fish", "isda", "hunger",
    "kadiwa", "nfa", "famine", "starv",
]

# ---------------------------------------------------------------------------
# Event-type taxonomy (regex families). First match wins for the primary
# event_type; a coarse but useful structured field for the thesis.
# ---------------------------------------------------------------------------

EVENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("typhoon",            ["typhoon", "bagyo", "storm signal", "super typhoon", "cyclone"]),
    ("flood",              ["flood", "baha", "flash flood", "binaha", "monsoon flood", "habagat"]),
    ("drought_elnino",     ["drought", "tagtuyot", "el nino", "el niño", "dry spell", "rainfall deficit"]),
    ("volcanic",           ["ashfall", "eruption", "taal volcano", "phivolcs", "lahar"]),
    ("fish_kill",          ["fish kill", "fishkill", "patay na isda", "red tide", "algal bloom"]),
    ("oil_spill",          ["oil spill", "oil slick", "sunken tanker", "fuel spill"]),
    ("animal_disease",     ["african swine fever", "asf", "bird flu", "avian flu", "hog cholera"]),
    ("price_increase",     ["price hike", "price spike", "price surge", "price increase",
                            "taas ng presyo", "food inflation", "inflation", "presyo"]),
    ("rice_supply",        ["rice supply", "rice shortage", "nfa rice", "smuggled rice",
                            "buffer stock", "rice importation", "rice tariff"]),
    ("crop_damage",        ["crop damage", "crop loss", "crop failure", "harvest loss",
                            "damaged crops", "palay damage", "nasira ang ani"]),
    ("malnutrition",       ["malnutrition", "malnutrisyon", "stunting", "wasting",
                            "undernourish", "underweight"]),
    ("feeding_program",    ["feeding program", "supplementary feeding", "nutribun",
                            "community pantry", "mobile kitchen"]),
    ("food_assistance",    ["ayuda", "food pack", "relief goods", "food aid", "kadiwa",
                            "libreng bigas", "food assistance", "rice subsidy"]),
    ("hunger_report",      ["hunger", "gutom", "food insecurity", "sws hunger",
                            "food deprivation", "famine"]),
    ("livelihood_shock",   ["loss of income", "nawalan ng kita", "job loss", "layoff",
                            "unemployment", "fisherfolk", "fishermen", "livelihood"]),
    ("agriculture_general",["farmer", "magsasaka", "harvest", "palay", "agriculture",
                            "farming", "vegetable", "poultry", "livestock"]),
]

# ---------------------------------------------------------------------------
# Severity cues (coarse heuristic — flagged as such in the reason).
# ---------------------------------------------------------------------------

SEVERITY_HIGH = [
    "state of calamity", "declared calamity", "millions in damage",
    "billion", "million worth", "dead", "deaths", "killed", "casualties",
    "destroyed", "wiped out", "devastated", "thousands of families",
    "widespread", "massive", "severe", "worst", "record", "emergency",
    "total loss", "wala nang", "lubog", "nalunod",
]
SEVERITY_MED = [
    "damaged", "affected", "loss", "losses", "nasira", "nasalanta",
    "apektado", "shortage", "hike", "surge", "displaced", "evacuated",
    "warning", "alert", "decline", "drop", "shortfall",
]

# ---------------------------------------------------------------------------
# Affected-population / commodity extraction vocab.
# ---------------------------------------------------------------------------

POP_TERMS = [
    ("farmers", ["farmer", "magsasaka", "rice farmer", "corn farmer"]),
    ("fisherfolk", ["fisherfolk", "fisherman", "fishermen", "fishers", "mangingisda"]),
    ("children", ["child", "children", "bata", "infant", "toddler", "students", "pupils"]),
    ("families", ["family", "families", "pamilya", "household", "sambahayan"]),
    ("residents", ["resident", "residents", "villagers", "community", "barangay"]),
    ("consumers", ["consumer", "consumers", "shoppers", "buyers", "mamimili"]),
    ("workers", ["worker", "workers", "laborer", "ofw", "employees"]),
    ("evacuees", ["evacuee", "evacuees", "displaced", "bakwit"]),
]

COMMODITY_TERMS = [
    ("rice", ["rice", "bigas", "palay"]),
    ("vegetables", ["vegetable", "gulay", "onion", "sibuyas", "tomato", "kamatis"]),
    ("fish", ["fish", "isda", "bangus", "tilapia", "milkfish", "galunggong"]),
    ("pork", ["pork", "baboy", "hog", "swine"]),
    ("chicken_egg", ["chicken", "manok", "poultry", "egg", "itlog"]),
    ("sugar", ["sugar", "asukal", "sugarcane"]),
    ("coconut", ["coconut", "niyog", "copra", "coco"]),
    ("corn", ["corn", "mais"]),
    ("coffee", ["coffee", "kape", "barako"]),
    ("cooking_oil", ["cooking oil", "mantika", "palm oil"]),
]


# ===========================================================================
# Text normalisation + compiled matchers
# ===========================================================================

def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace for robust matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower()).strip()


def _boundary_regex(terms: list[str]) -> re.Pattern:
    """Word-boundary alternation regex over the (already-normalised) terms."""
    # Escape, allow the ascii-folded forms; longer terms first so the regex
    # prefers the most specific match.
    parts = sorted({normalize(t) for t in terms if t}, key=len, reverse=True)
    esc = [re.escape(p) for p in parts]
    return re.compile(r"(?<![a-z])(" + "|".join(esc) + r")(?![a-z])")


_CITY_RE = _boundary_regex(list(CITY_TO_PROVINCE.keys()))
_SAFE_PROV_RE = _boundary_regex(list(SAFE_PROVINCE_TOKENS.keys()))
_ALIAS_RE = _boundary_regex(CALABARZON_ALIASES)
_QUEZON_FALSE_RE = re.compile("|".join(re.escape(normalize(t)) for t in QUEZON_FALSE))
_RIZAL_FALSE_RE = re.compile("|".join(re.escape(normalize(t)) for t in RIZAL_FALSE))
_NONCAL_RE = _boundary_regex(NON_CALABARZON)
_FOOD_RE = _boundary_regex(ALL_FOOD_KEYWORDS)
_CORE_FOOD_RE = re.compile("|".join(re.escape(normalize(t)) for t in CORE_FOOD_MARKERS))
_QUEZON_PROV_RE = re.compile(r"(?<![a-z])quezon(?![a-z])")
_RIZAL_PROV_RE = re.compile(r"(?<![a-z])rizal(?![a-z])")


# ===========================================================================
# Census-derived LGU matchers — all 142 CALABARZON cities / municipalities
# ===========================================================================
# precision_pass.py and reanalyze_calabarzon.py both used to build this map
# themselves, and both did it the same lossy way: they kept a name only if it
# was >= 5 characters and not on the ambiguous list, which silently discarded
# ten real LGUs (Bay, Famy, Imus, Lian, Lipa, Lobo, Naic, Pila, Taal, Tuy —
# Imus and Lipa are among the region's largest cities). Keying the dict on the
# bare name also collapsed the two Rosarios (Batangas and Cavite) into one.
#
# The rule here is "demote, never drop": a name that cannot stand on its own
# goes to the ambiguous map, where it still matches as long as its province is
# named in the same text. Every one of the 142 lands in exactly one map.

# Names that collide with a well-known place elsewhere in the Philippines.
# This is the list the two scripts already shared, plus "talisay" (Talisay
# City, Cebu and Talisay, Negros Occidental both dwarf Talisay, Batangas in
# news volume) and "sto. tomas" (the PSGC spelling of what this set previously
# only listed as "santo tomas"; Santo Tomas also exists in Pampanga, Isabela,
# La Union and Davao del Norte).
AMBIGUOUS_LGU = {
    "rizal", "san juan", "san pedro", "san pablo", "santa cruz", "sta. cruz",
    "san jose", "san antonio", "san francisco", "santa maria", "santa rosa",
    "rosario", "victoria", "san andres", "san narciso", "general luna",
    "san isidro", "santo tomas", "sto. tomas", "real", "mabini", "quezon",
    "candelaria", "dolores", "san nicolas", "magsaysay", "talisay",
}

# A handful of names are not saved by requiring their province either, because
# the name and the province are the same word: the municipality of Quezon in
# Quezon province. A bare "quezon" is overwhelmingly Quezon City, the province,
# or the president, so this LGU is admitted only in an explicitly qualified
# form. QUEZON_FALSE still screens the well-known non-CALABARZON senses.
REQUIRED_QUALIFIER: dict[str, re.Pattern] = {
    "quezon": re.compile(
        r"quezon,\s*quezon|municipality of quezon|bayan ng quezon|town of quezon"
    ),
}

# Short names are too collision-prone to match bare ("bay", "tuy", "pila").
_SHORT_NAME_MAX = 4

DEFAULT_CENSUS_PATH = "data/processed/lgu_census.parquet"


def build_lgu_matchers(census_path: str = DEFAULT_CENSUS_PATH) -> dict:
    """
    Build the CALABARZON LGU matchers from the LGU census.

    Returns a dict with:
      ``unambig``    {name: province}   — may match on their own.
      ``ambig``      {name: [province]} — only count when a province name from
                                          the list also appears in the text.
      ``unambig_re`` compiled alternation over ``unambig`` (or None if empty).

    Alternate spellings from the census ``lgu_aliases`` column are registered
    alongside the PSGC-canonical name, so "Mataas na Kahoy" resolves the same
    as "Mataasnakahoy".
    """
    import pandas as pd

    d = pd.read_parquet(census_path)

    # name -> set of provinces claiming it (Rosario is claimed by two)
    claims: dict[str, set[str]] = {}
    for row in d.itertuples():
        prov = str(row.province_name)
        names = [str(row.lgu_name)]
        aliases = str(getattr(row, "lgu_aliases", "") or "")
        names += [a for a in aliases.split("|") if a.strip()]
        for n in names:
            claims.setdefault(normalize(n), set()).add(prov)

    unambig: dict[str, str] = {}
    ambig: dict[str, list[str]] = {}
    for name, provs in claims.items():
        risky = (
            len(provs) > 1                   # same name in two provinces
            or len(name) <= _SHORT_NAME_MAX  # too short to stand alone
            or name in AMBIGUOUS_LGU         # famous namesake elsewhere
            or name in REQUIRED_QUALIFIER    # name collides with its own province
        )
        if risky:
            ambig[name] = sorted(provs)
        else:
            unambig[name] = next(iter(provs))

    unambig_re = None
    if unambig:
        parts = sorted((re.escape(n) for n in unambig), key=len, reverse=True)
        unambig_re = re.compile(r"(?<![a-z])(" + "|".join(parts) + r")(?![a-z])")

    return {"unambig": unambig, "ambig": ambig, "unambig_re": unambig_re}


def match_cal_lgu(norm_text: str, matchers: dict) -> tuple[bool, str, str]:
    """
    Find a CALABARZON city/municipality in already-normalised text.

    Returns ``(found, province, city)``. Unambiguous names match on sight;
    ambiguous ones require their province to be named too, which is also how
    the Batangas/Cavite Rosario ambiguity is resolved.
    """
    rx = matchers.get("unambig_re")
    if rx is not None:
        m = rx.search(norm_text)
        if m:
            city = m.group(0)
            return True, matchers["unambig"].get(city, ""), city.title()

    for name, provs in matchers.get("ambig", {}).items():
        qualifier = REQUIRED_QUALIFIER.get(name)
        if qualifier is not None:
            # Province co-occurrence is meaningless here (the name *is* the
            # province), so demand the explicit form instead.
            if qualifier.search(norm_text):
                return True, provs[0], name.title()
            continue
        if not re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", norm_text):
            continue
        for prov in provs:
            if prov.lower() in norm_text:
                return True, prov, name.title()
    return False, "", ""


# ===========================================================================
# STAGE 1 — geographic classification
# ===========================================================================

def geo_classify(norm_text: str, prior_province: str | None = None) -> dict:
    """
    Decide CALABARZON geographic relevance from normalised title+summary text.

    Returns dict:
      geo_level : 'strong' | 'region' | 'weak' | 'none'
      province  : one of the 5 CALABARZON provinces, 'CALABARZON', or ''
      city      : specific municipality/city if identified, else ''
      geo_note  : short trace of why
    """
    cities = _CITY_RE.findall(norm_text)
    if cities:
        # Pick the province of the first specific hit; record the city name.
        city = cities[0]
        prov = CITY_TO_PROVINCE.get(city, "")
        return {"geo_level": "strong", "province": prov,
                "city": city.title(), "geo_note": f"city:{city}"}

    # Safe bare province tokens (Cavite / Batangas / Laguna)
    safe = _SAFE_PROV_RE.findall(norm_text)
    if safe:
        prov = SAFE_PROVINCE_TOKENS[safe[0]]
        # Guard: a dominating other-province conflict with no CALABARZON city
        return {"geo_level": "strong", "province": prov,
                "city": "", "geo_note": f"province:{safe[0]}"}

    # Quezon province — only if it is NOT one of the false contexts and the
    # token appears (and ideally "quezon province" / with a food-agri frame).
    if _QUEZON_PROV_RE.search(norm_text) and not _QUEZON_FALSE_RE.search(norm_text):
        return {"geo_level": "strong", "province": "Quezon",
                "city": "", "geo_note": "province:quezon"}

    # Rizal province — only if not the hero / park / other-town contexts.
    if _RIZAL_PROV_RE.search(norm_text) and not _RIZAL_FALSE_RE.search(norm_text):
        return {"geo_level": "strong", "province": "Rizal",
                "city": "", "geo_note": "province:rizal"}

    # Region-level alias (CALABARZON / Region IV-A). Acceptable but not
    # province-specific.
    if _ALIAS_RE.search(norm_text):
        return {"geo_level": "region", "province": "CALABARZON",
                "city": "", "geo_note": "region_alias"}

    # Nothing in the text. Fall back to the weak prior from the recovered
    # GDELT pool (geocoded from the article body we no longer hold in full),
    # flagged 'weak' so Stage 4 downgrades / review-flags it.
    if prior_province:
        return {"geo_level": "weak", "province": prior_province,
                "city": "", "geo_note": "prior_province_only"}

    return {"geo_level": "none", "province": "", "city": "", "geo_note": "no_geo_signal"}


# ===========================================================================
# STAGE 2 — food-security candidate detection + signal extraction
# ===========================================================================

def food_signals(norm_text: str) -> dict:
    """
    Extract food-security signals from normalised text.

    Returns dict:
      is_food_candidate : bool  — any food keyword present
      has_core_food     : bool  — a strong 'actually about food' marker present
      food_hits         : list[str]  — matched food keywords (deduped, capped)
      dim_guess         : str   — first-hit dimension code (availability/...)
      event_type        : str
      severity          : 'high' | 'medium' | 'low'
      affected_population: str  (comma-joined)
      affected_commodity : str  (comma-joined)
    """
    hits = _FOOD_RE.findall(norm_text)
    is_cand = bool(hits)
    has_core = bool(_CORE_FOOD_RE.search(norm_text))

    # Dimension guess: first dimension whose keywords appear.
    dim_guess = ""
    for code, (_, kws) in FOOD_DIMENSIONS.items():
        if any(normalize(k) in norm_text for k in kws):
            dim_guess = code
            break

    # Event type: first matching family.
    event = ""
    for name, pats in EVENT_PATTERNS:
        if any(normalize(p) in norm_text for p in pats):
            event = name
            break

    # Severity heuristic.
    if any(normalize(s) in norm_text for s in SEVERITY_HIGH):
        severity = "high"
    elif any(normalize(s) in norm_text for s in SEVERITY_MED):
        severity = "medium"
    else:
        severity = "low"

    pops = [name for name, terms in POP_TERMS
            if any(normalize(t) in norm_text for t in terms)]
    comms = [name for name, terms in COMMODITY_TERMS
             if any(normalize(t) in norm_text for t in terms)]

    return {
        "is_food_candidate": is_cand,
        "has_core_food": has_core,
        "food_hits": sorted(set(hits))[:8],
        "dim_guess": dim_guess,
        "event_type": event,
        "severity": severity,
        "affected_population": ",".join(pops),
        "affected_commodity": ",".join(comms),
    }


def has_conflicting_province(norm_text: str, cal_province: str) -> bool:
    """True if a non-CALABARZON province clearly appears (used to downgrade
    region-alias / weak-prior articles that are really about elsewhere)."""
    return bool(_NONCAL_RE.search(norm_text))


# ===========================================================================
# Dimension mapping from the model's winning hypothesis to the thesis
# A–F food-security dimensions.
# ===========================================================================

HYPO_TO_DIMENSION: dict[str, str] = {
    "T1":  "availability_affordability",   # prices / supply / access
    "T2":  "utilization_nutrition",        # hunger / malnutrition / feeding
    "T3":  "assistance_access",            # food aid / subsidy / relief
    "T4":  "livelihood_access",            # poverty / unemployment
    "T5":  "stability_availability",       # transport / storage infra
    "T6":  "availability_production",      # crops / farmland / harvest
    "T7":  "stability",                    # displacement / evacuation
    "T8":  "stability",                    # unrest
    "T1b": "availability_production",      # fish kill / aquaculture
    "T9":  "livelihood_access",            # OFW remittance
}

# Which winning hypotheses count as DIRECT food-insecurity evidence
# (vs. an indirect causal driver).
DIRECT_HYPOTHESES = {"T1", "T2", "T3", "T6", "T1b"}
