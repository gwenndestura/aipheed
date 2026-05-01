"""
analyze_calabarzon_food_insecurity.py  (v2 — comprehensive revision)
----------------------------------------------------------------------
Deep analysis of all 1,557 Calabarzon news articles.
Determines:
  1. dswd_province_correct (Y/N/Y*)
  2. dswd_food_relevant (Y/N)

Scoring tiers:
  TIER 1 — Core food insecurity signals (weight 3):
    Direct hunger, malnutrition, food price/shortage, food distribution,
    government food programs, rice supply, food security.

  TIER 2 — Proximate food insecurity drivers (weight 2):
    Climate + agricultural damage, commodity price shock, fish kill,
    oil spill (fishermen/food supply impact), supply chain disruption,
    agricultural support (distribution of inputs, seedlings),
    poverty / livelihood loss directly tied to food access.

  TIER 3 — Contextual / latent food signals (weight 1):
    General typhoon/flood without food angle, agriculture/farming,
    fishermen/fisherfolk, food-industry news.

Verdicts:
  Y  — food insecurity relevant (TIER1 hit, or TIER2+ score >= 2,
        or TIER3 score >= 2 AND crop/agricultural keyword present)
  N  — not food insecurity relevant
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Province / municipality maps
# ---------------------------------------------------------------------------

PROVINCE_NAMES: dict[str, list[str]] = {
    "Batangas": [
        "batangas", "batangueño", "batangueno",
        "lipa", "tanauan", "sto. tomas", "santo tomas", "taal lake", "taal",
        "nasugbu", "san jose batangas", "mabini batangas", "lemery", "bauan",
        "laurel batangas", "malvar", "agoncillo", "alitagtag", "balayan",
        "cuenca", "ibaan", "lobo batangas", "mataas na kahoy", "padre garcia",
        "san juan batangas", "san luis batangas", "san nicolas batangas",
        "san pascual batangas", "santa teresita batangas", "talisay batangas",
        "tingloy", "calatagan", "batangas city", "calaca batangas",
    ],
    "Cavite": [
        "cavite", "caviteño", "caviteno",
        "bacoor", "dasmariñas", "dasmarinas", "imus", "general trias", "tagaytay",
        "trece martires", "kawit", "naic", "rosario cavite", "silang", "amadeo",
        "alfonso cavite", "carmona", "cavite city", "gen. mariano alvarez",
        "indang", "magallanes cavite", "mendez", "noveleta", "tanza", "ternate",
    ],
    "Laguna": [
        "laguna", "laguneño", "laguneno",
        "calamba", "santa rosa laguna", "biñan", "binan", "san pedro laguna",
        "cabuyao", "los baños", "los banos", "bay laguna", "calauan",
        "famy", "kalayaan laguna", "luisiana", "lumban", "mabitac",
        "magdalena laguna", "majayjay", "nagcarlan", "paete", "pagsanjan",
        "pakil", "pangil", "pila laguna", "san pablo laguna",
        "siniloan", "sta. cruz laguna", "santa cruz laguna", "victoria laguna",
        "laguna de bay", "sampaloc lake", "laguna lake",
    ],
    "Quezon": [
        "quezon province", "lucena", "tayabas", "candelaria quezon",
        "gumaca", "infanta quezon", "real quezon", "pagbilao",
        "padre burgos quezon", "mauban", "sariaya", "tiaong", "unisan",
        "agdangan", "alabat", "atimonan", "buenavista quezon",
        "calauag", "catanauan", "dolores quezon", "general luna quezon",
        "general nakar", "guinayangan", "jomalig", "lopez quezon",
        "lucban", "macalelon", "mulanay", "panukulan", "patnanungan",
        "perez quezon", "pitogo", "plaridel quezon", "polillo",
        "san andres quezon", "san antonio quezon", "san narciso quezon",
        "tagkawayan", "lucena city",
    ],
    "Rizal": [
        "rizal province", "antipolo", "cainta", "taytay rizal",
        "binangonan", "angono", "baras rizal", "cardona", "jala-jala",
        "morong rizal", "pililla", "rodriguez rizal", "san mateo rizal",
        "tanay", "teresa rizal", "rizal",
    ],
}

CALABARZON_ALIASES = [
    "calabarzon", "region iv-a", "region iva", "region 4a", "region iv a",
]

NON_CALABARZON_PROVINCES = [
    "ilocos", "cagayan valley", "nueva ecija", "pangasinan", "pampanga",
    "bulacan", "tarlac", "zambales", "nueva vizcaya", "aurora province",
    "metro manila", "quezon city", "makati",
    "bicol", "albay", "camarines", "sorsogon", "masbate", "catanduanes",
    "western visayas", "cebu", "iloilo", "bacolod", "negros",
    "eastern visayas", "leyte", "samar",
    "mindanao", "davao", "cagayan de oro", "zamboanga",
    "lanao", "cotabato", "maguindanao", "sultan kudarat",
    "palawan", "mindoro", "romblon", "marinduque",
    "aklan", "antique", "capiz", "guimaras",
    "surigao", "agusan", "bukidnon", "misamis",
    "benguet", "baguio", "ifugao", "kalinga", "mountain province",
]

# ---------------------------------------------------------------------------
# TIER 1 — Core food insecurity signals
# ---------------------------------------------------------------------------

TIER1_CORE: list[str] = [
    # ---- Hunger / malnutrition ----
    "hunger", "gutom", "nagugutom", "gutom na bata",
    "malnutrition", "malnutrisyon", "undernourish", "malnourish",
    "stunting", "wasting", "underweight", "undernutrition",
    "food insecurity", "food insecure", "food security",
    "food shortage", "food scarcity", "food crisis",
    "kakulangan ng pagkain", "walang makain", "wala nang makain",
    "hindi makakain", "pagkagutom",
    # ---- Children nutrition programs ----
    "feeding program", "supplementary feeding", "nutribun",
    "e-nutribun", "lugaw program", "batang busog", "malusog na bata",
    "batang matatag",  # anti-malnutrition campaign
    "nutrition gap", "nutrition program", "nutrisyon",
    # ---- Food prices — direct ----
    "food price", "presyo ng pagkain", "rice price", "presyo ng bigas",
    "price hike", "price spike", "price surge", "price increase",
    "price freeze", "basic necessities",
    "taas ng presyo", "pagmamahal ng bilihin", "mahal na pagkain",
    "food inflation", "food cpi", "palay price", "farmgate price",
    "vegetable price", "fish price", "presyo ng isda", "presyo ng gulay",
    "market price", "bilihin",
    # ---- Rice / staple supply ----
    "rice supply", "nfa rice", "nfa palay", "smuggled rice", "rice smuggl",
    "bigas suplay", "kakulangan ng bigas", "walang bigas",
    "food supply", "food distribution", "kakulangan ng suplay",
    "rice tariff", "rice importation", "NFA rice",
    # ---- Government food programs ----
    "ayuda", "food aid", "relief goods", "food pack", "food relief",
    "free rice", "libreng bigas", "dswd food", "da food",
    "food assistance", "food distribution", "food program",
    "4ps", "pantawid", "conditional cash transfer",
    "kadiwa", "kadiwa ng pangulo", "bigasang bayan",
    "national food authority", "mobile kitchen",
    # ---- Agricultural damage / supply disruption ----
    "crop damage", "crop loss", "crop failure", "pagkasira ng ani",
    "nasira ang ani", "damaged crops", "damaged rice", "palay damage",
    "harvest loss",
    # ---- Fish kill ----
    "fish kill", "fishkill", "patay na isda",
    # ---- Oil spill / fishing ban ----
    "fishing ban", "shellfish ban", "shellfish harvest ban",
    "fishermen affected", "fishers affected",
    # ---- Food production support ----
    "food productivity", "food development", "food security program",
    "food agri", "agri productivity",
    # ---- Food access / poverty ----
    "food access", "food affordab", "food expenditure",
    "subsistence", "pagkain ng isang beses",
    # ---- Specific food industry terms ----
    "p20 rice", "p20/kilo rice", "20 piso bigas",
    "DA readies", "da-apco", "da-calabarzon",
]

# ---------------------------------------------------------------------------
# TIER 2 — Proximate food insecurity drivers
# ---------------------------------------------------------------------------

TIER2_PROXIMATE: list[str] = [
    # ---- Commodity price shocks ----
    "commodity price", "price volatility", "price control", "price ceiling",
    "price monitoring", "price cap",
    "sibuyas presyo", "onion price", "sugar price", "presyo ng asukal",
    "cooking oil price", "pork price", "chicken price", "egg price",
    "presyo ng manok", "presyo ng baboy", "presyo ng itlog",
    "sugar mill", "sugar cane", "cane processing",
    "BOC finds", "imported sugar", "smuggled goods",
    # ---- Climate affecting food ----
    "typhoon damage", "flood damage", "crop loss", "typhoon victims food",
    "el nino drought", "el niño drought", "drought rice", "drought farm",
    "dry spell rice", "dry spell farm", "rainfall deficit",
    # ---- Agricultural support programs ----
    "vegetable seedling", "vegetable seeds", "sowing vegetable",
    "distribute seedling", "seedling distribution",
    "organic meat distribute", "distribute organic",
    "urban farming", "urban agriculture", "backyard farming",
    "farm-to-market road", "farm to market", "transport of goods",
    "supply chain disruption", "logistics disruption",
    "bantilan bridge", "bridge collapse",
    "road closed goods", "goods transport",
    # ---- Oil spill impact ----
    "oil spill", "oil spill cavite", "oil spill batangas",
    "fishing vessel sinks", "fishing vessel fire",
    # ---- Fishermen / fisherfolk livelihoods ----
    "fishermen affected", "fishermen income", "fishers income",
    "fisherfolk livelihood", "fishermen livelihood",
    "1,600 cavite fishermen",
    # ---- Poverty / livelihood affecting food ----
    "poverty", "kahirapan", "destitute",
    "livelihood", "hanapbuhay", "kabuhayan",
    "food expenditure", "household income", "purchasing power",
    "pababa ng kita", "nawalan ng kita",
    # ---- Unemployment ----
    "unemployment", "walang trabaho", "nawalan ng trabaho",
    "job loss", "layoff", "retrenchment",
    # ---- OFW / remittance ----
    "ofw remittance", "ofw food", "padala", "overseas worker food",
    "nabawasan ng kita", "nawalan ng remittance",
    # ---- Hunger survey ----
    "sws hunger", "hunger survey", "FIES", "hunger incidence",
    "food insecurity experience",
    # ---- Farm inputs affecting food ----
    "fertilizer price", "abono presyo", "farm input", "agricultural supply",
    "pesticide", "herbicide",
    # ---- ASF / disease affecting livestock ----
    "asf", "african swine fever", "bird flu", "avian flu",
    "hog ban", "pork ban",
]

# ---------------------------------------------------------------------------
# TIER 3 — Contextual / latent food signals
# ---------------------------------------------------------------------------

TIER3_CONTEXTUAL: list[str] = [
    # ---- Climate events (without direct food angle) ----
    "bagyo", "typhoon", "baha", "flood", "tagtuyot", "drought", "el nino",
    "el niño", "la nina", "la niña", "storm surge", "landslide",
    "signal number", "calamity", "kalamidad", "sakuna", "nasalanta",
    "binaha", "lumikas", "evacuation", "evacuees", "displaced",
    # ---- Agriculture / fishing (general) ----
    "harvest", "ani", "palay", "mais", "vegetable", "gulay", "agriculture",
    "magsasaka", "farmer", "fisherfolk", "mangingisda", "fishermen",
    "fishers", "fisher",
    "crop", "pananim", "sakahan", "taniman",
    "kapeng barako", "barako", "kape", "coffee farmer",
    "moringa", "aquaponics", "hydroponics",
    "seedling", "seeds",
    "mango", "saging", "niyog", "kamoteng kahoy",
    "baboy", "manok", "itlog", "karne", "isda",
    "shellfish", "tahong", "hipon",
    "tilapia", "bangus", "sardine",
    # ---- Food industry / production (general) ----
    "food produc", "food manufactur",
    "agribusiness", "agri-business", "agribiz",
    "food plant", "food factory",
    "meat produc", "egg produc", "dairy",
    "hog farm", "poultry farm", "livestock",
    # ---- Economic signals ----
    "inflation", "cpi", "presyo",
    "economy", "minimum wage", "sahod",
    "poverty rate",
    # ---- Relief / aid (general) ----
    "relief", "humanitarian", "aid", "tulong",
    "food", "pagkain",
    # ---- Supply chain ----
    "supply chain", "logistics", "import", "export",
    "port", "warehouse", "cold storage",
]

# ---------------------------------------------------------------------------
# Province keywords for province_correct check
# ---------------------------------------------------------------------------

FOOD_AGRI_CONTEXT_WORDS = [
    "crop", "palay", "ani", "harvest", "magsasaka", "farmer", "fishermen",
    "fishers", "fisher", "fisherfolk", "gulay", "agriculture", "sakahan",
    "food", "pagkain", "fish", "isda", "vegetable", "livestock", "poultry",
    "tilapia", "bangus", "sardine", "shellfish", "tahong", "hipon",
    "moringa", "rice", "bigas", "mais", "corn",
    # additional food/agricultural products
    "kapeng", "barako", "kape", "coffee",
    "mango", "saging", "niyog", "kamoteng",
    "baboy", "manok", "itlog", "karne",
    "cold storage", "grains",
    "egg", "pork", "chicken", "beef", "meat",
    "sardine", "tawilis", "bangus", "tilapia",
    "farm", "farming", "agri",
]


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def _matches(keywords: list[str], text: str) -> list[str]:
    return [kw for kw in keywords if kw.lower() in text]


def score_food_insecurity(title: str) -> tuple[str, int, list[str]]:
    """Returns (verdict, score, matched_keywords)."""
    text = title.lower()

    t1 = _matches(TIER1_CORE, text)
    t2 = _matches(TIER2_PROXIMATE, text)
    t3 = _matches(TIER3_CONTEXTUAL, text)

    score = len(t1) * 3 + len(t2) * 2 + len(t3) * 1

    # TIER 1 hit → always Y
    if t1:
        verdict = "Y"
    # TIER 2 hit → Y (score >= 2 guaranteed)
    elif t2:
        verdict = "Y"
    # TIER 3 only, high density (>=3): strong agricultural/food context even without exact match
    elif score >= 3:
        verdict = "Y"
    # TIER 3 only, score >= 2 AND recognized food/agricultural keyword present
    elif score >= 2 and any(k in text for k in FOOD_AGRI_CONTEXT_WORDS):
        verdict = "Y"
    # TIER 3 only: any hit with "food" or "pagkain" in title
    elif score >= 1 and ("food" in text or "pagkain" in text):
        verdict = "Y"
    else:
        verdict = "N"

    return verdict, score, t1 + t2 + t3


# ---------------------------------------------------------------------------
# Province correctness
# ---------------------------------------------------------------------------

def check_province_correct(title: str, assigned_province: str) -> tuple[str, str]:
    """Returns (verdict, note). verdict: 'Y' | 'Y*' | 'N'."""
    text = title.lower()

    # Direct assigned-province name / municipality match
    for kw in PROVINCE_NAMES.get(assigned_province, [assigned_province.lower()]):
        if kw.lower() in text:
            return "Y", f"direct:{assigned_province.lower()}"

    # CALABARZON alias
    if any(alias in text for alias in CALABARZON_ALIASES):
        return "Y*", "calabarzon_alias"

    # Quezon City ≠ Quezon Province (common geocoding error)
    if assigned_province == "Quezon" and "quezon city" in text:
        return "N", "quezon_city_not_quezon_province"

    # Rizal province ≠ Quezon City (common geocoding error)
    if assigned_province == "Rizal" and "quezon city" in text:
        return "N", "quezon_city_not_rizal_province"

    # Conflicting non-Calabarzon province
    conflicting = [p for p in NON_CALABARZON_PROVINCES if p in text]
    if conflicting:
        return "N", f"conflict:{','.join(conflicting[:3])}"

    # Philippines / national level — geocoded by query keyword
    if "philippines" in text or "pilipinas" in text:
        return "Y*", "national_article_geocoded"

    # No geographic signal in title — geocoding was from article body
    return "Y*", "no_title_geo_signal"


# ---------------------------------------------------------------------------
# Category labeler
# ---------------------------------------------------------------------------

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("food_price",          ["food price", "rice price", "price hike", "price spike",
                             "price surge", "price increase", "price freeze",
                             "basic necessities", "food inflation", "market price",
                             "onion price", "pork price", "chicken price", "egg price",
                             "sugar price", "taas ng presyo", "bilihin", "commodity price"]),
    ("hunger_malnutrition", ["hunger", "gutom", "malnutrition", "stunting", "wasting",
                             "underweight", "malnourish", "undernourish", "food insecurity",
                             "food insecure", "food shortage", "food scarcity", "food crisis",
                             "kakulangan ng pagkain", "walang makain"]),
    ("food_assistance",     ["ayuda", "food aid", "relief goods", "food pack", "food relief",
                             "feeding program", "nutribun", "batang busog", "mobile kitchen",
                             "4ps", "pantawid", "food assistance", "food distribution",
                             "kadiwa", "libreng bigas", "free rice", "food program"]),
    ("rice_supply",         ["rice supply", "nfa rice", "smuggled rice", "rice tariff",
                             "rice importation", "p20 rice", "p20/kilo"]),
    ("crop_damage",         ["crop damage", "crop loss", "harvest loss", "flood damage",
                             "typhoon damage", "palay damage", "nasira ang ani",
                             "kapeng barako", "ashfall ruins", "crop failure"]),
    ("fish_kill_oil_spill", ["fish kill", "fishkill", "oil spill", "fishing ban",
                             "shellfish ban", "fishermen affected", "fishers affected"]),
    ("climate_shock",       ["bagyo", "typhoon", "baha", "flood", "tagtuyot", "drought",
                             "el nino", "el niño", "la nina", "la niña", "calamity",
                             "nasalanta", "binaha"]),
    ("commodity_supply",    ["sugar mill", "imported sugar", "BOC finds sugar",
                             "asf", "african swine fever", "hog ban",
                             "supply chain", "bantilan bridge", "transport of goods",
                             "farm-to-market road"]),
    ("poverty_livelihood",  ["poverty", "kahirapan", "livelihood", "hanapbuhay",
                             "unemployment", "nawalan ng trabaho", "job loss", "layoff",
                             "ofw remittance", "padala"]),
    ("agriculture",         ["harvest", "ani", "palay", "agriculture", "magsasaka",
                             "farmer", "fisherfolk", "fishermen", "fishers",
                             "crop", "vegetable", "gulay", "seedling",
                             "aquaponics", "hydroponics", "urban farm", "urban agri"]),
]


def get_categories(title: str) -> list[str]:
    text = title.lower()
    return [cat for cat, kws in CATEGORY_RULES if any(k.lower() in text for k in kws)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    data_path = Path("dswd_article_sample.xlsx")
    output_path = Path("dswd_article_sample_analyzed.xlsx")

    print(f"Loading {data_path} ...")
    df = pd.read_excel(data_path)
    print(f"  {len(df)} articles loaded.\n")

    prov_results, fi_results, notes_results = [], [], []

    for _, row in df.iterrows():
        title = str(row.get("title", "") or "")
        province = str(row.get("province_name", "") or "")

        pv, pnote = check_province_correct(title, province)
        fv, fscore, fkws = score_food_insecurity(title)
        cats = get_categories(title)

        parts = []
        if cats:
            parts.append("cats:" + ",".join(cats))
        parts.append(f"fi_score:{fscore}")
        if fkws:
            parts.append("kw:" + "|".join(fkws[:5]))
        parts.append(f"prov:{pnote}")

        prov_results.append(pv)
        fi_results.append(fv)
        notes_results.append(" | ".join(parts))

    df["dswd_province_correct (Y/N)"] = prov_results
    df["dswd_food_relevant (Y/N)"] = fi_results
    df["dswd_notes"] = notes_results

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    total = len(df)
    food_y = (df["dswd_food_relevant (Y/N)"] == "Y").sum()
    food_n = (df["dswd_food_relevant (Y/N)"] == "N").sum()
    prov_hard = (df["dswd_province_correct (Y/N)"] == "Y").sum()
    prov_soft = (df["dswd_province_correct (Y/N)"] == "Y*").sum()
    prov_n    = (df["dswd_province_correct (Y/N)"] == "N").sum()

    print("=" * 72)
    print("CALABARZON FOOD INSECURITY ANALYSIS REPORT  (v2)")
    print("=" * 72)
    print(f"\nTotal articles analyzed   : {total}")
    print()
    print("FOOD INSECURITY RELEVANCE")
    print(f"  Relevant (Y)            : {food_y:>5}  ({food_y/total*100:.1f}%)")
    print(f"  Not relevant (N)        : {food_n:>5}  ({food_n/total*100:.1f}%)")
    print()
    print("PROVINCE CORRECTNESS")
    print(f"  Confirmed (Y)           : {prov_hard:>5}  ({prov_hard/total*100:.1f}%)  — province/municipality in title")
    print(f"  Probable  (Y*)          : {prov_soft:>5}  ({prov_soft/total*100:.1f}%)  — CALABARZON alias / national article")
    print(f"  Mismatch  (N)           : {prov_n:>5}  ({prov_n/total*100:.1f}%)  — conflicting non-Calabarzon province")
    print()
    print("BY PROVINCE")
    for prov in ["Batangas", "Cavite", "Laguna", "Quezon", "Rizal"]:
        sub = df[df["province_name"] == prov]
        fi = (sub["dswd_food_relevant (Y/N)"] == "Y").sum()
        pc = sub["dswd_province_correct (Y/N)"].isin(["Y", "Y*"]).sum()
        pn = (sub["dswd_province_correct (Y/N)"] == "N").sum()
        print(f"  {prov:<12}  total={len(sub):>4}  "
              f"food_relevant={fi:>4} ({fi/len(sub)*100:.0f}%)  "
              f"prov_ok={pc} ({pc/len(sub)*100:.0f}%)  "
              f"prov_mismatch={pn}")

    # ---- Food categories (food-relevant articles only) ----
    food_rel = df[df["dswd_food_relevant (Y/N)"] == "Y"]
    print()
    print(f"FOOD INSECURITY CATEGORIES (food-relevant articles, N={food_y})")
    cat_counts: dict[str, int] = {}
    for notes in food_rel["dswd_notes"]:
        for part in str(notes).split(" | "):
            if part.startswith("cats:"):
                for c in part[5:].split(","):
                    if c:
                        cat_counts[c] = cat_counts.get(c, 0) + 1
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<28} {cnt:>4}  ({cnt/food_y*100:.1f}%)")

    # ---- Province mismatches ----
    mismatches = df[df["dswd_province_correct (Y/N)"] == "N"]
    print()
    print(f"PROVINCE MISMATCHES (N={len(mismatches)})")
    for _, row in mismatches.head(30).iterrows():
        print(f"  [{row['province_name']}] {row['title'][:85]}")
        print(f"    note: {row['dswd_notes'].split('prov:')[-1]}")

    # ---- Non-food sample (so researcher can manually review) ----
    non_food = df[df["dswd_food_relevant (Y/N)"] == "N"]
    print()
    print(f"NON-FOOD-RELEVANT SAMPLE (showing 50 of {len(non_food)})")
    for _, row in non_food.head(50).iterrows():
        score_str = ""
        for part in str(row["dswd_notes"]).split(" | "):
            if part.startswith("fi_score:"):
                score_str = part
        print(f"  [{row['province_name'][:3]}] [{score_str}] {row['title'][:85]}")

    print()
    print(f"Saving to {output_path} ...")
    df.to_excel(output_path, index=False)
    print("Done.")
    return df


if __name__ == "__main__":
    main()
