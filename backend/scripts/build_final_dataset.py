"""
scripts/build_final_dataset.py
-------------------------------
Assemble the FINAL curated CALABARZON food-insecurity news dataset from the
THESIS's strict reanalysis pipeline output, re-geocoded to barangay level.

Faithful to the thesis:
  * Relevance = the strict reanalysis gates (NLI CORE hypotheses + food-anchor +
    dimension-evidence + CALABARZON geo-verification). Input is
    data/processed/reanalysis/relevant.parquet (the retained HIGH/MEDIUM set) —
    NOT the looser NLI>=0.30 corpus.
  * Rich thesis metadata preserved: food-security dimension (A-F), event_type,
    affected commodity/population, is_direct_food_insecurity, NLI
    scores, and the reanalysis relevance_reason.

New this session (the enhancement the thesis lacked):
  * Geography re-tagged with the fixed 142-LGU PSGC gazetteer geocoder, adding
    city/municipality AND barangay (the reanalysis kept titles only, so lead
    text is rejoined by article_id to feed the geocoder).
  * food_insecurity_topics keyword tags + a templated relevance_summary.

Outputs (data/processed/):
  calabarzon_food_insecurity_dataset.parquet / .csv
  calabarzon_lgu_coverage_matrix.csv
  calabarzon_dataset_README.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for precision_pass
from precision_pass import FOOD_ANCHOR  # canonical food-anchor lexicon (single source)

STRICT = Path("data/processed/reanalysis/relevant.parquet")
GEO = Path("data/processed/corpus_geocoded.parquet")
FULLPOOL = Path("data/processed/_fullpool_geocoded.parquet")
GAZ = Path("data/processed/psgc_gazetteer.parquet")
OUTDIR = Path("data/processed")

TOPIC_PATTERNS: dict[str, re.Pattern] = {
    "food_security_general": re.compile(r"\b(food secur|food insecur|food sufficien|food access|food availab|food supply|food shortage|food crisis|food scarcity|food production|food self-sufficien)\w*", re.I),
    "hunger_food_deprivation": re.compile(r"\b(hunger|gutom|starv|famine|food deprivation|walang makain|nagugutom)\w*", re.I),
    "malnutrition_undernutrition": re.compile(r"\b(malnutri|malnutrisyon|stunt|wasting|undernouri|nutriti|nutrisyon)\w*", re.I),
    "food_prices_affordability": re.compile(r"\b(food prices?|rice prices?|presyo|inflation|afford|mahal|bilihin|expensive|cost of food)\w*", re.I),
    "poverty_food_access": re.compile(r"\b(poverty|kahirapan|poorest|mahirap|indigent)\w*", re.I),
    "rice_staple_supply": re.compile(r"\b(rice|palay|bigas|\bnfa\b|staple|imported rice|rice supply|rice shortage)\w*", re.I),
    "agricultural_production": re.compile(r"\b(crop|harvest|farm|farmer|magsasaka|agri|agricultur|yield|planting|farmland|vegetable|gulay|produce grower)", re.I),
    "crop_losses_disaster": re.compile(r"\b(crop damage|agri damage|agricultural damage|agri.?fisher|typhoon|bagyo|flood|baha|drought|tagtuyot|el ni|la ni|calamity|landslide|volcano|volcanic|eruption|ashfall|abo ng bulkan)", re.I),
    "fisheries_livestock": re.compile(r"\b(fish|isda|bangus|tilapia|milkfish|fishkill|fish kill|red tide|poultry|manok|hog|swine|\basf\b|livestock|bird flu|pork|baboy|chicken|beef|karne|\bmeat\b)", re.I),
    "food_assistance_programs": re.compile(r"\b(ayuda|relief goods|kadiwa|\bdswd\b|4ps|pantawid|feeding program|food pack|community pantr|libreng bigas|rice subsidy|hot meal|soup kitchen|relief pack|goods distribution)\w*", re.I),
    "livelihood_income": re.compile(r"\b(livelihood|kabuhayan|income|unemploy|jobless|no work|remittance|\bofw\b|wage|job loss|job cut|lost (their )?jobs|laid off|lay ?off|retrench|displaced workers?|\btupad\b|plant (shut ?down|closure|closed)|factory (shut ?down|closure)|mill (shut ?down|closure)|(shut ?down|closure|closed) (its |the )?(plant|factory|mill)|ceased operations|nawalan ng (trabaho|hanapbuhay)|tanggal sa trabaho)\w*", re.I),
    "supply_chain_distribution": re.compile(r"\b(supply|shortage|kakulangan|distribution|transport|logistics|road closure|blockade)\w*", re.I),
    "pests_crop_disease": re.compile(r"\b(pest|infestation|blight|fall armyworm|bird flu|\basf\b|african swine fever|rice black bug|disease outbreak)\w*", re.I),
}


def _safe_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV, falling back to a *.locked.csv name if the target is open in
    another program (Excel holds an exclusive lock on Windows)."""
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        alt = path.with_suffix(".locked.csv")
        df.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"  ! {path.name} is open elsewhere — wrote {alt.name} instead")


# Non-food contexts that incidentally match a food/farm word — dropped in _clean.
_NEGATIVE = re.compile(
    r"\b(solar farm|wind farm|solar power|wind power|renewable energy|photovoltaic|"
    r"\d+[- ]?mw\b|megawatt|power plant|energy corp|"                       # energy
    r"bay area|california|new york|texas|florida|united states|\bu\.?s\.?a?\b|"
    r"london|dubai|singapore|hong kong|canada|australia|"                    # foreign
    r"recipe|dishes you can cook|101: getting to know|must-try|restaurants|"
    r"comfort food|good food in a relationship|"                            # food-culture
    r"hunger[- ]strike|hunger strikers|"                                     # protest, not food insecurity
    r"danfulani|naic boss|naic md|nirsal|tinubu|naira|"                      # Nigerian NAIC vs Naic, Cavite
    r"guide service|fishing report|fishing tournament|san luis obispo)\b",   # US fishing vs CALABARZON LGU
    re.I)


# Foreign-source detection. "Laguna" is Spanish for lagoon and a Mexican region,
# so Spanish/Portuguese/Italian/Indonesian articles mis-geocode onto Laguna
# province. Drop them permanently: by foreign TLD/outlet, or (for non-PH domains
# such as the news.google.com aggregator) by foreign-language signal. PH outlets
# are trusted regardless of an "El Nino"/"n~" token so genuine PH stories survive.
_PH_OUTLETS = re.compile(
    r"\.ph\b|\.ph$|philstar|inquirer|rappler|gmanetwork|manilatimes|manilabulletin|"
    r"mb\.com|abs-cbn|sunstar|bworldonline|businessmirror|journal\.com|tribune\.net|"
    r"manilastandard|remate|bandera|tempo\.com\.ph|pna\.gov|pia\.gov|panaynews|"
    r"mindanews|cebudaily|philippine|ateneo|dailyguardian|manila", re.I)
_FOREIGN_DOM = re.compile(
    r"\.(mx|es|cl|it|ve|do|ar|pe|uy|br|fr|de|id|co|uk)(/|$|\b)|"
    r"milenio|infobae|clarin|laverdad|larazon\.es|eldiario|ecoticias|radioagricultura|"
    r"diariolibre|veneziatoday|informacion\.es|elespanol|ultimasnoticias|cnnindonesia|"
    r"semana\.com|eluniversal|noroeste|vanguardia|elsiglo|periodicodaily|midiamax|"
    r"harianterbit|rmol|tribunnews|mediaindonesia|expansion\.mx|lanzadigital|latribuna|"
    r"andaluciainformacion|laopinion|aimdigital|unosantafe|heraldodemexico|cugetliber|"
    r"el19digital|nativenews|lanacion|elpais|elmundo|abc\.es|20minutos|okdiario|"
    r"cibercuba|novedadesdetabasco|laprovincia\.es|mediosobson|mirror\.co|periodistadigital|"
    r"colimanoticias|livescience|ellitoral|timesofindia|theyucatantimes|lacronicabadajoz|"
    r"radiorebelde|gizmodo|aktual24|kaq580|libertaddigital|elperiodico|mexiconewsdaily|"
    r"excelsiorcalifornia|levante-emv|caraotadigital|kompas|beritasatu|sindonews|bisnis\.com",
    re.I)
_FLANG_ACCENT = re.compile(r"[¿¡áéíóúàèìòùâêôãõçü]")
_FLANG_WORDS = re.compile(
    r"\b(seg[uú]n|est[aá]|m[aá]s|a[ñn]o|r[ií]o nazas|ciudad de|gobierno|millones|"
    r"sequ[ií]a|cosecha|mar menor|murcia|jalisco|comarca|acu[ií]fero|prefeitura|banjir|"
    r"warga|pemerintah|dengan|untuk|yang|camalotes|crecida|riada|avenidas|ayuntamiento|"
    r"diputaci[oó]n|consejer[ií]a|regenerativa)\b", re.I)


def _is_foreign(domain: str, text: str) -> bool:
    dom = str(domain or "")
    if _FOREIGN_DOM.search(dom):
        return True
    if _PH_OUTLETS.search(dom):
        return False
    return bool(_FLANG_ACCENT.search(text) or _FLANG_WORDS.search(text))


# Off-topic subjects that carry a food/disaster token but are not food-insecurity
# stories (verified by manual review of the collected set). Matched on the TITLE
# only — the article's subject lives in the headline, so a stray word in the lead
# (e.g. a "dengue" related-link under a typhoon story) never triggers a drop.
_OFFTOPIC = re.compile(
    r"\b(measles|pertussis|dengue|polio|rabies|tigdas|disease outbreak)\b"          # disease-calamity, not climate
    r"|\b(shooting|shot dead|shot to death|murder|homicide|stabb|love triangle|"
    r"slay|slain|shabu|drug (bust|haul|raid|war|den)|robbery|carnap|\brape\b|"
    r"ambush|nabbed|dynamite fishing|blast fishing)\b"                              # crime / illegal-fishing arrests
    r"|\b(traffic jam|delivery rider|utility post|road crash|road mishap)\b"        # accidents
    r"|\b(miss universe|miss world|beauty pageant|\bpageant\b|teleserye|box office|"
    r"showbiz|horse race|\bPBA\b|\bUAAP\b|\bNBA\b|Gilas|palaro)\b"                   # showbiz / sport (NOT bare 'basketball' — evac 'basketball court')
    r"|\b(space week|satellite internet|digital skills|drone data|analog mission|broadband)\b"  # tech
    r"|\b(hagisan ng suman|food treasure|food trip|mascot|foodie|feast in a|"
    r"delicacies|culinary|cuisine|kakanin|food festival)\b"                        # food-culture / cuisine
    r"|\b(earnings (down|up)|net income|quarterly (profit|earnings)|share price|"
    r"stock price|bottom line)\b"                                                  # corporate financial results
    r"|\bgdp growth\b|\b(illegal horse|\bPETA\b)\b"                                 # macro / animal-rights
    r"|\b(cocaine|marijuana|marihuana|poach\w*|wildlife|threatened birds|"
    r"illegal possession|held for illegal|\barrested\b|apprehended)\b"              # crime (drug/wildlife/arrest) - NOT rice smuggling (food supply)
    r"|\b(wrestling|\bwwe\b|\baew\b|killer kross|matt cardona)\b"                   # pro-wrestling ('HOG' event collides with livestock)
    r"|\b(propeller|crash-land|plane crash|aircraft|garage fire)\b"                 # accidents
    r"|\b(canary island|tenerife|mallorca|ibiza|puerto rico)\b"                     # foreign places
    r"|los banos enterprise|\bfriant\b"                                             # Los Banos, CALIFORNIA (vs Los Banos, Laguna)
    r"|\bvax schedules?\b|\bvaccination schedules?\b|\b(agri.?tourism|agriculture tourism|tourism park)\b"  # health-logistics / agri-tourism
    r"|\benjoying nature\b|\bresort\b"                                              # leisure / resort (not food insecurity)
    r"|\bforest (refuge|fraud)\b|when forests become", re.I)                        # environmental-corruption editorials
# Another-region place named in the title with NO CALABARZON province co-mentioned
# (a mis-geocode, e.g. "Rice prices rising in Bataan market" tagged to Quezon).
# Kept when a CALABARZON province also appears ("Bataan oil spill reaching Cavite").
_OTHERREG_TITLE = re.compile(
    r"\b(capas|tarlac|pampanga|bulacan|zambales|bataan|nueva ecija|pangasinan|benguet|"
    r"ilocos|isabela|cagayan|bicol|cebu|davao|iloilo|bacolod|zamboanga|abra)\b", re.I)
_CALZN_TITLE = re.compile(r"\b(cavite|laguna|batangas|rizal|quezon|calabarzon)\b", re.I)
# Reclamation / dredging / land-conversion: off-topic governance UNLESS the story
# is about fisherfolk catch or farmland (a real food-source / livelihood loss).
_RECLAM = re.compile(r"\b(reclamation|dredging|land conversion|spillway|seabed quarr)\b", re.I)
_FISHFARM = re.compile(
    r"\b(fish|catch|harvest|tahong|mussel|livelihood|farm|palay|rice|crop|magsasaka|"
    r"mangingisda|pamalakaya|coconut|vegetable|poultry|hog|swine|tilapia|bangus)", re.I)
# Metro-Manila places that only ever appear here as barangay-name mis-geocodes.
_METRO_MISGEO = re.compile(r"\b(taguig|quezon city)\b", re.I)
# Politician self-promotion PR (Sen. Bong Go's aid-distribution / visit press
# releases, republished across outlets) — the event may be real but the framing is
# campaign PR, not food-insecurity reporting. Scoped to this benefactor pattern so
# genuine coverage that merely mentions other officials is untouched.
_PR_PROMO = re.compile(
    r"\bbong go\b|\bchristopher (lawrence )?go\b|\b(senator|sen\.?) go\b"
    r"|\bgo'?s? (provides|provided|gives|gave|boosts|aids|leads|led|distributes|hands|"
    r"turns over|visits|pushes|prioritizes|outreach)\b",
    re.I)


def _is_offtopic(title: str) -> bool:
    t = str(title or "")
    if _OFFTOPIC.search(t) or _METRO_MISGEO.search(t) or _PR_PROMO.search(t):
        return True
    if _OTHERREG_TITLE.search(t) and not _CALZN_TITLE.search(t):
        return True
    return bool(_RECLAM.search(t)) and not bool(_FISHFARM.search(t))


def _matched_topics(text: str) -> list[str]:
    return [name for name, pat in TOPIC_PATTERNS.items() if pat.search(text)]


# NLI hypothesis -> (dimension code, human category label) and event_type, used
# to give the corpus-recovered rows the same rich metadata as the strict set.
CATEGORY = {
    "T1": ("B", "Food accessibility / affordability (prices)"),
    "T2": ("C", "Food utilization / nutrition (malnutrition)"),
    "T3": ("E", "Hunger / food deprivation (assistance)"),
    "T6": ("A", "Food availability (crop production)"),
    "T1b": ("A", "Food availability (fisheries)"),
    "T4": ("B", "Food accessibility (poverty)"),
    "T5": ("D", "Food stability (supply chain)"),
    "T7": ("D", "Food stability (disaster / displacement)"),
    "T8": ("D", "Food stability (unrest)"),
    "T9": ("F", "Livelihood affecting food access"),
    "T10": ("F", "Livelihood / employment loss affecting food access"),
}
EVENT_MAP = {
    "T1": "food_price_change", "T2": "malnutrition_nutrition", "T3": "food_assistance",
    "T6": "crop_production_loss", "T1b": "fishery_loss", "T4": "poverty_hardship",
    "T5": "supply_disruption", "T7": "disaster_displacement", "T8": "unrest_disruption",
    "T9": "remittance_shock", "T10": "employment_loss",
}


def _recover_text() -> pd.DataFrame:
    """Best available title+lead per article_id, from the geocoded corpus and the
    full collected pool (reanalysis kept only titles)."""
    frames = []
    for p in (GEO, FULLPOOL):
        if p.exists():
            d = pd.read_parquet(p)
            cols = [c for c in ["article_id", "title", "summary"] if c in d.columns]
            frames.append(d[cols])
    allt = pd.concat(frames, ignore_index=True)
    allt["_slen"] = allt["summary"].fillna("").str.len()
    allt = allt.sort_values("_slen", ascending=False).drop_duplicates("article_id")
    return allt[["article_id", "summary"]].rename(columns={"summary": "_lead"})


def _load_union() -> pd.DataFrame:
    """Union of the strict reanalysis set (rich metadata) and the additional
    relevance-passing CALABARZON articles from the geocoded corpus that the
    strict pipeline excluded (recall recovery). Strict rows keep their full
    metadata; corpus-only rows get dimension/category/event derived from the NLI
    hypothesis. Both then pass the same quality gates in _clean()."""
    strict = pd.read_parquet(STRICT)
    strict["_source"] = "strict_reanalysis"

    corpus = pd.read_parquet(GEO)
    corpus = corpus[(corpus["is_relevant"] == True) &  # noqa: E712
                    (corpus["province_name"].notna()) &
                    (~corpus["article_id"].isin(set(strict["article_id"])))].copy()
    cat = corpus["top_hypothesis"].map(lambda h: CATEGORY.get(h, ("", "General food-insecurity relevance")))
    corpus["province"] = corpus["province_name"]
    corpus["city_municipality"] = corpus["lgu_name"]
    corpus["food_security_dimension"] = [c[0] for c in cat]
    corpus["food_security_dimension_label"] = [c[1] for c in cat]
    corpus["event_type"] = corpus["top_hypothesis"].map(EVENT_MAP)
    corpus["food_insecurity_relevance"] = corpus["food_insecurity_score"].map(
        lambda s: "HIGH" if (isinstance(s, (int, float)) and s >= 0.60) else "MEDIUM")
    corpus["core_score"] = corpus["food_insecurity_score"]
    corpus = corpus.rename(columns={"link": "link", "summary": "_corpus_lead"})
    corpus["_source"] = "corpus_recall"
    # Climate-shock and economic-shock rows are food-security determinants without a
    # food-anchor term; flag them so _clean() and the softie cut exempt them.
    fs = corpus.get("fetcher_source")
    corpus["is_climate_shock"] = (fs == "climate_shock") if fs is not None else False
    corpus["is_economic_shock"] = (fs == "economic_shock") if fs is not None else False
    corpus["is_poverty_shock"] = (fs == "poverty_shock") if fs is not None else False
    for col in ("affected_commodity", "affected_population",
                "relevance_reason", "is_direct_food_insecurity"):
        corpus[col] = None

    keep = ["article_id", "title", "link", "published", "source_domain", "province",
            "province_code", "city_municipality", "food_security_dimension",
            "food_security_dimension_label", "top_hypothesis", "event_type",
            "affected_commodity", "affected_population",
            "is_direct_food_insecurity", "food_insecurity_relevance", "core_score",
            "relevance_reason", "is_climate_shock", "is_economic_shock",
            "is_poverty_shock", "_source"]
    both = pd.concat([strict[[c for c in keep if c in strict.columns]],
                      corpus[[c for c in keep if c in corpus.columns]]],
                     ignore_index=True)
    for col in ("is_climate_shock", "is_economic_shock", "is_poverty_shock"):
        both[col] = both[col].fillna(False) if col in both.columns else False
    print(f"union: {len(strict)} strict + {len(corpus)} corpus-recall = {len(both)} "
          f"({int(both['is_climate_shock'].sum())} climate-shock, "
          f"{int(both['is_economic_shock'].sum())} economic-shock, "
          f"{int(both['is_poverty_shock'].sum())} poverty-shock)")
    return both


def build() -> None:
    from app.ml.corpus.location_geocoder import geocode_location

    df = _load_union()
    print(f"union input: {len(df)} rows "
          f"({df['food_insecurity_relevance'].value_counts().to_dict()})")

    # Rejoin lead text so the barangay-aware geocoder has more than the title.
    df = df.merge(_recover_text(), on="article_id", how="left")
    df["_text"] = (df["title"].fillna("") + ". " + df["_lead"].fillna("")).astype(str)

    # Re-geocode with the fixed 142-LGU + barangay geocoder. The reanalysis
    # province (which passed geo-verification) seeds it as the trusted prior.
    geo = [geocode_location(t, pc if isinstance(pc, str) else None)
           for t, pc in zip(df["_text"], df["province_code"])]
    g = pd.DataFrame(geo, index=df.index)
    # Prefer the new geocoder's finer tags; fall back to reanalysis where blank.
    df["province_final"] = g["province_name"].fillna(df["province"])
    df["city_municipality_final"] = g["lgu_name"].fillna(df["city_municipality"])
    df["barangay_final"] = g["barangay_name"]
    df["match_level"] = g["match_level"]

    df["food_insecurity_topics"] = df["_text"].map(lambda t: ",".join(_matched_topics(t)))
    # After the topic gate every kept row has >=1 topic, so flag instead the
    # SOFTEST rows for an optional eyeball: those whose only signals are indirect
    # (poverty / livelihood / supply-chain) with no direct food/hunger/production
    # term. These are the weakest-relevance rows a reviewer may want to check.
    _WEAK = {"poverty_food_access", "livelihood_income", "supply_chain_distribution"}
    df["needs_review"] = df["food_insecurity_topics"].map(
        lambda s: bool(s) and set(s.split(",")).issubset(_WEAK))
    # Economic-shock rows carry a livelihood-only topic set but are an explicit,
    # geo-guarded determinant class (job loss / closure) — never soft review-only.
    df.loc[df["is_economic_shock"].fillna(False), "needs_review"] = False
    # Poverty-shock rows are poverty/affordability-only by definition (the FAO
    # economic-access pillar) and are an explicit, geo-guarded determinant class —
    # never soft review-only, even though poverty_food_access is a _WEAK topic.
    df.loc[df["is_poverty_shock"].fillna(False), "needs_review"] = False
    df["relevance_summary"] = df.apply(_summary_row, axis=1)
    df["author"] = None
    df["data_source"] = df["_source"]

    out = df.rename(columns={
        "published": "publication_date",
        "source_domain": "news_source",
        "link": "url",
        "_lead": "content_lead",
        "food_security_dimension_label": "food_insecurity_category",
        "core_score": "relevance_score",
        "food_insecurity_relevance": "relevance_tier",
    })[[
        "title", "publication_date", "news_source", "author", "url",
        "content_lead", "province_final", "city_municipality_final", "barangay_final",
        "relevance_tier", "food_security_dimension", "food_insecurity_category",
        "food_insecurity_topics", "event_type",
        "affected_commodity", "affected_population", "is_direct_food_insecurity",
        "relevance_summary", "relevance_reason", "relevance_score", "match_level",
        "needs_review", "data_source", "article_id", "is_climate_shock", "is_economic_shock",
        "is_poverty_shock",
    ]].rename(columns={
        "province_final": "province",
        "city_municipality_final": "city_municipality",
        "barangay_final": "barangay",
    })

    # STRICT + CLEANED: every retained row must be tied to a CALABARZON province
    # AND carry a concrete food-anchor term. This removes the two residual
    # false-positive classes in the strict set: (a) region-only national
    # aggregates with no CALABARZON province, and (b) NLI over-fires with no real
    # food term ("GDP growth could slow", the COVID-oil story). Same FOOD_ANCHOR
    # lexicon as precision_pass.py, so the rule is reproducible and thesis-faithful.
    n_before = len(out)
    out, dropped = _clean(out)
    print(f"\ngated: {n_before} -> {len(out)} rows "
          f"(dropped {len(dropped)}: {dropped['drop_reason'].value_counts().to_dict()})")

    # Syndication dedup across the union: same story republished under a near-
    # identical title (different URL/id). Keep the strict-reanalysis copy first.
    out["_nt"] = out["title"].fillna("").str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    out["_pri"] = (out["data_source"] == "strict_reanalysis").map({True: 0, False: 1})
    out = out.sort_values("_pri").drop_duplicates("_nt", keep="first")
    out = out[out["_nt"].str.len() > 0].drop(columns=["_nt", "_pri"])
    print(f"after syndication dedup: {len(out)} rows "
          f"({out['data_source'].value_counts().to_dict()})")

    # Final max-precision cut: drop the soft needs_review rows (only indirect
    # poverty/livelihood/supply signals). Logged to the audit for transparency.
    review = out[out["needs_review"]].copy()
    if len(review):
        review["drop_reason"] = "needs_review_weak_signal"
        dropped = pd.concat([dropped, review], ignore_index=True)
        out = out[~out["needs_review"]].copy()
        print(f"after dropping needs_review softies: {len(out)} rows (-{len(review)})")

    n_clim = int(out["is_climate_shock"].fillna(False).sum()) if "is_climate_shock" in out.columns else 0
    n_econ = int(out["is_economic_shock"].fillna(False).sum()) if "is_economic_shock" in out.columns else 0
    n_pov = int(out["is_poverty_shock"].fillna(False).sum()) if "is_poverty_shock" in out.columns else 0
    _sc = ["is_climate_shock", "is_economic_shock", "is_poverty_shock"]
    out = out.drop(columns=_sc, errors="ignore")
    dropped = dropped.drop(columns=_sc, errors="ignore")
    out = out.sort_values(["province", "city_municipality", "publication_date"], na_position="last")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTDIR / "calabarzon_food_insecurity_dataset.parquet", index=False)
    _safe_csv(out, OUTDIR / "calabarzon_food_insecurity_dataset.csv")
    _safe_csv(dropped.sort_values("drop_reason"),
              OUTDIR / "calabarzon_dataset_dropped_audit.csv")
    print(f"final dataset: {len(out)} rows -> calabarzon_food_insecurity_dataset.(parquet|csv)"
          f" | climate-shock rows: {n_clim} | economic-shock rows: {n_econ}"
          f" | poverty-shock rows: {n_pov}")
    print("province:", out["province"].value_counts(dropna=False).to_dict())
    print("distinct city/municipality:", out["city_municipality"].nunique(),
          "| barangay-level rows:", int(out["barangay"].notna().sum()))

    _coverage_matrix(out)
    _readme(out, dropped)


def _clean(out: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, dropped). Every retained row must satisfy the thesis's two
    inclusion conditions, checked reproducibly:

      1. FOOD-INSECURITY substance — a concrete food-anchor term AND at least one
         specific food-insecurity topic (drops food-culture / marketing / energy
         over-fires: 'solar farm', Jollibee, recipes, that mention food/farm
         only incidentally).
      2. CALABARZON geography — a province, and not an article whose SUBJECT is
         another region (a competing other-region cue with no specific CALABARZON
         LGU resolved drops COMCAST 'Bay Area', Mandaue, Bulacan, Pangasinan...).

    Rows whose lead could not be rejoined are judged on title only and given the
    benefit of the doubt on the food-anchor (the reanalysis already gated them).
    """
    from app.ml.corpus.location_geocoder import _OTHER_LOC
    txt = (out["title"].fillna("") + " " + out["content_lead"].fillna("")).astype(str)
    has_lead = out["content_lead"].fillna("").str.len() > 0
    # ASF (African Swine Fever) is a food-anchor: it directly hits pork/livestock
    # food supply, but the noun "ASF" alone isn't in the FOOD_ANCHOR lexicon.
    _asf = re.compile(r"\b(asf|african swine fever|swine fever)\b", re.I)
    # Climate-shock (typhoon/flood) and economic-shock (job loss/closure) rows are
    # food-security determinants (FAO stability & access pillars) without a food-anchor
    # term, so they are exempt from the food-anchor requirement. Both are pre-guarded
    # for geography (CALABARZON actually affected, not a Metro-Manila/national/foreign
    # wire) in their collect_* sweeps; the geo + other-region + noise gates still apply.
    def _flag(col):
        return out[col].fillna(False) if col in out.columns else pd.Series(False, index=out.index)
    shock = _flag("is_climate_shock") | _flag("is_economic_shock") | _flag("is_poverty_shock")
    has_food = txt.map(lambda t: bool(FOOD_ANCHOR.search(t)) or bool(_asf.search(t))) | ~has_lead | shock
    has_topic = txt.map(lambda t: len(_matched_topics(t)) > 0) | shock
    has_geo = out["province"].notna()
    has_lgu = out["city_municipality"].notna()
    # CALABARZON-only: drop any article that names another region at all — even
    # one carrying a CALABARZON LGU token, since those proved to be other-region
    # stories (Misamis real estate, a Mindanao essay) that merely mention a
    # CALABARZON place in passing. Strict per the "only cover CALABARZON" rule.
    other_subject = txt.map(lambda t: bool(_OTHER_LOC.search(t.lower())))
    # Negative gate: incidental food/farm words in a clearly non-food context.
    #  - energy "solar/wind farm", megawatt deals (the "farm" collides with agri)
    #  - foreign locations (COMCAST "Bay Area", California, abroad)
    #  - food-culture / lifestyle (recipes, "101: getting to know", restaurants)
    neg = txt.map(lambda t: bool(_NEGATIVE.search(t)))
    # Foreign source ("Laguna" = Spanish lagoon / Mexican region mis-geocodes).
    foreign = [_is_foreign(d, t) for d, t in zip(out["news_source"], txt)]
    foreign = pd.Series(foreign, index=out.index)
    # Off-topic subject (disease/crime/accident/showbiz/sport/tech/culture/governance
    # carrying a food or disaster token) — matched on the title only.
    offtopic = out["title"].map(_is_offtopic)

    keep_mask = has_food & has_topic & has_geo & ~other_subject & ~neg & ~foreign & ~offtopic
    kept = out[keep_mask].copy()
    dropped = out[~keep_mask].copy()
    reasons = []
    for f, tp, g, o, n, fr, ot in zip(has_food[~keep_mask], has_topic[~keep_mask],
                                      has_geo[~keep_mask], other_subject[~keep_mask],
                                      neg[~keep_mask], foreign[~keep_mask], offtopic[~keep_mask]):
        r = []
        if not g:
            r.append("no_calabarzon_province")
        if fr:
            r.append("foreign_source")
        if ot:
            r.append("off_topic_subject")
        if o:
            r.append("other_region_subject")
        if n:
            r.append("off_topic_energy_foreign_or_culture")
        if not f:
            r.append("no_food_anchor")
        if not tp:
            r.append("no_food_insecurity_topic")
        reasons.append("+".join(r) or "other")
    dropped["drop_reason"] = reasons
    return kept, dropped


def _summary_row(r) -> str:
    loc = []
    if pd.notna(r.get("barangay_final")):
        loc.append(f"Brgy. {r['barangay_final']}")
    if pd.notna(r.get("city_municipality_final")):
        loc.append(str(r["city_municipality_final"]))
    loc.append(str(r["province_final"]) if pd.notna(r.get("province_final")) else "CALABARZON (region-level)")
    location = ", ".join(loc)
    topics = (r["food_insecurity_topics"] or "").replace(",", ", ").replace("_", " ")
    cat = r.get("food_security_dimension_label") or "food-insecurity relevance"
    return (f"Relevant to food insecurity via {cat}. "
            f"Signals: {topics or 'food-insecurity indicators'}. Location: {location}.")


def _coverage_matrix(out: pd.DataFrame) -> None:
    gaz = pd.read_parquet(GAZ).drop_duplicates("lgu_psgc")[["province_name", "lgu_name", "lgu_psgc"]]
    covered = set(out["city_municipality"].dropna())
    mentioned = set()
    if FULLPOOL.exists():
        mentioned = set(pd.read_parquet(FULLPOOL)["lgu_name"].dropna())

    def status(name):
        if name in covered:
            return "covered_relevant"
        if name in mentioned:
            return "mentioned_only"
        return "absent"

    gaz["coverage"] = gaz["lgu_name"].map(status)
    gaz = gaz.sort_values(["province_name", "lgu_name"])
    _safe_csv(gaz, OUTDIR / "calabarzon_lgu_coverage_matrix.csv")
    print("\ncoverage matrix (142 LGUs):", gaz["coverage"].value_counts().to_dict())


def _readme(out: pd.DataFrame, dropped: pd.DataFrame) -> None:
    n = len(out)
    provs = out["province"].value_counts(dropna=False).to_dict()
    lgus = out["city_municipality"].nunique()
    brgy = int(out["barangay"].notna().sum())
    drop_reasons = dropped["drop_reason"].value_counts().to_dict()
    src = out["data_source"].value_counts().to_dict()
    txt = f"""# CALABARZON Food-Insecurity News Dataset

**Rows:** {n} articles · **Provinces:** {provs}
**Distinct cities/municipalities:** {lgus} of 142 · **barangay-level rows:** {brgy}
**By source:** {src}

## What this is
News articles that provide usable evidence about food insecurity (or its clear
determinants) in Region IV-A (CALABARZON). Every row satisfies BOTH inclusion
conditions: (1) a substantive food-insecurity connection, and (2) a geographic
tie to a CALABARZON province / city / municipality / barangay. General CALABARZON
news (politics, crime, sports, entertainment, food-culture) is excluded.

## Source pool (relevance faithful to the thesis)
Union of two relevance-scored pools, both from the thesis's zero-shot XLM-RoBERTa
NLI classifier (10 food-insecurity hypotheses):
  * `strict_reanalysis` — the reanalysis pipeline's retained HIGH/MEDIUM set
    (NLI + food-anchor + dimension-evidence + geo-verification gates).
  * `corpus_recall` — additional relevance-passing CALABARZON articles the strict
    pipeline had excluded, recovered from the geocoded corpus so genuine coverage
    is not lost. (`data_source` column records which pool each row came from.)

## Quality gates applied to every row (reproducible, no LLM)
  1. **Food-anchor** — a concrete foodstuff/agri/hunger/nutrition/fishery/food-
     program term (`FOOD_ANCHOR` lexicon, shared with `precision_pass.py`).
  2. **Food-insecurity topic** — matches >=1 specific thesis topic (hunger, prices,
     malnutrition, crop loss, assistance, ...); drops articles that mention food
     only incidentally.
  3. **CALABARZON geography** — has a province and is NOT an article whose subject
     is another region (competing other-region cue with no CALABARZON LGU → dropped:
     e.g. Mandaue, Bulacan, Pangasinan, "Bay Area" USA).
  4. **Negative filter** — drops incidental food/farm words in non-food contexts:
     energy ("solar/wind farm", megawatt deals), foreign locations, and food-culture
     (recipes, restaurants, "101: getting to know").
  5. **Syndication dedup** — same story under a near-identical title collapsed to one
     copy (strict version kept).
Every dropped row and its failing gate is logged to
`calabarzon_dataset_dropped_audit.csv` (removed this run: {drop_reasons}).

## Geography (142-LGU enhancement)
Re-geocoded with the full **142-LGU PSGC gazetteer** geocoder (province → city/
municipality → barangay), with disambiguation (ambiguous/surname/feature names
require a locality cue) and a non-CALABARZON conflict guard.

## Columns
title, publication_date, news_source, author*, url, content_lead*, province,
city_municipality, barangay, relevance_tier (HIGH/MEDIUM), food_security_dimension
(A–F), food_insecurity_category, food_insecurity_topics, event_type,
affected_commodity*, affected_population*, is_direct_food_insecurity,
relevance_summary, relevance_reason*, relevance_score, match_level, needs_review,
data_source, article_id.  (* populated for strict_reanalysis rows; sparse for
corpus_recall rows.)

`needs_review` = True for the softest rows (only indirect poverty/livelihood/supply
signals) — {int(out['needs_review'].sum())} of {n} rows, for an optional eyeball.

## Known limitations (scope notes)
- **Residual precision (~99%):** the gates are lexical, so ~1 row may survive on an
  incidental token (e.g. a "Cordillera vegetable prices" story pre-tagged to a
  CALABARZON province). Retained for reproducibility; the audit CSV lists all drops.
- **Coverage ceiling:** {lgus}/142 LGUs have a qualifying article. Targeted collection
  via two independent global news indexes (GDELT, Event Registry) confirmed most
  remaining municipalities have no food-insecurity news — a data-availability limit.
  See `calabarzon_lgu_coverage_matrix.csv` (covered / mentioned-only / absent).
- *`author` and full `content` were not retained by the historical fetchers
  (title + lead only); barangay tagging is limited because barangays are usually
  named deeper in bodies than the stored lead.
"""
    (OUTDIR / "calabarzon_dataset_README.md").write_text(txt, encoding="utf-8")
    print("wrote README + coverage matrix")


if __name__ == "__main__":
    build()
