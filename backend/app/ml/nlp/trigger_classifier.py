"""
app/ml/nlp/trigger_classifier.py
----------------------------------
Five-category bilingual trigger classifier (v5 / HungerGist 8+2 taxonomy).

Categories and keyword taxonomy (Ahn et al., 2023 + PH extensions):
    market          : food prices, rice, supply disruption (T1 Food Supply/Price)
    climate         : typhoons, rainfall, ENSO, drought (T2/T6/T7 overlap)
    employment      : joblessness, retrenchment, livelihood loss (T4/T8)
    ofw_remittance  : OFW income, remittances, foreign employment (T9)
    fish_kill       : aquaculture collapse, Laguna/Taal fish deaths (T1b)

Each article receives a list of triggered categories (multi-label).
Province-quarter proportions per trigger type are independent LightGBM features.

Usage:
    from app.ml.nlp.trigger_classifier import classify_triggers, compute_trigger_proportions
    triggers = classify_triggers("OFW remittances dropped amid pandemic")
    # → ["ofw_remittance", "employment"]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword taxonomy
# ---------------------------------------------------------------------------

TRIGGER_KEYWORDS: dict[str, list[str]] = {
    "market": [
        # English
        "rice price", "food price", "price hike", "price increase", "price spike",
        "price surge", "inflationary", "food inflation", "rice shortage", "food shortage",
        "supply disruption", "supply chain", "commodity price", "market price",
        "food cost", "grocery price", "vegetable price", "fish price", "pork price",
        "chicken price", "sugar price", "oil price", "cooking oil", "flour price",
        "food basket", "basic goods", "essential goods", "sembako", "nfa rice",
        "nfa price", "price control", "price ceiling", "palay", "farmgate price",
        # Filipino / Taglish
        "presyo ng bigas", "presyo ng pagkain", "taas ng presyo", "mataas na presyo",
        "pagmamahal ng bigas", "kakulangan ng bigas", "rice tariffication",
        "taripa sa bigas", "batas taripa",
    ],
    "climate": [
        # English
        "typhoon", "tropical storm", "cyclone", "bagyo", "landslide", "flood",
        "flooding", "inundation", "storm surge", "rainfall", "drought", "dry spell",
        "el nino", "la nina", "enso", "climate change", "extreme weather",
        "agricultural damage", "crop damage", "crop failure", "harvest failure",
        "irrigation failure", "water shortage", "dam release", "erosion",
        "coastal flooding", "storm damage", "weather disturbance",
        # Filipino / Taglish
        "baha", "pagbaha", "tagtuyot", "tag-tuyot", "tag-init", "pag-ulan",
        "malakas na ulan", "pagguho ng lupa", "sigwa", "lindol",
    ],
    "employment": [
        # English
        "unemployment", "underemployment", "layoff", "lay-off", "retrenchment",
        "redundancy", "job loss", "jobless", "no work", "wage cut", "salary cut",
        "minimum wage", "poverty", "livelihood loss", "kasambahay", "labor",
        "worker", "informal sector", "displaced worker", "contractual",
        "endo", "no job", "out of work", "economic hardship",
        # Filipino / Taglish
        "walang trabaho", "nawalan ng trabaho", "pagtanggal sa trabaho",
        "tanggal sa trabaho", "kahirapan", "gutom", "walang hanapbuhay",
        "pangkabuhayan", "hanapbuhay", "trabaho",
    ],
    "ofw_remittance": [
        # English (T9 — OFW Remittance Shock)
        "ofw", "overseas filipino", "overseas worker", "overseas contract worker",
        "ocw", "remittance", "overseas remittance", "dollar remittance",
        "ofws stranded", "deployment ban", "overseas employment",
        "foreign employment", "dmw", "poea", "migrant worker",
        "repatriated ofw", "ofws return", "bangko remittance",
        # Filipino / Taglish
        "pinadala", "padala ng pera", "ofws", "pilipino sa ibang bansa",
        "mangagawa sa ibang bansa", "remesa",
    ],
    "fish_kill": [
        # English (T1b — Philippines-specific aquaculture extension)
        "fish kill", "fishkill", "fish death", "dead fish", "fish mortality",
        "aquaculture damage", "fish farm", "fishing ban", "red tide", "algal bloom",
        "toxic algae", "paralytic shellfish poisoning", "psp", "harmful algae",
        "laguna lake", "taal lake", "lake sebu", "bangus", "tilapia kill",
        "shrimp kill", "oyster mortality", "mussel die-off",
        # Filipino / Taglish
        "patay na isda", "pamamatay ng isda", "pagkamatay ng isda",
        "lawa ng laguna", "red tide", "lawa ng taal", "hapon ng lawa",
        "namatay na isda", "bangus na patay",
    ],
}

# Pre-compile for speed
_COMPILED: dict[str, list[re.Pattern]] = {
    category: [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
               for kw in keywords]
    for category, keywords in TRIGGER_KEYWORDS.items()
}

ALL_TRIGGERS: list[str] = list(TRIGGER_KEYWORDS.keys())


# ---------------------------------------------------------------------------
# Article-level classification
# ---------------------------------------------------------------------------

def classify_triggers(text: str) -> list[str]:
    """
    Classify article text into triggered food-insecurity driver categories.

    Multi-label: an article can trigger multiple categories (e.g. an article
    about OFW job losses due to typhoon triggers both 'ofw_remittance' and
    'climate').

    Parameters
    ----------
    text : str
        Article title + summary (combined).

    Returns
    -------
    list[str]
        List of triggered category names (subset of ALL_TRIGGERS).
        Empty list if no keywords match.
    """
    if not text:
        return []

    triggered: list[str] = []
    for category, patterns in _COMPILED.items():
        for pattern in patterns:
            if pattern.search(text):
                triggered.append(category)
                break  # only count each category once per article

    return triggered


def classify_triggers_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply classify_triggers to every row in df.

    Adds:
        triggers          : list of triggered categories per article
        trigger_market    : 1/0
        trigger_climate   : 1/0
        trigger_employment: 1/0
        trigger_ofw_remittance: 1/0
        trigger_fish_kill : 1/0
        trigger_count     : number of categories triggered

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'title' and 'summary' columns.

    Returns
    -------
    DataFrame with trigger columns added.
    """
    out = df.copy()

    combined = (
        out.get("title", pd.Series("", index=out.index)).fillna("").astype(str)
        + " "
        + out.get("summary", pd.Series("", index=out.index)).fillna("").astype(str)
    )

    out["triggers"] = combined.apply(classify_triggers)

    for cat in ALL_TRIGGERS:
        out[f"trigger_{cat}"] = out["triggers"].apply(lambda t: int(cat in t))

    out["trigger_count"] = out["triggers"].apply(len)

    n_triggered = (out["trigger_count"] > 0).sum()
    logger.info(
        "classify_triggers_df: %d / %d articles triggered at least one category",
        n_triggered, len(out),
    )
    return out


# ---------------------------------------------------------------------------
# Province-quarter aggregation
# ---------------------------------------------------------------------------

def compute_trigger_proportions(
    articles_df: pd.DataFrame,
    save_path: Path | None = Path("data/processed/trigger_proportions.parquet"),
) -> pd.DataFrame:
    """
    Compute proportion of articles per trigger category for each province-quarter.

    Proportions are used as five independent LightGBM features.
    An article's trigger columns must already be present (from classify_triggers_df).

    Parameters
    ----------
    articles_df : pd.DataFrame
        Must have: province_code, quarter, trigger_market, trigger_climate,
        trigger_employment, trigger_ofw_remittance, trigger_fish_kill.
        province_code = None rows are excluded.

    save_path : Path | None
        Where to save the result. Pass None to skip saving.

    Returns
    -------
    pd.DataFrame
        Columns: province_code, quarter, article_count,
                 trigger_market, trigger_climate, trigger_employment,
                 trigger_ofw_remittance, trigger_fish_kill
        Each trigger column is the PROPORTION of articles in that
        province-quarter that triggered the category (0.0–1.0).
    """
    df = articles_df[articles_df["province_code"].notna()].copy()
    if df.empty:
        logger.warning("compute_trigger_proportions: no geocoded articles.")
        return pd.DataFrame()

    # Ensure trigger columns exist
    trigger_cols = [f"trigger_{c}" for c in ALL_TRIGGERS]
    for col in trigger_cols:
        if col not in df.columns:
            df[col] = 0

    agg = df.groupby(["province_code", "quarter"]).agg(
        article_count=("province_code", "count"),
        **{col: (col, "sum") for col in trigger_cols},
    ).reset_index()

    for col in trigger_cols:
        agg[col] = agg[col] / agg["article_count"]

    logger.info(
        "compute_trigger_proportions: %d province-quarter rows",
        len(agg),
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(save_path, index=False)
        logger.info("Trigger proportions saved → %s", save_path)

    return agg
