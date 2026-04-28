"""
app/ml/nlp/classifier.py
-------------------------
Zero-shot food insecurity classifier using XLM-RoBERTa.

DESIGN RATIONALE — HungerGist (Ahn et al., 2023)
-------------------------------------------------
HungerGist demonstrated that food insecurity is signaled at the SENTENCE
level across 8 distinct topics — not just food/agriculture text. Their model
identified "gists": individual sentences from healthcare, governance, civil
life, land use, and economic development articles that are just as predictive
of food crisis as direct food-price sentences.

Implication for aiPHeed: a single hypothesis template
    "This article describes food insecurity"
would miss sentences from Topics 2-8.  Instead we use a MULTI-HYPOTHESIS
approach with one hypothesis per HungerGist gist topic, then aggregate the
maximum score across all hypotheses as the article's food insecurity signal.

HYPOTHESIS TEMPLATES (aligned to HungerGist 8 topics + 2 PH extensions)
------------------------------------------------------------------------
HungerGist base taxonomy (Ahn et al., 2023):
T1  Food Supply/Price    — direct food scarcity / price signal
T2  Healthcare           — health crisis → food insecurity predictor
T3  Leadership/Policy    — governance affecting food access
T4  Finance/Economy      — budget allocations, economic shocks
T5  Regional Development — infrastructure affecting food distribution
T6  Land Use             — agricultural land loss / conversion
T7  Civil Life           — displacement, family misery, unrest
T8  Social Instability   — protest, unrest correlated with food prices

Philippines-specific extensions (not in original HungerGist):
T1b Fish Kill/Aquaculture — Laguna Lake / Taal Lake / coastal food supply loss
T9  OFW Remittance Shock  — overseas worker income loss → household food poverty

TRANSFERABILITY NOTE
--------------------
HungerGist was validated on 9 African IPC-tracked countries (2010–2021).
Its 8 gist topics represent universal food insecurity mechanisms, not
Africa-specific ones. Each topic has a direct Philippine analogue:
  - T2 Healthcare  → delayed 4Ps/Pantawid, malnutrition in CALABARZON
  - T3 Governance  → rice tariffication law, NFA price control orders
  - T6 Land Use    → CALABARZON agricultural land converted to subdivisions
  - T7 Civil Life  → typhoon evacuees with no food access
The two PH extensions cover mechanisms absent from African corpora:
aquaculture collapse (Laguna/Taal) and OFW remittance shocks (~10% PH GDP).

Usage:
    from app.ml.nlp.classifier import score_article, load_classifier
    model = load_classifier()
    result = score_article(model, title="...", summary="...")
    # result["food_insecurity_score"]  0–1
    # result["top_hypothesis"]         which HungerGist topic fired
    # result["all_scores"]             dict[topic -> score]
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HungerGist-aligned hypothesis templates
# ---------------------------------------------------------------------------

# Each tuple: (topic_id, topic_name, hypothesis_template)
HYPOTHESES: list[tuple[str, str, str]] = [
    (
        "T1_food_supply",
        "Food Supply / Price",
        "This text describes food price increases, food shortage, or disruption to food supply.",
    ),
    (
        "T2_healthcare",
        "Healthcare → Food Crisis",
        "This text describes a health crisis, delayed social assistance, or malnutrition among poor families.",
    ),
    (
        "T3_governance",
        "Governance / Policy",
        "This text describes government food policy, rice price regulation, or agricultural reform.",
    ),
    (
        "T4_finance",
        "Finance / Economy",
        "This text describes budget allocations for agriculture, social welfare spending, or economic hardship affecting food access.",
    ),
    (
        "T5_regional_dev",
        "Regional Development",
        "This text describes agricultural infrastructure, farm-to-market roads, or post-harvest facilities affecting food distribution.",
    ),
    (
        "T6_land_use",
        "Land Use / Agricultural Land",
        "This text describes conversion of farmland, agricultural land loss, or changes in land use that affect crop production.",
    ),
    (
        "T7_civil_life",
        "Civil Life / Displacement",
        "This text describes displaced families, evacuees, informal settlers, or communities facing food hardship.",
    ),
    (
        "T8_social_instability",
        "Social Instability",
        "This text describes social unrest, protests, or strikes related to food prices or economic conditions.",
    ),
    (
        "T1b_fish_kill",
        "Fish Kill / Aquaculture",
        "This text describes fish kills, red tide, aquaculture damage, or fishing bans that reduce food supply.",
    ),
    (
        "T9_ofw_remittance",
        "OFW / Remittance Shock",
        "This text describes overseas Filipino workers, OFW remittances declining, or foreign employment loss "
        "affecting household food budgets and purchasing power.",
    ),
]

# Minimum score threshold for an article to be labelled food-insecurity-relevant
RELEVANCE_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_classifier() -> Any:
    """
    Load XLM-RoBERTa zero-shot classification pipeline (cached singleton).

    Uses facebook/bart-large-mnli for English articles.
    XLM-RoBERTa (joeddav/xlm-roberta-large-xnli) handles Filipino/mixed text.

    Model choice follows thesis constraint: no fine-tuning permitted.
    """
    try:
        from transformers import pipeline
        logger.info("Loading XLM-RoBERTa zero-shot classifier…")
        clf = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
            device=-1,  # CPU; change to 0 for GPU
        )
        logger.info("XLM-RoBERTa classifier loaded.")
        return clf
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
        raise


# ---------------------------------------------------------------------------
# Article scoring
# ---------------------------------------------------------------------------

def score_article(
    classifier: Any,
    title: str,
    summary: str,
    multi_label: bool = True,
) -> dict:
    """
    Score an article against all 10 hypothesis templates (8 HungerGist topics + 2 PH extensions).

    The text fed to the model is: title + ". " + summary (first 512 chars).
    multi_label=True means each hypothesis is scored independently (MNLI
    entailment only), not competing — matches HungerGist's sentence-level
    independent scoring approach.

    Parameters
    ----------
    classifier  : transformers zero-shot pipeline
    title       : str — article headline
    summary     : str — first paragraph or RSS summary
    multi_label : bool — True = independent per-hypothesis scores (recommended)

    Returns
    -------
    dict with keys:
        food_insecurity_score : float   — max score across all hypotheses (0–1)
        is_relevant           : bool    — score >= RELEVANCE_THRESHOLD
        top_hypothesis        : str     — topic_id of highest-scoring hypothesis
        top_topic_name        : str     — human-readable topic name
        all_scores            : dict    — {topic_id: score} for all hypotheses
    """
    text = f"{title}. {summary}"[:512]
    hypothesis_templates = [h[2] for h in HYPOTHESES]

    try:
        result = classifier(
            text,
            candidate_labels=hypothesis_templates,
            multi_label=multi_label,
        )
    except Exception as exc:
        logger.error("Classifier error: %s", exc)
        return {
            "food_insecurity_score": 0.0,
            "is_relevant": False,
            "top_hypothesis": None,
            "top_topic_name": None,
            "all_scores": {},
        }

    # Map back label → (topic_id, topic_name, score)
    label_to_meta = {h[2]: (h[0], h[1]) for h in HYPOTHESES}
    all_scores: dict[str, float] = {}
    for label, score in zip(result["labels"], result["scores"]):
        topic_id, _ = label_to_meta.get(label, (label, label))
        all_scores[topic_id] = round(score, 4)

    max_score = max(all_scores.values(), default=0.0)
    top_id = max(all_scores, key=all_scores.get, default=None)
    top_name = next((h[1] for h in HYPOTHESES if h[0] == top_id), None)

    return {
        "food_insecurity_score": round(max_score, 4),
        "is_relevant": max_score >= RELEVANCE_THRESHOLD,
        "top_hypothesis": top_id,
        "top_topic_name": top_name,
        "all_scores": all_scores,
    }


# ---------------------------------------------------------------------------
# Batch scoring for corpus
# ---------------------------------------------------------------------------

def score_corpus(
    classifier: Any,
    records: list[dict],
    title_field: str = "title",
    summary_field: str = "summary",
) -> list[dict]:
    """
    Score a list of corpus records in-place, adding NLP score fields.

    Adds to each record:
        food_insecurity_score : float
        is_relevant           : bool
        top_hypothesis        : str
        top_topic_name        : str
        nlp_all_scores        : dict

    Parameters
    ----------
    records      : list of corpus dicts (from run_corpus_ingestion)
    title_field  : key for article title
    summary_field: key for article summary

    Returns the same list with score fields added.
    """
    total = len(records)
    logger.info("Scoring %d articles with XLM-RoBERTa…", total)

    for i, rec in enumerate(records):
        scores = score_article(
            classifier,
            title=rec.get(title_field, "") or "",
            summary=rec.get(summary_field, "") or "",
        )
        rec["food_insecurity_score"] = scores["food_insecurity_score"]
        rec["is_relevant"] = scores["is_relevant"]
        rec["top_hypothesis"] = scores["top_hypothesis"]
        rec["top_topic_name"] = scores["top_topic_name"]
        rec["nlp_all_scores"] = scores["all_scores"]

        if (i + 1) % 100 == 0:
            logger.info("  Scored %d / %d articles", i + 1, total)

    relevant = sum(1 for r in records if r.get("is_relevant"))
    logger.info(
        "score_corpus complete: %d / %d articles relevant (threshold=%.2f)",
        relevant, total, RELEVANCE_THRESHOLD,
    )
    return records
