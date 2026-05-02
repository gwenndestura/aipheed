"""
app/ml/nlp/classifier.py
-------------------------
HungerGist 10-hypothesis zero-shot food insecurity classifier.

Uses XLM-RoBERTa (joeddav/xlm-roberta-large-xnli) as the NLI backbone.
Per-article score = max across 10 HungerGist hypotheses.

Ten hypotheses (Ahn et al., 2023 + PH extensions):
  T1  : Food supply disruption or food price increases affecting access to food
  T2  : Health services or nutrition programs unavailable or unaffordable
  T3  : Government food security programs ineffective or unavailable
  T4  : Economic hardship reducing household income and food purchasing power
  T5  : Infrastructure failures limiting food transport or storage
  T6  : Agricultural land loss or conversion reducing food production
  T7  : Civil displacement or evacuation reducing food access
  T8  : Social unrest or conflict disrupting food systems
  T1b : Fish kill or aquaculture collapse reducing fish food supply
  T9  : OFW remittance reduction reducing household food purchasing power

Zero-shot rationale (Backend Guide v3, Critical Reminders):
  No fine-tuning — reproducibility without annotated Filipino food insecurity
  training data; DSWD can rerun the pipeline without model retraining.

Fallback:
  If the XLM-RoBERTa model is unavailable (no internet / memory constraint),
  the module falls back to keyword-based scoring transparently. The fallback
  is surfaced via top_hypothesis='keyword_bootstrap'.

Usage:
    from app.ml.nlp.classifier import load_classifier, score_article
    clf = load_classifier()
    result = score_article(clf, title="...", summary="...")
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "joeddav/xlm-roberta-large-xnli"

RELEVANCE_THRESHOLD = 0.30   # articles above this score are flagged is_relevant=True

# The 10 HungerGist hypotheses used for zero-shot classification
HYPOTHESES: dict[str, str] = {
    "T1":  "Food supply disruption or food price increases affecting access to food",
    "T2":  "Health services or nutrition programs unavailable or unaffordable",
    "T3":  "Government food security programs ineffective or unavailable",
    "T4":  "Economic hardship reducing household income and food purchasing power",
    "T5":  "Infrastructure failures limiting food transport or storage",
    "T6":  "Agricultural land loss or conversion reducing food production",
    "T7":  "Civil displacement or evacuation reducing food access",
    "T8":  "Social unrest or conflict disrupting food systems",
    "T1b": "Fish kill or aquaculture collapse reducing fish food supply in the Philippines",
    "T9":  "OFW remittance reduction reducing household food purchasing power",
}

# Whether to prefer keyword fallback (set AIPHEED_USE_KEYWORD_SCORER=1 in .env)
_USE_KEYWORD_FALLBACK = os.getenv("AIPHEED_USE_KEYWORD_SCORER", "0").strip() == "1"

# ---------------------------------------------------------------------------
# Keyword bank (mirrors sentiment.py _KEYWORD_BANK — kept here for fallback)
# ---------------------------------------------------------------------------

_KEYWORD_BANK: list[tuple[str, str]] = [
    # (keyword, hypothesis_id)
    ("rice price", "T1"), ("food price", "T1"), ("presyo ng bigas", "T1"),
    ("food shortage", "T1"), ("supply disruption", "T1"), ("price hike", "T1"),
    ("presyo ng pagkain", "T1"), ("kakulangan ng bigas", "T1"),
    ("fish kill", "T1b"), ("patay na isda", "T1b"), ("red tide", "T1b"),
    ("algal bloom", "T1b"), ("laguna lake", "T1b"), ("taal lake", "T1b"),
    ("malnutrition", "T2"), ("stunting", "T2"), ("malnourished", "T2"),
    ("nutrition program", "T2"), ("feeding program", "T2"),
    ("nfa", "T3"), ("kadiwa", "T3"), ("4ps", "T3"), ("dswd", "T3"),
    ("food subsidy", "T3"), ("ayuda", "T3"), ("pantawid", "T3"),
    ("unemployment", "T4"), ("poverty", "T4"), ("kahirapan", "T4"),
    ("jobless", "T4"), ("layoff", "T4"), ("walang trabaho", "T4"),
    ("farm to market", "T5"), ("post-harvest", "T5"), ("cold storage", "T5"),
    ("farmland", "T6"), ("agricultural land", "T6"), ("harvest loss", "T6"),
    ("evacuee", "T7"), ("evacuation", "T7"), ("displaced", "T7"), ("bakwit", "T7"),
    ("typhoon", "T7"), ("baha", "T7"), ("flood", "T7"),
    ("strike", "T8"), ("protest", "T8"), ("welga", "T8"), ("unrest", "T8"),
    ("ofw", "T9"), ("remittance", "T9"), ("pinadala", "T9"), ("overseas filipino", "T9"),
    ("food insecurity", "T1"), ("gutom", "T4"), ("hunger", "T4"),
]

_KW_BY_TOPIC: dict[str, list[str]] = {}
for _kw, _tid in _KEYWORD_BANK:
    _KW_BY_TOPIC.setdefault(_tid, []).append(_kw.lower())


def _keyword_score_article(title: str, summary: str) -> dict:
    """Keyword-based fallback scorer returning the same dict shape as score_article."""
    text = f"{title} {summary}".lower()
    scores: dict[str, float] = {}
    for topic_id, keywords in _KW_BY_TOPIC.items():
        hits = sum(1 for kw in keywords if kw in text)
        scores[topic_id] = min(hits * 0.15, 1.0)

    if not scores or max(scores.values()) == 0:
        # Try broad food signal
        food_words = ["food", "pagkain", "gutom", "hunger", "rice", "bigas"]
        broad_hit = any(w in text for w in food_words)
        best_score = 0.15 if broad_hit else 0.0
        best_topic = "T1" if broad_hit else "T4"
    else:
        best_topic = max(scores, key=lambda k: scores[k])
        best_score = scores[best_topic]

    return {
        "food_insecurity_score": round(best_score, 4),
        "is_relevant": best_score >= RELEVANCE_THRESHOLD,
        "top_hypothesis": f"{best_topic}_keyword",
        "top_topic_name": f"{HYPOTHESES.get(best_topic, best_topic)} [keyword]",
        "all_scores": {k: round(v, 4) for k, v in scores.items()},
    }


# ---------------------------------------------------------------------------
# XLM-RoBERTa classifier
# ---------------------------------------------------------------------------

class _KeywordClassifier:
    """Lightweight stand-in when XLM-RoBERTa is not available."""
    mode = "keyword"


class _XLMRobertaClassifier:
    """Wraps the transformers zero-shot-classification pipeline."""
    mode = "xlm-roberta"

    def __init__(self, pipeline):
        self._pipeline = pipeline

    def classify(self, text: str) -> list[dict]:
        """Run NLI inference for all hypotheses on text."""
        hypotheses_list = list(HYPOTHESES.values())
        result = self._pipeline(
            text,
            candidate_labels=hypotheses_list,
            multi_label=True,
            hypothesis_template="{}",
        )
        # Map back from hypothesis text → topic_id
        text_to_id = {v: k for k, v in HYPOTHESES.items()}
        return [
            {
                "topic_id": text_to_id.get(lbl, lbl),
                "score": float(sc),
            }
            for lbl, sc in zip(result["labels"], result["scores"])
        ]


@lru_cache(maxsize=1)
def load_classifier() -> Any:
    """
    Load and cache the food insecurity classifier.

    Returns an _XLMRobertaClassifier if the transformers pipeline loads
    successfully, otherwise returns _KeywordClassifier (keyword fallback).
    The caller does not need to know which mode is active.
    """
    if _USE_KEYWORD_FALLBACK:
        logger.info(
            "load_classifier: AIPHEED_USE_KEYWORD_SCORER=1 — using keyword fallback."
        )
        return _KeywordClassifier()

    try:
        from transformers import pipeline as hf_pipeline

        logger.info(
            "load_classifier: loading XLM-RoBERTa zero-shot pipeline (%s)...", MODEL_NAME
        )
        pipe = hf_pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=-1,            # CPU; change to 0 for CUDA
            multi_label=True,
        )
        clf = _XLMRobertaClassifier(pipe)
        logger.info("load_classifier: XLM-RoBERTa ready.")
        return clf

    except Exception as exc:
        logger.warning(
            "load_classifier: XLM-RoBERTa unavailable (%s) — falling back to keywords.",
            exc,
        )
        return _KeywordClassifier()


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------

def score_article(clf: Any, title: str, summary: str) -> dict:
    """
    Score one article for food insecurity relevance.

    Parameters
    ----------
    clf     : classifier returned by load_classifier()
    title   : article headline
    summary : article lead / snippet (up to ~500 chars)

    Returns
    -------
    dict with:
        food_insecurity_score : float  — max across 10 hypotheses (0–1)
        is_relevant           : bool   — True if score >= RELEVANCE_THRESHOLD
        top_hypothesis        : str    — topic_id of highest-scoring hypothesis
        top_topic_name        : str    — human-readable hypothesis text
        all_scores            : dict   — {topic_id: score} for all 10 hypotheses
    """
    if isinstance(clf, _KeywordClassifier):
        return _keyword_score_article(title, summary)

    # XLM-RoBERTa path
    text = f"{title}. {summary}"[:512]  # truncate to model context window
    try:
        raw = clf.classify(text)
    except Exception as exc:
        logger.warning("score_article: inference error (%s) — using keyword fallback.", exc)
        return _keyword_score_article(title, summary)

    scores: dict[str, float] = {r["topic_id"]: r["score"] for r in raw}

    if not scores:
        return _keyword_score_article(title, summary)

    best_topic = max(scores, key=lambda k: scores[k])
    best_score = scores[best_topic]

    return {
        "food_insecurity_score": round(best_score, 4),
        "is_relevant": best_score >= RELEVANCE_THRESHOLD,
        "top_hypothesis": best_topic,
        "top_topic_name": HYPOTHESES.get(best_topic, best_topic),
        "all_scores": {k: round(v, 4) for k, v in scores.items()},
    }
