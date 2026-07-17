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

# The 10 HungerGist hypotheses used for zero-shot classification.
#
# Phrased in the XNLI zero-shot topical form ("This article is about X") —
# the usage pattern this model is documented and trained for. The original
# failure-assertion phrasing ("...programs ineffective or unavailable")
# made strict entailment reject genuinely relevant articles (a DSWD
# food-pack delivery story does not entail "programs unavailable") while
# passing conflict stories with no food content. Validated on enriched
# CALABARZON samples: topical phrasing + raw P(entailment) + the 0.30
# threshold gives precision-first separation (0 false positives in
# stratified samples). Topic ids and coverage are unchanged.
HYPOTHESES: dict[str, str] = {
    "T1":  "This article is about food prices, food supply problems, or difficulty accessing food",
    "T2":  "This article is about hunger, malnutrition, or nutrition and feeding programs",
    "T3":  "This article is about government food assistance, rice subsidies, or relief distribution",
    "T4":  "This article is about poverty, unemployment, or economic hardship of families",
    "T5":  "This article is about roads, transport, or storage problems affecting food supply",
    "T6":  "This article is about farmland loss, crop damage, or reduced harvests",
    "T7":  "This article is about evacuation or displacement of families due to disaster",
    "T8":  "This article is about strikes, protests, or unrest disrupting food or livelihoods",
    "T1b": "This article is about fish kills, fishing bans, or aquaculture losses",
    "T9":  "This article is about overseas Filipino workers or remittances supporting families",
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
    """
    Direct NLI scoring: article text as premise, each HungerGist hypothesis
    as hypothesis, score = P(entailment) from the 3-way XNLI softmax
    (contradiction / neutral / entailment).

    This is the "maximum entailment probability" defined in the methodology.
    The zero-shot pipeline's multi_label mode is deliberately NOT used: it
    renormalises entailment against contradiction only, which inflates
    off-topic text (a sports article can score >0.8) and breaks the 0.30
    relevance threshold. Raw P(entailment) separates cleanly.
    """
    mode = "xlm-roberta"

    def __init__(self, model, tokenizer, entailment_idx: int):
        self._model = model
        self._tok = tokenizer
        self._ent_idx = entailment_idx

    def classify(self, text: str) -> list[dict]:
        """Run NLI inference for all 10 hypotheses in one batched forward pass."""
        import torch

        topic_ids = list(HYPOTHESES.keys())
        hypotheses = list(HYPOTHESES.values())
        enc = self._tok(
            [text] * len(hypotheses),
            hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, self._ent_idx]
        return [
            {"topic_id": tid, "score": float(p)}
            for tid, p in zip(topic_ids, probs)
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
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(
            "load_classifier: loading XLM-RoBERTa NLI model (%s)...", MODEL_NAME
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()

        label_to_idx = {
            str(lbl).lower(): int(idx)
            for idx, lbl in model.config.id2label.items()
        }
        entailment_idx = label_to_idx.get("entailment", 2)

        clf = _XLMRobertaClassifier(model, tokenizer, entailment_idx)
        logger.info(
            "load_classifier: XLM-RoBERTa ready (entailment index %d).", entailment_idx
        )
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
