"""
app/ml/nlp/sentiment.py
-----------------------
DataFrame-friendly wrapper around the HungerGist zero-shot classifier.

Loads the XLM-RoBERTa pipeline once (cached via lru_cache in classifier.py)
and exposes a pandas-compatible batch API for the corpus ingestion pipeline.

The per-article food_insecurity_score field produced here is the input to
fssi_builder.py, which aggregates it to province-quarter level.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from app.ml.nlp.classifier import (
    RELEVANCE_THRESHOLD,
    HYPOTHESES,
    load_classifier,
    score_article,
)

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class SentimentScorer:
    """
    Lazy-loaded singleton that wraps the HungerGist zero-shot pipeline.

    Usage:
        scorer = SentimentScorer()
        df = scorer.score_df(articles_df)
    """

    _instance: SentimentScorer | None = None
    _model: Any = None

    def __new__(cls) -> SentimentScorer:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> Any:
        """Load and cache the XLM-RoBERTa pipeline."""
        if self._model is None:
            logger.info("SentimentScorer: loading XLM-RoBERTa classifier...")
            self._model = load_classifier()
            logger.info("SentimentScorer: model ready.")
        return self._model

    def score_df(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
        summary_col: str = "summary",
        batch_log_every: int = 500,
    ) -> pd.DataFrame:
        """
        Score every row in df, adding five NLP columns in-place.

        New columns added:
            food_insecurity_score : float   (0–1, max across 10 hypotheses)
            is_relevant           : bool    (>= RELEVANCE_THRESHOLD)
            top_hypothesis        : str     (topic_id of highest-scoring hypothesis)
            top_topic_name        : str     (human-readable)
            nlp_all_scores        : object  (dict[topic_id → score])

        Articles that already have a non-null food_insecurity_score are skipped
        so the method is idempotent on partially scored DataFrames.

        Parameters
        ----------
        df             : DataFrame with at least title_col and summary_col.
        title_col      : column name for article headline.
        summary_col    : column name for article summary.
        batch_log_every: log progress every N articles.
        """
        clf = self.load()

        needs_scoring = (
            "food_insecurity_score" not in df.columns
            or df["food_insecurity_score"].isna().any()
        )
        if not needs_scoring:
            logger.info("score_df: all rows already scored — skipping model inference.")
            return df

        out = df.copy()
        for col in ("food_insecurity_score", "is_relevant", "top_hypothesis",
                    "top_topic_name", "nlp_all_scores"):
            if col not in out.columns:
                out[col] = None

        total = len(out)
        for i, idx in enumerate(out.index):
            if pd.notna(out.at[idx, "food_insecurity_score"]):
                continue

            result = score_article(
                clf,
                title=str(out.at[idx, title_col] or ""),
                summary=str(out.at[idx, summary_col] or ""),
            )
            out.at[idx, "food_insecurity_score"] = result["food_insecurity_score"]
            out.at[idx, "is_relevant"] = result["is_relevant"]
            out.at[idx, "top_hypothesis"] = result["top_hypothesis"]
            out.at[idx, "top_topic_name"] = result["top_topic_name"]
            out.at[idx, "nlp_all_scores"] = result["all_scores"]

            if (i + 1) % batch_log_every == 0:
                logger.info("score_df: %d / %d rows scored", i + 1, total)

        relevant = int(out["is_relevant"].sum())
        logger.info(
            "score_df complete: %d / %d articles relevant (threshold=%.2f)",
            relevant, total, RELEVANCE_THRESHOLD,
        )
        return out


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def score_articles_df(
    df: pd.DataFrame,
    title_col: str = "title",
    summary_col: str = "summary",
) -> pd.DataFrame:
    """
    Score a corpus DataFrame with the HungerGist multi-hypothesis pipeline.

    Thin wrapper around SentimentScorer().score_df() for import simplicity.

    Parameters
    ----------
    df         : corpus DataFrame — must have title_col and summary_col.
    title_col  : headline column name.
    summary_col: summary column name.

    Returns
    -------
    DataFrame with food_insecurity_score, is_relevant, top_hypothesis,
    top_topic_name, nlp_all_scores columns added.
    """
    scorer = SentimentScorer()
    return scorer.score_df(df, title_col=title_col, summary_col=summary_col)


# ---------------------------------------------------------------------------
# Keyword-based fallback scorer (no model download required)
# ---------------------------------------------------------------------------

# Food-insecurity signal keywords aligned to HungerGist topics (bilingual PH)
_KEYWORD_BANK: list[str] = [
    # T1 Food Supply / Price
    "bigas", "rice", "presyo", "food price", "food shortage", "pagkain",
    "gutom", "hunger", "malnutrition", "pagkukulang", "kakulangan ng pagkain",
    # T1b Fish Kill
    "fish kill", "patay na isda", "pamamatay ng isda", "red tide", "algal bloom",
    "laguna lake", "taal lake", "fishkill",
    # T2 Healthcare
    "malnutrition", "stunting", "underweight", "wasting", "malnourished",
    # T3 Governance / Policy
    "rice tariff", "batas taripa", "nfa", "price control", "food subsidy",
    "ayuda", "4ps", "pantawid",
    # T4 Finance / Economy
    "poverty", "kahirapan", "walang trabaho", "unemployment", "layoff",
    # T5 Infrastructure
    "farm to market road", "post-harvest", "cold storage",
    # T6 Land Use
    "farmland conversion", "agricultural land", "lupa",
    # T7 Civil Life
    "evacuee", "evacuation", "typhoon victim", "bakwit", "displaced",
    # T8 Social Instability
    "welga", "strike", "protest", "unrest",
    # T9 OFW Remittance
    "ofw", "remittance", "pinadala", "overseas filipino",
    # General food insecurity
    "food insecurity", "food crisis", "food security",
]

_KEYWORD_SET: set[str] = {k.lower() for k in _KEYWORD_BANK}


def keyword_score(title: str, summary: str) -> float:
    """
    Lightweight keyword-based food insecurity proxy score (0–1).

    Returns the proportion of matched keyword categories (capped at 1.0).
    Used as a bootstrap proxy when XLM-RoBERTa is unavailable.
    """
    text = f"{title} {summary}".lower()
    matched = sum(1 for kw in _KEYWORD_SET if kw in text)
    # Sigmoid-style scaling: 1 match ≈ 0.2, 5 matches ≈ 0.7, 10+ ≈ 0.9+
    return round(min(matched / 12.0, 1.0), 4)


def apply_keyword_scores_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply keyword_score to every row in df, adding food_insecurity_score.

    Used as a fast bootstrap when the transformer model is not available.
    Sets top_hypothesis = 'keyword_bootstrap' as a provenance flag.
    """
    out = df.copy()
    out["food_insecurity_score"] = out.apply(
        lambda r: keyword_score(
            str(r.get("title") or ""),
            str(r.get("summary") or ""),
        ),
        axis=1,
    )
    out["is_relevant"] = out["food_insecurity_score"] >= RELEVANCE_THRESHOLD
    out["top_hypothesis"] = "keyword_bootstrap"
    out["top_topic_name"] = "Keyword Bootstrap (no model)"
    out["nlp_all_scores"] = [{} for _ in range(len(out))]
    logger.info(
        "apply_keyword_scores_df: %d rows scored via keywords (%d relevant)",
        len(out),
        int(out["is_relevant"].sum()),
    )
    return out
