"""
app/ml/nlp/topic_model.py
--------------------------
BERTopic topic model for unsupervised narrative topic discovery.

Embedding model: paraphrase-multilingual-MiniLM-L12-v2
  - Multilingual (covers Filipino/Taglish/English mixed text)
  - Compact (117M params, ~420MB) — feasible on CPU
  - Sentence-level embeddings align with HungerGist's sentence-gist design

The model is trained once on the full 2020-2025 corpus, then used to:
  1. Assign topic labels to each article
  2. Compute province-quarter topic proportion features (supplementary)
  3. Generate topic labels for PDF report narrative enrichment

Model persistence: models/bertopic_model/
Topic proportions: data/processed/topic_proportions.parquet

Usage:
    from app.ml.nlp.topic_model import TopicModelPipeline
    tm = TopicModelPipeline()
    tm.fit(texts)
    tm.save()
    proportions_df = tm.compute_proportions(articles_df)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from bertopic import BERTopic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = Path("models/bertopic_model")
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MIN_TOPIC_SIZE = 10          # minimum articles per topic cluster
N_GRAM_RANGE = (1, 2)        # unigrams + bigrams for topic representation
TOP_N_WORDS = 10             # keywords per topic in representation


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class TopicModelPipeline:
    """
    BERTopic wrapper with a multilingual sentence embedding backbone.

    The pipeline follows BERTopic's recommended production usage:
    - UMAP for dimensionality reduction
    - HDBSCAN for clustering
    - KeyBERT-inspired c-TF-IDF for topic representation

    Topic -1 is BERTopic's "outlier" bucket — articles that don't fit any
    discovered topic. These are kept in proportion features as topic_-1.
    """

    def __init__(self) -> None:
        self._model: BERTopic | None = None
        self._embedding_model = None

    def _load_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model

        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded.")
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise
        return self._embedding_model

    def fit(
        self,
        texts: list[str],
        min_topic_size: int = MIN_TOPIC_SIZE,
    ) -> BERTopic:
        """
        Fit BERTopic on a list of texts.

        Parameters
        ----------
        texts          : list of article strings (title + summary combined)
        min_topic_size : minimum cluster size passed to HDBSCAN

        Returns
        -------
        BERTopic model fitted on texts.
        """
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer

        embedding_model = self._load_embedding_model()

        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        vectorizer_model = CountVectorizer(
            ngram_range=N_GRAM_RANGE,
            stop_words="english",
            min_df=2,
        )
        representation_model = KeyBERTInspired()

        self._model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            top_n_words=TOP_N_WORDS,
            verbose=True,
        )

        logger.info("Fitting BERTopic on %d texts...", len(texts))
        self._model.fit_transform(texts)

        n_topics = len(self._model.get_topic_info()) - 1  # -1 for outlier topic
        logger.info("BERTopic fit complete: %d topics discovered", n_topics)
        return self._model

    def save(self, save_dir: Path = MODEL_DIR) -> None:
        """Persist the fitted BERTopic model to disk."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        save_dir.mkdir(parents=True, exist_ok=True)
        self._model.save(
            str(save_dir),
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=EMBEDDING_MODEL_NAME,
        )
        logger.info("BERTopic model saved → %s", save_dir)

    def load(self, model_dir: Path = MODEL_DIR) -> BERTopic:
        """Load a previously saved BERTopic model."""
        from bertopic import BERTopic
        if not model_dir.exists():
            raise FileNotFoundError(f"BERTopic model directory not found: {model_dir}")
        self._model = BERTopic.load(str(model_dir))
        logger.info("BERTopic model loaded from %s", model_dir)
        return self._model

    def transform(self, texts: list[str]) -> tuple[list[int], list[float]]:
        """
        Assign topics to new texts.

        Returns
        -------
        tuple of (topics, probabilities) arrays.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call fit() or load() first.")
        topics, probs = self._model.transform(texts)
        return topics, probs

    def get_topic_info(self) -> pd.DataFrame:
        """Return BERTopic's topic info DataFrame."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        return self._model.get_topic_info()

    def compute_proportions(
        self,
        articles_df: pd.DataFrame,
        save_path: Path | None = Path("data/processed/topic_proportions.parquet"),
        max_topics: int = 20,
    ) -> pd.DataFrame:
        """
        Compute province-quarter topic proportion features.

        Assigns each article a topic via transform(), then aggregates to
        province-quarter proportions for the top `max_topics` topics.

        Parameters
        ----------
        articles_df : DataFrame with province_code, quarter, title, summary.
        save_path   : where to save the result Parquet.
        max_topics  : number of top topics to keep as separate proportion columns.
                      Topics beyond top-N are grouped into topic_other_pct.

        Returns
        -------
        DataFrame with columns: province_code, quarter, topic_0_pct,
        topic_1_pct, ..., topic_N_pct, topic_other_pct, topic_outlier_pct
        """
        df = articles_df[articles_df["province_code"].notna()].copy()
        if df.empty:
            logger.warning("compute_proportions: no geocoded articles.")
            return pd.DataFrame()

        texts = (
            df["title"].fillna("").astype(str)
            + " "
            + df["summary"].fillna("").astype(str)
        ).tolist()

        topics, _ = self.transform(texts)
        df["topic_id"] = topics

        # Get top N most frequent topics (excluding -1 outlier)
        topic_counts = (
            df[df["topic_id"] != -1]["topic_id"].value_counts().head(max_topics)
        )
        top_topic_ids: list[int] = list(topic_counts.index)

        def _calc_province_quarter_props(grp: pd.DataFrame) -> pd.Series:
            n = len(grp)
            props: dict[str, float] = {}
            for tid in top_topic_ids:
                props[f"topic_{tid}_pct"] = (grp["topic_id"] == tid).sum() / n
            other = grp[
                ~grp["topic_id"].isin(top_topic_ids) & (grp["topic_id"] != -1)
            ].shape[0]
            outlier = (grp["topic_id"] == -1).sum()
            props["topic_other_pct"] = other / n
            props["topic_outlier_pct"] = outlier / n
            return pd.Series(props)

        result = (
            df.groupby(["province_code", "quarter"])
            .apply(_calc_province_quarter_props)
            .reset_index()
        )

        logger.info(
            "compute_proportions: %d province-quarter rows, %d topic columns",
            len(result),
            len(top_topic_ids) + 2,
        )

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(save_path, index=False)
            logger.info("Topic proportions saved → %s", save_path)

        return result


# ---------------------------------------------------------------------------
# Standalone fit-and-save runner
# ---------------------------------------------------------------------------

def fit_topic_model(
    corpus_path: Path = Path("data/processed/corpus_geocoded.parquet"),
    model_dir: Path = MODEL_DIR,
    proportions_path: Path = Path("data/processed/topic_proportions.parquet"),
    sample_size: int | None = None,
) -> pd.DataFrame:
    """
    Fit BERTopic on the geocoded corpus and save model + proportions.

    Parameters
    ----------
    corpus_path      : geocoded corpus parquet
    model_dir        : where to save the BERTopic model
    proportions_path : where to save topic proportion features
    sample_size      : if set, subsample corpus for faster fit (dev mode)

    Returns
    -------
    topic_proportions DataFrame
    """
    df = pd.read_parquet(corpus_path)

    # Filter to 2020-Q1 → 2025-Q4
    df = df[df["quarter"].between("2020-Q1", "2025-Q4", inclusive="both")]

    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        logger.info("fit_topic_model: sampled to %d articles", len(df))

    texts = (
        df["title"].fillna("").astype(str)
        + " "
        + df["summary"].fillna("").astype(str)
    ).tolist()

    tm = TopicModelPipeline()
    tm.fit(texts)
    tm.save(model_dir)

    return tm.compute_proportions(df, save_path=proportions_path)
