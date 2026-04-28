"""
app/ml/features/fssi_builder.py
---------------------------------
Food Stress Sentiment Index (FSSI) builder — province-quarter level.

Formula (Backend Guide v3, Ahn et al. 2023):
    FSSI_p,t = w_p,t * (1/N) * sum_i max_h(score(x_i, h))

where:
    w_p,t   = bias weight from bias_weighter.py (rural undercoverage correction)
    N       = number of geocoded articles in province p, quarter t
    score(x_i, h) = per-hypothesis score from XLM-RoBERTa zero-shot classifier
    max_h   = maximum across all 10 HungerGist hypothesis scores per article
              (= food_insecurity_score in the scored corpus)

Feature engineering also computes:
    FSSI_lag1   : FSSI shifted one quarter back (t-1)
    FSSI_lag2   : FSSI shifted two quarters back (t-2)
    FSSI_accel  : FSSI_t - FSSI_{t-1}  (momentum / acceleration)

Usage:
    from app.ml.features.fssi_builder import compute_fssi
    fssi_df = compute_fssi(geocoded_df, weights_df)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_PATH = Path("data/processed/fssi_quarterly.parquet")

# Quarters in chronological order for proper lag computation
_QUARTER_SORT_KEY: dict[str, int] = {}


def _quarter_to_int(q: str) -> int:
    """Convert 'YYYY-QN' to a sortable integer (e.g. '2020-Q1' → 20201)."""
    try:
        year, qpart = q.split("-")
        return int(year) * 4 + int(qpart[1]) - 1
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_fssi(
    articles_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    save_path: Path | None = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Compute FSSI_p,t for all province-quarter combinations in articles_df.

    Parameters
    ----------
    articles_df : pd.DataFrame
        Must contain: province_code, quarter, food_insecurity_score.
        province_code = None rows are filtered out.
        food_insecurity_score must be numeric (result from classifier or keyword proxy).

    weights_df : pd.DataFrame
        Output from bias_weighter.compute_bias_weights().
        Must contain: province_code, quarter, bias_weight.

    save_path : Path | None
        If provided, saves result to this Parquet path.

    Returns
    -------
    pd.DataFrame
        Columns:
            province_code  : str
            quarter        : str       ("YYYY-QN")
            article_count  : int
            fssi_raw       : float     (unweighted mean of food_insecurity_score)
            bias_weight    : float
            FSSI           : float     (bias-weighted FSSI)
            FSSI_lag1      : float     (t-1 value; NaN for first available quarter)
            FSSI_lag2      : float     (t-2 value; NaN for first two quarters)
            FSSI_accel     : float     (FSSI_t - FSSI_{t-1}; NaN for first quarter)
    """
    df = articles_df.copy()

    # Keep only geocoded, in-range articles
    df = df[df["province_code"].notna()].copy()
    if df.empty:
        logger.warning("compute_fssi: no geocoded articles — returning empty DataFrame")
        return _empty_fssi_df()

    if "food_insecurity_score" not in df.columns:
        raise ValueError(
            "articles_df must have 'food_insecurity_score' column. "
            "Run sentiment.py first (transformer or keyword fallback)."
        )

    df["food_insecurity_score"] = pd.to_numeric(df["food_insecurity_score"], errors="coerce").fillna(0.0)

    # Aggregate to province-quarter: mean of per-article max-hypothesis scores
    agg = (
        df.groupby(["province_code", "quarter"])
        .agg(
            article_count=("food_insecurity_score", "count"),
            fssi_raw=("food_insecurity_score", "mean"),
        )
        .reset_index()
    )

    # Join bias weights
    w = weights_df[["province_code", "quarter", "bias_weight"]].copy()
    agg = agg.merge(w, on=["province_code", "quarter"], how="left")
    agg["bias_weight"] = agg["bias_weight"].fillna(1.0)  # neutral weight if missing

    # Apply bias correction: FSSI = w_p,t * fssi_raw
    agg["FSSI"] = agg["bias_weight"] * agg["fssi_raw"]

    # Sort chronologically per province for lag computation
    agg["_q_int"] = agg["quarter"].apply(_quarter_to_int)
    agg = agg.sort_values(["province_code", "_q_int"]).reset_index(drop=True)

    # Compute lags and acceleration per province
    agg["FSSI_lag1"] = agg.groupby("province_code")["FSSI"].shift(1)
    agg["FSSI_lag2"] = agg.groupby("province_code")["FSSI"].shift(2)
    agg["FSSI_accel"] = agg["FSSI"] - agg["FSSI_lag1"]

    agg = agg.drop(columns=["_q_int"])

    logger.info(
        "compute_fssi: %d province-quarter rows | FSSI range [%.4f, %.4f]",
        len(agg),
        agg["FSSI"].min(),
        agg["FSSI"].max(),
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(save_path, index=False)
        logger.info("FSSI saved → %s", save_path)

    return agg


def _empty_fssi_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "province_code", "quarter", "article_count",
        "fssi_raw", "bias_weight", "FSSI",
        "FSSI_lag1", "FSSI_lag2", "FSSI_accel",
    ])


# ---------------------------------------------------------------------------
# Standalone runner helper (called from scripts/run_w11_member_a.py)
# ---------------------------------------------------------------------------

def build_fssi_from_parquets(
    corpus_path: Path = Path("data/processed/corpus_geocoded.parquet"),
    weights_path: Path = Path("data/processed/bias_weights.parquet"),
    save_path: Path = OUTPUT_PATH,
    use_keyword_fallback: bool = True,
) -> pd.DataFrame:
    """
    Load geocoded corpus + bias weights parquets, score if needed, compute FSSI.

    Parameters
    ----------
    corpus_path         : path to corpus_geocoded.parquet
    weights_path        : path to bias_weights.parquet
    save_path           : where to save fssi_quarterly.parquet
    use_keyword_fallback: if True and no food_insecurity_score column exists,
                          applies keyword-based proxy scores before computing FSSI.
                          Set False when transformer scores are pre-populated.
    """
    corpus_df = pd.read_parquet(corpus_path)
    weights_df = pd.read_parquet(weights_path)

    if (
        "food_insecurity_score" not in corpus_df.columns
        or corpus_df["food_insecurity_score"].isna().all()
    ):
        if use_keyword_fallback:
            logger.warning(
                "build_fssi_from_parquets: no transformer scores found — "
                "applying keyword-based proxy scores (FSSI bootstrap mode). "
                "Re-run after XLM-RoBERTa scoring to get final FSSI values."
            )
            from app.ml.nlp.sentiment import apply_keyword_scores_df
            corpus_df = apply_keyword_scores_df(corpus_df)
        else:
            raise RuntimeError(
                "No food_insecurity_score in corpus and use_keyword_fallback=False. "
                "Run sentiment.score_articles_df() first."
            )

    # Filter to 2020-Q1 → 2025-Q4 model window
    corpus_df = corpus_df[
        corpus_df["quarter"].between("2020-Q1", "2025-Q4", inclusive="both")
    ]

    return compute_fssi(corpus_df, weights_df, save_path=save_path)
