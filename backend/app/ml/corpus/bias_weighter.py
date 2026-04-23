"""
app/ml/corpus/bias_weighter.py
-------------------------------
Inverse-coverage spatial bias weight computation.

Formula (from backend guide):
    w_p,t = log(1 + median_articles) / log(1 + articles_p,t)

Provinces with below-median article counts receive upweighted FSSI values
to counteract systematic rural undercoverage before features enter LightGBM.

Outputs:
  - Province-quarter bias weight Parquet at data/processed/bias_weights.parquet
  - Article count log for the data_sufficiency_flag threshold

Usage:
    from app.ml.corpus.bias_weighter import compute_bias_weights
    weights_df = compute_bias_weights(geocoded_df)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum article count below which a province-quarter gets data_sufficiency_flag
DATA_SUFFICIENCY_MIN_ARTICLES = 5

OUTPUT_PATH = Path("data/processed/bias_weights.parquet")


# ---------------------------------------------------------------------------
# Quarter helpers
# ---------------------------------------------------------------------------

def _month_to_quarter(month: int) -> str:
    """Return 'Q1'/'Q2'/'Q3'/'Q4' for a given month integer (1–12)."""
    return f"Q{(month - 1) // 3 + 1}"


def _parse_quarter(published: str) -> str | None:
    """
    Extract 'YYYY-QN' string from an ISO datetime string.
    Returns None if parsing fails.
    """
    try:
        dt = pd.Timestamp(published)
        return f"{dt.year}-{_month_to_quarter(dt.month)}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_bias_weights(
    articles_df: pd.DataFrame,
    save_path: Path | None = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Compute province-quarter bias weights from a geocoded article DataFrame.

    Parameters
    ----------
    articles_df : pd.DataFrame
        Must contain columns: 'province_code', 'published'.
        province_code = None rows are excluded from weight computation.

    save_path : Path | None
        If provided, writes the output Parquet to this path.
        Pass None to skip saving (useful in tests).

    Returns
    -------
    pd.DataFrame
        Columns:
            quarter             : str  — e.g. "2024-Q2"
            province_code       : str  — PSGC code
            article_count       : int  — articles for this province-quarter
            median_articles     : float — cross-province median for that quarter
            bias_weight         : float — w_p,t = log(1+median) / log(1+count)
            data_sufficiency    : str  — "OK" | "LIMITED_SIGNAL"
    """
    df = articles_df.copy()

    # Filter to geocoded articles only
    df = df[df["province_code"].notna()].copy()
    if df.empty:
        logger.warning("compute_bias_weights: no geocoded articles — returning empty DataFrame")
        return pd.DataFrame(columns=[
            "quarter", "province_code", "article_count",
            "median_articles", "bias_weight", "data_sufficiency",
        ])

    # Derive quarter from published timestamp
    df["quarter"] = df["published"].apply(_parse_quarter)
    df = df[df["quarter"].notna()]

    # Count articles per province-quarter
    counts = (
        df.groupby(["quarter", "province_code"])
        .size()
        .reset_index(name="article_count")
    )

    # Compute median article count per quarter (cross-province median)
    quarter_medians = (
        counts.groupby("quarter")["article_count"]
        .median()
        .reset_index(name="median_articles")
    )
    counts = counts.merge(quarter_medians, on="quarter", how="left")

    # Apply bias weight formula: w_p,t = log(1 + median) / log(1 + count)
    def _weight(row: pd.Series) -> float:
        denom = math.log(1 + row["article_count"])
        if denom == 0:
            return 1.0
        return math.log(1 + row["median_articles"]) / denom

    counts["bias_weight"] = counts.apply(_weight, axis=1)

    # Data sufficiency flag
    counts["data_sufficiency"] = counts["article_count"].apply(
        lambda n: "OK" if n >= DATA_SUFFICIENCY_MIN_ARTICLES else "LIMITED_SIGNAL"
    )

    limited = (counts["data_sufficiency"] == "LIMITED_SIGNAL").sum()
    if limited > 0:
        logger.warning(
            "compute_bias_weights: %d province-quarters flagged LIMITED_SIGNAL "
            "(< %d articles)",
            limited, DATA_SUFFICIENCY_MIN_ARTICLES,
        )

    logger.info(
        "compute_bias_weights: %d province-quarter rows, weight range [%.3f, %.3f]",
        len(counts),
        counts["bias_weight"].min(),
        counts["bias_weight"].max(),
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        counts.to_parquet(save_path, index=False)
        logger.info("Bias weights saved → %s", save_path)

    return counts


def get_weight(
    weights_df: pd.DataFrame,
    province_code: str,
    quarter: str,
    default: float = 1.0,
) -> float:
    """
    Look up the bias weight for a specific province-quarter.

    Returns `default` (1.0) if the combination is not found —
    a weight of 1.0 applies no correction (neutral).
    """
    mask = (
        (weights_df["province_code"] == province_code)
        & (weights_df["quarter"] == quarter)
    )
    row = weights_df[mask]
    if row.empty:
        logger.debug(
            "get_weight: no entry for %s / %s — returning default %.2f",
            province_code, quarter, default,
        )
        return default
    return float(row.iloc[0]["bias_weight"])
