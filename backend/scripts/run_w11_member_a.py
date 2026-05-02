"""
scripts/run_w11_member_a.py
----------------------------
Week 11 Member A pipeline runner.

Executes all four W11-N steps in order:
  W11-1: Geocoding + bias weights  → corpus_geocoded.parquet, bias_weights.parquet
  W11-2: FSSI computation          → fssi_quarterly.parquet
  W11-3: Trigger classification    → trigger_proportions.parquet
  W11-3b: BERTopic (optional)      → topic_proportions.parquet + models/bertopic_model/
  W11-4: Labels + feature matrix   → labels.parquet, label_distribution.json,
                                      features_fused.parquet

Usage (from backend/ directory):
    python scripts/run_w11_member_a.py
    python scripts/run_w11_member_a.py --skip-bertopic
    python scripts/run_w11_member_a.py --skip-bertopic --use-transformer
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_w11")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 11 Member A pipeline runner")
    p.add_argument(
        "--skip-bertopic", action="store_true",
        help="Skip BERTopic model fitting (faster; topic features will be absent from feature matrix)",
    )
    p.add_argument(
        "--use-transformer", action="store_true",
        help="Use XLM-RoBERTa for FSSI scoring (requires model download; default: keyword fallback)",
    )
    p.add_argument(
        "--corpus-path", type=Path, default=Path("data/raw/corpus_raw.parquet"),
        help="Input corpus parquet (default: data/raw/corpus_raw.parquet)",
    )
    p.add_argument(
        "--bertopic-sample", type=int, default=None,
        help="Subsample N articles for BERTopic fit (default: all). Use 10000 for dev.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_w11_1(corpus_path: Path) -> tuple[Path, Path]:
    """W11-1: Geocoding + bias weights."""
    logger.info("=== W11-1: Geocoding + Bias Weights ===")
    t0 = time.time()

    import pandas as pd
    from app.ml.corpus.geocoder import geocode_batch
    from app.ml.corpus.bias_weighter import compute_bias_weights

    corpus_df = pd.read_parquet(corpus_path)
    logger.info("Loaded %d articles from %s", len(corpus_df), corpus_path)

    # Geocode
    records = corpus_df.to_dict(orient="records")
    records = geocode_batch(records)
    geocoded_df = pd.DataFrame(records)

    geocoded_path = Path("data/processed/corpus_geocoded.parquet")
    geocoded_df.to_parquet(geocoded_path, index=False)

    matched = geocoded_df["province_code"].notna().sum()
    logger.info(
        "Geocoding complete: %d / %d articles matched (%.1f%%)",
        matched, len(geocoded_df), matched / len(geocoded_df) * 100,
    )
    logger.info("corpus_geocoded.parquet saved → %s", geocoded_path)

    # Bias weights
    weights_df = compute_bias_weights(
        geocoded_df,
        save_path=Path("data/processed/bias_weights.parquet"),
    )
    logger.info("W11-1 complete (%.1fs)", time.time() - t0)
    return geocoded_path, Path("data/processed/bias_weights.parquet")


def step_w11_2(geocoded_path: Path, weights_path: Path, use_transformer: bool) -> Path:
    """W11-2: FSSI computation."""
    logger.info("=== W11-2: FSSI Builder ===")
    t0 = time.time()

    from app.ml.features.fssi_builder import build_fssi_from_parquets

    fssi_path = Path("data/processed/fssi_quarterly.parquet")
    fssi_df = build_fssi_from_parquets(
        corpus_path=geocoded_path,
        weights_path=weights_path,
        save_path=fssi_path,
        use_keyword_fallback=not use_transformer,
    )
    logger.info(
        "FSSI complete: %d province-quarter rows (%.1fs)",
        len(fssi_df), time.time() - t0,
    )
    return fssi_path


def step_w11_3_triggers(geocoded_path: Path) -> Path:
    """W11-3: Trigger classification."""
    logger.info("=== W11-3: Trigger Classifier ===")
    t0 = time.time()

    import pandas as pd
    from app.ml.nlp.trigger_classifier import classify_triggers_df, compute_trigger_proportions

    geocoded_df = pd.read_parquet(geocoded_path)

    # Filter to model window
    geocoded_df = geocoded_df[geocoded_df["quarter"].between("2020-Q1", "2025-Q4")]

    # Score triggers
    scored_df = classify_triggers_df(geocoded_df)

    proportions_path = Path("data/processed/trigger_proportions.parquet")
    proportions_df = compute_trigger_proportions(scored_df, save_path=proportions_path)

    logger.info(
        "Triggers complete: %d province-quarter rows (%.1fs)",
        len(proportions_df), time.time() - t0,
    )
    return proportions_path


def step_w11_3b_bertopic(
    geocoded_path: Path,
    sample_size: int | None = None,
) -> Path | None:
    """W11-3b: BERTopic topic model (optional)."""
    logger.info("=== W11-3b: BERTopic ===")
    t0 = time.time()

    try:
        from app.ml.nlp.topic_model import fit_topic_model
        proportions_path = Path("data/processed/topic_proportions.parquet")
        fit_topic_model(
            corpus_path=geocoded_path,
            model_dir=Path("models/bertopic_model"),
            proportions_path=proportions_path,
            sample_size=sample_size,
        )
        logger.info("BERTopic complete (%.1fs)", time.time() - t0)
        return proportions_path
    except Exception as exc:
        logger.warning("BERTopic step failed (will continue without): %s", exc)
        return None


def step_w11_4(psa_path: Path = Path("data/processed/psa_indicators.parquet")) -> tuple[Path, Path, Path]:
    """W11-4: Label generation + feature matrix."""
    logger.info("=== W11-4: Labels + Feature Matrix ===")
    t0 = time.time()

    from app.ml.features.label_generator import generate_labels
    from app.ml.features.feature_matrix import build_feature_matrix

    labels_df = generate_labels()
    logger.info("Labels: %d rows | label_stress dist: %s",
                len(labels_df), labels_df["label_stress"].value_counts().to_dict())

    features_df = build_feature_matrix()
    logger.info(
        "Feature matrix: shape %s | %d feature columns",
        features_df.shape, len(features_df.columns) - 2,
    )

    logger.info("W11-4 complete (%.1fs)", time.time() - t0)
    return (
        Path("data/processed/labels.parquet"),
        Path("data/processed/label_distribution.json"),
        Path("data/processed/features_fused.parquet"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("=================================================")
    logger.info("  aiPHeed — Week 11 Member A Pipeline Runner")
    logger.info("=================================================")
    logger.info("Config: use_transformer=%s | skip_bertopic=%s",
                args.use_transformer, args.skip_bertopic)

    total_t0 = time.time()

    geocoded_path, weights_path = step_w11_1(args.corpus_path)
    fssi_path = step_w11_2(geocoded_path, weights_path, args.use_transformer)
    triggers_path = step_w11_3_triggers(geocoded_path)

    if not args.skip_bertopic:
        step_w11_3b_bertopic(geocoded_path, sample_size=args.bertopic_sample)
    else:
        logger.info("BERTopic skipped (--skip-bertopic).")

    step_w11_4()

    total_time = time.time() - total_t0
    logger.info("=================================================")
    logger.info("  Week 11 pipeline complete in %.1fs", total_time)
    logger.info("  Outputs:")
    logger.info("    data/processed/corpus_geocoded.parquet")
    logger.info("    data/processed/bias_weights.parquet")
    logger.info("    data/processed/fssi_quarterly.parquet")
    logger.info("    data/processed/trigger_proportions.parquet")
    logger.info("    data/processed/labels.parquet")
    logger.info("    data/processed/label_distribution.json")
    logger.info("    data/processed/features_fused.parquet")
    if not args.skip_bertopic:
        logger.info("    data/processed/topic_proportions.parquet")
        logger.info("    models/bertopic_model/")
    logger.info("=================================================")


if __name__ == "__main__":
    main()
