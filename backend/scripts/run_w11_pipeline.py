"""
scripts/run_w11_pipeline.py
----------------------------
Week 11 — NLP Feature Extraction Pipeline (Member A deliverables)

Executes all Week 11 steps in order:

  Step 1: Bias weights           → data/processed/bias_weights.parquet
  Step 2: NLP scoring            → corpus_geocoded.parquet (adds NLP columns)
  Step 3: FSSI computation       → data/processed/fssi_quarterly.parquet
  Step 4: Trigger classification → data/processed/trigger_proportions.parquet
  Step 5: BERTopic (optional)    → data/processed/topic_proportions.parquet
  Step 6: Label generation       → data/processed/labels.parquet
  Step 7: Feature matrix fusion  → data/processed/features_fused.parquet

Usage:
    cd backend
    python scripts/run_w11_pipeline.py

    # Skip BERTopic (fast mode):
    python scripts/run_w11_pipeline.py --no-bertopic

    # Use keyword scoring instead of XLM-RoBERTa:
    python scripts/run_w11_pipeline.py --keyword-scores

    # Re-score even if scores already exist:
    python scripts/run_w11_pipeline.py --force-rescore
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Make sure backend/ is on sys.path when running from backend/
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("w11_pipeline")

PROCESSED = Path("data/processed")
RAW = Path("data/raw")

CORPUS_PATH = PROCESSED / "corpus_geocoded.parquet"
BIAS_WEIGHTS_PATH = PROCESSED / "bias_weights.parquet"
FSSI_PATH = PROCESSED / "fssi_quarterly.parquet"
TRIGGERS_PATH = PROCESSED / "trigger_proportions.parquet"
TOPICS_PATH = PROCESSED / "topic_proportions.parquet"
LABELS_PATH = PROCESSED / "labels.parquet"
FEATURES_PATH = PROCESSED / "features_fused.parquet"


def step1_bias_weights(corpus_df):
    """Compute province-quarter bias weights."""
    logger.info("=" * 60)
    logger.info("STEP 1: Bias weight computation")
    logger.info("=" * 60)
    from app.ml.corpus.bias_weighter import compute_bias_weights
    weights_df = compute_bias_weights(corpus_df, save_path=BIAS_WEIGHTS_PATH)
    logger.info("Bias weights: %d province-quarter rows", len(weights_df))
    return weights_df


def step2_nlp_scoring(corpus_df, use_keyword: bool, force: bool):
    """Score corpus articles with food insecurity scores."""
    import pandas as pd

    logger.info("=" * 60)
    logger.info("STEP 2: NLP food insecurity scoring")
    logger.info("=" * 60)

    already_scored = (
        "food_insecurity_score" in corpus_df.columns
        and corpus_df["food_insecurity_score"].notna().sum() > 0
    )

    if already_scored and not force:
        n_scored = corpus_df["food_insecurity_score"].notna().sum()
        logger.info(
            "Corpus already has %d/%d scored articles — skipping (use --force-rescore to redo).",
            n_scored, len(corpus_df),
        )
        return corpus_df

    if use_keyword:
        logger.info("Using keyword-based scoring (fast, no model download).")
        from app.ml.nlp.sentiment import apply_keyword_scores_df
        scored_df = apply_keyword_scores_df(corpus_df)
    else:
        logger.info("Using XLM-RoBERTa zero-shot scoring (slow on first run — downloading model).")
        from app.ml.nlp.sentiment import score_articles_df
        scored_df = score_articles_df(corpus_df)

    # nlp_all_scores is a dict column — pyarrow can't serialize empty dicts as struct.
    # Convert to JSON string for parquet storage; drop if fully empty.
    if "nlp_all_scores" in scored_df.columns:
        try:
            import json as _json
            scored_df["nlp_all_scores"] = scored_df["nlp_all_scores"].apply(
                lambda x: _json.dumps(x) if isinstance(x, dict) else (x or "{}")
            )
        except Exception:
            scored_df = scored_df.drop(columns=["nlp_all_scores"])

    # Save updated corpus with NLP columns (scored version)
    scored_path = PROCESSED / "corpus_scored.parquet"
    scored_df.to_parquet(scored_path, index=False)
    n_relevant = int(scored_df["is_relevant"].sum()) if "is_relevant" in scored_df.columns else 0
    logger.info(
        "Corpus scored: %d total | %d relevant (food insecurity signal)",
        len(scored_df), n_relevant,
    )
    return scored_df


def step3_fssi(corpus_df, weights_df):
    """Compute FSSI province-quarter index."""
    logger.info("=" * 60)
    logger.info("STEP 3: FSSI computation")
    logger.info("=" * 60)
    import pandas as pd
    from app.ml.features.fssi_builder import compute_fssi

    # Filter to model window and geocoded rows
    model_df = corpus_df[corpus_df["province_code"].notna()].copy()
    # Add quarter column if not already present
    if "quarter" not in model_df.columns:
        def _to_quarter(pub):
            try:
                dt = pd.Timestamp(pub)
                return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
            except Exception:
                return None
        model_df["quarter"] = model_df["published"].apply(_to_quarter)

    model_df = model_df[model_df["quarter"].notna()]

    fssi_df = compute_fssi(model_df, weights_df, save_path=FSSI_PATH)
    logger.info("FSSI: %d province-quarter rows", len(fssi_df))
    return fssi_df


def step4_triggers(corpus_df):
    """Classify articles into 5-category trigger taxonomy."""
    logger.info("=" * 60)
    logger.info("STEP 4: Trigger classification (5 categories)")
    logger.info("=" * 60)
    import pandas as pd
    from app.ml.nlp.trigger_classifier import classify_triggers_df, compute_trigger_proportions

    # Filter to geocoded
    geo_df = corpus_df[corpus_df["province_code"].notna()].copy()
    if "quarter" not in geo_df.columns:
        def _to_quarter(pub):
            try:
                dt = pd.Timestamp(pub)
                return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
            except Exception:
                return None
        geo_df["quarter"] = geo_df["published"].apply(_to_quarter)
    geo_df = geo_df[geo_df["quarter"].notna()]

    classified_df = classify_triggers_df(geo_df)
    triggers_df = compute_trigger_proportions(classified_df, save_path=TRIGGERS_PATH)
    logger.info("Trigger proportions: %d province-quarter rows", len(triggers_df))
    return triggers_df


def step5_bertopic(corpus_df):
    """Fit BERTopic and compute province-quarter topic proportions."""
    logger.info("=" * 60)
    logger.info("STEP 5: BERTopic topic modeling (optional)")
    logger.info("=" * 60)
    try:
        from app.ml.nlp.topic_model import fit_topic_model, compute_proportions
        import pandas as pd

        geo_df = corpus_df[corpus_df["province_code"].notna()].copy()
        if "quarter" not in geo_df.columns:
            def _to_quarter(pub):
                try:
                    dt = pd.Timestamp(pub)
                    return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
                except Exception:
                    return None
            geo_df["quarter"] = geo_df["published"].apply(_to_quarter)

        geo_df = geo_df[geo_df["quarter"].notna()]
        texts = (geo_df["title"].fillna("") + " " + geo_df["summary"].fillna("")).tolist()

        logger.info("Fitting BERTopic on %d geocoded articles...", len(texts))
        model_path = Path("models/bertopic_model")
        topic_model, topics = fit_topic_model(texts, save_path=model_path)

        geo_df = geo_df.copy()
        geo_df["_topic"] = topics
        proportions_df = compute_proportions(geo_df, topic_model, save_path=TOPICS_PATH)
        logger.info("BERTopic: %d province-quarter rows, %d topic columns",
                    len(proportions_df), len(proportions_df.columns) - 2)
        return proportions_df
    except Exception as exc:
        logger.warning("BERTopic failed (%s) — skipping. Features will run without topics.", exc)
        return None


def step6_labels():
    """Generate FIES primary labels + CPI robustness labels."""
    logger.info("=" * 60)
    logger.info("STEP 6: Label generation (FIES primary + CPI robustness)")
    logger.info("=" * 60)
    from app.ml.features.label_generator import generate_labels
    labels_df = generate_labels()
    dist = labels_df["label_stress"].value_counts().to_dict()
    logger.info(
        "Labels: %d province-quarter rows | label_stress distribution: %s",
        len(labels_df), dist,
    )
    return labels_df


def step7_feature_matrix():
    """Fuse all sources into the training feature matrix."""
    logger.info("=" * 60)
    logger.info("STEP 7: Feature matrix fusion (all 11 sources + NLP)")
    logger.info("=" * 60)
    from app.ml.features.feature_matrix import build_feature_matrix
    features_df = build_feature_matrix()
    logger.info(
        "Feature matrix: %s | %d feature columns",
        features_df.shape, len(features_df.columns) - 2,
    )
    return features_df


def main():
    parser = argparse.ArgumentParser(description="Week 11 NLP Feature Pipeline")
    parser.add_argument("--no-bertopic", action="store_true",
                        help="Skip BERTopic (faster; features matrix will omit topic proportions)")
    parser.add_argument("--keyword-scores", action="store_true",
                        help="Use keyword-based NLP scoring instead of XLM-RoBERTa")
    parser.add_argument("--force-rescore", action="store_true",
                        help="Re-score even if food_insecurity_score already exists")
    args = parser.parse_args()

    t0 = time.time()

    logger.info("=" * 60)
    logger.info("WEEK 11 PIPELINE — NLP Feature Extraction")
    logger.info("Corpus: %s", CORPUS_PATH)
    logger.info("Mode: %s", "keyword" if args.keyword_scores else "XLM-RoBERTa")
    logger.info("=" * 60)

    import pandas as pd

    if not CORPUS_PATH.exists():
        logger.error("Corpus not found at %s. Run corpus collection first.", CORPUS_PATH)
        sys.exit(1)

    corpus_df = pd.read_parquet(CORPUS_PATH)
    logger.info("Corpus loaded: %d articles", len(corpus_df))

    # STEP 1
    weights_df = step1_bias_weights(corpus_df)

    # STEP 2
    corpus_df = step2_nlp_scoring(corpus_df, args.keyword_scores, args.force_rescore)

    # STEP 3
    step3_fssi(corpus_df, weights_df)

    # STEP 4
    step4_triggers(corpus_df)

    # STEP 5 (optional)
    if not args.no_bertopic:
        step5_bertopic(corpus_df)
    else:
        logger.info("Skipping BERTopic (--no-bertopic).")

    # STEP 6
    step6_labels()

    # STEP 7
    features_df = step7_feature_matrix()

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("WEEK 11 PIPELINE COMPLETE in %.1f seconds", elapsed)
    logger.info("Deliverables:")
    logger.info("  bias_weights.parquet    → %s", BIAS_WEIGHTS_PATH.exists())
    logger.info("  fssi_quarterly.parquet  → %s", FSSI_PATH.exists())
    logger.info("  trigger_proportions.parquet → %s", TRIGGERS_PATH.exists())
    logger.info("  labels.parquet          → %s", LABELS_PATH.exists())
    logger.info("  features_fused.parquet  → %s", FEATURES_PATH.exists())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
