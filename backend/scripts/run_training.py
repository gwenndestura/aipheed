"""
CLI entry point for aiPHeed LightGBM training.

Usage:
    python scripts/run_training.py

Steps:
    1. Load features_fused.parquet + labels.parquet
    2. Walk-forward Optuna 100-trial search
    3. Retrain final model on full data
    4. Save models/lgbm_best.pkl and models/optuna_study.pkl
    5. Log with MLflow
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_training")


def main():
    from app.ml.training.trainer import train_model

    logger.info("Starting aiPHeed LightGBM training...")
    result = train_model()

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Best CV F1:     %.4f", result["best_cv_f1"])
    logger.info("Final F1:       %.4f", result["final_f1"])
    logger.info("Final ROC-AUC:  %.4f", result["final_roc_auc"])
    logger.info("F1 target:      %s", "PASS" if result["meets_f1_target"] else "FAIL")
    logger.info("AUC target:     %s", "PASS" if result["meets_roc_auc_target"] else "FAIL")
    logger.info("Model saved to: models/lgbm_best.pkl")


if __name__ == "__main__":
    main()