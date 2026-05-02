"""
scripts/run_w12_training.py
----------------------------
Week 12 — LightGBM Training and 4-Baseline Benchmark (Member B deliverables)

Executes:
  Step 1: WalkForward LightGBM training + Optuna 100-trial search
           → models/lgbm_best.pkl, models/optuna_study.pkl
           → MLflow run logged
  Step 2: 4-baseline benchmark on same walk-forward folds
           → data/processed/eval_results.json

Targets (Backend Guide v3):
  Weighted F1  >= 0.75
  ROC-AUC      >= 0.80

Usage:
    cd backend
    python scripts/run_w12_training.py

    # Reduce Optuna trials for quick test:
    python scripts/run_w12_training.py --n-trials 20

    # Skip baseline evaluation:
    python scripts/run_w12_training.py --no-eval
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("w12_training")


def main():
    parser = argparse.ArgumentParser(description="Week 12 LightGBM Training")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Optuna trial count (default: 100)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip 4-baseline evaluation")
    args = parser.parse_args()

    t0 = time.time()

    features_path = Path("data/processed/features_fused.parquet")
    labels_path   = Path("data/processed/labels.parquet")

    if not features_path.exists():
        logger.error(
            "features_fused.parquet not found. Run scripts/run_w11_pipeline.py first."
        )
        sys.exit(1)
    if not labels_path.exists():
        logger.error(
            "labels.parquet not found. Run scripts/run_w11_pipeline.py first."
        )
        sys.exit(1)

    # ── STEP 1: LightGBM training ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: LightGBM training (Walk-Forward CV + Optuna)")
    logger.info("Trials: %d", args.n_trials)
    logger.info("=" * 60)

    from app.ml.training.trainer import train_model, N_TRIALS
    import app.ml.training.trainer as trainer_module

    # Override trial count if specified
    if args.n_trials != 100:
        trainer_module.N_TRIALS = args.n_trials

    result = train_model()

    logger.info("=" * 60)
    logger.info("TRAINING RESULTS")
    logger.info("  Best CV F1       : %.4f", result["best_cv_f1"])
    logger.info("  Holdout F1       : %.4f", result["holdout_f1"])
    logger.info("  Holdout ROC-AUC  : %.4f", result["holdout_roc_auc"])
    logger.info("  F1 target (>=0.75)  : %s", "PASS" if result["meets_f1_target"] else "FAIL")
    logger.info("  AUC target (>=0.80) : %s", "PASS" if result["meets_roc_auc_target"] else "FAIL")
    logger.info("=" * 60)

    # Save training summary
    result_path = Path("data/processed/training_results.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Training results saved → %s", result_path)

    # ── STEP 2: Baseline evaluation ───────────────────────────────────────
    if not args.no_eval:
        logger.info("=" * 60)
        logger.info("STEP 2: 4-baseline benchmark evaluation")
        logger.info("=" * 60)
        try:
            from app.ml.training.evaluator import run_baseline_evaluation
            eval_results = run_baseline_evaluation()
            logger.info("Baseline evaluation complete → data/processed/eval_results.json")
            # Print summary
            for baseline, metrics in eval_results.items():
                if isinstance(metrics, dict) and "mean_f1" in metrics:
                    logger.info(
                        "  %-30s F1=%.4f  AUC=%.4f",
                        baseline,
                        metrics.get("mean_f1", 0),
                        metrics.get("mean_roc_auc", 0),
                    )
        except Exception as exc:
            logger.error("Baseline evaluation failed: %s", exc)
    else:
        logger.info("Skipping baseline evaluation (--no-eval).")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("WEEK 12 COMPLETE in %.1f seconds (%.1f min)", elapsed, elapsed / 60)
    logger.info("Deliverables:")
    logger.info("  models/lgbm_best.pkl      → %s", Path("models/lgbm_best.pkl").exists())
    logger.info("  models/optuna_study.pkl   → %s", Path("models/optuna_study.pkl").exists())
    logger.info("  eval_results.json         → %s", Path("data/processed/eval_results.json").exists())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
