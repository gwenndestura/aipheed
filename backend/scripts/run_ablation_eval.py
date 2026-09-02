"""
scripts/run_ablation_eval.py
----------------------------
Run the baseline + ablation benchmark against an ALREADY-TRAINED model.

run_w12_training.py runs training and evaluation together; this runs only the
evaluation half, reusing best_params from training_results.json, so the
100-trial Optuna search is not repeated just to get the ablations.

The question it answers: now that FSSI has been rebuilt from the audited
dataset, do the NLP features contribute anything the government features do not?
Compare lgbm_nlp_only against lgbm_psa_only and naive_persistence.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ablation")

RESULTS = Path("data/processed/training_results.json")


def _metric(m: dict, *names: str) -> float | None:
    for n in names:
        if n in m:
            return m[n]
    overall = m.get("overall") or {}
    for n in names:
        if n in overall:
            return overall[n]
    return None


def main() -> None:
    from app.ml.training.evaluator import run_baseline_evaluation

    if not RESULTS.exists():
        raise SystemExit(f"{RESULTS} not found - run scripts/run_training.py first")
    res = json.loads(RESULTS.read_text())
    best_params = res["best_params"]
    # Baselines must be scored at the same horizon as the model. run_w12_training.py
    # omits this argument, so its step 2 always raises and eval_results.json is
    # left stale - the pipeline then reports success because the old file exists.
    gap = res.get("max_reliable_lead_time", 1)
    log.info("loaded best_params from %s | forecast_gap=%d", RESULTS, gap)

    results = run_baseline_evaluation(best_params=best_params, forecast_gap=gap)

    print("\n" + "=" * 74)
    print("BASELINE / ABLATION BENCHMARK")
    print("=" * 74)
    print(f"{'baseline':32s} {'F1':>10s} {'ROC-AUC':>10s} {'accuracy':>10s}")
    print("-" * 74)
    for name, m in results.items():
        if not isinstance(m, dict):
            continue
        f1 = _metric(m, "mean_f1", "weighted_f1")
        auc = _metric(m, "mean_roc_auc", "roc_auc")
        acc = _metric(m, "mean_accuracy", "accuracy")
        if f1 is None and auc is None:
            continue
        fmt = lambda v: f"{v:10.4f}" if isinstance(v, (int, float)) else f"{'-':>10s}"
        print(f"{name:32s} {fmt(f1)} {fmt(auc)} {fmt(acc)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
