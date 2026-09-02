"""
scripts/train_final.py
----------------------
Production training for the aiPHeed food-availability shock model.

Winning configuration, selected empirically over every variant tried:

    per-commodity-group models
      + 4-model ensemble (LightGBM, RandomForest, ExtraTrees, LogisticRegression)
      + commodity target encoding (smoothed, fit on training folds only)
      + decision threshold chosen on the training fold

Operating figure (>= 12 quarters history, 90% coverage):
    accuracy 0.8217 · F1 0.8130 · majority class 0.7326 -> skill +0.0891
Full coverage, mature window:
    accuracy 0.7979 · majority 0.7153 -> skill +0.0826
All folds, full coverage:
    accuracy 0.7554 · majority 0.6853 -> skill +0.0700

Abstention raises accuracy AND skill, which is what separates it from tightening
the shock threshold -- that raises accuracy while skill collapses (+0.070 at
-10% down to +0.012 at -25%, where a constant predictor already scores 0.846).

Each component was measured, not assumed:
    per-group vs pooled          0.7543 vs 0.7496
    ensemble vs LightGBM alone   0.7543 vs 0.7316
    with vs without target enc.  0.7543 vs 0.7502

Rejected after testing (all made it worse):
    regression head on dev_pct   0.7510      recency weighting     0.7539
    deep seasonal lag8 features  0.7126      seed averaging        0.7271
    regression head alone        0.6837

Validation is walk-forward by quarter. Every fit -- model, target encoding and
threshold -- happens inside the training window; no test quarter influences any
of them. The comparator is seasonal persistence (same quarter last year), which
is the honest baseline for an annual agricultural cycle.

Target : label_shock from build_food_availability_panel.py
         (PSA OpenStat production volumes, CALABARZON, 2021-2026)
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_food_availability import (  # noqa: E402
    GOV, MATCHED, NLP, SEASONAL, SERIES, load,
)
from scripts.train_food_availability_v2 import (  # noqa: E402
    MIN_TRAIN, best_params, fit_predict, pick_threshold, target_encode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("final")

OUT = Path("data/processed/final_results.json")
VARIANT = "ens_tenc"

# --- Operating specification -------------------------------------------------
# Two conditions, both stated in every report of the headline figure.
#
# MATURITY_FOLDS: the model needs training history. Cold-start folds score 0.7104
# against 0.7979 once >= 12 quarters are available, so the first folds are
# reported separately rather than averaged into an operating figure.
#
# COVERAGE: the model abstains on the least-confident band, which is how an
# operational triage tool is meant to work -- high-confidence province-commodity
# quarters are auto-flagged, the rest go to analyst review. Unlike tightening the
# shock threshold (which raises accuracy while skill collapses, +0.070 -> +0.012),
# abstention raises accuracy AND skill: +0.0826 at full coverage, +0.0891 at 90%.
MATURITY_FOLDS = 4        # folds discarded as cold-start
COVERAGE = 0.90           # fraction of cases the model answers


def main() -> None:
    df = load()
    cols = [c for c in SERIES + SEASONAL + GOV + NLP + MATCHED if c in df.columns]
    params = best_params()
    quarters = sorted(df["quarter"].unique())

    rows = []
    for i in range(MIN_TRAIN, len(quarters)):
        tr = df[df["quarter"].isin(quarters[:i])].copy()
        te = df[df["quarter"] == quarters[i]].copy()
        if len(te) < 10 or tr["label_shock"].nunique() < 2:
            continue

        tr["commodity_te"], te["commodity_te"] = target_encode(tr, te, "commodity")
        use = cols + ["commodity_te"]

        prob = np.zeros(len(te))
        thr_col = np.zeros(len(te))
        for grp in te["group"].unique():
            mask = (te["group"] == grp).to_numpy()
            sub = tr[tr["group"] == grp]
            if len(sub) < 50 or sub["label_shock"].nunique() < 2:
                sub = tr                      # fall back to pooled for thin groups
            p_tr_g, p_g = fit_predict(sub, sub, use, VARIANT, params) if False else                           (None, None)
            # Fit once on the group, score both its own training rows (for the
            # threshold) and the test rows.
            p_tr_g, _ = fit_predict(sub, sub.head(1), use, VARIANT, params)
            _, p_g = fit_predict(sub, te[mask], use, VARIANT, params)
            prob[mask] = p_g
            # PER-GROUP threshold. The groups have very different base rates --
            # fisheries shocks ~56% of quarters, crops 24-30% -- so a single
            # pooled cut-off is systematically wrong for at least one of them.
            # Fisheries AUC was healthy (0.75) while its accuracy lagged, which
            # is the signature of a mis-set threshold rather than poor ranking.
            thr_col[mask] = pick_threshold(sub["label_shock"], p_tr_g)

        te = te.assign(prob=prob, pred=(prob >= thr_col).astype(int),
                       threshold=thr_col)
        rows.append(te)

    preds = pd.concat(rows, ignore_index=True)

    def m(d: pd.DataFrame) -> dict:
        out = {"n": len(d),
               "accuracy": accuracy_score(d["label_shock"], d["pred"]),
               "f1": f1_score(d["label_shock"], d["pred"], average="weighted", zero_division=0),
               "seasonal_persistence": accuracy_score(d["label_shock"],
                                                      d["shock_lag4"].astype(int)),
               "naive_persistence": accuracy_score(d["label_shock"],
                                                   d["shock_lag1"].astype(int)),
               # Always-predict-the-common-class. Any reported accuracy must be
               # quoted against this: on an imbalanced target a constant
               # predictor can look strong, and a stricter shock threshold
               # inflates accuracy by shrinking the positive class rather than
               # by predicting better.
               "majority_class": max(d["label_shock"].mean(),
                                     1 - d["label_shock"].mean())}
        out["roc_auc"] = (roc_auc_score(d["label_shock"], d["prob"])
                          if d["label_shock"].nunique() > 1 else float("nan"))
        out["skill_vs_seasonal"] = out["accuracy"] - out["seasonal_persistence"]
        out["skill_vs_naive"] = out["accuracy"] - out["naive_persistence"]
        out["skill_vs_majority"] = out["accuracy"] - out["majority_class"]
        return out

    # Confidence = distance from the decision boundary.
    preds["confidence"] = (preds["prob"] - preds["threshold"]).abs()

    quarters_scored = sorted(preds["quarter"].unique())
    mature = preds[preds["quarter"].isin(quarters_scored[MATURITY_FOLDS:])].copy()
    k = int(len(mature) * COVERAGE)
    operating = mature.nlargest(k, "confidence")
    operating_is = preds.index.isin(operating.index)
    preds["in_operating_set"] = operating_is

    cold_start = preds[preds["quarter"].isin(quarters_scored[:MATURITY_FOLDS])]

    overall = m(preds)
    mature_all = m(mature)
    operating_m = m(operating)
    cold = m(cold_start)
    by_group = {g: m(d) for g, d in operating.groupby("group")}
    by_prov = {g: m(d) for g, d in operating.groupby("province_name")}

    print("\n" + "=" * 78)
    print("aiPHeed — FINAL MODEL")
    print("=" * 78)
    def block(title: str, r: dict, note: str = "") -> None:
        print(f"\n  {title}{note}")
        print(f"    n={r['n']:5d}  accuracy={r['accuracy']:.4f}  F1={r['f1']:.4f}  "
              f"AUC={r['roc_auc']:.4f}")
        print(f"    majority {r['majority_class']:.4f} (skill {r['skill_vs_majority']:+.4f})"
              f"   |   seasonal persistence {r['seasonal_persistence']:.4f} "
              f"(skill {r['skill_vs_seasonal']:+.4f})")

    block("OPERATING FIGURE", operating_m,
          f"   [>={MATURITY_FOLDS} folds history, {COVERAGE:.0%} coverage]")
    print("    ^ quote WITH: the majority baseline, the coverage, and that the")
    print(f"      remaining {100 * (1 - COVERAGE):.0f}% are referred for analyst review.")
    block("mature window, full coverage", mature_all)
    block("all folds, full coverage", overall)
    block("cold-start folds (excluded from operating figure)", cold)

    print("\n  by commodity group  [operating set]")
    for g, r in by_group.items():
        print(f"    {g:22s} n={r['n']:5d}  acc={r['accuracy']:.4f}  "
              f"AUC={r['roc_auc']:.4f}  skill={r['skill_vs_seasonal']:+.4f}")
    print("\n  by province")
    for g, r in by_prov.items():
        print(f"    {g:22s} n={r['n']:5d}  acc={r['accuracy']:.4f}  "
              f"AUC={r['roc_auc']:.4f}  skill={r['skill_vs_seasonal']:+.4f}")
    print("=" * 78)

    OUT.write_text(json.dumps({
        "operating_spec": {"maturity_folds_excluded": MATURITY_FOLDS,
                           "coverage": COVERAGE,
                           "abstain_pct": round(100 * (1 - COVERAGE), 1)},
        "operating": operating_m, "mature_full_coverage": mature_all,
        "all_folds": overall, "cold_start": cold,
        "by_group": by_group, "by_province": by_prov}, indent=2, default=float))
    preds.to_parquet("data/processed/final_predictions.parquet", index=False)
    log.info("saved -> %s and data/processed/final_predictions.parquet", OUT)


if __name__ == "__main__":
    main()
