"""
scripts/experiment_threshold.py
--------------------------------
Does the decision threshold need to be chosen out of sample?

THE DEFECT
----------
train_final.py picks each group's threshold like this:

    p_tr_g, _ = fit_predict(sub, sub.head(1), use, VARIANT, params)
    thr_col[mask] = pick_threshold(sub["label_shock"], p_tr_g)

fit_predict returns m.predict_proba(A) where A is the TRAINING matrix, so the
threshold is chosen on in-sample predictions. Two of the four ensemble members
are a RandomForest (max_depth 10) and an ExtraTrees (max_depth 12); both fit
training rows very tightly, so their in-sample probabilities sit far closer to
0 and 1 than anything the model will produce at test time.

A cut-off placed on that distribution does not transfer. If it lands too high
the model under-calls shocks, which is consistent with the measured shock recall
of 0.58 against a precision of 0.69 -- the ranking is decent but the boundary
is conservative.

THE FIX UNDER TEST
------------------
Choose the threshold on genuinely out-of-sample probabilities: hold out the last
few TRAINING quarters, fit on the rest, predict the held-out quarters, and pick
the cut-off there. The model that scores the actual test quarter is still fit on
all training data -- only the threshold changes.

This is a correctness fix, not a tuning knob: no test-quarter information is
used either way, so any gain is real rather than borrowed.

Folds are split SELECT / CONFIRM exactly as in experiment_accuracy.py, and only
a CONFIRM gain counts.

USAGE
-----
    python scripts/experiment_threshold.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_accuracy import operating, score  # noqa: E402
from scripts.train_food_availability import (  # noqa: E402
    GOV, MATCHED, NLP, SEASONAL, SERIES, load,
)
from scripts.train_food_availability_v2 import (  # noqa: E402
    MIN_TRAIN, best_params, fit_predict, pick_threshold, target_encode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("thr")

OUT = Path("data/processed/experiment_threshold.json")
VARIANT = "ens_tenc"
MATURITY_FOLDS = 4


def choose_threshold(sub: pd.DataFrame, cols: list[str], params: dict,
                     holdout: int) -> float:
    """
    holdout=0 reproduces the current in-sample behaviour.
    holdout=k reserves the last k training quarters to place the cut-off.
    """
    if holdout == 0:
        p_tr, _ = fit_predict(sub, sub.head(1), cols, VARIANT, params)
        return pick_threshold(sub["label_shock"], p_tr)

    qs = sorted(sub["quarter"].unique())
    if len(qs) < holdout + 4:
        p_tr, _ = fit_predict(sub, sub.head(1), cols, VARIANT, params)
        return pick_threshold(sub["label_shock"], p_tr)

    inner_tr = sub[sub["quarter"].isin(qs[:-holdout])]
    inner_te = sub[sub["quarter"].isin(qs[-holdout:])]
    if inner_tr["label_shock"].nunique() < 2 or len(inner_te) < 20 \
            or inner_te["label_shock"].nunique() < 2:
        p_tr, _ = fit_predict(sub, sub.head(1), cols, VARIANT, params)
        return pick_threshold(sub["label_shock"], p_tr)

    _, p_oof = fit_predict(inner_tr, inner_te, cols, VARIANT, params)
    return pick_threshold(inner_te["label_shock"], p_oof)


def walk_forward(df: pd.DataFrame, cols: list[str], params: dict,
                 holdout: int) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
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
        thr = np.zeros(len(te))
        for grp in te["group"].unique():
            mask = (te["group"] == grp).to_numpy()
            sub = tr[tr["group"] == grp]
            if len(sub) < 50 or sub["label_shock"].nunique() < 2:
                sub = tr
            # The scoring model still sees every training row; only the
            # threshold is placed on held-out quarters.
            _, p_te = fit_predict(sub, te[mask], use, VARIANT, params)
            prob[mask] = p_te
            thr[mask] = choose_threshold(sub, use, params, holdout)

        rows.append(te.assign(prob=prob, pred=(prob >= thr).astype(int),
                              threshold=thr, fold=i))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    df = load()
    cols = [c for c in SERIES + SEASONAL + GOV + NLP + MATCHED if c in df.columns]
    params = best_params()

    quarters = sorted(df["quarter"].unique())
    mature_q = quarters[MIN_TRAIN:][MATURITY_FOLDS:]
    split_at = mature_q[len(mature_q) // 2 - 1]
    log.info("SELECT <= %s | CONFIRM > %s", split_at, split_at)

    results = {}
    thresholds_seen = {}
    for holdout in (0, 2, 3, 4):
        name = "in-sample (current)" if holdout == 0 else f"out-of-sample, {holdout}q holdout"
        log.info("running %s", name)
        preds = walk_forward(df, cols, params, holdout)
        results[name] = score(preds, split_at)
        op = operating(preds)
        thresholds_seen[name] = {
            g: round(float(s["threshold"].mean()), 3)
            for g, s in op.groupby("group")
        }

    print("\n" + "=" * 100)
    print("THRESHOLD PLACEMENT — operating set, same model, same features")
    print(f"SELECT <= {split_at}   |   CONFIRM > {split_at} (never used to choose)")
    print("=" * 100)
    base = results["in-sample (current)"]
    print(f"{'threshold source':30s} {'sel acc':>9s} {'con acc':>9s} {'delta':>8s} "
          f"{'con prec':>9s} {'con rec':>9s} {'con F1':>9s} {'con AUC':>9s}")
    print("-" * 100)
    for name, r in results.items():
        c, s = r.get("confirm", {}), r.get("select", {})
        d = c.get("accuracy", float("nan")) - base.get("confirm", {}).get("accuracy", float("nan"))
        print(f"{name:30s} {s.get('accuracy', float('nan')):9.4f} "
              f"{c.get('accuracy', float('nan')):9.4f} {d:+8.4f} "
              f"{c.get('precision_shock', float('nan')):9.4f} "
              f"{c.get('recall_shock', float('nan')):9.4f} "
              f"{c.get('f1_shock', float('nan')):9.4f} "
              f"{c.get('roc_auc', float('nan')):9.4f}")
    print("=" * 100)
    print("mean threshold by group:")
    for name, t in thresholds_seen.items():
        print(f"  {name:30s} {t}")

    OUT.write_text(json.dumps({"split_at": split_at, "results": results,
                               "mean_thresholds": thresholds_seen}, indent=2))
    log.info("saved -> %s", OUT)


if __name__ == "__main__":
    main()
