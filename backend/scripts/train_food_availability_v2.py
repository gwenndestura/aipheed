"""
scripts/train_food_availability_v2.py
-------------------------------------
Best-effort training on the food-availability panel.

Everything here is fit inside the training folds only. Walk-forward by quarter;
no test quarter ever contributes to fitting, encoding, calibration, or threshold
selection. Seasonal persistence (same quarter last year) is the honest baseline
and is recomputed on identical folds.

Techniques v1 did not have:

1. Commodity target encoding. v1 identified a series only by `group_code` (3
   values) while the panel holds 104 distinct commodities that shock at very
   different rates. Each commodity's historical shock rate is encoded with
   smoothing toward the global mean, computed on training rows only.

2. Regression head. Binarising to a 0/1 shock discards the magnitude of the
   deviation. A LightGBM regressor on dev_pct, thresholded at the same cut-off,
   often recovers signal the classifier loses -- and it can be blended with the
   classifier's probability.

3. Recency weighting. Older quarters get exponentially less weight, so the model
   tracks the recent regime rather than averaging across it.

4. Per-group models. Fisheries, fruit and vegetables have different dynamics;
   one model per commodity group can beat a single pooled model.

Also carried over from v1: 4-model ensemble, and a decision threshold chosen on
the training fold instead of assuming 0.5.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_food_availability import (  # noqa: E402
    GOV, MATCHED, NLP, SEASONAL, SERIES, load,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v2")

OUT = Path("data/processed/food_availability_v2.json")
TUNED = Path("data/processed/food_availability_tuned.json")
MIN_TRAIN = 8
SHOCK_THRESHOLD = -10.0
SMOOTHING = 10.0          # target-encoding prior strength
HALF_LIFE = 6.0           # quarters; recency weight halves every this many


def best_params() -> dict:
    p = json.loads(TUNED.read_text())["ALL (series+seasonal+gov+NLP+matched)"]["best_params"]
    return dict(objective="binary", verbosity=-1, random_state=42,
                class_weight="balanced", **p)


def build_members(params: dict) -> list[tuple[str, object]]:
    """
    The four ensemble members, defined once so training and serving cannot drift.
    "rf"/"et" are marked because they need NaN filled before fit/predict.
    """
    return [
        ("lgbm", LGBMClassifier(**params)),
        ("rf", RandomForestClassifier(n_estimators=400, max_depth=10,
                                      min_samples_leaf=5, class_weight="balanced",
                                      random_state=42, n_jobs=-1)),
        ("et", ExtraTreesClassifier(n_estimators=400, max_depth=12,
                                    min_samples_leaf=4, class_weight="balanced",
                                    random_state=42, n_jobs=-1)),
        ("lr", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000,
                                                class_weight="balanced"))),
    ]


def target_encode(tr: pd.DataFrame, te: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """Smoothed mean-encode `col` by historical shock rate, fit on train only."""
    prior = tr["label_shock"].mean()
    stats = tr.groupby(col)["label_shock"].agg(["sum", "count"])
    enc = (stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)
    return tr[col].map(enc).fillna(prior), te[col].map(enc).fillna(prior)


def recency_weights(tr: pd.DataFrame, quarters: list[str]) -> np.ndarray:
    idx = {q: i for i, q in enumerate(quarters)}
    age = tr["quarter"].map(idx).max() - tr["quarter"].map(idx)
    return np.power(0.5, age / HALF_LIFE).to_numpy()


def pick_threshold(y: pd.Series, p: np.ndarray) -> float:
    grid = np.linspace(0.15, 0.85, 71)
    return float(grid[int(np.argmax([accuracy_score(y, (p >= t).astype(int)) for t in grid]))])


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str],
                variant: str, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_prob, test_prob) for one variant."""
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    w = recency_weights(tr, sorted(tr["quarter"].unique())) if "recency" in variant else None

    probs_tr, probs_te = [], []

    if "reg" in variant:
        # Regression head: predict the deviation, convert to a shock probability
        # by distance below the threshold, scaled by the training residual sd.
        r = LGBMRegressor(objective="regression", verbosity=-1, random_state=42,
                          n_estimators=params.get("n_estimators", 300),
                          learning_rate=params.get("learning_rate", 0.05),
                          num_leaves=params.get("num_leaves", 16),
                          max_depth=params.get("max_depth", 4),
                          min_child_samples=params.get("min_child_samples", 20),
                          subsample=params.get("subsample", 0.8),
                          colsample_bytree=params.get("colsample_bytree", 0.8))
        r.fit(Xtr, tr["dev_pct"], sample_weight=w)
        sd = float(np.std(tr["dev_pct"] - r.predict(Xtr))) or 1.0
        from scipy.stats import norm
        probs_tr.append(norm.cdf((SHOCK_THRESHOLD - r.predict(Xtr)) / sd))
        probs_te.append(norm.cdf((SHOCK_THRESHOLD - r.predict(Xte)) / sd))

    if "clf" in variant or "ens" in variant:
        members = build_members(params) if "ens" in variant             else [("lgbm", LGBMClassifier(**params))]
        for name, m in members:
            A = Xtr.fillna(-999) if name in ("rf", "et") else Xtr
            B = Xte.fillna(-999) if name in ("rf", "et") else Xte
            try:
                m.fit(A, tr["label_shock"], sample_weight=w) if w is not None and name != "lr" \
                    else m.fit(A, tr["label_shock"])
            except TypeError:
                m.fit(A, tr["label_shock"])
            probs_tr.append(m.predict_proba(A)[:, 1])
            probs_te.append(m.predict_proba(B)[:, 1])

    return np.mean(probs_tr, axis=0), np.mean(probs_te, axis=0)


def evaluate(df: pd.DataFrame, cols: list[str], variant: str,
             per_group: bool = False) -> dict:
    params = best_params()
    quarters = sorted(df["quarter"].unique())
    acc, f1s, aucs, p4 = [], [], [], []

    for i in range(MIN_TRAIN, len(quarters)):
        tr = df[df["quarter"].isin(quarters[:i])]
        te = df[df["quarter"] == quarters[i]]
        if len(te) < 10 or tr["label_shock"].nunique() < 2:
            continue

        tr, te = tr.copy(), te.copy()
        if "tenc" in variant:
            tr["commodity_te"], te["commodity_te"] = target_encode(tr, te, "commodity")
            use = cols + ["commodity_te"]
        else:
            use = cols

        if per_group:
            pte = np.zeros(len(te))
            for grp in te["group"].unique():
                mtr, mte = tr[tr["group"] == grp], te["group"] == grp
                if len(mtr) < 50 or mtr["label_shock"].nunique() < 2:
                    mtr = tr
                ptr_g, pte_g = fit_predict(mtr, te[mte], use, variant, params)
                pte[mte.to_numpy()] = pte_g
            ptr, _ = fit_predict(tr, tr.head(1), use, variant, params)
            thr = pick_threshold(tr["label_shock"], ptr)
        else:
            ptr, pte = fit_predict(tr, te, use, variant, params)
            thr = pick_threshold(tr["label_shock"], ptr)

        pred = (pte >= thr).astype(int)
        acc.append(accuracy_score(te["label_shock"], pred))
        f1s.append(f1_score(te["label_shock"], pred, average="weighted", zero_division=0))
        if te["label_shock"].nunique() > 1:
            aucs.append(roc_auc_score(te["label_shock"], pte))
        p4.append(accuracy_score(te["label_shock"], te["shock_lag4"].astype(int)))

    return {"accuracy": float(np.mean(acc)), "f1": float(np.mean(f1s)),
            "roc_auc": float(np.mean(aucs)), "seasonal_persistence": float(np.mean(p4)),
            "folds": len(acc)}


def main() -> None:
    df = load()
    cols = [c for c in SERIES + SEASONAL + GOV + NLP + MATCHED if c in df.columns]

    variants = {
        "v1 baseline (ens + thr)":        ("ens", False),
        "+ commodity target encoding":    ("ens_tenc", False),
        "+ regression head":              ("ens_reg_tenc", False),
        "+ recency weighting":            ("ens_reg_tenc_recency", False),
        "regression head only":           ("reg_tenc", False),
        "per-group models":               ("ens_tenc", True),
    }
    results = {}
    for name, (variant, per_group) in variants.items():
        try:
            results[name] = evaluate(df, cols, variant, per_group)
            r = results[name]
            log.info("%-32s acc=%.4f f1=%.4f auc=%.4f", name, r["accuracy"], r["f1"], r["roc_auc"])
        except Exception as exc:
            log.warning("%s failed: %s", name, str(exc)[:110])

    print("\n" + "=" * 90)
    print("V2 — best-effort configurations (walk-forward, all fitting inside train folds)")
    print("=" * 90)
    print(f"{'configuration':34s} {'acc':>8s} {'F1':>8s} {'AUC':>8s} {'lag4':>8s} {'skill4':>8s}")
    print("-" * 90)
    for name, r in results.items():
        print(f"{name:34s} {r['accuracy']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>8.4f} "
              f"{r['seasonal_persistence']:>8.4f} "
              f"{r['accuracy'] - r['seasonal_persistence']:>+8.4f}")
    print("=" * 90)

    OUT.write_text(json.dumps(results, indent=2))
    log.info("saved -> %s", OUT)


if __name__ == "__main__":
    main()
