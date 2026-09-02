"""
scripts/train_fisheries_shock.py
--------------------------------
Train and evaluate against the fisheries food-availability shock target.

Kept separate from trainer.py so the existing pipeline is untouched and the
comparison is explicit. Same walk-forward machinery, same right-sized LightGBM
search space, same skill-over-persistence reporting.

Target : label_shock from build_fisheries_shock_label.py
         (persistence 0.618, balance 0.500, provinces differ in 52/56 quarters)
Features: features_fused.parquet, restricted to non-leaking columns. Nothing in
         the matrix derives from fisheries production, so unlike the composite
         stress label there is no circularity to remove.

Reports three feature sets so the NLP contribution is isolated:
    government-only, NLP-only, and combined -- each against naive persistence.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.training.cross_validation import WalkForwardSplitter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fisheries_train")

FEATURES = Path("data/processed/features_fused.parquet")
LABELS = Path("data/processed/fisheries_shock_labels.parquet")
OUT = Path("data/processed/fisheries_shock_results.json")

MIN_TRAIN_QUARTERS = 8
FORECAST_GAP = 0          # nowcast; raise to test lead time
RANDOM_SEED = 42

NLP_FEATURES = ["FSSI", "FSSI_lag1", "FSSI_lag2", "FSSI_accel", "trigger_climate",
                "trigger_fish_kill", "trigger_market"]
GOV_FEATURES = [
    "commodity_livestock", "commodity_leafy_veg", "commodity_fruit_veg",
    "rice_price_regular_lag1", "ofw_remit_yoy_pct_lag1", "unemployment_rate_lag1",
    "headline_cpi", "diesel_php_per_l_lag1",
    "rainfall_anomaly_pct_lag1", "rainfall_anomaly_pct_accel",
    "tc_count", "tc_severe_flag", "enso_numeric", "drought_alert",
]
LAG_FEATURES = ["shock_lag1", "shock_lag2"]

PARAMS = dict(objective="binary", verbosity=-1, random_state=RANDOM_SEED,
              class_weight="balanced", n_estimators=250, learning_rate=0.03,
              num_leaves=8, max_depth=3, min_child_samples=10,
              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5)


def load() -> pd.DataFrame:
    feats = pd.read_parquet(FEATURES)
    labs = pd.read_parquet(LABELS)[["province_code", "quarter", "label_shock", "dev_pct"]]
    df = feats.merge(labs, on=["province_code", "quarter"], how="inner")
    df = df.sort_values(["province_code", "quarter"]).reset_index(drop=True)

    # Lagged target: persistence's information, given to the model too.
    g = df.groupby("province_code")["label_shock"]
    df["shock_lag1"] = g.shift(FORECAST_GAP + 1)
    df["shock_lag2"] = g.shift(FORECAST_GAP + 2)
    df = df.dropna(subset=["shock_lag1", "shock_lag2"]).reset_index(drop=True)

    log.info("merged: %d rows | %d quarters | balance %.3f",
             len(df), df["quarter"].nunique(), df["label_shock"].mean())
    return df


def walk_forward(df: pd.DataFrame, cols: list[str]) -> dict:
    cols = [c for c in cols if c in df.columns]
    dfq = pd.DataFrame({"quarter": df["quarter"].values}, index=df.index)
    splitter = WalkForwardSplitter(min_train_quarters=MIN_TRAIN_QUARTERS,
                                   forecast_gap=FORECAST_GAP)
    acc, f1s, aucs, pers = [], [], [], []
    for tr, te in splitter.split(dfq):
        if len(tr) < 10 or len(te) < 2:
            continue
        ytr, yte = df.loc[tr, "label_shock"], df.loc[te, "label_shock"]
        if ytr.nunique() < 2:
            continue
        m = LGBMClassifier(**PARAMS).fit(df.loc[tr, cols], ytr)
        pred = m.predict(df.loc[te, cols])
        acc.append(accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, average="weighted", zero_division=0))
        if yte.nunique() > 1:
            aucs.append(roc_auc_score(yte, m.predict_proba(df.loc[te, cols])[:, 1]))
        pers.append(accuracy_score(yte, df.loc[te, "shock_lag1"].astype(int)))
    return {
        "n_folds": len(acc),
        "accuracy": float(np.mean(acc)) if acc else float("nan"),
        "f1": float(np.mean(f1s)) if f1s else float("nan"),
        "roc_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "persistence_accuracy": float(np.mean(pers)) if pers else float("nan"),
        "n_features": len(cols),
    }


def main() -> None:
    df = load()
    sets = {
        "persistence_only  (lagged target)": LAG_FEATURES,
        "government only":                   GOV_FEATURES,
        "NLP only":                          NLP_FEATURES,
        "government + NLP":                  GOV_FEATURES + NLP_FEATURES,
        "government + NLP + lags":           GOV_FEATURES + NLP_FEATURES + LAG_FEATURES,
    }
    results = {name: walk_forward(df, cols) for name, cols in sets.items()}

    print("\n" + "=" * 88)
    print("FISHERIES FOOD-AVAILABILITY SHOCK — walk-forward, gap=0")
    print("=" * 88)
    print(f"{'feature set':36s} {'nfeat':>6s} {'acc':>8s} {'F1':>8s} {'AUC':>8s} "
          f"{'persist':>9s} {'skill':>8s}")
    print("-" * 88)
    for name, r in results.items():
        skill = r["accuracy"] - r["persistence_accuracy"]
        print(f"{name:36s} {r['n_features']:>6d} {r['accuracy']:>8.4f} {r['f1']:>8.4f} "
              f"{r['roc_auc']:>8.4f} {r['persistence_accuracy']:>9.4f} {skill:>+8.4f}")
    print("=" * 88)
    print(f"folds: {results['government + NLP']['n_folds']}   rows: {len(df)}")

    OUT.write_text(json.dumps(results, indent=2))
    log.info("saved -> %s", OUT)


if __name__ == "__main__":
    main()
