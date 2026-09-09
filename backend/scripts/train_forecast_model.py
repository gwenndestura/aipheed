"""
scripts/train_forecast_model.py
--------------------------------
Train the FORECAST variant: the same model, restricted to information that
exists before the quarter it scores.

Why a second model. train_final.py builds a same-quarter nowcast: 17 of its 33
features describe the quarter being scored -- that quarter's CPI, typhoon
count, ENSO phase, news volume. None of those exist until the quarter is over,
so that model can never be run ahead of the data, whatever else is collected.
It answers "what happened last quarter", which is a monitoring question.

This variant scores a quarter only from information that existed before it
began, in two steps:

    load(feature_lag=1)  attaches the whole 19-column government/climate matrix
                         a quarter late, so headline_cpi, tc_count, ENSO,
                         drought, FSSI and the price series all arrive at t-1.
    shifted here         matched_articles and total_articles, the two features
                         built on the panel itself rather than the matrix.

Everything else was already lag-safe: shock_lag*, dev_lag*, dev_roll4,
series_vol and prov_shock_rate_lag1 are all built with .shift(); group_code,
province_idx, quarter_num and commodity_te carry no future information.

Measured cost, full coverage, same rows and baselines as the nowcast:

    model                accuracy    AUC    vs majority   vs seasonal
    nowcast (as-built)     0.7873   0.8052    +0.0857       +0.0054
    forecast (lagged)      0.7567   0.7969    +0.0551       -0.0252

Read that honestly. The forecast model does NOT beat seasonal persistence on
binary accuracy -- but neither did the nowcast by much (+0.0054), so removing
same-quarter information reveals a thin edge rather than destroying a large
one. What survives is the ranking: AUC 0.797 against 0.805. Seasonal
persistence emits a bare yes/no and cannot rank at all, so it cannot answer
"which of this province's series should be looked at first" -- which is the
question a triage tool is actually for.

Coverage bonus. Because the government features are joined at t-1, this model
can score one quarter PAST the feature matrix's edge: 2026-Q1 is scoreable
from 2025-Q4 features, where the nowcast needs 2026-Q1 features that do not
exist yet.

Outputs
-------
    models/food_availability_forecast_model.joblib
    data/processed/forecast_results.json

Usage
-----
    python scripts/train_forecast_model.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_food_availability import (  # noqa: E402
    GOV, MATCHED, NLP, SEASONAL, SERIES, classification_metrics, load,
    seasonal_baseline_acc,
)
from scripts.train_food_availability_v2 import (  # noqa: E402
    MIN_TRAIN, SMOOTHING, best_params, build_members, fit_predict,
    pick_threshold, target_encode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("forecast")

OUT = Path("data/processed/forecast_results.json")
BUNDLE = Path("models/food_availability_forecast_model.joblib")
VARIANT = "ens_tenc"
MATURITY_FOLDS = 4
COVERAGE = 0.90

# The government/climate matrix is attached at t-1 by load(feature_lag=1), so
# all 19 of its columns are already forecast-safe. Only these two are built on
# the panel itself at quarter t and still need shifting.
CONTEMPORANEOUS = ["matched_articles", "total_articles"]

# Attaching the whole matrix a quarter late also makes the features that were
# already lagged effectively lag-2. That is more conservative than strictly
# necessary -- rainfall at t-1 IS knowable before t begins -- but it is the
# price of reaching a quarter past the matrix's own edge, which is what lets
# this model score 2026-Q1 when the nowcast cannot.
FEATURE_LAG = 1

LAG_SUFFIX = "__prev"


def lag_contemporaneous(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Shift each contemporaneous feature one quarter within its own series."""
    d = df.sort_values(["province_code", "commodity", "quarter"]).copy()
    g = d.groupby(["province_code", "commodity"], sort=False)
    lagged = []
    for c in CONTEMPORANEOUS:
        if c in d.columns:
            d[f"{c}{LAG_SUFFIX}"] = g[c].shift(1)
            lagged.append(c)
    return d, lagged


def forecast_feature_cols(df: pd.DataFrame, lagged: list[str]) -> list[str]:
    base = [c for c in SERIES + SEASONAL + GOV + NLP + MATCHED if c in df.columns]
    return [f"{c}{LAG_SUFFIX}" if c in lagged else c for c in base]


def metrics(d: pd.DataFrame) -> dict:
    maj = max(d["label_shock"].mean(), 1 - d["label_shock"].mean())
    seasonal, seasonal_n = seasonal_baseline_acc(d)
    naive = accuracy_score(d["label_shock"], d["shock_lag1"].astype(int))
    out = {"n": len(d)}
    # accuracy, precision, recall, F1 (positive class and weighted), ROC-AUC
    # and the confusion counts.
    out.update(classification_metrics(d["label_shock"], d["pred"], d["prob"]))
    acc = out["accuracy"]
    out.update({
        "seasonal_persistence": seasonal,
        # 2021 rows have no fourth lag, so the seasonal baseline is scored on
        # fewer rows than the model. Quote this alongside skill_vs_seasonal.
        "seasonal_persistence_n": seasonal_n,
        "naive_persistence": naive,
        "majority_class": maj,
    })
    out["skill_vs_seasonal"] = acc - seasonal
    out["skill_vs_naive"] = acc - naive
    out["skill_vs_majority"] = acc - maj
    return out


def persist(df: pd.DataFrame, cols: list[str], params: dict) -> None:
    """Fit on the full panel and store a servable bundle."""
    prior = df["label_shock"].mean()
    stats = df.groupby("commodity")["label_shock"].agg(["sum", "count"])
    encoding = ((stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)).to_dict()

    fitted = df.copy()
    fitted["commodity_te"] = fitted["commodity"].map(encoding).fillna(prior)
    use = cols + (["commodity_te"] if "commodity_te" not in cols else [])

    groups: dict[str, dict] = {}
    for grp, sub in fitted.groupby("group"):
        train = sub if (len(sub) >= 50 and sub["label_shock"].nunique() > 1) else fitted
        members = build_members(params)
        for name, m in members:
            X = train[use].fillna(-999) if name in ("rf", "et") else train[use]
            m.fit(X, train["label_shock"])
        p_tr = np.mean([
            m.predict_proba(train[use].fillna(-999) if n in ("rf", "et") else train[use])[:, 1]
            for n, m in members], axis=0)
        groups[grp] = {"members": members,
                       "threshold": pick_threshold(train["label_shock"], p_tr)}
        log.info("fitted %-22s n=%5d threshold=%.3f", grp, len(train),
                 groups[grp]["threshold"])

    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "groups": groups,
        "feature_cols": use,
        "commodity_encoding": encoding,
        "encoding_prior": float(prior),
        "trained_through": max(df["quarter"]),
        "n_train_rows": len(fitted),
        "horizon": "one-quarter-ahead",
        "lagged_features": CONTEMPORANEOUS,
        "feature_lag": FEATURE_LAG,
        "lag_suffix": LAG_SUFFIX,
        "note": ("Forecast variant: every contemporaneous feature is taken at t-1, "
                 "so a quarter is scored only from information available before it "
                 "begins. Performance figures come from the walk-forward evaluation "
                 "in forecast_results.json, not from this fit."),
    }, BUNDLE)
    log.info("forecast bundle -> %s", BUNDLE)


def main() -> None:
    raw = load(feature_lag=FEATURE_LAG)
    df, lagged = lag_contemporaneous(raw)
    cols = forecast_feature_cols(df, lagged)
    log.info("lagged %d contemporaneous features: %s", len(lagged), lagged)

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
        thr = np.zeros(len(te))
        for grp in te["group"].unique():
            mask = (te["group"] == grp).to_numpy()
            sub = tr[tr["group"] == grp]
            if len(sub) < 50 or sub["label_shock"].nunique() < 2:
                sub = tr
            p_tr_g, _ = fit_predict(sub, sub.head(1), use, VARIANT, params)
            _, p_g = fit_predict(sub, te[mask], use, VARIANT, params)
            prob[mask] = p_g
            thr[mask] = pick_threshold(sub["label_shock"], p_tr_g)

        rows.append(te.assign(prob=prob, pred=(prob >= thr).astype(int), threshold=thr))

    preds = pd.concat(rows, ignore_index=True)
    preds["confidence"] = (preds["prob"] - preds["threshold"]).abs()

    scored = sorted(preds["quarter"].unique())
    mature = preds[preds["quarter"].isin(scored[MATURITY_FOLDS:])].copy()
    operating = mature.nlargest(int(len(mature) * COVERAGE), "confidence")
    cold = preds[preds["quarter"].isin(scored[:MATURITY_FOLDS])]

    results = {
        "horizon": "one-quarter-ahead",
        "lagged_features": CONTEMPORANEOUS,
        "feature_lag": FEATURE_LAG,
        "operating_spec": {"maturity_folds_excluded": MATURITY_FOLDS,
                           "coverage": COVERAGE,
                           "abstain_pct": round((1 - COVERAGE) * 100, 1)},
        "operating": metrics(operating),
        "mature_full_coverage": metrics(mature),
        "all_folds": metrics(preds),
        "cold_start": metrics(cold),
        "by_group": {g: metrics(d) for g, d in operating.groupby("group")},
        "by_province": {g: metrics(d) for g, d in operating.groupby("province_name")},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    print("aiPHeed - FORECAST MODEL (one quarter ahead)")
    print("=" * 78)
    for title, r in [("OPERATING FIGURE  [>=4 folds, 90% coverage]", results["operating"]),
                     ("mature window, full coverage", results["mature_full_coverage"]),
                     ("all folds, full coverage", results["all_folds"])]:
        print(f"\n  {title}")
        # Shock class first: at a ~0.29 positive rate the weighted averages are
        # carried by the easy majority class.
        print(f"    n={r['n']:5d}   accuracy={r['accuracy']:.4f}")
        print(f"      shock class    precision={r['precision_shock']:.4f}  "
              f"recall={r['recall_shock']:.4f}  F1={r['f1_shock']:.4f}")
        print(f"      weighted avg   precision={r['precision_weighted']:.4f}  "
              f"recall={r['recall_weighted']:.4f}  F1={r['f1_weighted']:.4f}")
        print(f"      ROC-AUC={r['roc_auc']:.4f}   confusion "
              f"TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']}")
        print(f"    majority {r['majority_class']:.4f} (skill {r['skill_vs_majority']:+.4f})"
              f"   |   seasonal persistence {r['seasonal_persistence']:.4f} "
              f"(skill {r['skill_vs_seasonal']:+.4f})")
    print("\n  Quote WITH: that this model does NOT beat seasonal persistence on")
    print("  accuracy, and that its value is the forward RANKING (AUC) which a")
    print("  persistence rule cannot provide at all.")

    persist(df, cols, params)


if __name__ == "__main__":
    main()
