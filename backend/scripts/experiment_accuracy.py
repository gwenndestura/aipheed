"""
scripts/experiment_accuracy.py
-------------------------------
Candidate improvements to the food-availability model, measured honestly.

WHY A SEPARATE SCRIPT
---------------------
Picking a feature set by running every candidate over the same folds you then
report is how a number gets talked up without the model getting better. The
folds are therefore split:

    SELECT  the earlier test quarters -- candidates are compared here
    CONFIRM the later test quarters   -- never consulted while choosing

A candidate is only worth keeping if it improves on CONFIRM. A candidate that
wins on SELECT and loses on CONFIRM was fitting the selection folds, and is
reported as such rather than quietly dropped.

WHAT IS DELIBERATELY NOT TRIED
------------------------------
Three levers would raise accuracy while making the model less useful, and are
excluded on principle:

  * Raising the shock threshold. At -25% a constant "no shock" predictor already
    scores 0.846 while skill over it collapses from +0.070 to +0.012. The
    threshold sensitivity analysis in build_food_availability_panel.py settles
    this: -10% is retained because it maximises demonstrable skill.
  * Lowering coverage below 90%. Abstention is legitimate and already applied,
    but pushing it further buys accuracy by answering fewer questions.
  * Widening the cold-start exclusion past 4 folds. That discards the hard
    quarters rather than predicting them.

CANDIDATES
----------
All are seasonal-memory features, because the ablation is unambiguous that
seasonality carries this target: `seasonal only` (5 features) reaches AUC 0.761
while `government only` (14) reaches 0.552 and `NLP only` (5) reaches 0.500.

  qoy      Quarter-of-year specific history. shock_rate_hist already exists but
           pools every quarter together, so a series that reliably fails in Q3
           and never in Q1 gets one blended rate. These split it by
           quarter-of-year: for a 2025-Q3 row, the mean over 2024-Q3, 2023-Q3,
           2022-Q3 and no others.
  lag12    Three years back, same quarter. Only became reachable when the window
           grew to 20 quarters; at 16 it would have emptied the panel.
  pgrate   Province x commodity-group shock rate at t-1. prov_shock_rate_lag1
           exists but pools groups, and fisheries shocks at ~56% against 24-30%
           for crops, so the pooled rate is wrong for both.
  devqoy   How far the last observed deviation sits from what this series
           normally does in that quarter of the year.

Every one is built with .shift() inside the series, so a row sees only quarters
that precede it.

USAGE
-----
    python scripts/experiment_accuracy.py            # feature candidates
    python scripts/experiment_accuracy.py --retune   # + Optuna on current window
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_food_availability import (  # noqa: E402
    GOV, MATCHED, NLP, SEASONAL, SERIES, classification_metrics, load,
)
from scripts.train_food_availability_v2 import (  # noqa: E402
    MIN_TRAIN, best_params, fit_predict, pick_threshold, target_encode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("experiment")

OUT = Path("data/processed/experiment_accuracy.json")
VARIANT = "ens_tenc"
MATURITY_FOLDS = 4
COVERAGE = 0.90

KEY = ["group", "commodity", "province_code"]

# "qoy" was ADOPTED on 2026-09-09 and now lives in SEASONAL / load(), so it is
# no longer a candidate -- listing it here would duplicate the column names in
# base_cols and LightGBM rejects a duplicated feature name outright.
# Kept for the record: it scored +0.0102 SELECT / +0.0133 CONFIRM.
CANDIDATES = {
    "lag12":  ["shock_lag12", "dev_lag12"],
    "pgrate": ["pg_shock_rate_lag1"],
    "devqoy": ["dev_vs_qoy"],
}


def add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every candidate, built from strictly prior quarters."""
    d = df.sort_values(KEY + ["year", "quarter_num"]).reset_index(drop=True)
    g = d.groupby(KEY)
    qoy = d.groupby(KEY + ["quarter_num"])

    # Quarter-of-year history: shift(1) inside (series, quarter-of-year) means
    # the 2025-Q3 row sees 2024-Q3 and earlier Q3s only.
    d["shock_rate_qoy"] = qoy["label_shock"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    d["dev_qoy_mean"] = qoy["dev_pct"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())

    d["shock_lag12"] = g["label_shock"].shift(12)
    d["dev_lag12"] = g["dev_pct"].shift(12)

    # Province x group spillover at t-1. Built on the panel's own labels the
    # same way prov_shock_rate_lag1 is, then merged back.
    pg = (d.groupby(["province_code", "group", "quarter"])["label_shock"]
            .mean().rename("pg_rate").reset_index()
            .sort_values(["province_code", "group", "quarter"]))
    pg["pg_shock_rate_lag1"] = pg.groupby(["province_code", "group"])["pg_rate"].shift(1)
    d = d.merge(pg[["province_code", "group", "quarter", "pg_shock_rate_lag1"]],
                on=["province_code", "group", "quarter"], how="left")

    d["dev_vs_qoy"] = d["dev_lag1"] - d["dev_qoy_mean"]
    return d


def walk_forward(df: pd.DataFrame, cols: list[str], params: dict) -> pd.DataFrame:
    """The train_final configuration: ensemble, per-group models and thresholds."""
    # De-duplicate while preserving order: a feature promoted from CANDIDATES
    # into the baseline set would otherwise appear twice, and LightGBM raises
    # "Feature appears more than one time" rather than ignoring it.
    seen: set[str] = set()
    cols = [c for c in cols
            if c in df.columns and not (c in seen or seen.add(c))]
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
            p_tr, _ = fit_predict(sub, sub.head(1), use, VARIANT, params)
            _, p_te = fit_predict(sub, te[mask], use, VARIANT, params)
            prob[mask] = p_te
            thr[mask] = pick_threshold(sub["label_shock"], p_tr)

        rows.append(te.assign(prob=prob, pred=(prob >= thr).astype(int),
                              threshold=thr, fold=i))
    return pd.concat(rows, ignore_index=True)


def operating(preds: pd.DataFrame) -> pd.DataFrame:
    """Mature folds, then the most-confident COVERAGE fraction."""
    folds = sorted(preds["fold"].unique())
    mature = preds[preds["fold"] >= folds[MATURITY_FOLDS]] if len(folds) > MATURITY_FOLDS else preds
    conf = (mature["prob"] - mature["threshold"]).abs()
    cut = conf.quantile(1 - COVERAGE)
    return mature[conf >= cut]


def score(preds: pd.DataFrame, select_until: str) -> dict:
    """Metrics on the SELECT half and the untouched CONFIRM half."""
    op = operating(preds)
    sel = op[op["quarter"] <= select_until]
    con = op[op["quarter"] > select_until]
    out = {}
    for name, d in (("select", sel), ("confirm", con)):
        if len(d) < 20:
            out[name] = {"n": len(d)}
            continue
        m = classification_metrics(d["label_shock"], d["pred"], d["prob"])
        out[name] = {"n": len(d), "accuracy": m["accuracy"],
                     "precision_shock": m["precision_shock"],
                     "recall_shock": m["recall_shock"],
                     "f1_shock": m["f1_shock"], "roc_auc": m["roc_auc"],
                     "majority": float(max(d["label_shock"].mean(),
                                           1 - d["label_shock"].mean()))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retune", action="store_true",
                    help="also run an Optuna search on the current window")
    ap.add_argument("--retune-only", action="store_true",
                    help="skip the feature trials and only run the search")
    args = ap.parse_args()

    df = add_candidate_features(load())
    base_cols = [c for c in SERIES + SEASONAL + GOV + NLP + MATCHED if c in df.columns]
    params = best_params()

    quarters = sorted(df["quarter"].unique())
    test_quarters = quarters[MIN_TRAIN:]
    mature_q = test_quarters[MATURITY_FOLDS:]
    split_at = mature_q[len(mature_q) // 2 - 1]
    log.info("operating quarters %s .. %s | SELECT <= %s, CONFIRM > %s",
             mature_q[0], mature_q[-1], split_at, split_at)

    baseline_key = f"baseline ({len(base_cols)} features)"
    trials: dict[str, list[str]] = {baseline_key: base_cols}
    if not args.retune_only:
        for name, feats in CANDIDATES.items():
            trials[f"+ {name}"] = base_cols + feats
        trials["+ all candidates"] = base_cols + [f for v in CANDIDATES.values()
                                                  for f in v]

    results = {}
    for name, cols in trials.items():
        log.info("running %s (%d features)", name, len(cols))
        results[name] = score(walk_forward(df, cols, params), split_at)

    if args.retune:
        log.info("retuning hyperparameters on the current window")
        results["retuned params (baseline features)"] = retune(df, base_cols, split_at)

    print("\n" + "=" * 104)
    print("ACCURACY EXPERIMENTS — operating set, ensemble + per-group thresholds")
    print(f"SELECT quarters <= {split_at}   |   CONFIRM quarters > {split_at} "
          f"(never used for choosing)")
    print("=" * 104)
    base = results[baseline_key]
    hdr = f"{'candidate':34s}"
    for half in ("select", "confirm"):
        hdr += f" {half+' acc':>12s} {'delta':>8s}"
    print(hdr + f" {'conf rec':>9s} {'conf AUC':>9s}")
    print("-" * 104)
    for name, r in results.items():
        line = f"{name:34s}"
        for half in ("select", "confirm"):
            a = r.get(half, {}).get("accuracy")
            b = base.get(half, {}).get("accuracy")
            if a is None:
                line += f" {'--':>12s} {'--':>8s}"
            else:
                line += f" {a:12.4f} {a - b:+8.4f}"
        c = r.get("confirm", {})
        line += f" {c.get('recall_shock', float('nan')):9.4f} {c.get('roc_auc', float('nan')):9.4f}"
        print(line)
    print("=" * 104)
    print(f"majority baseline on CONFIRM: "
          f"{base.get('confirm', {}).get('majority', float('nan')):.4f}")
    print("Keep a candidate only if CONFIRM improves. A SELECT-only gain is "
          "selection noise.")

    OUT.write_text(json.dumps({"split_at": split_at, "results": results}, indent=2))
    log.info("saved -> %s", OUT)


def retune(df: pd.DataFrame, cols: list[str], split_at: str) -> dict:
    """
    Optuna over the SELECT folds only; the CONFIRM folds score the winner.

    The stored parameters were searched on the previous 16-quarter window with
    six folds. The window is now 20 quarters, so they are stale by construction.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sel_df = df[df["quarter"] <= split_at]

    def objective(trial: optuna.Trial) -> float:
        p = dict(objective="binary", verbosity=-1, random_state=42,
                 class_weight="balanced",
                 n_estimators=trial.suggest_int("n_estimators", 120, 500, step=20),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 8, 48),
                 max_depth=trial.suggest_int("max_depth", 2, 7),
                 min_child_samples=trial.suggest_int("min_child_samples", 10, 60),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                 reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 0.0, 1.0))
        preds = walk_forward(sel_df, cols, p)
        op = operating(preds)
        return float((op["label_shock"] == op["pred"]).mean()) if len(op) else 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=25, show_progress_bar=False)
    log.info("best SELECT accuracy %.4f with %s", study.best_value, study.best_params)

    p = dict(objective="binary", verbosity=-1, random_state=42,
             class_weight="balanced", **study.best_params)
    out = score(walk_forward(df, cols, p), split_at)
    out["best_params"] = study.best_params
    return out


if __name__ == "__main__":
    main()
