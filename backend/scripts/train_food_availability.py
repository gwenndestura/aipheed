"""
scripts/train_food_availability.py
----------------------------------
Train and evaluate on the pooled province-quarter-commodity food-availability
panel (2021-2026, entirely inside the five-year data window).

Training starts 2021-Q3 -- the first quarter with both short lags. The 2021 rows
enter with their seasonal lags (shock_lag4, dev_lag4) absent, because the panel
begins at 2021-Q1 and the baseline year is never retained; LightGBM splits on
NaN, so they contribute rather than being dropped.

Unit of observation is province x quarter x commodity, so the government feature
matrix repeats across the commodities sharing a province-quarter. The
commodity's own history is what varies within that cell, which is why the
series-level lags matter here.

Splits are by QUARTER, walk-forward, so no commodity from a test quarter appears
in training. Compared against naive persistence on the same folds.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("food_train")

PANEL = Path("data/processed/food_availability_panel.parquet")
FEATURES = Path("data/processed/features_fused.parquet")
OUT = Path("data/processed/food_availability_results.json")

MIN_TRAIN_QUARTERS = 8
RANDOM_SEED = 42

NLP = ["FSSI", "FSSI_lag1", "FSSI_lag2", "FSSI_accel", "trigger_climate"]
MATCHED = ["matched_articles", "matched_lag1", "total_articles"]
GOV = ["commodity_livestock", "commodity_leafy_veg", "commodity_fruit_veg",
       "rice_price_regular_lag1", "ofw_remit_yoy_pct_lag1", "unemployment_rate_lag1",
       "headline_cpi", "diesel_php_per_l_lag1",
       "rainfall_anomaly_pct_lag1", "rainfall_anomaly_pct_accel",
       "tc_count", "tc_severe_flag", "enso_numeric", "drought_alert"]
SERIES = ["shock_lag1", "shock_lag2", "dev_lag1", "group_code", "province_idx"]
# Seasonal history: the annual cycle plus where in the year we are.
# shock_rate_qoy / dev_qoy_mean are quarter-of-year specific rather than pooled
# across quarters -- see the note in load(). Added 2026-09-09 after measuring on
# held-out folds; the only one of four candidates to improve on both halves.
SEASONAL = ["shock_lag4", "dev_lag4", "dev_roll4", "series_vol", "quarter_num",
            "shock_rate_qoy", "dev_qoy_mean"]
DEEP = ["shock_lag8", "dev_lag8", "dev_trend4", "dev_roll_min4", "shock_rate_hist",
        "prov_shock_rate_lag1"]

PARAMS = dict(objective="binary", verbosity=-1, random_state=RANDOM_SEED,
              class_weight="balanced", n_estimators=300, learning_rate=0.05,
              num_leaves=16, max_depth=4, min_child_samples=20,
              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5)


DATASET = Path("data/processed/calabarzon_food_insecurity_dataset.parquet")
CENSUS = Path("data/processed/lgu_census.parquet")

# event_type -> the commodity group whose shocks that event describes. The
# government features are province-quarter, so they are identical across the ~45
# commodities sharing a cell and cannot distinguish which commodity is shocked.
# The news corpus can: a fish-kill article is about fisheries, a crop-damage
# article about crops. matched_articles gives the model that resolution.
EVENT_TO_GROUP = {
    "fishery_loss": "fisheries",
    "crop_production_loss": ("vegetables_rootcrops", "fruit_crops"),
}


def build_matched_news() -> pd.DataFrame:
    """Article counts per province-quarter, split by the commodity group they concern."""
    art = pd.read_parquet(DATASET)
    art = art[art["geographic_scope"].isin(["city_municipality", "province"])].copy()
    d = pd.to_datetime(art["publication_date"], errors="coerce")
    art["quarter"] = d.dt.year.astype("Int64").astype(str) + "-Q" + d.dt.quarter.astype("Int64").astype(str)

    census = pd.read_parquet(CENSUS)[["province_code", "province_name"]].drop_duplicates()
    art = art.merge(census.rename(columns={"province_name": "province"}),
                    on="province", how="left")

    rows = []
    for group in ("fisheries", "vegetables_rootcrops", "fruit_crops"):
        events = [e for e, g in EVENT_TO_GROUP.items()
                  if (g == group if isinstance(g, str) else group in g)]
        sub = art[art["event_type"].isin(events)]
        c = (sub.groupby(["province_code", "quarter"]).size()
                .reset_index(name="matched_articles"))
        c["group"] = group
        rows.append(c)
    out = pd.concat(rows, ignore_index=True)

    # total food-insecurity articles in the cell, regardless of commodity
    tot = (art.groupby(["province_code", "quarter"]).size()
              .reset_index(name="total_articles"))
    return out.merge(tot, on=["province_code", "quarter"], how="outer")


def load(feature_lag: int = 0) -> pd.DataFrame:
    """
    Build the modelling frame.

    feature_lag=0 (default) joins the government/climate feature matrix on the
    same quarter as the row being modelled -- the nowcast specification.

    feature_lag=1 joins it on the PREVIOUS quarter, which is what the forecast
    variant needs: a quarter can then be scored from information that already
    existed before it began, and coverage reaches one quarter past the feature
    matrix's own edge.
    """
    panel = pd.read_parquet(PANEL)
    feats = pd.read_parquet(FEATURES)

    key = ["group", "commodity", "province_code"]
    panel = panel.sort_values(key + ["year", "quarter_num"]).reset_index(drop=True)
    g = panel.groupby(key)
    panel["shock_lag1"] = g["label_shock"].shift(1)
    panel["shock_lag2"] = g["label_shock"].shift(2)
    panel["dev_lag1"] = g["dev_pct"].shift(1)
    # Annual cycle. Production is seasonal, so the same quarter one year back is
    # far more informative than the previous quarter: lag4 agrees with the label
    # 75.0% of the time against 66.5% for lag1. Omitting it was the single
    # largest gap in the feature set.
    panel["shock_lag4"] = g["label_shock"].shift(4)
    panel["dev_lag4"] = g["dev_pct"].shift(4)
    panel["dev_roll4"] = g["dev_pct"].transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
    panel["series_vol"] = g["dev_pct"].transform(lambda x: x.shift(1).expanding(min_periods=3).std())

    # Quarter-of-year specific history. shock_rate_hist below pools every
    # quarter into one number, so a series that reliably fails in Q3 and never
    # in Q1 gets a single blended rate that describes neither. Grouping by
    # quarter_num as well means a 2025-Q3 row averages 2024-Q3, 2023-Q3 and
    # 2022-Q3 and nothing else. shift(1) inside that group keeps it strictly
    # backward-looking.
    #
    # Measured on held-out folds never used for selection (2025-Q3..2026-Q2):
    # accuracy +0.0133, shock recall +0.1041, AUC +0.0316 over the 33-feature
    # baseline -- the only candidate of four to improve on both fold halves.
    # See scripts/experiment_accuracy.py.
    qoy = panel.groupby(key + ["quarter_num"])
    panel["shock_rate_qoy"] = qoy["label_shock"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean())
    panel["dev_qoy_mean"] = qoy["dev_pct"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean())

    # Deeper seasonal memory: two years back in the same quarter, and how often
    # this series has shocked in that quarter historically.
    panel["shock_lag8"] = g["label_shock"].shift(8)
    panel["dev_lag8"] = g["dev_pct"].shift(8)
    panel["dev_trend4"] = panel["dev_lag1"] - panel["dev_roll4"]
    panel["dev_roll_min4"] = g["dev_pct"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).min())
    panel["shock_rate_hist"] = g["label_shock"].transform(
        lambda x: x.shift(1).expanding(min_periods=2).mean())
    # Only the short lags are required. Demanding shock_lag4 here discarded
    # every 2021 row: the panel's first quarter is 2021-Q1 and the baseline year
    # is never retained, so nothing inside the window can supply a fourth lag
    # for that year. All of 2021 fell out of training despite its labels being
    # sound. LightGBM splits on NaN natively, so those rows now enter with the
    # seasonal lags absent rather than not entering at all. Training starts
    # 2021-Q3, the first quarter with both short lags available.
    panel = panel.dropna(subset=["shock_lag1", "shock_lag2"]).reset_index(drop=True)

    # Cross-commodity spillover, lagged. A typhoon or a dry spell hits many
    # commodities in a province at once, so how much of the province was shocked
    # LAST quarter is informative about this one. Built from t-1 only.
    prov_q = (panel.groupby(["province_code", "quarter"])["label_shock"]
                   .mean().rename("prov_shock_rate").reset_index())
    prov_q = prov_q.sort_values(["province_code", "quarter"])
    prov_q["prov_shock_rate_lag1"] = prov_q.groupby("province_code")["prov_shock_rate"].shift(1)
    panel = panel.merge(prov_q[["province_code", "quarter", "prov_shock_rate_lag1"]],
                        on=["province_code", "quarter"], how="left")

    panel["group_code"] = panel["group"].astype("category").cat.codes
    panel["province_idx"] = panel["province_code"].astype("category").cat.codes

    news = build_matched_news()
    panel = panel.merge(news, on=["province_code", "quarter", "group"], how="left")
    panel["matched_articles"] = panel["matched_articles"].fillna(0.0)
    panel["total_articles"] = panel["total_articles"].fillna(0.0)
    panel = panel.sort_values(key + ["year", "quarter_num"]).reset_index(drop=True)
    panel["matched_lag1"] = panel.groupby(key)["matched_articles"].shift(1).fillna(0.0)

    if feature_lag:
        # Attach features from `feature_lag` quarters earlier by relabelling the
        # feature matrix forward, so quarter Q carries the values observed at
        # Q-lag. Panel rows past the feature edge become scoreable.
        shifted = feats.copy()
        idx = shifted["quarter"].str[:4].astype(int) * 4 +             shifted["quarter"].str[-1].astype(int) - 1 + feature_lag
        shifted["quarter"] = (idx // 4).astype(str) + "-Q" + (idx % 4 + 1).astype(str)
        feats = shifted

    df = panel.merge(feats, on=["province_code", "quarter"], how="inner")
    df = df.sort_values(["quarter", "group", "commodity", "province_code"]).reset_index(drop=True)
    log.info("panel %d rows -> %d after feature merge | quarters %s..%s | balance %.3f",
             len(panel), len(df), df["quarter"].min(), df["quarter"].max(),
             df["label_shock"].mean())
    return df


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    The four metrics the CS department requires, plus ROC-AUC and the raw
    confusion counts.

    Reported twice, because on this target the two averagings say different
    things and quoting only one is misleading:

      *_shock     the positive class alone -- of the quarters flagged as a
                  production shock, how many were (precision), and of the real
                  shocks, how many were caught (recall). The positive rate is
                  about 0.29, so these are the operationally meaningful figures
                  for a triage tool: they describe the alerts themselves.

      *_weighted  support-weighted across both classes, which is what
                  app/ml/training/trainer.py already reports and what sklearn
                  gives by default for multiclass. Higher than the shock-only
                  figures, because the majority "no shock" class is easy.

    `f1` is retained as an alias of f1_weighted so existing references to it do
    not silently change meaning.
    """
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_shock": float(precision_score(y_true, y_pred, pos_label=1,
                                                 zero_division=0)),
        "recall_shock": float(recall_score(y_true, y_pred, pos_label=1,
                                           zero_division=0)),
        "f1_shock": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred,
                                                    average="weighted",
                                                    zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted",
                                              zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
    }
    out["f1"] = out["f1_weighted"]

    # Confusion counts, so any of the above can be recomputed or re-averaged
    # from the results file without re-running the model.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    (tn, fp), (fn, tp) = cm
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "support_shock": int(y_true.sum()), "support_total": int(len(y_true))})

    if y_prob is not None and y_true.nunique() > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
    return out


def seasonal_baseline_acc(d: pd.DataFrame) -> tuple[float, int]:
    """Accuracy of the same-quarter-last-year baseline, over the rows that have one.

    2021 rows carry no fourth lag: the panel starts at 2021-Q1 and the baseline
    year is never retained. Casting that NaN to 0 would score the baseline as
    having correctly called "no shock" on rows it cannot see, flattering it on
    exactly the quarters where it is blind. Those rows are excluded instead, and
    the number actually compared comes back with the figure so the coverage can
    be reported next to it.
    """
    m = d["shock_lag4"].notna()
    if not m.any():
        return float("nan"), 0
    return (float(accuracy_score(d.loc[m, "label_shock"],
                                 d.loc[m, "shock_lag4"].astype(int))), int(m.sum()))


def walk_forward(df: pd.DataFrame, cols: list[str]) -> dict:
    cols = [c for c in cols if c in df.columns]
    quarters = sorted(df["quarter"].unique())

    # Out-of-fold predictions are POOLED and scored once, rather than scored
    # per fold and averaged. Precision and recall on the shock class are
    # unstable per fold -- a quarter with few or no shocks gives a degenerate
    # value that then carries equal weight in the mean. Pooling also weights
    # each observation equally instead of each quarter, which is what the
    # headline evaluation in train_final.py does.
    y_true, y_pred, y_prob = [], [], []
    pers, pers4, folds, n4 = [], [], 0, 0

    for i in range(MIN_TRAIN_QUARTERS, len(quarters)):
        tr_q, te_q = quarters[:i], quarters[i]
        tr = df[df["quarter"].isin(tr_q)]
        te = df[df["quarter"] == te_q]
        if len(te) < 10 or tr["label_shock"].nunique() < 2:
            continue
        m = LGBMClassifier(**PARAMS).fit(tr[cols], tr["label_shock"])
        y_true.append(te["label_shock"])
        y_pred.append(pd.Series(m.predict(te[cols]), index=te.index))
        y_prob.append(pd.Series(m.predict_proba(te[cols])[:, 1], index=te.index))
        pers.append(accuracy_score(te["label_shock"], te["shock_lag1"].astype(int)))
        s4, s4n = seasonal_baseline_acc(te)
        if s4n:
            pers4.append(s4)
            n4 += s4n
        folds += 1

    if not folds:
        return {"folds": 0, "test_rows": 0, "n_features": len(cols)}

    yt = pd.concat(y_true)
    out = {"folds": folds, "test_rows": int(len(yt)), "n_features": len(cols)}
    out.update(classification_metrics(yt, pd.concat(y_pred), pd.concat(y_prob)))
    out.update({
        "persistence": float(np.mean(pers)) if pers else float("nan"),
        "seasonal_persistence": float(np.mean(pers4)) if pers4 else float("nan"),
        # Rows the seasonal baseline could actually be scored on. Lower than
        # test_rows because 2021 has no fourth lag inside the window.
        "seasonal_rows": n4,
    })
    return out


def main() -> None:
    df = load()
    sets = {
        "persistence only":        ["shock_lag1", "shock_lag2"],
        "series history only":     SERIES,
        "government only":         GOV,
        "NLP only":                NLP,
        "government + NLP":        GOV + NLP,
        "series + government":     SERIES + GOV,
        "series + government + NLP": SERIES + GOV + NLP,
        "matched news only":        MATCHED,
        "series + matched news":    SERIES + MATCHED,
        "series + gov + NLP + matched": SERIES + GOV + NLP + MATCHED,
        "seasonal only":                 SEASONAL,
        "series + seasonal":             SERIES + SEASONAL,
        "series + seasonal + matched":   SERIES + SEASONAL + MATCHED,
        "ALL (series+seasonal+gov+NLP+matched)": SERIES + SEASONAL + GOV + NLP + MATCHED,
        "series + seasonal + deep": SERIES + SEASONAL + DEEP,
        "EVERYTHING": SERIES + SEASONAL + DEEP + GOV + NLP + MATCHED,
    }
    res = {k: walk_forward(df, v) for k, v in sets.items()}

    span = f"{df['quarter'].min()}-{df['quarter'].max()}"
    print("\n" + "=" * 108)
    print(f"FOOD-AVAILABILITY SHOCK — pooled panel, walk-forward by quarter, {span}")
    print("precision / recall / F1 are for the SHOCK class; lag1 and lag4 are "
          "baseline accuracies")
    print("=" * 108)
    print(f"{'feature set':38s} {'nfeat':>6s} {'acc':>8s} {'prec':>8s} {'rec':>8s} "
          f"{'F1':>8s} {'AUC':>8s} {'lag4':>8s} {'skill4':>8s}")
    print("-" * 108)
    for k, r in res.items():
        print(f"{k:38s} {r['n_features']:>6d} {r['accuracy']:>8.4f} "
              f"{r['precision_shock']:>8.4f} {r['recall_shock']:>8.4f} "
              f"{r['f1_shock']:>8.4f} {r['roc_auc']:>8.4f} "
              f"{r['seasonal_persistence']:>8.4f} "
              f"{r['accuracy'] - r['seasonal_persistence']:>+8.4f}")
    print("=" * 108)
    any_r = res["government + NLP"]
    print(f"folds: {any_r['folds']}   test observations: {any_r['test_rows']}   "
          f"panel rows: {len(df)}")

    OUT.write_text(json.dumps(res, indent=2))
    log.info("saved -> %s", OUT)


if __name__ == "__main__":
    main()
