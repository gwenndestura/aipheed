"""
scripts/train_food_availability.py
----------------------------------
Train and evaluate on the pooled province-quarter-commodity food-availability
panel (2022-2026, entirely inside the five-year data window).

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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
SEASONAL = ["shock_lag4", "dev_lag4", "dev_roll4", "series_vol", "quarter_num"]
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


def load() -> pd.DataFrame:
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

    # Deeper seasonal memory: two years back in the same quarter, and how often
    # this series has shocked in that quarter historically.
    panel["shock_lag8"] = g["label_shock"].shift(8)
    panel["dev_lag8"] = g["dev_pct"].shift(8)
    panel["dev_trend4"] = panel["dev_lag1"] - panel["dev_roll4"]
    panel["dev_roll_min4"] = g["dev_pct"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).min())
    panel["shock_rate_hist"] = g["label_shock"].transform(
        lambda x: x.shift(1).expanding(min_periods=2).mean())
    panel = panel.dropna(subset=["shock_lag1", "shock_lag2", "shock_lag4"]).reset_index(drop=True)

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

    df = panel.merge(feats, on=["province_code", "quarter"], how="inner")
    df = df.sort_values(["quarter", "group", "commodity", "province_code"]).reset_index(drop=True)
    log.info("panel %d rows -> %d after feature merge | quarters %s..%s | balance %.3f",
             len(panel), len(df), df["quarter"].min(), df["quarter"].max(),
             df["label_shock"].mean())
    return df


def walk_forward(df: pd.DataFrame, cols: list[str]) -> dict:
    cols = [c for c in cols if c in df.columns]
    quarters = sorted(df["quarter"].unique())
    acc, f1s, aucs, pers, pers4, n = [], [], [], [], [], 0
    for i in range(MIN_TRAIN_QUARTERS, len(quarters)):
        tr_q, te_q = quarters[:i], quarters[i]
        tr = df[df["quarter"].isin(tr_q)]
        te = df[df["quarter"] == te_q]
        if len(te) < 10 or tr["label_shock"].nunique() < 2:
            continue
        m = LGBMClassifier(**PARAMS).fit(tr[cols], tr["label_shock"])
        pred = m.predict(te[cols])
        acc.append(accuracy_score(te["label_shock"], pred))
        f1s.append(f1_score(te["label_shock"], pred, average="weighted", zero_division=0))
        if te["label_shock"].nunique() > 1:
            aucs.append(roc_auc_score(te["label_shock"], m.predict_proba(te[cols])[:, 1]))
        pers.append(accuracy_score(te["label_shock"], te["shock_lag1"].astype(int)))
        pers4.append(accuracy_score(te["label_shock"], te["shock_lag4"].astype(int)))
        n += len(te)
    return {"folds": len(acc), "test_rows": n, "n_features": len(cols),
            "accuracy": float(np.mean(acc)) if acc else float("nan"),
            "f1": float(np.mean(f1s)) if f1s else float("nan"),
            "roc_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "persistence": float(np.mean(pers)) if pers else float("nan"),
            "seasonal_persistence": float(np.mean(pers4)) if pers4 else float("nan")}


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

    print("\n" + "=" * 92)
    print("FOOD-AVAILABILITY SHOCK — pooled panel, walk-forward by quarter, 2022-2025")
    print("=" * 92)
    print(f"{'feature set':38s} {'nfeat':>6s} {'acc':>8s} {'F1':>8s} {'AUC':>8s} "
          f"{'lag1':>8s} {'lag4':>8s} {'skill4':>8s}")
    print("-" * 92)
    for k, r in res.items():
        print(f"{k:38s} {r['n_features']:>6d} {r['accuracy']:>8.4f} {r['f1']:>8.4f} "
              f"{r['roc_auc']:>8.4f} {r['persistence']:>8.4f} {r['seasonal_persistence']:>8.4f} "
              f"{r['accuracy'] - r['seasonal_persistence']:>+8.4f}")
    print("=" * 92)
    any_r = res["government + NLP"]
    print(f"folds: {any_r['folds']}   test observations: {any_r['test_rows']}   "
          f"panel rows: {len(df)}")

    OUT.write_text(json.dumps(res, indent=2))
    log.info("saved -> %s", OUT)


if __name__ == "__main__":
    main()
