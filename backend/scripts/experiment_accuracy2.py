"""
scripts/experiment_accuracy2.py
--------------------------------
Second round of accuracy candidates, all extending the one idea that worked.

BACKGROUND
----------
Round one (scripts/experiment_accuracy.py) tested four feature candidates and an
alternative threshold placement. Exactly one survived: `qoy`, which splits a
series' shock history by quarter-of-year instead of pooling every quarter into a
single rate. It gained +0.0133 accuracy, +0.1041 shock recall and +0.0316 AUC on
folds never used to choose it.

Everything here follows from that result: if quarter-of-year structure is what
the model was missing, then the other places the pipeline pools across quarters
are the places left to look.

CANDIDATES
----------
  teqoy    Target encoding by (commodity, quarter-of-year) instead of commodity
           alone. target_encode currently gives Ampalaya one shock rate across
           all four quarters -- the same pooling qoy just fixed in the feature
           set, still present in the encoding.
  qoystd   Volatility of the series within its quarter-of-year. shock_rate_qoy
           says how often this commodity fails in Q3; this says how consistent
           that is, so the model can discount an unreliable seasonal pattern.
  provqoy  Province-level quarter-of-year shock rate. Captures "Q3 is bad in
           Quezon" independently of the individual commodity.
  hgb      A fifth ensemble member, HistGradientBoosting. Different inductive
           bias from the existing LGBM/RF/ExtraTrees/LogReg, and it splits on
           NaN natively -- which matters more now that the honest-NaN change
           left real gaps in the matrix instead of zeros.

EXCLUDED ON PRINCIPLE, as in round one: raising the shock threshold, lowering
coverage below 90%, widening the cold-start exclusion. Each raises accuracy by
making the task easier.

FOLD DISCIPLINE
---------------
SELECT / CONFIRM exactly as round one. Note honestly that CONFIRM has now been
consulted across roughly a dozen configurations, so it is no longer a clean
out-of-sample surface; a gain that shows on both halves and follows from a
stated prior is worth more here than a large gain on one half.

USAGE
-----
    python scripts/experiment_accuracy2.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_accuracy import operating, score  # noqa: E402
from scripts.train_food_availability import (  # noqa: E402
    GOV, MATCHED, NLP, SEASONAL, SERIES, load,
)
from scripts.train_food_availability_v2 import (  # noqa: E402
    MIN_TRAIN, SMOOTHING, best_params, build_members, pick_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("exp2")

OUT = Path("data/processed/experiment_accuracy2.json")
MATURITY_FOLDS = 4
KEY = ["group", "commodity", "province_code"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Round-two candidates, all built from strictly prior quarters."""
    d = df.sort_values(KEY + ["year", "quarter_num"]).reset_index(drop=True)
    qoy = d.groupby(KEY + ["quarter_num"])

    # How consistent is this series in this quarter of the year? min_periods=2
    # because a standard deviation over one prior observation is meaningless.
    d["dev_qoy_std"] = qoy["dev_pct"].transform(
        lambda s: s.shift(1).expanding(min_periods=2).std())

    # Province-wide shock rate for this quarter-of-year, lagged. Built on the
    # province-quarter aggregate then merged back, the same shape as the
    # existing prov_shock_rate_lag1.
    pq = (d.groupby(["province_code", "quarter_num", "year"])["label_shock"]
            .mean().rename("pq_rate").reset_index()
            .sort_values(["province_code", "quarter_num", "year"]))
    pq["prov_shock_rate_qoy"] = (pq.groupby(["province_code", "quarter_num"])["pq_rate"]
                                   .transform(lambda s: s.shift(1)
                                              .expanding(min_periods=1).mean()))
    d = d.merge(pq[["province_code", "quarter_num", "year", "prov_shock_rate_qoy"]],
                on=["province_code", "quarter_num", "year"], how="left")
    return d


def target_encode(tr: pd.DataFrame, te: pd.DataFrame,
                  by: list[str]) -> tuple[pd.Series, pd.Series]:
    """Smoothed mean encoding over `by`, fit on training rows only."""
    prior = tr["label_shock"].mean()
    stats = tr.groupby(by)["label_shock"].agg(["sum", "count"])
    enc = (stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)
    if len(by) == 1:
        return (tr[by[0]].map(enc).fillna(prior), te[by[0]].map(enc).fillna(prior))
    tr_idx = pd.MultiIndex.from_frame(tr[by])
    te_idx = pd.MultiIndex.from_frame(te[by])
    return (pd.Series(enc.reindex(tr_idx).to_numpy(), index=tr.index).fillna(prior),
            pd.Series(enc.reindex(te_idx).to_numpy(), index=te.index).fillna(prior))


def members(params: dict, with_hgb: bool) -> list[tuple[str, object]]:
    m = build_members(params)
    if with_hgb:
        m.append(("hgb", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4,
            min_samples_leaf=20, l2_regularization=0.5,
            class_weight="balanced", random_state=42)))
    return m


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str],
                params: dict, with_hgb: bool) -> tuple[np.ndarray, np.ndarray]:
    Xtr, Xte = tr[cols], te[cols]
    ptr, pte = [], []
    for name, m in members(params, with_hgb):
        A = Xtr.fillna(-999) if name in ("rf", "et") else Xtr
        B = Xte.fillna(-999) if name in ("rf", "et") else Xte
        m.fit(A, tr["label_shock"])
        ptr.append(m.predict_proba(A)[:, 1])
        pte.append(m.predict_proba(B)[:, 1])
    return np.mean(ptr, axis=0), np.mean(pte, axis=0)


def walk_forward(df: pd.DataFrame, cols: list[str], params: dict,
                 te_by: list[str], with_hgb: bool) -> pd.DataFrame:
    seen: set[str] = set()
    cols = [c for c in cols if c in df.columns and not (c in seen or seen.add(c))]
    quarters = sorted(df["quarter"].unique())
    rows = []
    for i in range(MIN_TRAIN, len(quarters)):
        tr = df[df["quarter"].isin(quarters[:i])].copy()
        te = df[df["quarter"] == quarters[i]].copy()
        if len(te) < 10 or tr["label_shock"].nunique() < 2:
            continue
        tr["commodity_te"], te["commodity_te"] = target_encode(tr, te, te_by)
        use = cols + ["commodity_te"]

        prob = np.zeros(len(te))
        thr = np.zeros(len(te))
        for grp in te["group"].unique():
            mask = (te["group"] == grp).to_numpy()
            sub = tr[tr["group"] == grp]
            if len(sub) < 50 or sub["label_shock"].nunique() < 2:
                sub = tr
            p_tr, _ = fit_predict(sub, sub.head(1), use, params, with_hgb)
            _, p_te = fit_predict(sub, te[mask], use, params, with_hgb)
            prob[mask] = p_te
            thr[mask] = pick_threshold(sub["label_shock"], p_tr)

        rows.append(te.assign(prob=prob, pred=(prob >= thr).astype(int),
                              threshold=thr, fold=i))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    df = add_features(load())
    base = [c for c in SERIES + SEASONAL + GOV + NLP + MATCHED if c in df.columns]
    params = best_params()

    quarters = sorted(df["quarter"].unique())
    mature_q = quarters[MIN_TRAIN:][MATURITY_FOLDS:]
    split_at = mature_q[len(mature_q) // 2 - 1]
    log.info("SELECT <= %s | CONFIRM > %s", split_at, split_at)

    trials = {
        f"baseline ({len(base)} feat)": (base, ["commodity"], False),
        "+ teqoy":  (base, ["commodity", "quarter_num"], False),
        "+ qoystd": (base + ["dev_qoy_std"], ["commodity"], False),
        "+ provqoy": (base + ["prov_shock_rate_qoy"], ["commodity"], False),
        "+ hgb":    (base, ["commodity"], True),
    }

    results = {}
    for name, (cols, te_by, hgb) in trials.items():
        log.info("running %s", name)
        results[name] = score(walk_forward(df, cols, params, te_by, hgb), split_at)

    base_key = next(iter(trials))
    b = results[base_key]
    print("\n" + "=" * 104)
    print("ACCURACY EXPERIMENTS, ROUND 2 — operating set")
    print(f"SELECT <= {split_at}   |   CONFIRM > {split_at}")
    print("=" * 104)
    print(f"{'candidate':26s} {'sel acc':>9s} {'delta':>8s} {'con acc':>9s} "
          f"{'delta':>8s} {'con prec':>9s} {'con rec':>9s} {'con AUC':>9s}")
    print("-" * 104)
    for name, r in results.items():
        s, c = r.get("select", {}), r.get("confirm", {})
        ds = s.get("accuracy", float("nan")) - b["select"]["accuracy"]
        dc = c.get("accuracy", float("nan")) - b["confirm"]["accuracy"]
        print(f"{name:26s} {s.get('accuracy', float('nan')):9.4f} {ds:+8.4f} "
              f"{c.get('accuracy', float('nan')):9.4f} {dc:+8.4f} "
              f"{c.get('precision_shock', float('nan')):9.4f} "
              f"{c.get('recall_shock', float('nan')):9.4f} "
              f"{c.get('roc_auc', float('nan')):9.4f}")
    print("=" * 104)
    print("Keep only what improves on BOTH halves.")

    OUT.write_text(json.dumps({"split_at": split_at, "results": results}, indent=2))
    log.info("saved -> %s", OUT)


if __name__ == "__main__":
    main()
