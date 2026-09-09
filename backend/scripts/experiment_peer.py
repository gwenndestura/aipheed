"""
scripts/experiment_peer.py
---------------------------
Peer-series shocks: does the same commodity failing elsewhere predict it failing here?

WHY THIS AXIS
-------------
The ablation splits cleanly by what a feature resolves to:

    series-level   seasonal only 0.761 AUC, series history 0.652
    province-level government 0.552, NLP 0.500, matched news 0.453,
                   and agro-climatic extremes, which failed outright

Province-level features hand identical values to all ~110 commodities sharing a
province-quarter, so they cannot say which one is shocked. Every spillover
feature currently in the model has that shape: prov_shock_rate_lag1 pools
commodities WITHIN a province.

Nothing pools one commodity ACROSS provinces. If ampalaya is failing in Cavite,
Laguna and Batangas at once, that is a regional crop event -- a pest, a variety
problem, a planting-window washout -- and it varies by commodity, which is the
property that predicts success here.

LEAKAGE CONTROL, WHICH IS THE WHOLE DIFFICULTY
----------------------------------------------
Two rules, both enforced below:

  1. LEAVE ONE OUT. The peer rate for province p excludes p itself. Including it
     would put the row's own label into its own feature.

  2. STRICTLY LAGGED. Only quarters before the one being scored are used. A
     same-quarter peer rate is tempting -- other provinces' production is
     published simultaneously -- but at prediction time no province has a label
     for the current quarter yet, so it is not available and is not tested.

CANDIDATES
----------
  peer1     same commodity, other provinces, previous quarter
  peer4     same commodity, other provinces, same quarter last year
  peerdev   same commodity, other provinces, mean deviation, previous quarter
  grouppeer same commodity GROUP, other provinces, previous quarter -- a
            deliberately coarser control. If this matches the commodity-level
            version, the signal is regional rather than crop-specific.

FOLD DISCIPLINE
---------------
Selection on SELECT only; CONFIRM is withheld and read once, later, for the
winner. Pass --confirm after the choice is final.

USAGE
-----
    python scripts/experiment_peer.py
    python scripts/experiment_peer.py --confirm
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

from scripts.experiment_accuracy import score, walk_forward  # noqa: E402
from scripts.train_food_availability import (  # noqa: E402
    GOV, MATCHED, NLP, SEASONAL, SERIES, load,
)
from scripts.train_food_availability_v2 import MIN_TRAIN, best_params  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("peer")

OUT = Path("data/processed/experiment_peer.json")
MATURITY_FOLDS = 4
KEY = ["group", "commodity", "province_code"]

CANDIDATES = {
    "peer1":     ["peer_shock_lag1"],
    "peer4":     ["peer_shock_lag4"],
    "peerdev":   ["peer_dev_lag1"],
    "grouppeer": ["group_peer_shock_lag1"],
    "peer1+peer4": ["peer_shock_lag1", "peer_shock_lag4"],
    "peer1+peerdev": ["peer_shock_lag1", "peer_dev_lag1"],
}


def _loo(df: pd.DataFrame, by: list[str], col: str) -> pd.Series:
    """
    Leave-one-out mean of `col` within `by`, for the CURRENT quarter.

    Never used as a feature directly -- it contains the row's own peers at the
    same time as the row's own label. It exists only to be shifted backwards.
    """
    g = df.groupby(by)[col]
    total, count = g.transform("sum"), g.transform("count")
    return pd.Series(np.where(count > 1, (total - df[col]) / (count - 1), np.nan),
                     index=df.index)


def add_peer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(KEY + ["year", "quarter_num"]).reset_index(drop=True)

    # Same commodity, other provinces, this quarter -- then pushed into the past.
    same_commodity = ["group", "commodity", "quarter"]
    d["_peer_now"] = _loo(d, same_commodity, "label_shock")
    d["_peer_dev_now"] = _loo(d, same_commodity, "dev_pct")

    # Same group, other provinces: the coarser control.
    d["_group_peer_now"] = _loo(d, ["group", "quarter"], "label_shock")

    g = d.groupby(KEY)
    d["peer_shock_lag1"] = g["_peer_now"].shift(1)
    d["peer_shock_lag4"] = g["_peer_now"].shift(4)
    d["peer_dev_lag1"] = g["_peer_dev_now"].shift(1)
    d["group_peer_shock_lag1"] = g["_group_peer_now"].shift(1)

    # The unshifted columns are same-quarter and must never reach the model.
    return d.drop(columns=["_peer_now", "_peer_dev_now", "_group_peer_now"])


def verify_no_leakage(d: pd.DataFrame) -> None:
    """A peer feature that correlates with the label like a copy of it is one."""
    print("\nleakage check -- correlation with the label it predicts:")
    for c in ["peer_shock_lag1", "peer_shock_lag4", "peer_dev_lag1",
              "group_peer_shock_lag1", "shock_lag4"]:
        if c not in d.columns:
            continue
        m = d[c].notna()
        r = d.loc[m, c].corr(d.loc[m, "label_shock"])
        print(f"   {c:24s} corr={r:+.4f}  coverage {100*m.mean():.0f}%")
    print("   (shock_lag4 is the incumbent seasonal feature, shown for scale;")
    print("    anything approaching 1.0 would indicate the label leaked in)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true",
                    help="read CONFIRM once, for the configuration SELECT chose")
    args = ap.parse_args()

    df = add_peer_features(load())
    base = [c for c in SERIES + SEASONAL + GOV + NLP + MATCHED if c in df.columns]
    params = best_params()
    verify_no_leakage(df)

    quarters = sorted(df["quarter"].unique())
    mature_q = quarters[MIN_TRAIN:][MATURITY_FOLDS:]
    split_at = mature_q[len(mature_q) // 2 - 1]
    log.info("SELECT <= %s | CONFIRM > %s", split_at, split_at)

    trials = {f"baseline ({len(base)} feat)": base}
    for name, feats in CANDIDATES.items():
        trials[f"+ {name}"] = base + feats

    results = {}
    for name, cols in trials.items():
        log.info("running %s (%d features)", name, len(cols))
        results[name] = score(walk_forward(df, cols, params), split_at)

    base_key = next(iter(trials))
    b = results[base_key]
    print("\n" + "=" * 92)
    print("PEER-SERIES SHOCKS — same commodity, other provinces, strictly lagged")
    print(f"Selection on SELECT (<= {split_at}) only. CONFIRM withheld.")
    print("=" * 92)
    print(f"{'candidate':26s} {'sel acc':>9s} {'delta':>9s} {'sel prec':>9s} "
          f"{'sel rec':>9s} {'sel F1':>9s} {'sel AUC':>9s}")
    print("-" * 92)
    for name, r in results.items():
        s = r.get("select", {})
        d = s.get("accuracy", float("nan")) - b["select"]["accuracy"]
        print(f"{name:26s} {s.get('accuracy', float('nan')):9.4f} {d:+9.4f} "
              f"{s.get('precision_shock', float('nan')):9.4f} "
              f"{s.get('recall_shock', float('nan')):9.4f} "
              f"{s.get('f1_shock', float('nan')):9.4f} "
              f"{s.get('roc_auc', float('nan')):9.4f}")
    print("=" * 92)

    best = max((n for n in results if n != base_key),
               key=lambda n: results[n]["select"]["accuracy"])
    gain = results[best]["select"]["accuracy"] - b["select"]["accuracy"]
    print(f"best on SELECT: {best} ({gain:+.4f})")
    if gain <= 0:
        print("No candidate beats the baseline on SELECT — nothing to confirm.")
    elif args.confirm:
        print("\nCONFIRM, read once for the SELECT winner:")
        for half in ("select", "confirm"):
            r = results[best].get(half, {})
            print(f"  {half:8s} n={r.get('n')} acc={r.get('accuracy', float('nan')):.4f} "
                  f"P={r.get('precision_shock', float('nan')):.4f} "
                  f"R={r.get('recall_shock', float('nan')):.4f} "
                  f"F1={r.get('f1_shock', float('nan')):.4f} "
                  f"AUC={r.get('roc_auc', float('nan')):.4f}")
    else:
        print("CONFIRM withheld. Re-run with --confirm once this choice is final.")

    OUT.write_text(json.dumps({"split_at": split_at, "best_on_select": best,
                               "results": results}, indent=2))
    log.info("saved -> %s", OUT)


if __name__ == "__main__":
    main()
