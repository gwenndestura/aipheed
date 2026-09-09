"""
scripts/experiment_climate.py
------------------------------
Agro-climatic extremes, tested at growing-season lags.

TWO CHANGES AT ONCE, BECAUSE THEY ONLY MAKE SENSE TOGETHER
----------------------------------------------------------
A. WHAT the model sees about weather.
   rainfall_anomaly_pct is a quarterly mean of monthly means, which is close to
   blind to crop damage: a quarter with three catastrophic days and one with
   uniformly damp weather have nearly the same mean. build_climate_extremes.py
   replaces that with the conventional extremes indices -- longest dry spell,
   largest 5-day rainfall, very-heavy-rain days, heat-stress days -- each as an
   anomaly against the province's own normal for the same quarter of the year.

B. WHEN the model sees it.
   Climate features are currently joined on the PRODUCTION quarter. But a
   harvest is set during the growing season, which for most of these commodities
   began one or two quarters earlier. Rain that falls after the crop is in is
   not what damaged it. So each index is tested at lag 0, 1 and 2 quarters and
   the data is allowed to say which window matters.

Neither is worth testing without the other: the right variable at the wrong time
is still the wrong feature.

FOLD DISCIPLINE
---------------
CONFIRM has already been read across roughly a dozen configurations in earlier
rounds, so it is no longer a clean surface. Selection here happens on SELECT
ONLY and this script prints SELECT alone. CONFIRM is read exactly once, later,
for whichever configuration SELECT picks -- run with --confirm to do that, and
only after the choice is made.

USAGE
-----
    python scripts/experiment_climate.py             # choose on SELECT
    python scripts/experiment_climate.py --confirm   # score the winner once
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_accuracy import operating, score, walk_forward  # noqa: E402
from scripts.train_food_availability import (  # noqa: E402
    GOV, MATCHED, NLP, SEASONAL, SERIES, load,
)
from scripts.train_food_availability_v2 import MIN_TRAIN, best_params  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("climate")

EXTREMES = Path("data/processed/province_climate_extremes.parquet")
OUT = Path("data/processed/experiment_climate.json")
MATURITY_FOLDS = 4

# The anomaly forms are used rather than the raw counts: a province with a
# structurally drier climate should not read as permanently shocked.
INDEX_COLS = ["cdd_max_anom", "rx5day_anom", "r20mm_anom",
              "heat_days_anom", "precip_total_anom"]


def shift_quarter(q: pd.Series, k: int) -> pd.Series:
    """Relabel a quarter label forward by k quarters."""
    idx = q.str[:4].astype(int) * 4 + q.str[-1].astype(int) - 1 + k
    return (idx // 4).astype(str) + "-Q" + (idx % 4 + 1).astype(str)


def attach(df: pd.DataFrame, lags: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """
    Merge the extremes at each requested lag.

    lag=k relabels the climate quarter forward by k, so a row for quarter Q
    carries the weather observed at Q-k. lag=0 is the current behaviour.
    """
    ext = pd.read_parquet(EXTREMES)
    keep = ["province_code", "quarter"] + INDEX_COLS
    ext = ext[[c for c in keep if c in ext.columns]]

    out, added = df, []
    for k in lags:
        e = ext.copy()
        e["quarter"] = shift_quarter(e["quarter"], k)
        suffix = f"_lag{k}"
        e = e.rename(columns={c: c + suffix for c in INDEX_COLS})
        added += [c + suffix for c in INDEX_COLS]
        out = out.merge(e, on=["province_code", "quarter"], how="left")
    return out, added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true",
                    help="read CONFIRM once, for the configuration SELECT chose")
    args = ap.parse_args()

    if not EXTREMES.exists():
        raise SystemExit(f"{EXTREMES} not found -- run "
                         "scripts/build_climate_extremes.py first")

    base_df = load()
    base = [c for c in SERIES + SEASONAL + GOV + NLP + MATCHED if c in base_df.columns]
    params = best_params()

    quarters = sorted(base_df["quarter"].unique())
    mature_q = quarters[MIN_TRAIN:][MATURITY_FOLDS:]
    split_at = mature_q[len(mature_q) // 2 - 1]
    log.info("SELECT <= %s | CONFIRM > %s", split_at, split_at)

    trials: dict[str, list[int] | None] = {
        "baseline (no extremes)": None,
        "extremes lag0 (production quarter)": [0],
        "extremes lag1 (previous quarter)": [1],
        "extremes lag2 (two quarters back)": [2],
        "extremes lag0+lag1": [0, 1],
        "extremes lag1+lag2 (growing season)": [1, 2],
        "extremes lag0+lag1+lag2": [0, 1, 2],
    }

    results = {}
    for name, lags in trials.items():
        if lags is None:
            df, cols = base_df, base
        else:
            df, added = attach(base_df, lags)
            cols = base + added
        log.info("running %s (%d features)", name, len(cols))
        results[name] = score(walk_forward(df, cols, params), split_at)

    b = results["baseline (no extremes)"]
    print("\n" + "=" * 92)
    print("AGRO-CLIMATIC EXTREMES AT GROWING-SEASON LAGS")
    print(f"Selection on SELECT (<= {split_at}) only. CONFIRM is NOT shown -- it has")
    print("been read too often in earlier rounds to serve as a clean surface here.")
    print("=" * 92)
    print(f"{'configuration':38s} {'sel acc':>9s} {'delta':>9s} {'sel rec':>9s} "
          f"{'sel F1':>9s} {'sel AUC':>9s}")
    print("-" * 92)
    for name, r in results.items():
        s = r.get("select", {})
        d = s.get("accuracy", float("nan")) - b["select"]["accuracy"]
        print(f"{name:38s} {s.get('accuracy', float('nan')):9.4f} {d:+9.4f} "
              f"{s.get('recall_shock', float('nan')):9.4f} "
              f"{s.get('f1_shock', float('nan')):9.4f} "
              f"{s.get('roc_auc', float('nan')):9.4f}")
    print("=" * 92)

    best = max((n for n in results if n != "baseline (no extremes)"),
               key=lambda n: results[n]["select"]["accuracy"])
    gain = results[best]["select"]["accuracy"] - b["select"]["accuracy"]
    print(f"best on SELECT: {best}  ({gain:+.4f})")

    if args.confirm:
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
