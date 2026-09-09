"""
scripts/experiment_price.py
----------------------------
Does a commodity's own price movement predict that commodity's production shock?

THE REASONING, WHICH CAME FROM A MEASUREMENT NOT A HUNCH
--------------------------------------------------------
Decomposing where shock variance lives:

    commodity x quarter   0.354      <- the biggest pocket
    series identity       0.207
    province x quarter    0.105

The commodity-quarter effect is the largest single source of structure, and it
is CONTEMPORANEOUS: provinces agree within a quarter (0.714 against 0.597 under
independence) but lagged peer features carry correlation ~0.05 and all six
failed. So the signal exists, and nothing in the model can currently see it,
because the only thing measuring it is production itself -- which is the label.

A price is a different observable of the same event. Scarcity shows up in price,
PSA publishes prices monthly by province and commodity, and for a NOWCAST a
same-quarter price is legitimate: it is released alongside or ahead of the
production figures, unlike another province's label, which does not exist yet at
prediction time.

STAGED ON PURPOSE
-----------------
The production and price tables use different taxonomies -- production tracks
"Banana", prices track "Banana Lakatan, ripe". A token-subset crosswalk matches
33 of 110 commodities, 38% of panel rows. That is not enough to move a headline
figure, but it is plenty to answer the only question that matters right now:
does the mechanism work at all? If price movement does not predict shocks where
the match is clean, hand-mapping the remaining 77 commodities would be wasted
effort.

So the comparison here is deliberately restricted to the matched subset, with
and without the price features, on identical rows.

FEATURES
--------
    price_qoy_dev    price this quarter vs this series' own normal for the same
                     quarter of the year -- the framing that made shock_rate_qoy
                     work, applied to price. SAME QUARTER: nowcast only.
    price_yoy        price vs the same quarter last year. SAME QUARTER.
    price_dev_lag1   the previous quarter's deviation. Forecast-safe.

USAGE
-----
    python scripts/experiment_price.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_accuracy import score, walk_forward  # noqa: E402
from scripts.train_food_availability import (  # noqa: E402
    GOV, MATCHED, NLP, SEASONAL, SERIES, load,
)
from scripts.train_food_availability_v2 import MIN_TRAIN, best_params  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("price")

OPENSTAT = "https://openstat.psa.gov.ph/PXWeb/api/v1/en"
CROSSWALK = Path("data/reference/_price_crosswalk.json")
CACHE = Path("data/processed/commodity_prices_matched.parquet")
OUT = Path("data/processed/experiment_price.json")

PROVINCES = {"Batangas": "PH040500000", "Cavite": "PH040100000",
             "Laguna": "PH040200000", "Quezon": "PH040300000",
             "Rizal": "PH040400000"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
START_YEAR, END_YEAR = 2021, 2026
MATURITY_FOLDS = 4
KEY = ["group", "commodity", "province_code"]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aiPHeed/1.0 (+research; DLSU-D thesis)"})


def _req(method: str, url: str, **kw):
    for attempt in range(5):
        time.sleep(1.5)
        r = SESSION.request(method, url, **kw)
        if r.status_code in (429, 503):
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return json.loads(r.content.decode("utf-8-sig"))
    raise RuntimeError(f"throttled: {url}")


def fetch_prices() -> pd.DataFrame:
    """Monthly province prices for every crosswalked commodity, cached."""
    if CACHE.exists():
        df = pd.read_parquet(CACHE)
        log.info("price cache: %d rows, %s .. %s", len(df),
                 df["quarter"].min(), df["quarter"].max())
        return df

    xwalk = json.loads(CROSSWALK.read_text())
    by_table: dict[str, dict[str, list[str]]] = {}
    for panel_commodity, entries in xwalk.items():
        for table, code, _raw in entries:
            by_table.setdefault(table, {}).setdefault(panel_commodity, []).append(code)

    rows = []
    for table, wanted in by_table.items():
        meta = _req("GET", f"{OPENSTAT}/DB/2M/NWSNEW/{table}", timeout=90)
        variables = meta["variables"]
        codes = [v["code"] for v in variables]
        gi = next(i for i, v in enumerate(variables) if "geo" in v["code"].lower())
        ci = next(i for i, v in enumerate(variables) if v["code"].lower() == "commodity")
        yi = next(i for i, v in enumerate(variables) if v["code"].lower() == "year")
        pi = next(i for i, v in enumerate(variables)
                  if v["code"].lower() in ("period", "quarter"))

        geo = variables[gi]
        prov = {c: t.strip(". ") for c, t in zip(geo["values"], geo["valueTexts"])
                if t.strip(". ") in PROVINCES}
        yv, pv = variables[yi], variables[pi]
        years = [c for c, t in zip(yv["values"], yv["valueTexts"])
                 if t.strip().isdigit() and START_YEAR <= int(t) <= END_YEAR]
        ytxt = dict(zip(yv["values"], yv["valueTexts"]))
        months = [c for c, t in zip(pv["values"], pv["valueTexts"]) if t.strip() in MONTHS]
        ptxt = dict(zip(pv["values"], pv["valueTexts"]))

        all_codes = sorted({c for v in wanted.values() for c in v})
        query = []
        for i, code in enumerate(codes):
            if i == gi:   sel = list(prov)
            elif i == ci: sel = all_codes
            elif i == yi: sel = years
            elif i == pi: sel = months
            else:         sel = [variables[i]["values"][0]]
            query.append({"code": code, "selection": {"filter": "item", "values": sel}})

        data = _req("POST", f"{OPENSTAT}/DB/2M/NWSNEW/{table}",
                    json={"query": query, "response": {"format": "json"}}, timeout=180)
        # one price code can serve several panel commodities (Banana Lakatan
        # feeds both "Banana" and "Banana Lakatan"), so fan the row out
        code_to_panel: dict[str, list[str]] = {}
        for pc, cl in wanted.items():
            for c in cl:
                code_to_panel.setdefault(c, []).append(pc)

        for item in data["data"]:
            k = item["key"]
            try:
                val = float(item["values"][0])
            except (TypeError, ValueError):
                continue
            if val <= 0:
                continue
            pname = prov[k[gi]]
            for panel_commodity in code_to_panel.get(k[ci], []):
                rows.append({
                    "commodity": panel_commodity,
                    "province_code": PROVINCES[pname],
                    "year": int(ytxt[k[yi]]),
                    "month": MONTHS.index(ptxt[k[pi]].strip()) + 1,
                    "price": val,
                })
        log.info("%s: %d monthly observations so far", table, len(rows))

    m = pd.DataFrame(rows)
    if m.empty:
        raise RuntimeError("no price observations returned")
    m["quarter_num"] = (m["month"] - 1) // 3 + 1
    q = (m.groupby(["commodity", "province_code", "year", "quarter_num"],
                   as_index=False)["price"].mean())
    q["quarter"] = q["year"].astype(str) + "-Q" + q["quarter_num"].astype(str)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    q.to_parquet(CACHE, index=False)
    log.info("wrote %d province-commodity-quarters -> %s", len(q), CACHE)
    return q


def add_price_features(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    d = df.merge(prices[["commodity", "province_code", "quarter", "price"]],
                 on=["commodity", "province_code", "quarter"], how="left")
    d = d.sort_values(KEY + ["year", "quarter_num"]).reset_index(drop=True)

    # Price against this series' own normal for the same quarter of the year --
    # the framing that worked for shock_rate_qoy. Expanding and shifted, so the
    # norm itself never contains the current quarter.
    qoy = d.groupby(KEY + ["quarter_num"])["price"]
    norm = qoy.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    d["price_qoy_dev"] = 100 * (d["price"] - norm) / norm

    g = d.groupby(KEY)["price"]
    d["price_yoy"] = 100 * (d["price"] / g.shift(4) - 1)
    d["price_dev_lag1"] = d.groupby(KEY)["price_qoy_dev"].shift(1)
    return d


def main() -> None:
    prices = fetch_prices()
    base_df = load()
    d = add_price_features(base_df, prices)

    matched = d[d["price"].notna()].copy()
    log.info("matched subset: %d of %d rows (%.0f%%), %d commodities",
             len(matched), len(d), 100 * len(matched) / len(d),
             matched["commodity"].nunique())

    base = [c for c in SERIES + SEASONAL + GOV + NLP + MATCHED if c in d.columns]
    params = best_params()
    quarters = sorted(matched["quarter"].unique())
    mature_q = quarters[MIN_TRAIN:][MATURITY_FOLDS:]
    if len(mature_q) < 2:
        raise SystemExit("not enough quarters in the matched subset")
    split_at = mature_q[len(mature_q) // 2 - 1]
    log.info("SELECT <= %s | CONFIRM > %s", split_at, split_at)

    trials = {
        "baseline, matched rows only": base,
        "+ price_qoy_dev (same quarter)": base + ["price_qoy_dev"],
        "+ price_yoy (same quarter)": base + ["price_yoy"],
        "+ both same-quarter": base + ["price_qoy_dev", "price_yoy"],
        "+ price_dev_lag1 (forecast-safe)": base + ["price_dev_lag1"],
    }
    results = {}
    for name, cols in trials.items():
        log.info("running %s", name)
        results[name] = score(walk_forward(matched, cols, params), split_at)

    b = results["baseline, matched rows only"]
    print("\n" + "=" * 96)
    print("COMMODITY PRICE AS A CONTEMPORANEOUS SHOCK SIGNAL")
    print(f"Matched subset only: {len(matched)} rows, "
          f"{matched['commodity'].nunique()} commodities. Identical rows across trials.")
    print(f"Selection on SELECT (<= {split_at}).")
    print("=" * 96)
    print(f"{'configuration':34s} {'sel acc':>9s} {'delta':>9s} {'sel rec':>9s} "
          f"{'sel F1':>9s} {'sel AUC':>9s}")
    print("-" * 96)
    for name, r in results.items():
        s = r.get("select", {})
        dd = s.get("accuracy", float("nan")) - b["select"]["accuracy"]
        print(f"{name:34s} {s.get('accuracy', float('nan')):9.4f} {dd:+9.4f} "
              f"{s.get('recall_shock', float('nan')):9.4f} "
              f"{s.get('f1_shock', float('nan')):9.4f} "
              f"{s.get('roc_auc', float('nan')):9.4f}")
    print("=" * 96)
    best = max((n for n in results if n != "baseline, matched rows only"),
               key=lambda n: results[n]["select"]["accuracy"])
    gain = results[best]["select"]["accuracy"] - b["select"]["accuracy"]
    print(f"best on SELECT: {best} ({gain:+.4f})")
    print("A gain here justifies hand-mapping the remaining commodities;")
    print("no gain means the mechanism does not work and the mapping is not worth it.")

    OUT.write_text(json.dumps({"split_at": split_at, "n_matched": len(matched),
                               "results": results}, indent=2))
    log.info("saved -> %s", OUT)


if __name__ == "__main__":
    main()
