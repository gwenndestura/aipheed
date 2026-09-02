"""
scripts/build_fisheries_shock_label.py
--------------------------------------
A province-quarter food-availability target that actually exists.

Why this label. The composite stress label could not be supported: SWS hunger is
published for Balance Luzon with no province cut and only four quarters in the
model window are currently verifiable; PSA poverty and subsistence are triennial;
FNRI FIES is biennial and regional. The only series carrying both quarterly time
and provincial resolution was food_cpi_yoy -- which is an input to the label
itself, so using it was leakage.

PSA publishes fisheries production by province and quarter back to 1980
(OpenStat DB/2E/FS/0132E4GVFP1.px). A sharp fall in production against a
province's own seasonal norm is a food-AVAILABILITY shock -- one of the four FAO
food-security pillars, officially measured, at exactly the resolution the model
operates at. It is also what the news corpus predominantly records: 135 of 371
audited articles are fishery_loss and 30 are crop_production_loss.

Label
    baseline_p,q = mean production for province p in the same quarter-of-year
                   over the previous 3 years (shifted, so never self-referential)
    dev_pct      = 100 * (volume - baseline) / baseline
    shock        = 1 if dev_pct < SHOCK_THRESHOLD_PCT

Measured properties at -5%: 280 province-quarters, balance 0.500,
persistence 0.618, provinces differ in 52 of 56 quarters. For comparison the
previous label had persistence 0.809 -- copying last quarter answered it.

No leakage: no feature in the matrix derives from fisheries production.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fisheries_shock")

OPENSTAT = "https://openstat.psa.gov.ph/PXWeb/api/v1/en"
TABLE = "DB/2E/FS/0132E4GVFP1.px"
OUT = Path("data/processed/fisheries_shock_labels.parquet")

START_YEAR, END_YEAR = 2010, 2025
SHOCK_THRESHOLD_PCT = -5.0
BASELINE_YEARS = 3

PROVINCE_PSGC = {
    "Batangas": "PH040500000", "Cavite": "PH040100000", "Laguna": "PH040200000",
    "Quezon": "PH040300000", "Rizal": "PH040400000",
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aiPHeed/1.0 (+research; DLSU-D thesis)"})


def _json(r: requests.Response):
    return json.loads(r.content.decode("utf-8-sig"))


def fetch_production() -> pd.DataFrame:
    meta = _json(SESSION.get(f"{OPENSTAT}/{TABLE}", timeout=60))
    gv, sv, yv, qv = (v["code"] for v in meta["variables"])
    geo, sub, yr, qt = meta["variables"]

    prov = {c: t.strip(". ") for c, t in zip(geo["values"], geo["valueTexts"])
            if t.strip(". ") in PROVINCE_PSGC}
    # Subsector index 0 is the "FISHERIES" total across commercial, municipal
    # and aquaculture -- the whole provincial supply, which is what a
    # food-availability shock should be measured against.
    total_subsector = sub["values"][0]
    ytxt = {c: t for c, t in zip(yr["values"], yr["valueTexts"])}
    years = [c for c, t in zip(yr["values"], yr["valueTexts"])
             if START_YEAR <= int(t) <= END_YEAR]
    qtxt = {c: t for c, t in zip(qt["values"], qt["valueTexts"])}
    quarters = [c for c, t in zip(qt["values"], qt["valueTexts"]) if "Quarter" in t]

    body = {"query": [
        {"code": gv, "selection": {"filter": "item", "values": list(prov)}},
        {"code": sv, "selection": {"filter": "item", "values": [total_subsector]}},
        {"code": yv, "selection": {"filter": "item", "values": years}},
        {"code": qv, "selection": {"filter": "item", "values": quarters}},
    ], "response": {"format": "json"}}

    data = _json(SESSION.post(f"{OPENSTAT}/{TABLE}", json=body, timeout=180))
    rows = []
    for item in data["data"]:
        g, _s, y, q = item["key"]
        try:
            vol = float(item["values"][0])
        except (TypeError, ValueError):
            continue
        name = prov[g]
        rows.append({
            "province_code": PROVINCE_PSGC[name], "province_name": name,
            "year": int(ytxt[y]), "quarter_num": int(qtxt[q][-1]),
            "volume_mt": vol,
        })
    df = pd.DataFrame(rows)
    log.info("fetched %d province-quarters (%d-%d)", len(df), START_YEAR, END_YEAR)
    return df


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["province_code", "year", "quarter_num"]).reset_index(drop=True)

    # Seasonal baseline: same province, same quarter-of-year, previous N years.
    # shift(1) before rolling so the current quarter never enters its own baseline.
    df["baseline_mt"] = (
        df.groupby(["province_code", "quarter_num"])["volume_mt"]
          .transform(lambda s: s.shift(1).rolling(BASELINE_YEARS, min_periods=2).mean())
    )
    df["dev_pct"] = 100.0 * (df["volume_mt"] - df["baseline_mt"]) / df["baseline_mt"]

    before = len(df)
    df = df.dropna(subset=["dev_pct"]).copy()
    log.info("dropped %d rows without a baseline (first years)", before - len(df))

    df["quarter"] = df["year"].astype(str) + "-Q" + df["quarter_num"].astype(str)
    df["label_shock"] = (df["dev_pct"] < SHOCK_THRESHOLD_PCT).astype(int)
    df["dev_pct"] = df["dev_pct"].round(2)
    df["shock_threshold_pct"] = SHOCK_THRESHOLD_PCT
    df["source_url"] = f"https://openstat.psa.gov.ph/PXWeb/pxweb/en/{TABLE}"
    df["source_note"] = (
        "PSA Fisheries: Volume of Production by Geolocation, Subsector, Year and Quarter "
        "(all subsectors). Shock = production below the province's own 3-year seasonal "
        "baseline for the same quarter-of-year by more than "
        f"{abs(SHOCK_THRESHOLD_PCT):.0f}%."
    )
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return df


def main() -> None:
    df = build_labels(fetch_production())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    d = df.sort_values(["province_code", "year", "quarter_num"])
    d["lag"] = d.groupby("province_code")["label_shock"].shift(1)
    v = d.dropna(subset=["lag"])
    persistence = float((v["label_shock"] == v["lag"]).mean())
    differ = int((df.groupby("quarter")["label_shock"].nunique() > 1).sum())

    log.info("saved %d province-quarters -> %s", len(df), OUT)
    print(f"\nrows            : {len(df)}   ({df['quarter'].nunique()} quarters x "
          f"{df['province_name'].nunique()} provinces)")
    print(f"balance         : {df['label_shock'].mean():.3f}")
    print(f"persistence     : {persistence:.3f}   (previous label: 0.809)")
    print(f"province-varying: provinces differ in {differ} of {df['quarter'].nunique()} quarters")
    print("\nshocks per province:")
    print(df.groupby("province_name")["label_shock"].agg(["sum", "count", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
