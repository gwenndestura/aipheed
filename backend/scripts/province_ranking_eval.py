"""
scripts/province_ranking_eval.py
--------------------------------
Can news-derived signal recover the official province food-poverty RANKING?

This is the province-ranking reframe of the research question. Instead of asking
"when will food stress rise in province X" -- which no available label supports,
because SWS supplies time without province and PSA supplies province without
time -- it asks:

    Which CALABARZON provinces are most food-insecure, and can the news signal
    recover that ordering without waiting three years for the next FIES round?

Target   : PSA subsistence incidence among families (food poverty), by province.
           OpenStat DB/1F/FY/0051F3DF030.px. Years 2018, 2021, 2023.
Predictor: news signal aggregated to province-year from the audited dataset and
           the FSSI chain.

Two honesty constraints are built in:

1. Only 2021 and 2023 are usable. The news corpus starts in 2020, so 2018 has no
   predictor. That leaves 5 provinces x 2 years = 10 observations. Every number
   below should be read against that n.

2. The target's own precision limits what "correct" means. PSA province CVs run
   24-59% and the confidence intervals overlap heavily, so for many province
   pairs the true ordering is not established by the source. Pairwise accuracy is
   therefore reported twice: over all pairs, and over only those pairs whose
   published 95% CIs do not overlap -- the pairs where a right answer exists.
"""
from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("province_ranking")

SUBSISTENCE = Path("data/processed/psa_subsistence.parquet")
DATASET = Path("data/processed/calabarzon_food_insecurity_dataset.parquet")
FSSI = Path("data/processed/fssi_quarterly.parquet")
CENSUS = Path("data/processed/lgu_census.parquet")

EVAL_YEARS = [2021, 2023]


def load_target() -> pd.DataFrame:
    df = pd.read_parquet(SUBSISTENCE)
    p = df[df["geographic_level"] == "province"].copy()
    return p[["psgc_code", "area_name", "year", "subsistence_pct",
              "subsistence_cv", "subsistence_ci_low", "subsistence_ci_high"]]


def load_predictors() -> pd.DataFrame:
    """News signal aggregated to province-year."""
    art = pd.read_parquet(DATASET)
    art = art[art["geographic_scope"].isin(["city_municipality", "province"])].copy()
    art["year"] = pd.to_datetime(art["publication_date"], errors="coerce").dt.year

    census = pd.read_parquet(CENSUS)
    prov = (census.groupby(["province_code", "province_name"])["population_2020"]
                  .sum().reset_index())
    name_to_code = dict(zip(prov["province_name"], prov["province_code"]))
    art["psgc_code"] = art["province"].map(name_to_code)

    counts = (art.groupby(["psgc_code", "year"])
                 .agg(article_count=("article_id", "count"),
                      high_tier=("relevance_tier", lambda s: (s == "HIGH").sum()))
                 .reset_index())
    counts["high_share"] = counts["high_tier"] / counts["article_count"]

    pop = dict(zip(prov["province_code"], prov["population_2020"]))
    counts["articles_per_100k"] = counts.apply(
        lambda r: 1e5 * r["article_count"] / pop.get(r["psgc_code"], np.nan), axis=1)

    fssi = pd.read_parquet(FSSI)
    fssi["year"] = fssi["quarter"].str[:4].astype(int)
    fy = (fssi.groupby(["province_code", "year"])
              .agg(fssi_mean=("FSSI", "mean"), fssi_max=("FSSI", "max"))
              .reset_index().rename(columns={"province_code": "psgc_code"}))

    return counts.merge(fy, on=["psgc_code", "year"], how="outer")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    tgt = load_target()
    pred = load_predictors()
    df = tgt.merge(pred, on=["psgc_code", "year"], how="left")
    df = df[df["year"].isin(EVAL_YEARS)].copy()

    log.info("evaluation set: %d province-years (%s)", len(df), EVAL_YEARS)

    signals = ["article_count", "articles_per_100k", "high_share", "fssi_mean", "fssi_max"]

    print("\n" + "=" * 78)
    print("TARGET — PSA subsistence incidence among families (%), with precision")
    print("=" * 78)
    for y in EVAL_YEARS:
        sub = df[df["year"] == y].sort_values("subsistence_pct", ascending=False)
        print(f"\n{y}:")
        for _, r in sub.iterrows():
            print(f"   {r['area_name']:10s} {r['subsistence_pct']:5.1f}%  "
                  f"CV={r['subsistence_cv']:5.1f}  CI [{r['subsistence_ci_low']:.1f}, "
                  f"{r['subsistence_ci_high']:.1f}]")

    # --- which pairwise orderings are actually established? ----------------
    print("\n" + "=" * 78)
    print("HOW MUCH OF THE TARGET RANKING IS STATISTICALLY ESTABLISHED?")
    print("=" * 78)
    established: dict[int, list[tuple]] = {}
    for y in EVAL_YEARS:
        sub = df[df["year"] == y]
        pairs, ok = [], []
        for (_, a), (_, b) in itertools.combinations(sub.iterrows(), 2):
            pairs.append((a, b))
            if a["subsistence_ci_low"] > b["subsistence_ci_high"] or \
               b["subsistence_ci_low"] > a["subsistence_ci_high"]:
                ok.append((a, b))
        established[y] = ok
        print(f"  {y}: {len(ok):2d} of {len(pairs)} province pairs have non-overlapping 95% CIs")

    # --- ranking performance ----------------------------------------------
    print("\n" + "=" * 78)
    print("NEWS SIGNAL vs TARGET RANKING")
    print("=" * 78)
    print(f"{'signal':20s} {'year':>6s} {'spearman':>10s} {'pairwise_all':>14s} {'pairwise_established':>22s}")
    print("-" * 78)
    for sig in signals:
        for y in EVAL_YEARS:
            sub = df[df["year"] == y].dropna(subset=[sig, "subsistence_pct"])
            if len(sub) < 3:
                print(f"{sig:20s} {y:>6d} {'n/a':>10s} {'n/a':>14s} {'n/a':>22s}")
                continue
            rho = spearman(sub[sig].to_numpy(), sub["subsistence_pct"].to_numpy())

            def pair_acc(pairs) -> str:
                hits = tot = 0
                for a, b in pairs:
                    va = sub[sub["psgc_code"] == a["psgc_code"]][sig]
                    vb = sub[sub["psgc_code"] == b["psgc_code"]][sig]
                    if va.empty or vb.empty:
                        continue
                    tot += 1
                    same = (float(va.iloc[0]) > float(vb.iloc[0])) == \
                           (a["subsistence_pct"] > b["subsistence_pct"])
                    hits += int(same)
                return f"{hits}/{tot}" if tot else "n/a"

            allp = list(itertools.combinations(df[df['year'] == y].to_dict('records'), 2))
            allp = [(pd.Series(a), pd.Series(b)) for a, b in allp]
            print(f"{sig:20s} {y:>6d} {rho:>10.3f} {pair_acc(allp):>14s} "
                  f"{pair_acc(established[y]):>22s}")

    print("\nn = 5 provinces per year. Spearman on 5 points is extremely noisy; "
          "\ntreat these as descriptive, not as evidence of predictive skill.")


if __name__ == "__main__":
    main()
