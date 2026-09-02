"""
scripts/compare_fssi_rebuild.py
-------------------------------
Compare the rebuilt FSSI chain against the pre-audit baseline archived in
data/archive/pre_fssi_rebuild_20260901/.

Reports what changed in the five NLP training features
(FSSI, FSSI_lag1, FSSI_lag2, FSSI_accel, trigger_climate) and flags the
province-quarter cells whose FSSI is carried rather than measured.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OLD = Path("data/archive/pre_fssi_rebuild_20260901")
NEW = Path("data/processed")
NLP_FEATURES = ["FSSI", "FSSI_lag1", "FSSI_lag2", "FSSI_accel", "trigger_climate"]
KEYS = ["province_code", "quarter"]


def section(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main() -> None:
    section("ARTICLE BASE")
    of = pd.read_parquet(OLD / "fssi_quarterly.parquet")
    nf = pd.read_parquet(NEW / "fssi_quarterly.parquet")
    print(f"{'':22s} {'before':>12s} {'after':>12s}")
    print(f"{'articles':22s} {int(of['article_count'].sum()):>12d} {int(nf['article_count'].sum()):>12d}")
    print(f"{'province-quarter cells':22s} {len(of):>12d} {len(nf):>12d}")
    print(f"{'cells with >=5 articles':22s} "
          f"{int((of['article_count'] >= 5).sum()):>12d} {int((nf['article_count'] >= 5).sum()):>12d}")

    section("FSSI DISTRIBUTION (province-quarter level)")
    cmp = pd.DataFrame({
        "before": of["FSSI"].describe(),
        "after": nf["FSSI"].describe(),
    })
    print(cmp.round(4).to_string())

    section("COVERAGE OF THE 5x24 MODEL GRID")
    ofx = pd.read_parquet(OLD / "features_fused.parquet")
    nfx = pd.read_parquet(NEW / "features_fused.parquet")
    grid = nfx[KEYS].copy()
    measured = grid.merge(nf[KEYS + ["article_count"]], on=KEYS, how="left")
    empty = int(measured["article_count"].isna().sum())
    print(f"grid cells:                 {len(grid)}")
    print(f"cells with >=1 article:     {len(grid) - empty}")
    print(f"cells with NO article:      {empty}  <- FSSI here is forward/back-filled, not measured")
    print(f"cells >=5 articles:         {int((measured['article_count'] >= 5).sum())}")
    print("\narticles per province-quarter (0 = no article):")
    piv = measured.pivot_table(index="quarter", columns="province_code",
                               values="article_count", aggfunc="sum")
    print(piv.fillna(0).astype(int).to_string())

    section("TRAINING FEATURE SHIFT (features_fused, 120 rows)")
    m = ofx[KEYS + NLP_FEATURES].merge(nfx[KEYS + NLP_FEATURES], on=KEYS, suffixes=("_old", "_new"))
    rows = []
    for f in NLP_FEATURES:
        o, n = m[f"{f}_old"], m[f"{f}_new"]
        rows.append({
            "feature": f,
            "mean_before": o.mean(), "mean_after": n.mean(),
            "sd_before": o.std(), "sd_after": n.std(),
            "corr": o.corr(n),
            "cells_changed": int((~o.round(6).eq(n.round(6))).sum()),
        })
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    section("LARGEST PER-CELL FSSI CHANGES")
    m["delta"] = m["FSSI_new"] - m["FSSI_old"]
    top = m.reindex(m["delta"].abs().sort_values(ascending=False).index).head(12)
    print(top[KEYS + ["FSSI_old", "FSSI_new", "delta"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
