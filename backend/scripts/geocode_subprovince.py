"""
scripts/geocode_subprovince.py
-------------------------------
Re-tag an existing corpus parquet with sub-province geography (city /
municipality / barangay) using the full Region IV-A PSGC gazetteer.

The historical pipeline geocoded only to province_code. This adds the
specific-locality columns the thesis requires, honouring any trusted prior
province tag already on the rows (never overwriting it, only refining below it).

Usage
-----
  venv\\Scripts\\python scripts\\geocode_subprovince.py \
      --in  data/processed/corpus_geocoded.parquet \
      --out data/processed/corpus_geocoded.parquet

Adds/overwrites columns: province_code, province_name, lgu_name, lgu_psgc,
barangay_name, barangay_psgc, match_level, geo_specificity. Prints a coverage
report by province, LGU, and match level.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.corpus.location_geocoder import geocode_location_batch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/corpus_geocoded.parquet")
    ap.add_argument("--out", dest="out", default="data/processed/corpus_geocoded.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    print(f"input: {len(df)} rows from {args.inp}")

    # Preserve the trusted prior province (from GDELT etc.) as the seed; the
    # batch geocoder reads it from the province_code column when present.
    tagged = geocode_location_batch(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tagged.to_parquet(out_path, index=False)
    print(f"saved: {out_path}")

    # ── Coverage report ────────────────────────────────────────────────────
    print("\n=== SUB-PROVINCE COVERAGE ===")
    lvl = tagged["match_level"].value_counts(dropna=False)
    for k, v in lvl.items():
        print(f"  {k:10s} {v:6d}  ({v/len(tagged)*100:4.1f}%)")

    has_lgu = tagged["lgu_name"].notna().sum()
    has_brgy = tagged["barangay_name"].notna().sum()
    print(f"\n  rows tagged to an LGU:      {has_lgu} ({has_lgu/len(tagged)*100:.1f}%)")
    print(f"  rows tagged to a barangay:  {has_brgy} ({has_brgy/len(tagged)*100:.1f}%)")

    prov = tagged[tagged["province_name"].notna()]
    print("\n  distinct LGUs covered per province (of gazetteer total):")
    gaz = pd.read_parquet("data/processed/psgc_gazetteer.parquet")
    totals = gaz.groupby("province_name")["lgu_psgc"].nunique()
    for p, sub in prov.groupby("province_name"):
        covered = sub["lgu_psgc"].nunique()
        print(f"    {p:10s} {covered:3d} / {int(totals.get(p, 0)):3d} LGUs   "
              f"({sub['lgu_name'].notna().sum()} rows, "
              f"{sub['barangay_name'].notna().sum()} barangay-level)")

    top = tagged["lgu_name"].value_counts().head(15)
    print("\n  top 15 LGUs by article count:")
    for name, cnt in top.items():
        print(f"    {name:20s} {cnt}")


if __name__ == "__main__":
    main()
