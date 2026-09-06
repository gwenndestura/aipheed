"""
scripts/build_lgu_reference.py
------------------------------
Build the canonical CALABARZON LGU reference — all **142** cities /
municipalities — at ``data/reference/calabarzon_lgus.csv``.

WHY THIS EXISTS
---------------
``lgu_census.parquet`` and ``lgu_poverty.parquet`` were previously built from a
hand-typed 137-LGU list in ``fix_primary_data.py``. Region IV-A has 142 LGUs
(Batangas 34, Cavite 23, Laguna 30, Quezon 41, Rizal 14), so five were missing
entirely — Taal and Talisay (Batangas), Mauban, Pagbilao and Quezon (Quezon).
Those five had no population, no poverty value, and therefore no municipal
forecast from the disaggregator, and they were invisible to the geo-verification
gazetteer that ``precision_pass.py`` / ``reanalyze_calabarzon.py`` build from the
census.

This script makes ``psgc_gazetteer.parquet`` (the PSGC-official 142-LGU list) the
single source of truth for *which* LGUs exist and *what they are called*, then
attaches population / land area to each one. Downstream code reads the CSV, so
the LGU roster is no longer duplicated across modules.

SOURCES
-------
* Roster, PSGC codes, official names, city/municipality class
      ``data/processed/psgc_gazetteer.parquet`` (PSA PSGC).
* Population (2020 CPH) and land area (km²)
      ``data/reference/cph2020_calabarzon_municipal.csv`` — the full PSA 2020
      Census of Population and Housing municipal table for all 142 LGUs. The
      build aborts unless its per-province sums reconcile exactly with PSA's
      published province totals (and the 16,195,042 regional total), so a
      dropped or mistyped row cannot reach the model.

``pop_source`` records provenance per row: ``cph2020`` = exact PSA 2020 CPH
total, ``estimate`` = rounded working figure from the superseded build. With
the CPH table in place every row is ``cph2020``; the ``estimate`` path only
fires if an LGU is missing from it and has to fall back to the old census.

OUTPUT
------
data/reference/calabarzon_lgus.csv — 142 rows:
    province_code, province_name, lgu_psgc, lgu_name, lgu_aliases, lgu_type,
    population_2020, land_area_km2, pop_source

Run:  python scripts/build_lgu_reference.py
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAZETTEER = ROOT / "data" / "processed" / "psgc_gazetteer.parquet"
CPH_TABLE = ROOT / "data" / "reference" / "cph2020_calabarzon_municipal.csv"
LEGACY_CENSUS = ROOT / "data" / "processed" / "lgu_census.parquet"
OUTPUT = ROOT / "data" / "reference" / "calabarzon_lgus.csv"

EXPECTED_TOTAL = 142
EXPECTED_BY_PROVINCE = {
    "Batangas": 34, "Cavite": 23, "Laguna": 30, "Quezon": 41, "Rizal": 14,
}

# PSA 2020 CPH official province totals — the checksum that proves the
# municipal table is complete and unaltered. Quezon is published WITHOUT
# Lucena City (highly urbanised, administratively independent), so Lucena is
# subtracted before comparing; the roster still carries it under Quezon.
PSA_PROVINCE_TOTALS = {
    "Batangas": 2_908_494, "Cavite": 4_344_829, "Laguna": 3_382_193,
    "Quezon": 1_950_459, "Rizal": 3_330_143,
}
LUCENA_POP = 278_924
PSA_REGION_TOTAL = 16_195_042

PROVINCE_CODE = {
    "Cavite": "PH040100000",
    "Laguna": "PH040200000",
    "Quezon": "PH040300000",
    "Rizal": "PH040400000",
    "Batangas": "PH040500000",
}

# ---------------------------------------------------------------------------
# Cityhood corrections. The bundled PSGC snapshot predates two conversions, so
# taking its lgu_type verbatim would demote both back to municipality:
#   Calaca, Batangas  — city since RA 11544 (ratified 2022)
#   Carmona, Cavite   — city since RA 11938 (ratified 2023)
# With these applied CALABARZON has 22 cities and 120 municipalities.
# (Candelaria, Quezon is NOT here: it is a municipality, and the old 137-LGU
# table had it mislabelled as a city.)
# ---------------------------------------------------------------------------
CITY_OVERRIDES: set[tuple[str, str]] = {
    ("Batangas", "Calaca"),
    ("Cavite", "Carmona"),
}

EXPECTED_CITIES = 22

# ---------------------------------------------------------------------------
# Alternate spellings that appear in news text / legacy tables, keyed by the
# PSGC-canonical name. Consumed by the geo-verification gazetteers so that a
# story saying "Mataas na Kahoy" still resolves to Mataasnakahoy.
# Deliberately NOT included: "GMA" for Gen. Mariano Alvarez — it collides with
# the GMA broadcast network and would flood the corpus with false positives.
# ---------------------------------------------------------------------------
ALIASES: dict[str, list[str]] = {
    "Mataasnakahoy":            ["Mataas na Kahoy", "Mataas Na Kahoy"],
    "Sto. Tomas":               ["Santo Tomas", "Sto Tomas"],
    "Gen. Mariano Alvarez":     ["General Mariano Alvarez"],
    "General Emilio Aguinaldo": ["Gen. Emilio Aguinaldo"],
    "General Trias":            ["Gen. Trias"],
    "General Luna":             ["Gen. Luna"],
    "General Nakar":            ["Gen. Nakar"],
    "Santa Cruz":               ["Sta. Cruz", "Sta Cruz"],
    "Santa Maria":              ["Sta. Maria", "Sta Maria"],
    "Santa Rosa":               ["Sta. Rosa", "Sta Rosa"],
    "Santa Teresita":           ["Sta. Teresita", "Sta Teresita"],
    "Los Baños":                ["Los Banos"],
    "Biñan":                    ["Binan"],
    "Dasmariñas":               ["Dasmarinas"],
    "Jala-Jala":                ["Jalajala", "Jala Jala"],
    "Padre Garcia":             ["Pdre. Garcia"],
    "Padre Burgos":             ["Pdre. Burgos"],
}


def norm(name: str) -> str:
    """Fold a place name to a comparison key: accent-free, punctuation-free,
    lowercase, with the Sto./Sta./Gen. abbreviations spelled out."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    for abbr, full in (("sto.", "santo"), ("sta.", "santa"), ("gen.", "general")):
        s = s.replace(abbr, full)
    for word, full in ((" sto ", " santo "), (" sta ", " santa "), (" gen ", " general ")):
        s = f" {s} ".replace(word, full).strip()
    return "".join(ch for ch in s if ch.isalnum())


def _load_pop_area(path: Path, label: str) -> dict[tuple[str, str], tuple[int, float]]:
    """Read a (province, LGU) -> (population, land area) table, CSV or parquet."""
    if not path.exists():
        return {}
    d = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    out = {
        (str(r.province_name), norm(r.lgu_name)):
            (int(r.population_2020), float(r.land_area_km2))
        for r in d.itertuples()
    }
    print(f"[info] {label}: {len(out)} LGUs from {path.name}")
    return out


def _verify_psa_totals(pop_area: dict[tuple[str, str], tuple[int, float]]) -> None:
    """
    Check the municipal table against PSA's published province totals.

    142 independently transcribed figures summing to five official totals is a
    strong guarantee that nothing was dropped, duplicated or mistyped, so a
    mismatch aborts the build rather than quietly shipping bad denominators.
    """
    sums: dict[str, int] = {}
    for (prov, _), (pop, _area) in pop_area.items():
        sums[prov] = sums.get(prov, 0) + pop

    bad = []
    for prov, official in PSA_PROVINCE_TOTALS.items():
        got = sums.get(prov, 0)
        if prov == "Quezon":
            # PSA publishes Quezon without Lucena; the roster keeps Lucena.
            got -= LUCENA_POP
        if got != official:
            bad.append(f"{prov}: {got:,} != {official:,} (diff {got - official:+,})")

    region = sum(sums.values())
    if region != PSA_REGION_TOTAL:
        bad.append(f"CALABARZON: {region:,} != {PSA_REGION_TOTAL:,}")

    if bad:
        sys.exit("[fail] CPH table does not reconcile with PSA totals:\n  "
                 + "\n  ".join(bad))
    print(f"[ok] PSA checksum: all 5 province totals match; "
          f"region = {region:,}")


def build() -> pd.DataFrame:
    if not GAZETTEER.exists():
        sys.exit(f"[fail] missing {GAZETTEER} — run psgc_gazetteer_fetcher.py first")

    gaz = pd.read_parquet(GAZETTEER)
    roster = (
        gaz[["province_name", "lgu_psgc", "lgu_name", "lgu_name_official", "lgu_type"]]
        .drop_duplicates()
        .sort_values(["province_name", "lgu_name"])
        .reset_index(drop=True)
    )

    if len(roster) != EXPECTED_TOTAL:
        sys.exit(f"[fail] gazetteer has {len(roster)} LGUs, expected {EXPECTED_TOTAL}")
    counts = roster.groupby("province_name").size().to_dict()
    if counts != EXPECTED_BY_PROVINCE:
        sys.exit(f"[fail] per-province counts {counts} != {EXPECTED_BY_PROVINCE}")

    # Population / land area from the full PSA 2020 CPH municipal table, keyed
    # on (province, folded name) so spelling drift does not drop a row. The
    # legacy census is only a fallback for LGUs the CPH table somehow misses.
    cph = _load_pop_area(CPH_TABLE, "CPH 2020 municipal table")
    legacy = _load_pop_area(LEGACY_CENSUS, "legacy census")

    if cph:
        _verify_psa_totals(cph)

    rows, sources = [], {"cph2020": 0, "estimate": 0}
    for r in roster.itertuples():
        prov, name = r.province_name, r.lgu_name
        key = (prov, norm(name))

        if key in cph:
            pop, area = cph[key]
            source = "cph2020"
        elif key in legacy:
            pop, area = legacy[key]
            # Rounded working figures in the legacy table are flagged as such.
            source = "estimate" if (pop % 1000 == 0 and float(area).is_integer()) else "cph2020"
            print(f"[warn] {prov}/{name} not in the CPH table — fell back to the "
                  f"legacy census ({source})")
        else:
            sys.exit(f"[fail] no population/area for {prov}/{name} — add it to "
                     f"{CPH_TABLE.name}")
        sources[source] += 1

        lgu_type = "city" if (prov, name) in CITY_OVERRIDES else r.lgu_type

        rows.append({
            "province_code": PROVINCE_CODE[prov],
            "province_name": prov,
            "lgu_psgc": str(r.lgu_psgc),
            "lgu_name": name,
            "lgu_aliases": "|".join(ALIASES.get(name, [])),
            "lgu_type": lgu_type,
            "population_2020": int(pop),
            "land_area_km2": round(float(area), 2),
            "pop_source": source,
        })

    df = pd.DataFrame(rows)

    assert len(df) == EXPECTED_TOTAL, len(df)
    assert df["lgu_psgc"].is_unique, "duplicate PSGC code"
    assert not df[["province_name", "lgu_name"]].duplicated().any(), "duplicate LGU"
    assert (df["population_2020"] > 0).all() and (df["land_area_km2"] > 0).all()
    n_cities = int((df["lgu_type"] == "city").sum())
    assert n_cities == EXPECTED_CITIES, f"{n_cities} cities, expected {EXPECTED_CITIES}"

    print(f"[info] population source: {sources}")
    print(f"[info] {n_cities} cities / {len(df) - n_cities} municipalities")
    print(f"[info] pop_source: {df['pop_source'].value_counts().to_dict()}")
    return df


def main() -> None:
    df = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"[ok] wrote {OUTPUT} — {len(df)} LGUs")
    print(df.groupby("province_name").size().to_string())


if __name__ == "__main__":
    main()
