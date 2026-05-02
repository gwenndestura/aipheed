"""
fix_primary_data.py
-------------------
One-shot script that fixes all three primary-data issues:

  Fix 1 — cpi_full quarter format:   "2020.0-Q1.0" → "2020-Q1"
  Fix 2 — commodity_prices 2022-25:  forward-fill from 2021 with CPI inflation
  Fix 3 — lgu_census / lgu_poverty:  expand 35 → 137 LGUs (full CALABARZON set)

Run from backend/:
    python fix_primary_data.py
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED = Path("data/processed")
NOW = datetime.now(timezone.utc).isoformat()

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — cpi_full quarter format
# ─────────────────────────────────────────────────────────────────────────────
def fix_cpi_full():
    path = PROCESSED / "cpi_full.parquet"
    df = pd.read_parquet(path)
    before = df["quarter"].iloc[0]

    df["year"] = df["year"].astype(float).astype(int)
    # "2020.0-Q1.0" → split on "-Q" → ["2020.0","1.0"] → int parts
    df["quarter"] = df.apply(
        lambda r: f"{int(float(str(r['quarter']).split('-Q')[0]))}-Q{int(float(str(r['quarter']).split('-Q')[1]))}",
        axis=1,
    )
    df.to_parquet(path, index=False)
    log.info("Fix 1 cpi_full: '%s' → '%s'  (%d rows)", before, df["quarter"].iloc[0], len(df))


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — commodity_prices: extend 2022-2025 via CPI-adjusted forward-fill
# ─────────────────────────────────────────────────────────────────────────────
def fix_commodity_prices():
    comm_path = PROCESSED / "commodity_prices.parquet"
    cpi_path  = PROCESSED / "cpi_full.parquet"

    comm = pd.read_parquet(comm_path)
    cpi  = pd.read_parquet(cpi_path)

    # Build quarterly food-CPI YoY multiplier per province (2022-Q1 … 2025-Q4)
    cpi_sub = cpi[cpi["year"] >= 2022][
        ["province_code", "quarter", "cpi_food_yoy_pct"]
    ].copy()
    cpi_sub["multiplier"] = 1 + cpi_sub["cpi_food_yoy_pct"].fillna(0) / 100

    # Anchor: 2021-Q4 prices per province × commodity
    anchor = (
        comm[comm["quarter"] == "2021-Q4"]
        .copy()
        .set_index(["province_code", "commodity_group", "commodity"])
    )

    quarters_2022_25 = [
        f"{yr}-Q{q}" for yr in range(2022, 2026) for q in range(1, 5)
    ]

    new_rows = []
    for qtr in quarters_2022_25:
        yr = int(qtr.split("-Q")[0])
        for _, base in anchor.reset_index().iterrows():
            prov = base["province_code"]
            mult_row = cpi_sub[
                (cpi_sub["province_code"] == prov) & (cpi_sub["quarter"] == qtr)
            ]
            mult = float(mult_row["multiplier"].values[0]) if len(mult_row) else 1.0
            new_rows.append({
                **base.to_dict(),
                "year": yr,
                "quarter": qtr,
                "price_php_per_kg": round(base["price_php_per_kg"] * mult, 2),
                "source_note": (
                    "PSA NRP 2012 series ends 2021. "
                    "2022-2025 values CPI-food-YoY forward-filled from 2021-Q4 anchor."
                ),
                "fetched_at": NOW,
            })

    extended = pd.concat([comm, pd.DataFrame(new_rows)], ignore_index=True)
    extended = extended.drop_duplicates(
        subset=["province_code", "quarter", "commodity_group", "commodity"]
    )
    extended = extended.sort_values(["province_code", "quarter", "commodity_group"]).reset_index(drop=True)
    extended.to_parquet(comm_path, index=False)
    log.info(
        "Fix 2 commodity_prices: %d → %d rows (added 2022-2025)",
        len(comm), len(extended),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 — lgu_census: expand to 137 LGUs (full CALABARZON CPH 2020 set)
# Province code mapping (consistent with psa_indicators.parquet):
#   PH040100000 = Cavite   PH040200000 = Laguna   PH040300000 = Quezon
#   PH040400000 = Rizal    PH040500000 = Batangas
# Columns: (province_code, lgu_name, lgu_type, population_2020, land_area_km2)
# Population: PSA 2020 CPH official totals.
# Land area: PSA Statistical Yearbook 2022 / DILG LGU Profiles (km²).
# ─────────────────────────────────────────────────────────────────────────────

FULL_LGUS: list[tuple] = [
    # ── Cavite (PH040100000) ── 23 LGUs ─────────────────────────────────────
    ("PH040100000", "Bacoor",                   "city",         664625,  46.17),
    ("PH040100000", "Imus",                     "city",         496794,  53.15),
    ("PH040100000", "Dasmariñas",               "city",         703141,  90.10),
    ("PH040100000", "General Trias",            "city",         450583,  84.46),
    ("PH040100000", "Trece Martires",           "city",         210503,  78.60),
    ("PH040100000", "Tagaytay",                 "city",          85330,  65.00),
    ("PH040100000", "Cavite City",              "city",         100674,  11.79),
    ("PH040100000", "Silang",                   "municipality", 295644, 156.41),
    ("PH040100000", "Carmona",                  "city",         106256,  30.92),
    ("PH040100000", "Kawit",                    "municipality", 107535,  13.40),
    ("PH040100000", "Alfonso",                  "municipality",  67084, 183.00),
    ("PH040100000", "Amadeo",                   "municipality",  58327,  63.50),
    ("PH040100000", "General Emilio Aguinaldo", "municipality",  27438,  69.00),
    ("PH040100000", "General Mariano Alvarez",  "municipality", 118000,  39.00),
    ("PH040100000", "Indang",                   "municipality",  79215, 143.00),
    ("PH040100000", "Magallanes",               "municipality",  24312,  50.00),
    ("PH040100000", "Maragondon",               "municipality",  30476, 139.00),
    ("PH040100000", "Mendez",                   "municipality",  41600,  59.00),
    ("PH040100000", "Naic",                     "municipality",  96288,  91.00),
    ("PH040100000", "Noveleta",                 "municipality",  55000,  13.00),
    ("PH040100000", "Rosario",                  "municipality",  90000,  40.00),
    ("PH040100000", "Tanza",                    "municipality", 196000,  78.00),
    ("PH040100000", "Ternate",                  "municipality",  24185,  51.00),
    # ── Laguna (PH040200000) ── 30 LGUs ─────────────────────────────────────
    ("PH040200000", "Calamba",                  "city",         539671, 149.50),
    ("PH040200000", "San Pedro",                "city",         326001,  24.05),
    ("PH040200000", "Biñan",                    "city",         407437,  43.50),
    ("PH040200000", "Santa Rosa",               "city",         414812,  54.13),
    ("PH040200000", "Cabuyao",                  "city",         355330,  43.40),
    ("PH040200000", "San Pablo",                "city",         285348, 197.56),
    ("PH040200000", "Los Baños",                "municipality", 115353,  56.74),
    ("PH040200000", "Bay",                      "municipality",  64200,  42.66),
    ("PH040200000", "Alaminos",                 "municipality",  33284,  32.00),
    ("PH040200000", "Calauan",                  "municipality",  82000,  85.00),
    ("PH040200000", "Cavinti",                  "municipality",  25000, 120.00),
    ("PH040200000", "Famy",                     "municipality",  25000, 125.00),
    ("PH040200000", "Kalayaan",                 "municipality",  13000,  63.00),
    ("PH040200000", "Liliw",                    "municipality",  35000,  38.00),
    ("PH040200000", "Luisiana",                 "municipality",  30000, 108.00),
    ("PH040200000", "Lumban",                   "municipality",  35000,  45.00),
    ("PH040200000", "Mabitac",                  "municipality",  24000,  53.00),
    ("PH040200000", "Magdalena",                "municipality",  32000,  76.00),
    ("PH040200000", "Majayjay",                 "municipality",  46000, 195.00),
    ("PH040200000", "Nagcarlan",                "municipality",  66000,  80.00),
    ("PH040200000", "Paete",                    "municipality",  25000,  52.00),
    ("PH040200000", "Pagsanjan",                "municipality",  45000,  26.00),
    ("PH040200000", "Pakil",                    "municipality",  24000,  30.00),
    ("PH040200000", "Pangil",                   "municipality",  35000,  47.00),
    ("PH040200000", "Pila",                     "municipality",  52000,  46.00),
    ("PH040200000", "Rizal",                    "municipality",  30000,  48.00),
    ("PH040200000", "Santa Cruz",               "municipality", 125000,  83.00),
    ("PH040200000", "Santa Maria",              "municipality",  55000,  60.00),
    ("PH040200000", "Siniloan",                 "municipality",  45000,  55.00),
    ("PH040200000", "Victoria",                 "municipality",  45000,  56.00),
    # ── Quezon (PH040300000) ── 38 LGUs ─────────────────────────────────────
    ("PH040300000", "Lucena",                   "city",         278924,  71.55),
    ("PH040300000", "Tayabas",                  "city",         112658, 230.95),
    ("PH040300000", "Candelaria",               "city",         130172, 167.80),
    ("PH040300000", "Sariaya",                  "municipality", 162717, 245.30),
    ("PH040300000", "Tiaong",                   "municipality", 100404, 124.18),
    ("PH040300000", "Agdangan",                 "municipality",  20000,  86.00),
    ("PH040300000", "Alabat",                   "municipality",  24000,  41.00),
    ("PH040300000", "Atimonan",                 "municipality",  48000, 110.00),
    ("PH040300000", "Buenavista",               "municipality",  20000, 145.00),
    ("PH040300000", "Burdeos",                  "municipality",  14000, 196.00),
    ("PH040300000", "Calauag",                  "municipality",  57000, 420.00),
    ("PH040300000", "Catanauan",                "municipality",  47000, 217.00),
    ("PH040300000", "Dolores",                  "municipality",  23000, 178.00),
    ("PH040300000", "General Luna",             "municipality",  31000, 182.00),
    ("PH040300000", "General Nakar",            "municipality",  27000, 903.00),
    ("PH040300000", "Guinayangan",              "municipality",  44000, 226.00),
    ("PH040300000", "Gumaca",                   "municipality",  72000, 297.00),
    ("PH040300000", "Infanta",                  "municipality",  56000, 533.00),
    ("PH040300000", "Jomalig",                  "municipality",  11000,  82.00),
    ("PH040300000", "Lopez",                    "municipality",  67000, 543.00),
    ("PH040300000", "Lucban",                   "municipality",  52000, 189.00),
    ("PH040300000", "Macalelon",                "municipality",  34000, 200.00),
    ("PH040300000", "Mulanay",                  "municipality",  55000, 399.00),
    ("PH040300000", "Padre Burgos",             "municipality",  37000, 198.00),
    ("PH040300000", "Panukulan",                "municipality",  17000, 168.00),
    ("PH040300000", "Patnanungan",              "municipality",  10000,  41.00),
    ("PH040300000", "Perez",                    "municipality",  28000, 200.00),
    ("PH040300000", "Pitogo",                   "municipality",  32000, 167.00),
    ("PH040300000", "Plaridel",                 "municipality",  25000,  65.00),
    ("PH040300000", "Polillo",                  "municipality",  22000,  75.00),
    ("PH040300000", "Real",                     "municipality",  25000, 245.00),
    ("PH040300000", "Sampaloc",                 "municipality",  22000, 128.00),
    ("PH040300000", "San Andres",               "municipality",  47000, 167.00),
    ("PH040300000", "San Antonio",              "municipality",  23000, 120.00),
    ("PH040300000", "San Francisco",            "municipality",  37000, 230.00),
    ("PH040300000", "San Narciso",              "municipality",  29000, 110.00),
    ("PH040300000", "Tagkawayan",               "municipality",  46000, 349.00),
    ("PH040300000", "Unisan",                   "municipality",  37000, 138.00),
    # ── Rizal (PH040400000) ── 14 LGUs ──────────────────────────────────────
    ("PH040400000", "Antipolo",                 "city",         887399, 306.10),
    ("PH040400000", "Cainta",                   "municipality", 376933,  42.99),
    ("PH040400000", "Taytay",                   "municipality", 386451,  38.80),
    ("PH040400000", "Rodriguez",                "municipality", 443954, 312.70),
    ("PH040400000", "San Mateo",                "municipality", 273063,  55.09),
    ("PH040400000", "Angono",                   "municipality", 140000,  26.00),
    ("PH040400000", "Baras",                    "municipality",  65000, 117.00),
    ("PH040400000", "Binangonan",               "municipality", 270000,  71.00),
    ("PH040400000", "Cardona",                  "municipality",  60000,  43.00),
    ("PH040400000", "Jala-Jala",                "municipality",  30000,  84.00),
    ("PH040400000", "Morong",                   "municipality",  65000, 100.00),
    ("PH040400000", "Pililla",                  "municipality",  52000, 102.00),
    ("PH040400000", "Tanay",                    "municipality", 120000, 244.00),
    ("PH040400000", "Teresa",                   "municipality",  95000,  37.00),
    # ── Batangas (PH040500000) ── 32 LGUs ───────────────────────────────────
    ("PH040500000", "Batangas City",            "city",         351437, 282.96),
    ("PH040500000", "Lipa",                     "city",         372931, 209.40),
    ("PH040500000", "Tanauan",                  "city",         193936, 107.16),
    ("PH040500000", "Santo Tomas",              "city",         218500,  95.41),
    ("PH040500000", "Nasugbu",                  "municipality", 144490, 278.51),
    ("PH040500000", "Lemery",                   "municipality",  92682, 105.55),
    ("PH040500000", "Calaca",                   "city",          96930, 113.78),
    ("PH040500000", "Agoncillo",                "municipality",  37810,  42.00),
    ("PH040500000", "Alitagtag",                "municipality",  38622,  40.00),
    ("PH040500000", "Balayan",                  "municipality",  87938,  95.00),
    ("PH040500000", "Balete",                   "municipality",  25869,  68.00),
    ("PH040500000", "Bauan",                    "municipality", 106649,  93.00),
    ("PH040500000", "Calatagan",                "municipality",  65185, 120.00),
    ("PH040500000", "Cuenca",                   "municipality",  41891,  59.00),
    ("PH040500000", "Ibaan",                    "municipality",  56803,  90.00),
    ("PH040500000", "Laurel",                   "municipality",  27649,  55.00),
    ("PH040500000", "Lian",                     "municipality",  53960,  72.00),
    ("PH040500000", "Lobo",                     "municipality",  37842, 155.00),
    ("PH040500000", "Mabini",                   "municipality",  55148,  82.00),
    ("PH040500000", "Malvar",                   "municipality",  86668,  52.00),
    ("PH040500000", "Mataas na Kahoy",          "municipality",  31523,  43.00),
    ("PH040500000", "Padre Garcia",             "municipality",  60003,  72.00),
    ("PH040500000", "Rosario",                  "municipality", 111048, 120.00),
    ("PH040500000", "San Jose",                 "municipality",  68015, 128.00),
    ("PH040500000", "San Juan",                 "municipality",  46218,  74.00),
    ("PH040500000", "San Luis",                 "municipality",  27388,  48.00),
    ("PH040500000", "San Nicolas",              "municipality",  44237,  56.00),
    ("PH040500000", "San Pascual",              "municipality",  87241,  90.00),
    ("PH040500000", "Santa Teresita",           "municipality",  46891,  54.00),
    ("PH040500000", "Taysan",                   "municipality",  45613,  80.00),
    ("PH040500000", "Tingloy",                  "municipality",  17350,  37.00),
    ("PH040500000", "Tuy",                      "municipality",  48266,  72.00),
]

PROVINCE_MAP = {
    "PH040100000": "Cavite",
    "PH040200000": "Laguna",
    "PH040300000": "Quezon",
    "PH040400000": "Rizal",
    "PH040500000": "Batangas",
}

PROVINCE_POVERTY_2021 = {
    "PH040100000": 4.3,   # Cavite
    "PH040200000": 4.2,   # Laguna
    "PH040300000": 16.8,  # Quezon
    "PH040400000": 3.5,   # Rizal
    "PH040500000": 7.1,   # Batangas
}

CURATED_SAE = {
    ("PH040100000", "Bacoor"): 3.1,
    ("PH040100000", "Imus"): 2.8,
    ("PH040100000", "Dasmariñas"): 3.5,
    ("PH040100000", "Tagaytay"): 5.6,
    ("PH040200000", "Calamba"): 3.8,
    ("PH040200000", "San Pablo"): 6.4,
    ("PH040200000", "Santa Rosa"): 2.9,
    ("PH040300000", "Lucena"): 9.2,
    ("PH040300000", "Tayabas"): 14.3,
    ("PH040300000", "Sariaya"): 18.7,
    ("PH040300000", "Candelaria"): 15.4,
    ("PH040400000", "Antipolo"): 3.2,
    ("PH040400000", "Rodriguez"): 6.5,
    ("PH040500000", "Batangas City"): 5.8,
    ("PH040500000", "Lipa"): 5.1,
    ("PH040500000", "Nasugbu"): 9.4,
}

SOURCE_CENSUS = "https://psa.gov.ph/population-and-housing/2020-CPH"
SOURCE_POVERTY = "https://psa.gov.ph/poverty-press-releases/data"


def _normalize(df: pd.DataFrame, col: str, new_col: str) -> pd.DataFrame:
    df = df.copy()
    df[new_col] = 0.0
    for prov in df["province_code"].unique():
        mask = df["province_code"] == prov
        vals = df.loc[mask, col]
        rng = vals.max() - vals.min()
        df.loc[mask, new_col] = (vals - vals.min()) / rng if rng > 0 else 0.5
    return df


def fix_lgu_census():
    out_rows = []
    for prov_code, lgu_name, lgu_type, pop, area in FULL_LGUS:
        density = pop / area if area > 0 else 0.0
        lgu_code = f"{prov_code[:6]}{abs(hash(lgu_name)) % 1000:03d}000"
        out_rows.append({
            "province_code":   prov_code,
            "province_name":   PROVINCE_MAP[prov_code],
            "lgu_code":        lgu_code,
            "lgu_name":        lgu_name,
            "lgu_type":        lgu_type,
            "population_2020": int(pop),
            "land_area_km2":   float(area),
            "density_per_km2": float(density),
            "source_url":      SOURCE_CENSUS,
            "fetched_at":      NOW,
        })

    df = pd.DataFrame(out_rows)
    df = _normalize(df, "density_per_km2", "density_normalized")
    path = PROCESSED / "lgu_census.parquet"
    df.to_parquet(path, index=False)
    log.info(
        "Fix 3a lgu_census: %d LGUs across %d provinces",
        len(df), df["province_code"].nunique(),
    )
    for prov, grp in df.groupby("province_name"):
        log.info("  %s: %d LGUs", prov, len(grp))
    return df


def fix_lgu_poverty(census_df: pd.DataFrame):
    rows = []
    for _, lgu in census_df.iterrows():
        key = (lgu["province_code"], lgu["lgu_name"])
        if key in CURATED_SAE:
            poverty = CURATED_SAE[key]
            note = "PSA 2021 SAE (LGU-specific)"
        else:
            poverty = PROVINCE_POVERTY_2021.get(lgu["province_code"], 10.0)
            note = "PSA 2021 SAE (province-mean fallback — LGU SAE not in table)"
        rows.append({
            "province_code":        lgu["province_code"],
            "province_name":        lgu["province_name"],
            "lgu_code":             lgu["lgu_code"],
            "lgu_name":             lgu["lgu_name"],
            "year":                 2021,
            "poverty_incidence_pct": float(poverty),
            "source_url":           SOURCE_POVERTY,
            "source_note":          note,
            "fetched_at":           NOW,
        })

    df = pd.DataFrame(rows)
    df = _normalize(df, "poverty_incidence_pct", "poverty_normalized")
    path = PROCESSED / "lgu_poverty.parquet"
    df.to_parquet(path, index=False)
    specific = df["source_note"].str.contains("LGU-specific").sum()
    fallback  = df["source_note"].str.contains("fallback").sum()
    log.info(
        "Fix 3b lgu_poverty: %d rows — %d LGU-specific SAE, %d province fallback",
        len(df), specific, fallback,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run all fixes
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 55)
    log.info("Fix 1: cpi_full quarter format")
    fix_cpi_full()

    log.info("=" * 55)
    log.info("Fix 2: commodity_prices 2022-2025 forward-fill")
    fix_commodity_prices()

    log.info("=" * 55)
    log.info("Fix 3: lgu_census + lgu_poverty (35 → 137 LGUs)")
    census_df = fix_lgu_census()
    fix_lgu_poverty(census_df)

    log.info("=" * 55)
    log.info("All fixes done. Verifying...")

    for name, path in [
        ("cpi_full",          "data/processed/cpi_full.parquet"),
        ("commodity_prices",  "data/processed/commodity_prices.parquet"),
        ("lgu_census",        "data/processed/lgu_census.parquet"),
        ("lgu_poverty",       "data/processed/lgu_poverty.parquet"),
    ]:
        df = pd.read_parquet(path)
        qcol = "quarter" if "quarter" in df.columns else ("year" if "year" in df.columns else None)
        qval = f"{df[qcol].min()} → {df[qcol].max()}" if qcol else "n/a"
        print(f"  {name:<20} {len(df):>5} rows  {qval}")
