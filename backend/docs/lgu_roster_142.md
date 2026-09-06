# The 142-LGU CALABARZON roster

Region IV-A has **142 cities and municipalities**:

| Province | LGUs | Cities |
|---|---|---|
| Batangas | 34 | 5 |
| Cavite | 23 | 8 |
| Laguna | 30 | 6 |
| Quezon | 41 | 2 |
| Rizal | 14 | 1 |
| **Total** | **142** | **22** |

## What was wrong

`lgu_census.parquet` and `lgu_poverty.parquet` were built from a hand-typed
list in `fix_primary_data.py` that held only **137** LGUs. Five were missing
outright:

| Province | LGU | PSGC | 2020 CPH pop. | Land area |
|---|---|---|---|---|
| Batangas | Taal | 041029000 | 61,460 | 29.76 km² |
| Batangas | Talisay | 041030000 | 46,238 | 28.20 km² |
| Quezon | Mauban | 045627000 | 71,081 | 415.98 km² |
| Quezon | Pagbilao | 045630000 | 78,700 | 170.96 km² |
| Quezon | Quezon | 045637000 | 15,886 | 71.22 km² |

Consequences:

* The **disaggregator** iterates the census, so it emitted 137 municipal
  forecasts while its own docstring promised 142. Those five LGUs got no
  municipal risk index at all.
* `precision_pass.py` and `reanalyze_calabarzon.py` build their
  geo-verification gazetteer from the census, so the five were invisible to
  the retention gate.
* Three LGUs were spelled differently from the PSGC gazetteer
  (`Mataas na Kahoy`/`Mataasnakahoy`, `Santo Tomas`/`Sto. Tomas`,
  `General Mariano Alvarez`/`Gen. Mariano Alvarez`), so census rows did not
  join cleanly to the gazetteer or the coverage matrix.
* `lgu_code` was a `hash()`-derived placeholder, not a PSGC code — and
  `hash()` is salted per process, so codes were not stable across runs.

## How it works now

`data/reference/calabarzon_lgus.csv` is the single roster. Build it with:

```bash
python scripts/build_lgu_reference.py
```

It takes the roster, PSGC codes, canonical names and city/municipality class
from `psgc_gazetteer.parquet`, and population and land area from
`cph2020_calabarzon_municipal.csv`. It fails rather than writes if the result
is not 142 LGUs / 22 cities, or if the population figures do not reconcile with
PSA's published province totals.

Then rebuild the panels:

```bash
python -m app.ml.corpus.lgu_census_fetcher && python -m app.ml.corpus.lgu_poverty_fetcher
```

`lgu_census_fetcher` raises if the CSV is not 142 rows, so a short roster
fails loudly instead of silently shrinking the region.

### Columns added to `lgu_census.parquet`

* `lgu_aliases` — pipe-separated alternate spellings, used by the matchers so
  "Mataas na Kahoy" resolves to Mataasnakahoy.
* `population_source` — `cph2020` (exact PSA total) or `estimate` (rounded
  working figure). Every row is `cph2020`; `estimate` only appears if an LGU
  is missing from the CPH table and falls back to the old census.

### Matcher changes

`reanalyze_lib.build_lgu_matchers()` now owns matcher construction for both
`precision_pass.py` and `reanalyze_calabarzon.py`, which previously each built
their own and both dropped any name shorter than 5 characters. That silently
discarded ten real LGUs — **Bay, Famy, Imus, Lian, Lipa, Lobo, Naic, Pila,
Taal, Tuy** — including two of the region's largest cities. Keying on the bare
name also collapsed the two Rosarios (Batangas and Cavite) into one entry.

The rule is now **demote, never drop**: a name that cannot stand alone moves to
the ambiguous map, where it still matches whenever its province is named in the
same text. All 142 are reachable.

Two special cases:

* **Quezon** (municipality, Quezon province) — the name *is* its province, so
  requiring the province cue is vacuous and a bare "quezon" is far more often
  Quezon City, the province, or the president. It matches only via an explicit
  form: `quezon, quezon`, `municipality of quezon`, `bayan ng quezon`,
  `town of quezon`.
* **Talisay** — Talisay City (Cebu) and Talisay (Negros Occidental) dominate
  news volume, so Talisay, Batangas requires the province cue.

### Cityhood

`build_lgu_reference.py` carries a `CITY_OVERRIDES` set because the bundled
PSGC snapshot predates two conversions: **Calaca** (city since RA 11544, 2022)
and **Carmona** (city since RA 11938, 2023). Without the override both would be
demoted back to municipality. Conversely **Candelaria, Quezon** was mislabelled
a city in the old 137-LGU table and is correctly a municipality here.

## Population and land area — full PSA 2020 CPH table

All **142 of 142** rows now carry exact PSA 2020 Census of Population and
Housing figures (`population_source == "cph2020"`). The municipal table lives
at `data/reference/cph2020_calabarzon_municipal.csv` and is the input
`build_lgu_reference.py` reads.

### The build validates itself

`build_lgu_reference.py` aborts unless the municipal table reconciles exactly
with PSA's published province totals:

| Province | Sum of LGUs | PSA official |
|---|---|---|
| Batangas | 2,908,494 | 2,908,494 |
| Cavite | 4,344,829 | 4,344,829 |
| Laguna | 3,382,193 | 3,382,193 |
| Quezon (excl. Lucena) | 1,950,459 | 1,950,459 |
| Rizal | 3,330,143 | 3,330,143 |
| **CALABARZON** | **16,195,042** | **16,195,042** |

PSA publishes Quezon without Lucena City (highly urbanised, administratively
independent); the roster still carries Lucena under Quezon, so the checksum
subtracts it before comparing. 142 figures summing to five official totals is
a strong guarantee that nothing was dropped, duplicated or mistyped, so a
mismatch fails the build rather than shipping bad denominators.

### How wrong the old table was

Of the 137 LGUs carried over from the previous build, **123 had an incorrect
population, land area, or both**. The old regional total was 15,579,854 —
short of the true 16,195,042 by **615,188 people (3.8%)**.

The errors were not confined to the obviously-rounded rows. Several
precise-looking figures were simply wrong:

| LGU | Old pop. | Correct pop. | Old area | Correct area |
|---|---|---|---|---|
| Tanza, Cavite | 196,000 | 312,116 | 78.00 | 78.33 |
| San Juan, Batangas | 46,218 | 114,068 | 74.00 | 273.40 |
| Naic, Cavite | 96,288 | 160,987 | 91.00 | 75.81 |
| Gen. Mariano Alvarez, Cavite | 118,000 | 172,433 | 39.00 | 9.40 |
| Teresa, Rizal | 95,000 | 64,072 | 37.00 | 18.61 |
| Santa Teresita, Batangas | 46,891 | 21,559 | 54.00 | 16.30 |

This matters directly: density is 40% of the disaggregation weight
(`y_m = y_province × (0.6 × poverty_m + 0.4 × density_m)`). San Juan's area was
understated 3.7×, inflating its density by the same factor. Gen. Mariano
Alvarez was wrong in both directions at once — its corrected density is roughly
6× the old value, and it now ranks highest in CALABARZON on the municipal risk
index instead of sitting mid-table.

## Remaining limitation — poverty

Poverty is now the weaker input. Only **16 of 142** LGUs have an LGU-specific
2021 SAE value; the other 126 inherit their province mean, which flattens
within-province variation and leaves poverty contributing almost nothing to
the ranking inside a province. Dropping a full SAE table at
`data/reference/sae_2021_calabarzon.csv` (columns `province_code`, `lgu_name`,
`poverty_incidence_pct`) makes `lgu_poverty_fetcher` pick it up automatically —
no code change needed.

## Corpus coverage is a separate number

142 is the roster. **98 of 142** LGUs have at least one qualifying article in
`calabarzon_food_insecurity_dataset.csv` — that is a news-availability limit,
not a roster gap, and it is tracked in `calabarzon_lgu_coverage_matrix.csv`.
Re-running the reanalysis pipeline with the corrected matchers may raise it,
since ten LGUs (Imus and Lipa among them) were previously unmatchable.
