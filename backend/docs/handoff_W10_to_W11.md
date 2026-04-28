# Week 10 → Week 11 Handoff Document

**From:** Member A (Angel) — NLP, Features & API Routes
**To:** Member B (Nina) — Training, SHAP & Infrastructure
**Date:** End of Week 10 (Apr 4, 2026)
**Branch:** `week10`

---

## 1. Purpose of this handoff

Per the Backend Guide and Task Assignment v4 closing line of Week 10:

> *"Member A delivers confirmed PSA field names (extracted from statistical reports) and `corpus_raw.parquet` structure to Member B by Friday. Member B needs these for DB schema design and Pydantic contract design in Week 11."*

This document delivers exactly that, plus the v2 expansion work (ten additional primary-data parquets) so that Member B's W11 schema design can cover the full feature surface from day one.

---

## 2. Corpus parquet schema — `data/raw/corpus_raw.parquet`

**Status:** 59,878 rows ingested via Google News RSS, deduplicated by URL.
**Gitignored:** yes (under `data/raw/`).

| Column | Type | Null count | Description |
|---|---|---|---|
| `title` | string | 0 | Article headline |
| `link` | string | 0 | Canonical article URL (deduplication key) |
| `article_id` | string | 0 | SHA-256 hash of `link` — stable across pipeline runs |
| `published` | string | 0 | ISO-8601 timestamp from RSS feed |
| `summary` | string | 0 | RSS summary / first 300 characters |
| `source_domain` | string | 0 | Outlet domain (e.g., `rappler.com`, `pna.gov.ph`) — must match `CREDIBLE_DOMAINS` allowlist |
| `fetcher_source` | string | 0 | Currently `gnews_rss` for all rows; `rss`, `gnews`, `wayback` reserved for additional fetchers |
| `province_code` | string | **59,878** | **Null until W11-1 geocoding runs** — not a defect at end of W10 |
| `quarter` | string | 0 | Derived from `published`, format `"YYYY-QN"` |

**Quarter coverage:** 2014-Q2 through 2026-Q2 (filtered to 2020-Q1 through 2025-Q4 in W11 feature engineering).

**Top source domains:**
| Outlet | Articles |
|---|---|
| rappler.com | 6,959 |
| pna.gov.ph | 6,762 |
| gmanetwork.com | 6,187 |
| philstar.com | 6,151 |
| mb.com.ph | 4,749 |
| businessmirror.com.ph | 4,517 |
| pia.gov.ph | 3,642 |
| newsinfo.inquirer.net | 3,070 |

---

## 3. PSA OpenStat indicator schema — `data/processed/psa_indicators.parquet`

**Status:** 120 rows — 5 provinces × 24 quarters (2020-Q1 to 2025-Q4).
**Source:** PSA OpenStat PXWeb API (the working channel; PSA RSSO04A is Cloudflare-blocked).

| Column | Type | Description |
|---|---|---|
| `province_code` | string | PSGC 9-digit code (e.g., `PH040100000` = Cavite) |
| `province_name` | string | One of {Cavite, Laguna, Quezon, Rizal, Batangas} |
| `quarter` | string | `"YYYY-QN"` |
| `food_cpi` | float | Food + Non-alcoholic Beverages CPI, 2018=100 |
| `food_cpi_yoy` | float | Food CPI year-on-year % change |
| `rice_price_regular` | float | Regular-milled rice retail, ₱/kg (2020-21 from PSA NRP, 2022-25 backfilled from Ricelytics) |
| `rice_price_well` | float | Well-milled rice retail, ₱/kg |
| `unemployment_rate` | float | National quarterly LFS unemployment % (inherited to provinces) |
| `poverty_incidence` | float | **Poverty Incidence among Families (%)** — fixed from earlier bug where column was poverty threshold ₱; values now 3-17% range, not 4500-7000 |

**Field-name confirmation for DB schema:** column names above are stable. Member B's `models.py` should treat them as canonical for the `ForecastRecord.feature_*` columns and Pydantic schemas.

---

## 4. v2 primary-data expansion — ten additional parquets

Beyond the original Backend Guide spec (which covered news + PSA only), the v2 collection added nine credible Filipino primary sources to replicate the feature taxonomy of WFP HungerMap LIVE, FAO GIEWS, and Balashankar et al. (2023). All eleven parquets are documented in `data/processed/PRIMARY_DATA_MANIFEST.md`.

| Parquet | Rows | Issuer | What it provides |
|---|--:|---|---|
| `psa_indicators.parquet` | 120 | PSA OpenStat | food CPI, unemployment, rice 2020-21, poverty incidence (%) |
| `cpi_full.parquet` | 120 | PSA OpenStat | headline + food CPI for the food-vs-headline gap feature |
| `commodity_prices.parquet` | 496 | PSA NRP | corn, fish, vegetables, livestock, poultry, rootcrops |
| `ricelytics_prices.parquet` | 240 | PhilRice + PSA Monthly Price Survey | rice prices 2022-2025 (gap fill) |
| `nns_fies.parquet` | 10 | DOST-FNRI ENNS | **PRIMARY LABEL** — FAO SDG 2.1.2 FIES prevalence (NNS 2021 + 2023) |
| `sws_hunger.parquet` | 120 | Social Weather Stations | quarterly self-reported hunger — secondary quarterly label |
| `pagasa_climate.parquet` | 120 | PAGASA | tropical cyclones, rainfall anomaly, ENSO phase, drought alert |
| `bsp_macro.parquet` | 120 | Bangko Sentral ng Pilipinas | OFW remittance growth, PHP-USD exchange rate |
| `oil_prices.parquet` | 120 | DOE Oil Industry Update + EIA Brent | diesel, gasoline retail, Brent crude reference |
| `lgu_census.parquet` | 35 | PSA 2020 CPH | LGU population + density (disaggregator weight) |
| `lgu_poverty.parquet` | 35 | PSA 2021 SAE | LGU poverty incidence (disaggregator weight) |

**Schemas of the major join keys (province-quarter level):**

All province-quarter parquets share the same join keys:

```
province_code  : string  (PSGC 9-digit)
province_name  : string  (Cavite | Laguna | Quezon | Rizal | Batangas)
quarter        : string  ("YYYY-QN")
```

LGU-level parquets (`lgu_census`, `lgu_poverty`) join on `(province_code, lgu_code, lgu_name)`.
ENNS FIES joins on `(survey_cycle, province_code)` and is broadcast to all quarters within the cycle window via the W11 label generator.

---

## 5. Implication for Member B's Week 11 work

### `app/db/models.py`

Two new ORM tables on top of the originals:

```python
class MunicipalForecastRecord(Base):
    __tablename__ = "municipal_forecast_records"
    id                      : int (PK)
    quarter                 : str
    lgu_code                : str
    lgu_name                : str
    province_code           : str
    risk_index              : float
    disaggregation_label    : str   # required per Backend Guide Rule #8
    data_sufficiency_flag   : str   # inherits from province
    created_at              : datetime
```

Existing `ForecastRecord`, `SHAPRecord`, `AlertRecord` remain unchanged.

### Composite indexes (apply at table creation, per Backend Guide)

```python
Index("ix_forecast_quarter_province", "quarter", "province_code")
Index("ix_municipal_quarter_lgu", "quarter", "lgu_code")
Index("ix_shap_quarter_province", "quarter", "province_code")
Index("ix_alert_quarter_province", "quarter", "province_code")
```

### Pydantic schemas (`app/schemas/forecasts.py`)

`ForecastLatestResponse`, `MunicipalForecastResponse` — the latter must include `disaggregation_label` in every row (Backend Guide Rule #8).

---

## 6. Trigger taxonomy update (heads-up for `app/db/models.py` `feature_name` enum)

The Backend Guide v1 trigger taxonomy was three categories — market, climate, employment.
The v2 spec uses the HungerGist 8+2 taxonomy (Ahn et al., 2023), which collapses to **five trigger driver groups** at the feature level:

```
trigger_market
trigger_climate
trigger_employment
trigger_ofw_remittance     # NEW — T9 in HungerGist
trigger_fish_kill          # NEW — T1b PH-specific extension
```

Member B's `SHAPRecord.feature_name` should accept these five names plus the macro/climate/price feature names.

---

## 7. FastAPI Swagger URL — for the React frontend teammate

```
http://127.0.0.1:8000/docs
```

CORS is configured for `http://localhost:5173` (Vite default). All `/v1/*` endpoints will appear here as soon as Week 11 routes are added; current endpoints (`/v1/health`, `/v1/readiness`) are live.

---

## 8. What is **not** in this handoff (but is on the books for W11)

| Pending W11-1 (Member A) | Status |
|---|---|
| Geocoding — populate `corpus_raw.parquet.province_code` via PSGC fuzzy match | Geocoder code exists at `app/ml/corpus/geocoder.py`; not yet executed against the corpus |
| Bias weights — w_p,t per province-quarter | `bias_weighter.py` exists; not yet executed |
| FSSI builder — wraps `app/ml/nlp/classifier.py` (HungerGist 10-hypothesis multi-template) | classifier ready; FSSI builder not written |

---

## 9. Confirmation checklist for Member B

- [ ] Read this document
- [ ] Confirm `MunicipalForecastRecord` field names with Member A before W11-5
- [ ] Confirm five trigger feature names with Member A before W11-6 schema design
- [ ] Run `pytest tests/integration/test_api_health.py` once it's stubbed in W12-4 (smoke check)
- [ ] Reply to Member A with any field-name disagreements **before Monday of W11**

---

*End of W10 → W11 Handoff*
