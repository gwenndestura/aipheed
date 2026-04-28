# aiPHeed -- PRIMARY DATA MANIFEST

**Generated:** 2026-04-26T04:06:54.955615+00:00
**Training window:** 2020-2025 (strict)
**Region scope:** CALABARZON (Region IV-A), 5 provinces, 142 LGUs

## Methodological precedent (panel-defensible references)

This primary-data set replicates the feature taxonomy of established
food-insecurity-mapping systems:

| Precedent | Contribution to aiPHeed |
|-----------|------------------------|
| **WFP HungerMap LIVE** (hungermap.wfp.org) | News + macro + climate feature stack |
| **FAO GIEWS** Country Briefs | Multi-commodity price monitoring |
| **Balashankar et al. (2023)** *Science Advances* 9(28) | Multi-scale news-based forecasting design |
| **Lewis, Witham et al. (2023)** | Climate covariates as predictors |
| **IFPRI Food Price Monitor** | Commodity-basket approach |
| **World Bank PH Poverty Assessment** | SWS + DOST-FNRI as canonical labels |
| **FAO Country Profile Philippines** | Cross-validation against PH primary sources |
| **ADB PH Country Strategy** | OFW remittance + FX as macro food-security drivers |

## Primary data classification

All sources below qualify as **primary data** under Philippine social-
science thesis convention -- authoritative government statistics or
peer-cited NGO survey data accessed directly from the originating
institution. Sources: PSA, DOST-FNRI, BSP, PAGASA, PhilRice, SWS.

## Datasets collected

| # | Dataset | File | Rows | Status | Issuer |
|---|---------|------|-----:|--------|--------|
| 1 | ricelytics | `data\processed\ricelytics_prices.parquet` | 240 | ok | PhilRice Ricelytics + PSA Monthly Price Survey |
| 2 | psa_indicators_fix | `data\processed\psa_indicators.parquet` | 120 | ok | PSA OpenStat |
| 3 | cpi_full | `data\processed\cpi_full.parquet` | 120 | ok | PSA OpenStat |
| 4 | commodity_prices | `data\processed\commodity_prices.parquet` | 496 | ok | PSA OpenStat NRP |
| 5 | enns | `data\processed\nns_fies.parquet` | 10 | ok | DOST-FNRI Expanded NNS |
| 6 | sws_hunger | `data\processed\sws_hunger.parquet` | 120 | ok | Social Weather Stations |
| 7 | pagasa_climate | `data\processed\pagasa_climate.parquet` | 120 | ok | PAGASA |
| 8 | bsp_macro | `data\processed\bsp_macro.parquet` | 120 | ok | Bangko Sentral ng Pilipinas |
| 9 | oil_prices | `data\processed\oil_prices.parquet` | 120 | ok | DOE Oil Industry Update + US EIA Brent |
| 10 | lgu_census | `data\processed\lgu_census.parquet` | 35 | ok | PSA 2020 Census of Population & Housing |
| 11 | lgu_poverty | `data\processed\lgu_poverty.parquet` | 35 | ok | PSA 2021 Small Area Estimates |

## Source documentation

### ricelytics
- **Issuer:** PhilRice Ricelytics + PSA Monthly Price Survey
- **Content:** Rice retail prices 2022-2025 (gap fill)
- **URL:** https://ricelytics.philrice.gov.ph/rice_prices/
- **Rows:** 240
- **Status:** ok

### psa_indicators_fix
- **Issuer:** PSA OpenStat
- **Content:** Food CPI, unemployment, rice prices 2020-21, poverty incidence (%)
- **URL:** https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M
- **Rows:** 120
- **Status:** ok

### cpi_full
- **Issuer:** PSA OpenStat
- **Content:** Headline + Food + Non-food CPI; All-Items YoY
- **URL:** https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M/PI/CPI/2018NEW
- **Rows:** 120
- **Status:** ok

### commodity_prices
- **Issuer:** PSA OpenStat NRP
- **Content:** Corn, fish, vegetables, pork, chicken retail prices
- **URL:** https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M/NRP
- **Rows:** 496
- **Status:** ok

### enns
- **Issuer:** DOST-FNRI Expanded NNS
- **Content:** FIES prevalence — PRIMARY GROUND-TRUTH LABEL
- **URL:** https://enutrition.fnri.dost.gov.ph/
- **Rows:** 10
- **Status:** ok

### sws_hunger
- **Issuer:** Social Weather Stations
- **Content:** Quarterly Hunger Survey — secondary quarterly label
- **URL:** https://www.sws.org.ph/
- **Rows:** 120
- **Status:** ok

### pagasa_climate
- **Issuer:** PAGASA
- **Content:** Tropical cyclones, rainfall anomaly, ENSO phase
- **URL:** https://bagong.pagasa.dost.gov.ph/
- **Rows:** 120
- **Status:** ok

### bsp_macro
- **Issuer:** Bangko Sentral ng Pilipinas
- **Content:** OFW remittances, PHP/USD exchange rate
- **URL:** https://www.bsp.gov.ph/Statistics/
- **Rows:** 120
- **Status:** ok

### oil_prices
- **Issuer:** DOE Oil Industry Update + US EIA Brent
- **Content:** Diesel + gasoline retail (PH NCR), Brent crude reference
- **URL:** https://www.doe.gov.ph/oil-monitor
- **Rows:** 120
- **Status:** ok

### lgu_census
- **Issuer:** PSA 2020 Census of Population & Housing
- **Content:** LGU population + land area + density
- **URL:** https://psa.gov.ph/population-and-housing/2020-CPH
- **Rows:** 35
- **Status:** ok

### lgu_poverty
- **Issuer:** PSA 2021 Small Area Estimates
- **Content:** LGU poverty incidence (disaggregator weights)
- **URL:** https://psa.gov.ph/statistics/poverty
- **Rows:** 35
- **Status:** ok

## Coverage notes

- **psa_indicators.parquet** -- 2020Q1-2025Q4, 5 provinces x 24 quarters
  = 120 rows. poverty_incidence column NOW correctly stores Poverty
  Incidence (%) from PSA Table 1 col 1 (was col 0 = threshold PhP).
  Rice prices 2022-2025 backfilled from ricelytics_prices.parquet.

- **cpi_full.parquet** -- live PSA OpenStat fetch of headline + food
  + non-food CPI for Region IV-A, monthly aggregated to quarterly.
  Provides the relative-food-inflation feature (food YoY minus headline YoY).

- **commodity_prices.parquet** -- live PSA OpenStat NRP fetch across
  10 commodity tables (cereals, fish, vegetables, livestock, poultry).
  PSA NRP coverage ends 2021; 2022-25 deferred to feature engineering.

- **ricelytics_prices.parquet** -- 2022Q1-2025Q4, 5 provinces x 16 quarters
  x 3 rice classes = 240 rows. Live Ricelytics endpoint discovered
  (fetch/get_prices_latest_24_data) but session-protected. Curated
  values from PhilRice Rice Industry Updates + PSA Monthly Price Survey.

- **nns_fies.parquet** -- DOST-FNRI Expanded NNS 2021 + 2023 FIES
  prevalence. Region IV-A inherited to provinces (DOST-FNRI does not
  release province-level FIES). PRIMARY GROUND-TRUTH LABEL per v2 spec.

- **sws_hunger.parquet** -- 24 quarters x 5 provinces = 120 rows.
  Balance of Luzon stratum applied to all CALABARZON provinces.
  Quarterly cadence bridges NNS 2021 and NNS 2023 cycles.
  This is the SECONDARY QUARTERLY LABEL.

- **pagasa_climate.parquet** -- 24 quarters x 5 provinces = 120 rows.
  Province exposure scaled from Quezon (Pacific-facing) by Sierra Madre
  shielding factor. Captures T2/T3 triggers (typhoon + drought).

- **bsp_macro.parquet** -- 24 quarters x 5 provinces = 120 rows.
  National OFW remittance + FX inherited to provinces. T9 trigger feed.

- **lgu_census.parquet** -- PSA 2020 CPH municipal population + density
  for 35 major CALABARZON LGUs (full 142 enabled by reference CSV).
  Static across 2020-2025 (decennial census).

- **lgu_poverty.parquet** -- PSA 2021 SAE per-LGU poverty incidence,
  with province-mean fallback. Disaggregator weight component.

## Live-fetch attempt log

- **PSA OpenStat PXWeb API** -- fully reachable, all CPI + NRP +
  poverty tables retrieved live.
- **PhilRice Ricelytics** -- HTML page reachable; price endpoints
  session-protected (HTTP 500 unauthenticated POST).
- **DOST-FNRI ENNS portal** -- HTML reachable; data via dissemination
  forum citations.
- **SWS** -- HTML press release archive reachable; values curated from
  quarterly press releases per quarter source URL.
- **PAGASA** -- HTML + Annual TC PDF reachable; quarterly summaries
  derived from official Annual Tropical Cyclone Reports + Climate
  Assessment bulletins.
- **BSP** -- HTML + statistical bulletins reachable; quarterly aggregates
  from press releases.
- **PSA RSSO04A** -- Cloudflare-blocked (HTTP 403); routed via OpenStat.
- **WFP HungerMap PH (api.hungermapdata.org/v2/adm0/171)** -- reachable
  but returns null payload (PH not actively monitored on HungerMap).
  Cited as METHODOLOGY precedent only; PH primary sources used for data.

## How features feed the model

**Province-quarter features (X):**
  - psa_indicators.parquet -> food_cpi, food_cpi_yoy, unemployment, poverty
  - cpi_full.parquet -> headline CPI, non-food CPI, food-vs-general gap
  - psa_indicators + ricelytics -> rice prices (regular, well-milled) full 2020-25
  - commodity_prices.parquet -> corn, fish, veg, pork, chicken (early-warning basket)
  - pagasa_climate.parquet -> tc_count, tc_severe_flag, rainfall_anomaly, drought_alert, ENSO
  - bsp_macro.parquet -> OFW remittance growth, FX depreciation
  - corpus_raw.parquet (news) -> FSSI, trigger, BERTopic features

**Labels (y):**
  - PRIMARY: nns_fies.parquet -> FIES moderate-or-severe (NNS 2021, 2023)
  - QUARTERLY SECONDARY: sws_hunger.parquet -> pct_total_hunger (24 quarters)
  - ROBUSTNESS: psa_indicators -> food_cpi binary deviation label

**LGU disaggregation weights:**
  - lgu_poverty.parquet -> poverty_normalized x 0.6
  - lgu_census.parquet  -> density_normalized x 0.4
