# aiPHeed — Feature & Data Source Documentation

> **Building the API? Start with [`BACKEND_API_SPEC.md`](BACKEND_API_SPEC.md)** —
> the endpoint-by-endpoint contract (forecasting, SHAP explainability, charts,
> timeline/history, news, admin, auth, feedback). This file is the narrative
> "where does every number come from today" inventory; parts of §3–§4 predate the
> Sept-2026 refactor (SHAP is now the 5-trigger `getTriggerBreakdown`; the
> timeline is real-time; the basemap is self-hosted — see
> [`basemap.md`](basemap.md)). The API spec reflects current code.

**Purpose of this document:** a complete inventory of every feature in the aiPHeed frontend prototype, and — most importantly — where each piece of data on screen actually comes from. This is written for the backend developer picking up the project.

**TL;DR — the most important fact in this whole document:**
> **There is currently no backend.** This is a pure static React/Vite single-page app. Every number, chart, risk score, article, and login credential you see is either **hardcoded in TypeScript source files**, **randomly generated in the browser**, or **stored in the browser's `localStorage`/`sessionStorage`**. Nothing is fetched from a database or API (aside from static GeoJSON map-boundary files and third-party map tiles). "Real" data integration is the single biggest remaining task.

---

## 1. Tech stack

- **Framework:** React 18 + TypeScript, built with Vite.
- **Routing:** `react-router-dom` (`BrowserRouter`), all client-side, no server routing.
- **UI:** Tailwind CSS + shadcn/ui (Radix primitives) in [src/components/ui/](src/components/ui/).
- **Maps:** Leaflet (`leaflet` + `react-leaflet`-style manual integration in [PhilippineMap.tsx](src/components/PhilippineMap.tsx)), tiles from **CARTO** (`basemaps.cartocdn.com`), boundaries from **static GeoJSON files** in [public/](public/).
- **Charts:** `recharts`.
- **PDF/PNG export:** `jspdf` + `html2canvas`.
- **Data fetching layer:** `@tanstack/react-query` is installed and wired into `App.tsx`, but **no query actually calls anything** — there is no API client, no `fetch` to any backend endpoint anywhere in the app. It's present but unused (presumably scaffolded for future backend integration).
- **Bot protection:** Cloudflare Turnstile widget exists ([TurnstileGate.tsx](src/components/TurnstileGate.tsx)) but **is not mounted anywhere** in the app — dead code, not currently gating anything.

---

## 2. Routes (from [App.tsx](src/App.tsx))

| Path | Page | Notes |
|---|---|---|
| `/` | [Landing.tsx](src/pages/Landing.tsx) | Public marketing/about page |
| `/dashboard` | [Index.tsx](src/pages/Index.tsx) → `MapDashboard` | The main risk map dashboard |
| `/data` | [Data.tsx](src/pages/Data.tsx) | Tabular data explorer + CSV export |
| `/visualization` | [Visualization.tsx](src/pages/Visualization.tsx) | Trend & SHAP chart builder + PNG/PDF export |
| `/login` | [Login.tsx](src/pages/Login.tsx) | DA admin sign-in (standalone page) |
| `/admin`, `/admin/review`, `/admin/feedback` | [Admin.tsx](src/pages/Admin.tsx) | Admin console (map, review queue, feedback) |
| `*` | [NotFound.tsx](src/pages/NotFound.tsx) | 404 |

---

## 3. Data architecture — where every number comes from

This is the section the backend developer needs most. There are **three tiers** of "data" in this app today:

### 3a. Hardcoded province-level mock data — [src/data/quarterData.ts](src/data/quarterData.ts)

This is the **current single source of truth** for almost everything the dashboard displays. It is a plain TypeScript file, not a database:

- `PROVINCE_QUARTER_DATA`: an array of the 5 CALABARZON provinces (Quezon, Batangas, Rizal, Cavite, Laguna), each with a hand-written `scoresByQuarter` map (`"2025-Q1": 0.54`, etc.) covering 9 quarters (`2025-Q1` → `2027-Q1`) and a fixed `articles` count (e.g. Quezon = 142). **These numbers were typed in by hand**, not computed by any model.
- `DEFAULT_SHAP`: 6 hardcoded SHAP-style "feature contribution" values (Engel Coefficient, FPSI Price Stress, Dependency Rate, Income Decile, Household Size, Income Sources) used for the whole-region summary.
- `getShapForProvince(id)`: takes `DEFAULT_SHAP` and perturbs it with a **deterministic hash of the province id string** (`charCodeAt` based) so each province's SHAP bars look slightly different. This is a cosmetic pseudo-randomization, not a real explainability output.
- `TRIGGER_COMPOSITION_BY_PROVINCE`: hardcoded percentage splits (market/climate/employment/OFW/fish-kill) per province, typed in by hand.
- `SAMPLE_ARTICLES`: 5 hardcoded news articles (title/source/date/excerpt/real URL) shown identically for every province in the "News Articles Analyzed" card — they do **not** change based on the selected province or quarter.
- `QUEZON_TOP_MUNIS`, `REGION_OPTIONS`: more static arrays.
- Threshold constants used across the UI: `RISK_DISPLAY_CUTOFF = 0.5` (HIGH/LOW pill), `ALERT_THRESHOLD = 0.6` (active-alert badge), `LIMITED_SIGNAL_MIN_ARTICLES = 5`.
- `isLimitedSignal()`: flags a province as "Limited signal" using the rule `articles < 80 && quarter starts with "2027"` — an arbitrary demo rule, not a real article count from any data pipeline.
- `getQuarterMeta()`: computes "Generated on" / "Verification by" dates by simple date arithmetic (3 months before/after the quarter) — not from any actual model run log.

### 3b. Hardcoded + randomly-generated municipality/province data — [src/data/mockData.ts](src/data/mockData.ts)

This is a **second, partially overlapping** dataset, used by the map and left panel:

- `regionsData`: 5 provinces with hand-typed stats (population, poverty rate, unemployment, crop yield index, access to food, households at risk, FPSI, `momChange`, lat/lng). **Note:** these `riskScore` values (e.g. Quezon 0.66) are a *different set of numbers* from `PROVINCE_QUARTER_DATA` in quarterData.ts — the two files are not kept in sync, so the same province can show slightly different scores depending on which UI card reads which file.
- `historicalTrend`, `featureImportance`, `shapValues`: generated **at module load time using `Math.random()`** (see `generateTrend()`, `defaultShapValues()`, `defaultFeatures()`) — meaning these values are **different every time the app reloads** (no persistence, no server-side source).
- `municipalitiesData`: built by mapping over `src/data/calabarzon-municipalities.json` (a static list of ~140 municipality names/ids/province links) and computing each municipality's risk score/poverty/population/etc. via a **deterministic string-hash function** (`hash(id)`) seeded off the id/name — so scores are stable across reloads (same input → same output) but are **entirely synthetic**, not sourced from any survey or dataset.
- `nationalTrendData`: 12 months of `rfii`/`fpsi` values generated with `Math.random()` at load time — **changes every reload**. (Currently unused by any live page — was previously used by the now-orphaned `NationalTrendChart` component, see §6.)

### 3c. Real data: static geography files

- [public/ph-provinces.json](public/ph-provinces.json), [public/ph-regions.json](public/ph-regions.json), [public/calabarzon-municities.json](public/calabarzon-municities.json) — these **are real** GeoJSON polygon boundaries for Philippine provinces/regions/municipalities, fetched client-side via plain `fetch("/ph-provinces.json")` calls in [PhilippineMap.tsx](src/components/PhilippineMap.tsx#L364-L367). They're static files served from Vite's public folder, not an API.
- Map tiles (the visual basemap) are loaded live from CARTO's public tile CDN (`basemaps.cartocdn.com`, dark/light variants) — the only genuinely "live" network dependency in the whole app besides fonts/Turnstile's script.

### 3d. Browser-storage "state" (not a backend)

Several features use `localStorage`/`sessionStorage` as a fake persistence layer. **None of this syncs across devices/users/browsers** — it's purely local to whoever's browser is open.

| Key | Used by | Purpose |
|---|---|---|
| `aipheed_user` (sessionStorage) | [Login.tsx](src/pages/Login.tsx), [Admin.tsx](src/pages/Admin.tsx), [TopNavbar.tsx](src/components/TopNavbar.tsx) | "Logged in" flag: `{ email, role: "admin" }`. Cleared on tab close / logout. |
| `aipheed_rejections` | [src/lib/rejections.ts](src/lib/rejections.ts) | Admin's "reject this province's forecast" log — drives the "No forecast available" state shown to the public on the map. |
| `aipheed_rejections_seeded_v2` | [Admin.tsx](src/pages/Admin.tsx#L121-L147) | One-time flag so a deterministic set of "pre-existing" rejections gets seeded into `aipheed_rejections` the first time Admin loads (demo data). |
| `aipheed_feedback_v2` | [FeedbackModal.tsx](src/components/FeedbackModal.tsx) | Array of submitted System Usability Scale (SUS) survey responses. |
| `aipheed_theme` | [TopNavbar.tsx](src/components/TopNavbar.tsx), [Admin.tsx](src/pages/Admin.tsx), [Landing.tsx](src/pages/Landing.tsx) | Dark/light theme preference. |
| `aipheed_focus` (sessionStorage) | [Data.tsx](src/pages/Data.tsx) | One-shot "jump to this province on the map" signal when navigating from the data table. |

---

## 4. Feature-by-feature breakdown

### 4.1 Landing page (`/`) — [Landing.tsx](src/pages/Landing.tsx)
Static marketing page: hero, methodology explainer sections, research-team bios/photos (hardcoded names/images in `src/assets/researchers/`), no dynamic data. Purely presentational, no backend need beyond maybe a contact-form endpoint if one gets added.

### 4.2 Dashboard / Map (`/dashboard`) — [Index.tsx](src/pages/Index.tsx) + [MapDashboard.tsx](src/components/MapDashboard.tsx)

This is the core product screen. It's a full-bleed Leaflet map with floating cards on the left and right.

**Map itself** ([PhilippineMap.tsx](src/components/PhilippineMap.tsx)):
- Renders CALABARZON's 5 provinces from the GeoJSON, colored by `riskLevel` (from `regionsData` in mockData.ts — only two colors are actually used: yellow for "low", red for "high"; `RiskLevel` type has `moderate`/`severe` too but `getRiskLevel()` never returns them — see §7).
- Click a province → flies/zooms in, loads that province's municipalities (from `calabarzon-municities.json` + `municipalitiesData`), colors them the same way.
- Rejected provinces (per the admin's rejection log) render gray with a "No Forecast Available" tooltip and no pulse marker.
- Hover shows a tooltip with the risk score; a pulsing dot marker sits on each non-rejected province.

**Left column** ([LeftPanel.tsx](src/components/LeftPanel.tsx)):
- **No selection state** — three summary cards:
  - `RfiiScoreCardBody`: big animated regional-average score, computed as `average(PROVINCE_QUARTER_DATA[*].scoresByQuarter[currentQuarter])`. HIGH/LOW counts and "Limited" count are derived the same way.
  - `RiskDriversCardBody` ("Why is this Province at Risk?"): **5 hardcoded percentages** (Market 78%, Climate 66%, Fish Kill 57%, Employment 42%, OFW 38%) — these are **static and identical regardless of which province/quarter is selected**. Not wired to any real driver data.
  - `ProvinceRankingCardBody`: sorts `regionsData` (mockData.ts) by `riskScore` — note this reads the *other* mock dataset, not `PROVINCE_QUARTER_DATA`.
- **Province selected** (`RegionDetail`): risk score, QoQ change (`momChange` from mockData.ts), population, a "Trigger Composition" stacked bar (from `TRIGGER_COMPOSITION_BY_PROVINCE`), and the list of municipalities (from `municipalitiesData`) each showing a `getVulnerabilityDriver()` label computed from normalized poverty rate vs. population density (a simple heuristic, not a model output).
- **Municipality selected** (`MunicipalityDetail`): risk score, QoQ change, population, poverty rate — all from `municipalitiesData` (mockData.ts, hash-derived).
- **Rejected province/municipality**: shows `RejectedForecastPlaceholder` instead of any data.

**Right column** ([RightAnalyticsPanel.tsx](src/components/RightAnalyticsPanel.tsx)):
- `ShapNarrativeCardBody`: shows the SHAP-derived "Why is the risk high? / What is helping reduce the risk?" narrative. Pulls `getShapForProvince(id)` from quarterData.ts, then **rescales those SHAP values** so `baseline (0.32) + Σφ` exactly equals the risk score shown elsewhere for that province+quarter (a calibration hack, not a real Shapley decomposition). Each feature name maps to canned English sentences via the `FEATURE_PHRASING` dictionary in the same file.
- `NewsArticlesCardBody`: shows an article count (`PROVINCE_QUARTER_DATA[id].articles`, or the sum across all provinces if none selected), a **hardcoded** topic breakdown (`NEWS_TOPICS`: Food prices 42%, Typhoon 31%, Workers 18%, OFW 6%, Fish kill 3% — same for every province), and the 5 `SAMPLE_ARTICLES` (also identical regardless of province). Clicking an article opens a dialog with its (real, external) URL.

**Quarter time slider** ([QuarterTimeSlider.tsx](src/components/QuarterTimeSlider.tsx)): `buildQuarters()` hardcodes 6 quarters, `2025-Q3` → `2026-Q4`, with `2026-Q2` marked `current` and `2026-Q3`/`2026-Q4` marked `forecast`. **Note:** this only covers 6 of the 9 quarters that exist in `quarterData.ts`'s `QUARTER_IDS` (`2025-Q1` and `2027-Q1` are in the data but unreachable from this UI control).

**Map search** ([MapSearch.tsx](src/components/MapSearch.tsx)): client-side substring filter over `regionsData` + `municipalitiesData` names; picking a result dispatches a `window` CustomEvent (`aipheed:focus-region`) that `MapDashboard` listens for to select that region.

**Map controls / legend** ([MapControls.tsx](src/components/MapControls.tsx), [MapLegend.tsx](src/components/MapLegend.tsx)): zoom in/out (dispatches `aipheed:zoom-in`/`zoom-out` events consumed by the Leaflet map), collapse/expand both side panels, static color-key legend.

**About / Feedback** ([AboutModal.tsx](src/components/AboutModal.tsx), [FeedbackModal.tsx](src/components/FeedbackModal.tsx)): About is static text. Feedback opens the SUS survey — see §4.5.

### 4.3 Data page (`/data`) — [Data.tsx](src/pages/Data.tsx)

A sortable/filterable table built from `PROVINCE_QUARTER_DATA` × the first 6 `QUARTER_IDS`, with a per-row `articles` count synthesized as `Math.round(baseArticles * (0.7 + 0.05*i))` and a `trigger`/`status` value **derived from a formula on the province name length + row index** (`TRIGGERS[(name.length + i) % 3]`) — i.e., there is no real "trigger category" or "status" data behind this table; it's cosmetically varied per row. "Export CSV" builds a CSV client-side from the currently filtered rows and triggers a browser download — no server involved. Clicking a row stores `{ provinceId }` into `sessionStorage.aipheed_focus` and navigates to `/dashboard` (the dashboard doesn't currently appear to read `aipheed_focus` on mount — worth checking if this wiring is complete).

### 4.4 Visualization page (`/visualization`) — [Visualization.tsx](src/pages/Visualization.tsx)

Two sub-tabs, both reading from `quarterData.ts`:
- **Province Risk Level Trend**: pick province (or "All"), municipality, and a from/to quarter range → line chart. Province lines read `PROVINCE_QUARTER_DATA[id].scoresByQuarter[q]` directly. A single municipality's line is **not real data** — it's the parent province's score plus a small deterministic offset derived from hashing the municipality name (`((seed % 17) - 8) / 100`).
- **Feature Contribution Breakdown**: pick province + quarter → horizontal SHAP bar chart via `getShapForProvince()` (same synthetic SHAP as §4.2; note the quarter picker here doesn't actually affect the SHAP values shown, since `getShapForProvince` only takes the province id).
- Both charts support **Download PNG** (html2canvas snapshot of the chart DOM node) and **Download PDF** (same snapshot embedded into a jsPDF page) — fully client-side, no server rendering.

### 4.5 Feedback (SUS survey) — [FeedbackModal.tsx](src/components/FeedbackModal.tsx)

A 10-question System Usability Scale survey (standard SUS wording) plus demographics (name/email/agency/designation/age/sex/client type/province/municipality — province/municipality options come from a **third, separate** hardcoded list in [src/data/calabarzonProvinces.ts](src/data/calabarzonProvinces.ts), which does not match the GeoJSON-derived municipality list used by the map). Score is computed client-side with the standard SUS formula (odd items: `response-1`, even items: `5-response`, summed × 2.5). On submit, the whole entry is pushed into `localStorage.aipheed_feedback_v2` — **nothing is sent to a server**. Available from the public map (bottom of left panel / floating card footer).

### 4.6 Admin console (`/admin`) — [Admin.tsx](src/pages/Admin.tsx)

**Access gate:** requires `sessionStorage.aipheed_user.role === "admin"` (set only by the login flows below) *and* an email ending in `@calabarzon.da.gov.ph` (checked again client-side via `isAuthorizedAdmin()`). If either check fails, redirects to `/login` or shows an "Access denied" card. **This is entirely client-side — there is no server session, JWT, or cookie.** Anyone can open devtools and run `sessionStorage.setItem("aipheed_user", JSON.stringify({email:"x@calabarzon.da.gov.ph", role:"admin"}))` to get in; this needs a real backend-verified auth system.

Three sections (tab state only, same URL for `/admin`):
- **Mapping**: identical `MapDashboard` component as the public dashboard (`showAboutFeedback={false}`).
- **Review**: a queue of `ReviewItem`s **synthetically generated** in [Admin.tsx](src/pages/Admin.tsx#L91-L116) — for every year in `[2024, 2025, 2026]` × every quarter × every province in `regionsData`, a deterministic seed (`(year + quarterChar + provinceIndex) % 5`) decides whether the row starts as `"Staged"` or `"Approved"`, with a couple of special-cased rows for Quezon. Risk score is `regionsData[r].riskScore` plus a small per-quarter "drift" fudge factor. **None of this reflects a real review/approval pipeline** — it's demo data generated fresh from the same seed every page load.
  - **Approve / Reject / Undo Publishing** buttons mutate only React state (`items`) plus, for reject, call `addRejection()` which writes to `localStorage.aipheed_rejections` (§3d) — this is the *only* admin action that has any lasting effect, and it only affects the current browser.
  - **Rejection Log** tab lists everything in `aipheed_rejections`, with a "Restore" button that calls `removeRejection()`.
  - **Review detail modal**: shows municipality scores (again hash-derived from `calabarzon-municipalities.json`), a SHAP bar chart, human-readable SHAP narrative sentences (`SHAP_NARRATIVES` dictionary), and the province ranking — all recomputed from the same mock sources above.
- **Feedback**: reads `loadFeedback()` from `localStorage.aipheed_feedback_v2` and lists all SUS submissions with a detail modal. This is literally reading back what `FeedbackModal` (§4.5) wrote — **there is no aggregation across users/devices**, each admin only sees feedback submitted from their own browser.

### 4.7 Admin login — **two separate, inconsistent implementations**

This is worth flagging explicitly for the backend developer, since it's a real inconsistency in the current code:

1. **[Login.tsx](src/pages/Login.tsx)** (`/login` page): only checks that the email **ends with `@calabarzon.da.gov.ph`**. The password field exists in the form but **is never read or validated** — any password works, or none.
2. **[TopNavbar.tsx](src/components/TopNavbar.tsx#L119-L153)** (`AdminLoginModal`, opened from the hamburger menu on any public page): checks the email is **exactly** `admin@calabarzon.da.gov.ph` **and** the password is **exactly** `admin2026` — both are hardcoded in plaintext in client-side JS (visible to anyone reading the bundled JS), and shown in the UI itself as demo credentials.

Both paths, on "success", just write the same `sessionStorage.aipheed_user` object and navigate to `/admin`. **Neither talks to a server.** When real auth is built, both entry points need to be unified against it, and the password fields need to actually do something.

---

## 5. Orphaned / dead code (not reachable from any route)

These files exist in the repo but are **not imported by any active page or component** — they were built for an earlier iteration of the dashboard and are effectively unused today. Worth knowing about so the backend developer doesn't assume they're live:

- [src/components/DashboardOverview.tsx](src/components/DashboardOverview.tsx)
- [src/components/NationalTrendChart.tsx](src/components/NationalTrendChart.tsx)
- [src/components/RegionalRankingsTable.tsx](src/components/RegionalRankingsTable.tsx)
- [src/components/AnimatedStatWidgets.tsx](src/components/AnimatedStatWidgets.tsx)
- [src/components/WhatIfSandbox.tsx](src/components/WhatIfSandbox.tsx) — a full "what-if" slider sandbox for adjusting features and seeing a simulated risk score recompute; not wired into any page.
- [src/components/CalabarzonSilhouette.tsx](src/components/CalabarzonSilhouette.tsx)
- [src/components/TurnstileGate.tsx](src/components/TurnstileGate.tsx) — Cloudflare Turnstile bot-check widget, not mounted anywhere.
- [src/lib/pdfReport.ts](src/lib/pdfReport.ts) (843 lines) — a fully-built multi-page PDF report generator (`downloadRegionReport`, `downloadProvinceReport`, `downloadMunicipalityReport`) with cover pages, executive summary, municipal breakdown, and actions sections. **Not called from any button in the current UI** — the PDF export that *is* wired up (§4.4) is a much simpler chart-snapshot export in `Visualization.tsx`, unrelated to this file.
- `nationalTrendData` export in [mockData.ts](src/data/mockData.ts) — only consumer was the now-orphaned `NationalTrendChart`.

---

## 6. Known inconsistencies to resolve when wiring a real backend

- **Two parallel province datasets** that disagree: `regionsData` (mockData.ts) vs. `PROVINCE_QUARTER_DATA` (quarterData.ts). Different UI cards read from different ones for the "same" risk score.
- **`RiskLevel` type has 4 values** (`low`/`moderate`/`high`/`severe`) but **only 2 are ever produced** (`getRiskLevel()` in mockData.ts and `getRiskLevelFromScore()` in quarterData.ts both only return `"low"` or `"high"`) — moderate/severe colors/labels exist in [types.ts](src/data/types.ts) but are dead branches.
- **Municipality province/name lists disagree** between `calabarzon-municipalities.json` (used by the map + admin review) and `calabarzonProvinces.ts` (used only by the feedback demographics dropdown).
- **Quarter ranges disagree**: `QUARTER_IDS` has 9 quarters (`2025-Q1`→`2027-Q1`); `buildQuarters()` for the on-map slider only exposes 6 (`2025-Q3`→`2026-Q4`); `Data.tsx` only uses the first 6 of `QUARTER_IDS` too.
- **News articles, news topic %, and risk-driver %s are identical for every province/quarter** — none of them are actually filtered by the selected province, despite the UI implying they are.
- **Random values regenerate on every page reload** for anything sourced from `Math.random()` in mockData.ts (`historicalTrend`, `featureImportance`, `shapValues`, `nationalTrendData`) — so those specific fields are not stable/reproducible today.
- **No real authentication or authorization** — see §4.7. Admin gating can currently be bypassed via browser devtools.
- **No cross-device data**: rejections and feedback submissions live only in the browser that created them. An admin rejecting a forecast on one machine is invisible to anyone on another machine/browser.

---

## 7. What a backend needs to provide (suggested integration points)

Based on the above, a real backend would most usefully replace:

1. **A provinces/municipalities risk-score API** — replacing `mockData.ts` + `quarterData.ts`'s `PROVINCE_QUARTER_DATA`, keyed by province/municipality id × quarter, returning risk score, risk level, population, poverty rate, QoQ change, article count.
2. **A SHAP/feature-contribution API** — per province (and ideally per municipality) × quarter, replacing `getShapForProvince()`.
3. **A news-articles API** — per province × quarter, replacing the fixed `SAMPLE_ARTICLES` and `NEWS_TOPICS`.
4. **A trigger-composition API** — per province × quarter, replacing `TRIGGER_COMPOSITION_BY_PROVINCE`.
5. **Real authentication** (email+password or SSO against `@calabarzon.da.gov.ph`, with a server-issued session/JWT) to replace both ad-hoc client-side login checks.
6. **A review/approval/rejection API** — persisting `ReviewItem` status and the rejection log server-side (with audit trail: who rejected, when, why) instead of `localStorage`.
7. **A feedback-submission API** — persisting SUS survey responses server-side so admins see aggregate feedback across all users, not just their own browser.
8. **(Optional) A "what-if" simulation endpoint** if the orphaned `WhatIfSandbox.tsx` (§5) gets revived — would need a live model endpoint that accepts adjusted feature values and returns a recomputed risk score.

Everything geographic (the GeoJSON boundary files, the Leaflet/CARTO map rendering) can stay exactly as-is — that part is already "real" and doesn't depend on a backend.
