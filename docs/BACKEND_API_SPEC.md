# aiPHeed — Frontend Functionality & Backend API Specification

**Audience:** backend developers building the API the aiPHeed frontend will consume.

**What this document is:** every user-facing capability in the frontend, the exact
data each one needs, and a concrete REST contract to provide it. Where a feature
is currently faked, the mock source is named so you can see the shape the UI
already expects.

**Companion doc:** [`FEATURES.md`](FEATURES.md) is the narrative inventory of
"where every on-screen number comes from today." This doc is the forward-looking
API contract. Where they disagree, this doc wins (it reflects the current code
after the Sept‑2026 refactor of SHAP/triggers, the dynamic timeline, and the
self‑hosted basemap).

**Current state:** there is **no backend**. The app is a static React/Vite SPA;
all data is hardcoded in `src/data/*.ts`, hash-generated in the browser, or in
`localStorage`. `@tanstack/react-query` is wired but calls nothing. Your job is to
stand up the API and swap the mock modules for a thin API client.

---

## 0. Conventions

| Topic | Convention |
|---|---|
| Base path | `/api/v1` |
| Format | JSON. `Content-Type: application/json; charset=utf-8` |
| Auth | `Authorization: Bearer <jwt>` for admin/write endpoints. Public read endpoints need no auth. See §J. |
| Province ID | lowercase slug: `quezon`, `batangas`, `rizal`, `cavite`, `laguna` |
| Municipality ID | `"{provinceId}-{muniSlug}"`, e.g. `quezon-infanta`, `cavite-tagaytay` |
| Region ID | `calabarzon` (only one region in scope) |
| Quarter ID | `"{YYYY}-Q{n}"`, `n` ∈ 1..4, e.g. `2026-Q3`. Chronological sort = `year*4 + (n-1)` |
| Score / index | float `0.0`–`1.0`, 2 decimals for display |
| Percentages | integer 0–100 unless noted |
| Dates | ISO‑8601 (`2026-09-02T04:00:00Z`) for timestamps; `YYYY-MM-DD` for calendar dates |
| Timezone | Philippine Standard Time (UTC+8) for all display dates; store UTC |
| Errors | `{ "error": { "code": "string", "message": "human text", "details": {} } }` with appropriate HTTP status |
| Pagination | `?page=1&pageSize=50` → `{ "data": [...], "page": 1, "pageSize": 50, "total": 512 }` |
| Caching | Forecast/SHAP/news are quarter-stable → `Cache-Control: public, max-age=3600` + `ETag`. Admin data → `no-store`. |
| Missing data | If a subject×quarter has no published forecast, return `404` with `error.code = "forecast_not_found"` (distinct from `rejected`, see §K). |

### 0.1 Enums (frozen — the UI has hardcoded branches for these)

```jsonc
// RiskLevel — NOTE: UI type has 4 values but only 2 are ever produced today.
// Decide with the modeling team whether to activate "moderate"/"severe".
RiskLevel = "low" | "moderate" | "high" | "severe"

// Trigger categories — EXACTLY these 5, in this key order. The SHAP
// explainability and the "Why is this Province at Risk?" panel are built on them.
TriggerKey = "market" | "climate" | "employment" | "ofw" | "fishkill"
TriggerLabel = {
  market:     "Market / Prices",
  climate:    "Climate Stress",
  employment: "Employment",
  ofw:        "OFW Remittance",
  fishkill:   "Fish Kill"
}

ReviewStatus   = "Staged" | "Approved" | "Rejected"
RejectionReason = "Data quality issue"
  | "Insufficient signal / limited articles"
  | "Conflicts with field intelligence"
  | "Model anomaly / outlier prediction"
  | "Sensitive context — withhold publication"
  | "Other"
```

### 0.2 Thresholds (frontend constants — send them so the UI stops hardcoding)

| Constant | Value today | Meaning |
|---|---|---|
| `RISK_DISPLAY_CUTOFF` | `0.50` | score ≥ ⇒ **HIGH** pill, else **LOW** |
| `ALERT_THRESHOLD` | `0.60` | score ≥ ⇒ "Risk jumped suddenly" active-alert badge |
| `TRIGGER_RED_CUTOFF` | `20` | a trigger contributing **> 20 %** of the risk renders **red**, else **yellow** (20 % = even share of 5) |
| `LIMITED_SIGNAL_MIN_ARTICLES` | `5` | below this article count ⇒ "Limited signal" badge |

Expose via `GET /api/v1/config` (§E.3) so thresholds are server-controlled.

---

## A. Geography

Static GeoJSON boundary files (`public/ph-provinces.json`,
`public/calabarzon-municities.json`) are **real and can stay as-is** — served
from the frontend's own origin, no API needed. The map also needs the
**self-hosted vector basemap** `public/basemap.pmtiles` (see
[`basemap.md`](basemap.md)); that is an ops/CDN concern, not an API.

You only need to provide the **attribute** data keyed by the IDs in those files.

### A.1 `GET /api/v1/provinces`
List of provinces in scope with current-quarter headline stats.
Consumed by: province ranking card, map colouring, search index.

```jsonc
{
  "data": [
    {
      "id": "quezon",
      "name": "Quezon",
      "centroid": { "lat": 14.0313, "lng": 122.1106 },
      "population": 2122830,
      "povertyRate": 19.5,          // %
      "currentQuarter": "2026-Q3",
      "riskScore": 0.66,            // forecast index for currentQuarter
      "riskLevel": "high",
      "qoqChange": 4.6,             // % change vs previous quarter
      "articleCount": 142           // news articles analysed this quarter
    }
    // ...5 provinces
  ]
}
```
Replaces: `regionsData` in `src/data/mockData.ts` **and** `PROVINCE_QUARTER_DATA`
in `src/data/quarterData.ts` (today these are two disagreeing datasets — unify).

### A.2 `GET /api/v1/provinces/{provinceId}/municipalities`
Consumed by: province drill-down list, municipality colouring on the map.

```jsonc
{
  "data": [
    {
      "id": "quezon-infanta",
      "name": "Infanta",
      "provinceId": "quezon",
      "classification": "Municipality",   // "City" | "Municipality"
      "population": 74200,
      "povertyRate": 12.8,
      "currentQuarter": "2026-Q3",
      "riskScore": 0.71,
      "riskLevel": "high",
      "qoqChange": 2.1,
      "vulnerabilityDriver": "poverty-led" // "poverty-led"|"density-led"|"balanced"
    }
  ]
}
```
Replaces: `municipalitiesData` (hash-derived) in `mockData.ts`.
`vulnerabilityDriver` is currently a frontend heuristic (`getVulnerabilityDriver`,
poverty-norm vs density-norm); either keep it client-side or return it.

---

## B. Forecasting — risk scores

The forecast index is a 3-month-ahead binary food-insecurity risk probability,
`0..1`, published quarterly per province and spatially disaggregated to
municipalities.

### B.1 `GET /api/v1/forecast?scope={scope}&id={id}&quarter={quarterId}`
The single most-used endpoint. `scope` ∈ `region | province | municipality`.
`quarter` optional (defaults to current). Consumed by: the big "Forecast Index"
gauge, the HIGH/LOW pill, the alert badge, map colour, every summary card.

```jsonc
// GET /api/v1/forecast?scope=region&id=calabarzon&quarter=2026-Q3
{
  "scope": "region",
  "id": "calabarzon",
  "name": "CALABARZON",
  "quarter": "2026-Q3",
  "quarterLabel": "Q3 2026 · Jul–Sep",
  "generatedOn": "2026-04-01",       // model run date (3 months before quarter)
  "verificationBy": "2026-09-30",    // date actual will be checked against
  "horizon": "3-month-ahead",
  "riskScore": 0.47,                 // region = mean of the 5 province scores
  "riskLevel": "low",
  "qoqChange": 0.01,                 // absolute delta vs previous quarter
  "isForecast": false,              // true if quarter is in the future
  "isCurrent": true,
  "limitedSignal": false,            // articleCount < LIMITED_SIGNAL_MIN_ARTICLES
  "alert": false,                    // riskScore >= ALERT_THRESHOLD
  "provinceCounts": { "high": 1, "low": 4, "limited": 0 } // region scope only
}
```
For `scope=province` / `municipality`, drop `provinceCounts`, add `provinceId`
(muni scope). Replaces: `getProvinceScore`, `getCalabarzonAverage`,
`getRiskLabel`, `isLimitedSignal`, `getQuarterMeta` in `quarterData.ts`.

### B.2 `GET /api/v1/forecast/timeseries?scope={}&id={}&from={quarterId}&to={quarterId}`
A score per quarter over a range. Consumed by: **Visualization → Province Risk
Level Trend** chart, Forecast History, the trend explanation text.

```jsonc
{
  "scope": "province", "id": "quezon",
  "series": [
    { "quarter": "2025-Q3", "riskScore": 0.58, "riskLevel": "high", "isForecast": false },
    { "quarter": "2025-Q4", "riskScore": 0.61, "riskLevel": "high", "isForecast": false },
    { "quarter": "2026-Q3", "riskScore": 0.65, "riskLevel": "high", "isForecast": false },
    { "quarter": "2026-Q4", "riskScore": 0.66, "riskLevel": "high", "isForecast": true }
  ]
}
```
The trend chart also supports a **single municipality** series — today it's faked
(parent province ± hash offset). Provide real municipal timeseries or document
the disaggregation.

### B.3 Municipality disaggregation (methodology to confirm)
The PDF report text states municipal values are derived from the province
forecast using **PSA poverty incidence (60 %) + population density (40 %)**.
Either compute server-side and expose via B.1/B.2, or return the province
forecast + the weights and let the client disaggregate. **Decision needed.**

---

## C. SHAP Explainability — the 5-trigger breakdown

**This is the core explainability contract. Read carefully.**

The frontend collapsed per-feature SHAP into **5 fixed trigger categories**
(§0.1). For a given subject × quarter it needs each trigger's **share of the
total risk contribution**, where the 5 shares **sum to 1.0 (100 %)**.

Consumed by, all from one source of truth (`getTriggerBreakdown` in
`quarterData.ts`):
- Left panel **"Why is this Province at Risk?"** — ranked list of all 5
- Right panel **"…Summary" SHAP card** — same 5, same numbers, same colours
- **Visualization → "SHAP Explainability"** tab — ranked bar chart + narrative,
  and its PNG/PDF export
- **PDF report** "Why this forecast" section

Rules the UI enforces (keep them server-side so every surface agrees):
1. **Always all 5** categories, even at 0 %. Never omit one.
2. `pct` are integers **summing to exactly 100** (use largest-remainder rounding).
3. Sorted by `pct` descending.
4. Colour: `pct > TRIGGER_RED_CUTOFF (20)` ⇒ red, else yellow. Return it or let
   the client derive it — but the rule is fixed.

### C.1 `GET /api/v1/explainability?scope={}&id={}&quarter={quarterId}`

```jsonc
// GET /api/v1/explainability?scope=region&id=calabarzon&quarter=2026-Q3
{
  "scope": "region",
  "id": "calabarzon",
  "quarter": "2026-Q3",
  "baseline": 0.32,              // model base rate (optional, for waterfall)
  "riskScore": 0.47,            // baseline + sum(signedContribution) should ≈ this
  "triggers": [
    {
      "key": "market",
      "label": "Market / Prices",
      "share": 0.41,             // float, the 5 sum to 1.0
      "pct": 41,                 // int, the 5 sum to 100
      "signedContribution": 0.062, // optional: + pushes risk up, - pulls down
      "featureCount": 7,         // optional: # of model features rolled up here
      "topFeatures": ["Engel coefficient", "Food CPI YoY"] // optional
    },
    { "key": "climate",     "label": "Climate Stress", "share": 0.22, "pct": 22 },
    { "key": "ofw",         "label": "OFW Remittance", "share": 0.14, "pct": 14 },
    { "key": "employment",  "label": "Employment",     "share": 0.13, "pct": 13 },
    { "key": "fishkill",    "label": "Fish Kill",      "share": 0.10, "pct": 10 }
  ],
  "narrative": "This SHAP explainability view breaks CALABARZON's Q3 2026 food-insecurity risk into all five trigger categories, scaled so the shares add up to 100%. Ranked contribution this quarter: Market / Prices 41%, Climate Stress 22%, OFW Remittance 14%, Employment 13%, Fish Kill 10%. Market / Prices sits above the 20% even-share line and is flagged red; the rest are yellow. Shares are AI estimates from news-signal and indicator data, recomputed every quarter."
}
```

- `narrative` is optional — the frontend can compose it (`explainTriggerBreakdown`).
  If you return it, it's used verbatim in the chart's "What this shows" box and
  the PDF. Keep it plain ASCII (goes into jsPDF).
- The **feature → trigger-category mapping** is your responsibility. Document it
  (e.g. Engel coefficient, food CPI → `market`; FPSI/typhoon signals → `climate`;
  dependency ratio, income decile → `employment`; remittance indicators → `ofw`;
  fish-kill news flags → `fishkill`).
- Must resolve for **every** subject × every quarter in the timeline (past +
  current + forecast), because Forecast History and the Visualization tab iterate
  all quarters.

### C.2 (Optional) `GET /api/v1/explainability/features?scope={}&id={}&quarter={quarterId}`
Raw per-feature SHAP values, if you want to power a future feature-level
waterfall (the orphaned `WhatIfSandbox.tsx` / `pdfReport.ts` waterfall). Not
consumed by any live screen today.

```jsonc
{ "baseline": 0.32,
  "features": [
    { "feature": "Engel coefficient", "value": 0.192, "triggerKey": "market" },
    { "feature": "Income decile",     "value": -0.122, "triggerKey": "employment" }
  ] }
```

---

## D. Trigger Composition (province detail)

The province drill-down has a separate **"Trigger Composition"** stacked bar
(`TriggerCompositionBar`, `getTriggerComposition`). Today it's per-province only,
not quarter-aware, and uses its own colours. Recommendation: **fold it into
C.1** — it's the same 5 shares. If kept separate:

### D.1 `GET /api/v1/trigger-composition?id={provinceId}&quarter={quarterId}`
```jsonc
{ "id": "quezon", "quarter": "2026-Q3",
  "composition": [
    { "key": "market", "share": 0.32 },
    { "key": "climate", "share": 0.34 },
    { "key": "employment", "share": 0.16 },
    { "key": "ofw", "share": 0.10 },
    { "key": "fishkill", "share": 0.08 }
  ] }
```

---

## E. Timeline / quarters

### E.1 `GET /api/v1/quarters`
The set of quarters the UI shows on the time slider + Forecast History. The
frontend derives `current` from the browser clock today (`buildQuarters()` uses
`new Date()` → e.g. September ⇒ `Q3`); it should instead trust the server.

```jsonc
{
  "current": "2026-Q3",          // quarter containing "now" (PST), clamped to range
  "serverTime": "2026-09-02T11:13:09+08:00",
  "quarters": [
    { "id": "2025-Q1", "year": 2025, "q": 1, "label": "Q1", "monthsLabel": "Jan–Mar",
      "state": "actual" },       // "actual" | "current" | "forecast"
    { "id": "2026-Q3", "year": 2026, "q": 3, "label": "Q3", "monthsLabel": "Jul–Sep",
      "state": "current", "generatedOn": "2026-04-01", "verificationBy": "2026-09-30" },
    { "id": "2027-Q1", "year": 2027, "q": 1, "label": "Q1", "monthsLabel": "Jan–Mar",
      "state": "forecast", "generatedOn": "2026-10-01", "verificationBy": "2027-03-31" }
  ]
}
```
Rule: quarters before `current` = `actual` (history), `current` = the live one,
after = `forecast`. Replaces `buildQuarters()` / `currentQuarterId()` /
`getQuarterMeta()` in `QuarterTimeSlider.tsx` + `quarterData.ts`.

### E.2 `GET /api/v1/history?scope=region&id=calabarzon`
Backs the **Forecast History** modal. One row per non-future quarter.

```jsonc
{
  "rows": [
    {
      "quarter": "2026-Q3", "label": "Q3 2026", "state": "current",
      "riskScore": 0.47, "riskLevel": "low",
      "qoqChange": 0.01,
      "topTrigger": { "key": "market", "label": "Market / Prices", "pct": 41 },
      "generatedOn": "2026-04-01", "verificationBy": "2026-09-30"
    }
    // ...older quarters, newest first
  ]
}
```
Today the modal composes this client-side from B/C/E. A single endpoint is
cleaner but optional.

### E.3 `GET /api/v1/config`
```jsonc
{
  "thresholds": { "riskDisplayCutoff": 0.5, "alertThreshold": 0.6,
                  "triggerRedCutoff": 20, "limitedSignalMinArticles": 5 },
  "riskLevelsActive": ["low", "high"],   // or all 4 if activated
  "region": "calabarzon"
}
```

---

## F. News Articles & Topic Analysis

Right panel **"News Articles Analyzed"** card: an article count, a topic-mix bar,
and a clickable article list. Today all of it (`SAMPLE_ARTICLES`, `NEWS_TOPICS`)
is hardcoded and **identical for every province/quarter**.

### F.1 `GET /api/v1/news?scope={}&id={}&quarter={quarterId}&page=1&pageSize=20`
```jsonc
{
  "scope": "province", "id": "quezon", "quarter": "2026-Q3",
  "articleCount": 142,             // total analysed for this subject×quarter
  "topics": [                       // shares of the analysed corpus, sum ~100
    { "key": "food_prices", "label": "Food prices", "pct": 42 },
    { "key": "typhoon",     "label": "Typhoon",     "pct": 31 },
    { "key": "workers",     "label": "Workers",     "pct": 18 },
    { "key": "ofw",         "label": "OFW",         "pct": 6  },
    { "key": "fishkill",    "label": "Fish kill",   "pct": 3  }
  ],
  "data": [
    {
      "id": "art_8f2c1",
      "title": "Rice still above P50/kilo levels; no dip in sight",
      "source": "Inquirer.net",
      "date": "2026-08-12",
      "url": "https://newsinfo.inquirer.net/...",
      "excerpt": "Retail rice prices remain stubbornly high...",
      "topicKey": "food_prices",
      "sentiment": "negative",       // optional
      "triggerKey": "market"          // optional: which trigger it informs
    }
  ],
  "page": 1, "pageSize": 20, "total": 142
}
```
Notes: `url` is an external link opened in a new tab. `topics[].key` set can be
open-ended; the UI renders whatever comes back. Article corpus should match the
`articleCount` that B.1/A.1 report for the same subject×quarter.

---

## G. Charts

All charts are rendered client-side with `recharts` from the endpoints above:

| Chart | Screen | Data source |
|---|---|---|
| Forecast Index gauge / slider | dashboard left | B.1 |
| Province Risk Level Trend (line) | Visualization tab A | B.2 (province or "All" = 5 series; municipality = single series) |
| SHAP Explainability (ranked bar) | Visualization tab B | C.1 (`pct` per trigger, `color` rule, `narrative`) |
| "Why is this Province at Risk?" bars | dashboard left | C.1 |
| Trigger Composition (stacked bar) | province detail | C.1 (or D.1) |
| News topic mix (bars) | dashboard right | F.1 `topics` |
| Forecast History table | History modal | E.2 |

**Exports (PNG / PDF)** are 100 % client-side (`html2canvas` + `jspdf` snapshot of
the chart DOM, including the narrative box). **No server rendering endpoint is
required.** If you later want server-generated report PDFs, that's §M.

---

## H. Search / Autocomplete

Map search box: substring match over province + municipality names; selecting a
result focuses the map. Currently a client-side filter over the mock arrays +
a `window` CustomEvent (`aipheed:focus-region`).

### H.1 `GET /api/v1/search?q={text}&limit=10`
```jsonc
{ "data": [
  { "type": "province",     "id": "quezon",         "name": "Quezon" },
  { "type": "municipality", "id": "quezon-infanta", "name": "Infanta",
    "provinceId": "quezon", "provinceName": "Quezon" }
] }
```
Optional — a client-side index built from A.1/A.2 is fine at this scale (~145
items). Provide the endpoint only if the dataset grows.

---

## J. Authentication & Authorization

**Today: no real auth.** Two inconsistent client-side checks
(`Login.tsx` = any `@calabarzon.da.gov.ph` email, password ignored;
`TopNavbar.tsx` `AdminLoginModal` = hardcoded `admin@calabarzon.da.gov.ph` /
`admin2026`). Session = `sessionStorage.aipheed_user = { email, role: "admin" }`.
The `/admin` route is gated purely client-side and is trivially bypassable via
devtools.

Provide real auth. Both frontend entry points must be unified against it.

### J.1 `POST /api/v1/auth/login`
```jsonc
// request
{ "email": "analyst@calabarzon.da.gov.ph", "password": "..." }
// 200
{ "token": "<jwt>", "expiresIn": 3600,
  "user": { "id": "u_123", "email": "analyst@calabarzon.da.gov.ph",
            "name": "J. Dela Cruz", "role": "admin" } }
// 401 -> { "error": { "code": "invalid_credentials", "message": "..." } }
```
Rules: restrict to `@calabarzon.da.gov.ph` (config-driven domain allowlist);
enforce the password; issue a short-lived JWT (or httpOnly cookie session).

### J.2 `GET /api/v1/auth/me`
Returns `user` for a valid token, `401` otherwise. Frontend calls this on
`/admin` mount instead of trusting `sessionStorage`.

### J.3 `POST /api/v1/auth/logout`
Invalidate the token/session. Frontend clears its copy regardless.

### J.4 Authorization
`role` ∈ `admin` (only role today). All `/api/v1/admin/**` endpoints require a
valid `admin` token; return `401` (no token) / `403` (wrong role).

---

## K. Admin — Review & Publication Pipeline

The admin console has a review queue and a rejection mechanism. **The rejection
log has a public-facing effect:** a rejected province×quarter renders as
"No forecast available" on the public map. Today it's `localStorage`
(`aipheed_rejections`) + a same-tab CustomEvent — not shared across
devices/users.

### K.1 `GET /api/v1/admin/review?quarter={quarterId}&status={ReviewStatus}`
Auth: admin. The review queue.
```jsonc
{ "data": [
  { "id": "rv_2026Q3_quezon",
    "provinceId": "quezon", "province": "Quezon",
    "quarter": "2026-Q3",
    "riskScore": 0.66, "riskLevel": "high",
    "status": "Staged",                     // Staged | Approved | Rejected
    "generatedOn": "2026-04-01",
    "updatedAt": "2026-04-05T02:11:00Z",
    "updatedBy": "u_123"
  }
] }
```
Replaces the synthetic `ReviewItem[]` generated in `Admin.tsx`.

### K.2 `POST /api/v1/admin/review/{id}/approve`
Auth: admin. → `{ "id": "...", "status": "Approved", "updatedAt": "...", "updatedBy": "..." }`

### K.3 `POST /api/v1/admin/review/{id}/reject`
Auth: admin.
```jsonc
// request
{ "reason": "Model anomaly / outlier prediction",   // one of RejectionReason
  "notes": "Score spike not seen in field reports." } // optional free text
// 200
{ "id": "...", "status": "Rejected", "rejection": { ...see K.5 row... } }
```

### K.4 `POST /api/v1/admin/review/{id}/undo`  (a.k.a. "Undo Publishing")
Auth: admin. Moves `Approved`/`Rejected` back to `Staged`; if it was rejected,
removes the matching rejection-log entry.

### K.5 `GET /api/v1/rejections?quarter={quarterId}`
**Public** (no auth) — the map needs it to grey out provinces. Also feeds the
admin "Rejection Log" tab.
```jsonc
{ "data": [
  { "provinceId": "quezon", "province": "Quezon", "quarter": "2026-Q4",
    "reason": "Sensitive context — withhold publication",
    "notes": null,
    "rejectedBy": "u_123", "rejectedByName": "J. Dela Cruz",
    "timestamp": "2026-06-30T08:00:00Z" }
] }
```
Frontend keys on `(provinceId, quarter)` — at most one active rejection per pair.
Replaces `src/lib/rejections.ts` (`loadRejections`, `addRejection`,
`removeRejection`, `isRejected`, `getRejectedProvinceIdsForQuarter`). Keep the
push/refresh semantics: after any K.2–K.4 call the frontend re-fetches K.5 (or
subscribe via SSE/websocket if you want live multi-admin updates).

### K.6 `DELETE /api/v1/admin/rejections/{provinceId}/{quarter}`  ("Restore")
Auth: admin. Removes a rejection so the forecast publishes again.

### K.7 Audit trail
Every approve/reject/undo should persist `{ actor, action, subject, quarter,
reason?, notes?, timestamp }`. Optional `GET /api/v1/admin/audit` for a future
audit view.

---

## L. Feedback — System Usability Scale (SUS) survey

Public "Send Feedback" modal: 10 standard SUS Likert questions + demographics +
two free-text fields. Scored client-side (standard SUS formula, 0–100). Today it
`push`es to `localStorage.aipheed_feedback_v2`; the admin Feedback tab only sees
submissions from its own browser.

### L.1 `POST /api/v1/feedback`
**Public** (no auth).
```jsonc
{
  "demographics": {
    "fullName": "", "email": "", "agency": "", "designation": "",
    "age": "26-35",          // "≤25"|"26-35"|"36-45"|"46-60"|">60"
    "sex": "Female",         // "Male"|"Female"
    "clientType": "Government", // "Citizen"|"Business"|"Government"|"Others"
    "province": "Quezon",    // free-form today; align with province list
    "municipality": "Lucena"
  },
  "answers": { "0": 4, "1": 2, "2": 5, "3": 2, "4": 4,
               "5": 2, "6": 5, "7": 1, "8": 4, "9": 2 }, // qIndex -> 1..5
  "liked": "The map is clear.",
  "improvements": "More municipal detail."
}
// 201
{ "id": "fb_a1b2", "date": "2026-09-02T11:20:00Z", "score": 78 }
```
Server should recompute `score` (don't trust the client): positively-worded
items (0-indexed `0,2,4,6,8`) contribute `answer-1`; negatively-worded items
(`1,3,5,7,9`) contribute `5-answer`; sum the 10, then `× 2.5` → 0–100.
The 10 SUS question strings are in `FeedbackModal.tsx` (`SUS_QUESTIONS`) — keep
them versioned so historical responses stay interpretable.

### L.2 `GET /api/v1/admin/feedback?page=1&pageSize=50`
Auth: admin. All submissions + aggregates.
```jsonc
{
  "summary": { "count": 37, "meanScore": 74.2, "medianScore": 77.5,
               "byClientType": { "Government": 21, "Citizen": 9, "Business": 4, "Others": 3 } },
  "data": [ { "id": "fb_a1b2", "date": "...", "score": 78,
              "demographics": { ... }, "answers": { ... },
              "liked": "...", "improvements": "..." } ],
  "page": 1, "pageSize": 50, "total": 37
}
```
Replaces `loadFeedback()` in `FeedbackModal.tsx` + the admin Feedback reader.

---

## M. Optional / future

### M.1 Server-generated PDF report
`src/lib/pdfReport.ts` is a complete multi-page report generator
(`downloadRegionReport` / `downloadProvinceReport` / `downloadMunicipalityReport`)
**not currently wired to any button**. If you productionise it, it needs, per
scope × quarter: the forecast (B.1), the 5-trigger explainability (C.1), a
municipal breakdown table (A.2 + B.2), top news evidence (F.1), and recommended
actions (new — no data source exists yet). Could be a client build (as written)
or `GET /api/v1/report?scope=&id=&quarter=&format=pdf` returning a rendered file.

### M.2 What-if simulation
`src/components/WhatIfSandbox.tsx` (orphaned) adjusts feature sliders and expects
a recomputed score. Needs a live model endpoint:
`POST /api/v1/simulate { scope, id, quarter, featureOverrides: {..} } -> { riskScore, triggers[] }`.

### M.3 Real-time multi-admin updates
If two analysts review concurrently, K.1/K.5 changes should push. SSE
(`GET /api/v1/admin/stream`) or websockets; otherwise the frontend polls K.5.

---

## N. Frontend consumer → endpoint → mock-it-replaces map

| Frontend surface | Component | Endpoint | Currently faked by |
|---|---|---|---|
| Forecast Index gauge, HIGH/LOW pill, alert badge | `RfiiScoreCardBody`, `ShapNarrativeCardBody` | B.1 | `getCalabarzonAverage`, `getProvinceScore`, `getRiskLabel`, `isLimitedSignal` (`quarterData.ts`) |
| Province ranking card | `ProvinceRankingCardBody` | A.1 | `regionsData` (`mockData.ts`) |
| Map province/municipality colours + drill-down | `PhilippineMap`, `LeftPanel` | A.1, A.2 | `regionsData`, `municipalitiesData` |
| "Why is this Province at Risk?" (5 triggers) | `RiskDriversCardBody` | C.1 | `getTriggerBreakdown` (`quarterData.ts`) |
| Right-panel SHAP card (5 triggers) | `ShapNarrativeCardBody` | C.1 | `getTriggerBreakdown` |
| Trigger Composition stacked bar | `TriggerCompositionBar` | C.1 / D.1 | `getTriggerComposition` |
| Visualization → Trend line | `Visualization.tsx` `TrendView` | B.2 | `PROVINCE_QUARTER_DATA.scoresByQuarter` |
| Visualization → SHAP Explainability | `Visualization.tsx` `ShapView` | C.1 | `getTriggerBreakdown` + `explainTriggerBreakdown` |
| News count / topics / articles | `NewsArticlesCardBody` | F.1 | `SAMPLE_ARTICLES`, `NEWS_TOPICS`, `PROVINCE_QUARTER_DATA.articles` |
| Time slider + Current/Forecast/Actual state | `QuarterTimeSlider` | E.1 | `buildQuarters`, `currentQuarterId` |
| Forecast History modal | `ForecastHistoryModal` | E.2 (or B.2+C.1+E.1) | client-composed from `quarterData.ts` |
| Data table + CSV export | `Data.tsx` | B.2 (+ F.1 for counts) | `PROVINCE_QUARTER_DATA`, `trigger`/`status` name-hash |
| Map search | `MapSearch` | H.1 (optional) | client filter over mock arrays |
| Admin login / gate | `Login.tsx`, `TopNavbar` `AdminLoginModal`, `Admin.tsx` | J.1–J.4 | `sessionStorage.aipheed_user`, hardcoded creds |
| Admin review queue + approve/reject/undo | `Admin.tsx` | K.1–K.4, K.6 | synthetic `ReviewItem[]` |
| "No forecast available" on public map | `MapDashboard`, `PhilippineMap` | K.5 | `src/lib/rejections.ts` + `localStorage` |
| Feedback submit | `FeedbackModal` | L.1 | `localStorage.aipheed_feedback_v2` |
| Admin feedback viewer | `Admin.tsx` Feedback tab | L.2 | `loadFeedback()` |

---

## O. Cross-cutting requirements

1. **Unify the two province datasets.** `regionsData` (`mockData.ts`) and
   `PROVINCE_QUARTER_DATA` (`quarterData.ts`) currently give different scores for
   the same province. The API must be the single source; delete both mocks.
2. **Decide `RiskLevel` cardinality.** The UI type/colours/labels support
   `low|moderate|high|severe` but only `low|high` are ever produced. Either
   activate the middle bands (define cutoffs) or formally drop them.
3. **Every subject × every quarter must resolve** for forecast, explainability
   and news — the History modal and Visualization tab iterate the full quarter
   list including forecast quarters.
4. **`pct` integrity:** trigger `pct` must sum to exactly 100 per response;
   news `topics` pct should sum to ~100 (±1 ok).
5. **Stable values.** Anything the frontend renders must be deterministic for a
   given subject×quarter (today several mock fields use `Math.random()` at page
   load and change every refresh).
6. **Rejection is not the same as missing.** `404 forecast_not_found` (never
   generated) vs. a K.5 rejection entry (generated then withheld by an admin) —
   the UI shows different messages.
7. **Server owns "now".** Ship `serverTime` + `current` (E.1) so the timeline
   doesn't depend on the client clock.
8. **Auth is real and enforced server-side.** No client-only gates.
9. **CORS** for the frontend origin(s); admin endpoints `no-store`.

---

## P. Open questions for the modeling / data team

- Exact **feature → trigger-category** mapping for C.1 (and the rollup math:
  share of Σ|contribution|? of Σ positive contribution?).
- Municipality forecast: real per-muni model output, or the 60/40
  poverty/density disaggregation from the province score? (B.3)
- Are `moderate` / `severe` risk bands in scope? If so, the cutoffs.
- Should `region` score be the mean of province scores (current UI behaviour) or
  its own model output?
- News: is the article corpus per province, or region-wide tagged by province?
  Does `articleCount` include non-geolocated articles?
- Forecast `generatedOn` / `verificationBy` — real model-run + validation dates
  from a run log, or keep the "3 months before/after quarter" convention?
- Retention: how many historical quarters should `/history` and `/quarters`
  expose? (Data currently spans `2025-Q1`…`2027-Q1`.)
