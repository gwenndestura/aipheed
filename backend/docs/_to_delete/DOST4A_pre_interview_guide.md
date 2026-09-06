# Pre-Interview Key-Informant Guide — DOST Region IV-A
### aiPHeed: A Food-Insecurity Risk-Indicator System for CALABARZON

**Instrument type:** Pre-development (foundational) key-informant interview
**Respondent:** DOST Region IV-A (and, where relevant, DOST-FNRI / RNC nutrition focal persons)
**Purpose of this interview:** This is the *foundation* interview. Its outputs define what
aiPHeed must become — the problem it should address, the indicators experts consider valid,
and the outputs that would make it credible and usable as a **reference tool**. The product
is built *from* this conversation, not shown for approval after the fact.

---

## Framing statement (read aloud before starting)

> aiPHeed is a **research and prediction tool only**. It produces **potential risk
> indicators** for food insecurity across CALABARZON's provinces, cities, and municipalities.
> It **does not** issue official food-insecurity classifications and **cannot replace** the
> official assessment procedures of government agencies. We are asking DOST Region IV-A to
> act as the **expert validator** of the system's design and outputs — confirming what is
> sound, correcting what is not, and telling us what the tool would need to be trustworthy.
> Whether or how any agency acts on the outputs is entirely outside the scope of this study.

*Consent:* Confirm permission to take notes / record, and permission to cite the office
(anonymised on request) in the thesis. Note expected duration (~45–60 min).

---

## SECTION 0 — Respondent & mandate (warm-up, 5 min)

0.1 Please describe your office/unit and your role in it.
0.2 What is your unit's mandate with respect to community vulnerability, food security, or
    nutrition in Region IV-A?
0.3 Which provinces / LGUs fall under your monitoring responsibility?
0.4 Roughly how far ahead does your unit try to anticipate food-security or welfare problems
    (weeks? months? a quarter? reactive/after-the-fact)?

---

## SECTION A — Current monitoring tools & practice *(asked first, by request)*

*Goal: establish the baseline the thesis is positioned against — what exists, and where the gaps are.*

A.1 **What tools, systems, or platforms does your office currently use to monitor community
    vulnerability and the welfare of communities across the provinces?**
    (Prompt for names: dashboards, MIS, GIS, spreadsheets, survey platforms, DRRM systems,
    nutrition surveillance systems, etc.)
A.2 For each tool named — is it **real-time / periodic / survey-based**? How often is it updated?
A.3 What is the **finest geographic level** you can actually monitor down to — region, province,
    city/municipality, or barangay?
A.4 How does information reach you today when a food or welfare problem is developing in a
    community? (official reports, field officers, news, LGU escalation, social media…)
A.5 What are the **biggest gaps or frustrations** with the current tools? (timeliness, coverage
    of far-flung LGUs, granularity, manual effort, siloed data…)
A.6 Is any current tool **predictive / early-warning**, or are they mostly descriptive
    (reporting what already happened)?
A.7 If a reliable *3-month-ahead* risk signal existed for each province/LGU, where in your
    workflow would it fit — and where would it *not* be appropriate to use it?

> *Researcher note (defends: problem statement & research gap). A.5–A.7 establish that a
> forward-looking, LGU-level indicator addresses a real, expert-confirmed gap.*

---

## SECTION B — Data sources & indicators experts trust

*Goal: validate the government/official data sources feeding the model, and surface any we're missing.*

B.1 When your office assesses whether a community is food-insecure or vulnerable, **which data
    sources do you rely on?** (See Appendix 1 for the sources aiPHeed currently uses — ask them
    to react to each.)
B.2 Which **official indicators** do you consider the most reliable signals of emerging food
    insecurity? (e.g., food inflation/CPI, hunger incidence, rainfall/weather anomalies, crop
    or fishery damage, remittances, prevalence of undernutrition…)
B.3 Of the sources in Appendix 1 (PSA food CPI & commodity prices, BSP remittances, PAGASA
    rainfall, SWS hunger, DOST-FNRI ENNS/FIES nutrition), **which do you trust most, and are any
    you would weight less or question?**
B.4 Are there **authoritative sources we are not using** that you would expect a credible system
    to include? (DA/BFAR damage reports, DSWD DROMIC/4Ps, DILG, PhilRice, LGU nutrition action
    plans, DRRM bulletins…)
B.5 At what **time lag** do these official statistics typically become available to you
    (weeks, a quarter, a year)? — *This validates the thesis's use of news as a faster proxy.*
B.6 Do you agree that **local news reporting** can act as an *early* signal of food-security
    stress *before* official statistics are published? Where would that assumption break down?

> *Researcher note (defends: data-source selection, secondary-vs-primary data design, and the
> core justification for a news-based leading indicator).*

---

## SECTION C — Validating the food-insecurity concept & triggers

*Goal: expert-validate the classification scheme (dimensions A–F, the 10 hypotheses, severity).*

C.1 aiPHeed classifies each news signal into **six food-security dimensions** (see Appendix 2).
    Do these dimensions reasonably capture how food insecurity manifests in CALABARZON? Anything
    missing or mislabeled?
C.2 We reduce these to **10 detectable "trigger" hypotheses** (Appendix 2). For your region,
    which of these are the **most common / most important** drivers of food insecurity?
    Which are rare or irrelevant here?
C.3 Are there **CALABARZON-specific triggers** we should add? (e.g., Taal Volcano activity,
    Laguna Lake / Taal Lake fish kills & red tide, ASF/bird flu, typhoon corridors in Quezon,
    agricultural-land conversion in Cavite/Laguna.)
C.4 We tag severity as **High / Medium / Low** (Appendix 2). Do our severity cut-offs match how
    your office would triage the same event?
C.5 Which **commodities** matter most for food security in each province (rice, fish/milkfish,
    vegetables, poultry/hog…)? — *validates commodity-price features.*
C.6 Are there local terms / Filipino/Tagalog phrasings for hunger and food stress our text
    classifier should recognise? *(gutom, walang makain, kakulangan ng bigas, ayuda…)*

> *Researcher note (defends: the 10-hypothesis taxonomy, the A–F dimension mapping, and
> region-specific validity of the trigger design).*

---

## SECTION D — Validating aiPHeed's method & design choices

*Goal: get expert sign-off (or challenge) on the modelling decisions you must defend.*

D.1 aiPHeed produces a **quarterly, 3-month-ahead risk ranking** for the 5 provinces (extending
    toward city/municipality/barangay). Is a **quarterly cadence** and **one-quarter lead time**
    operationally meaningful for your work, or would you need it faster/finer?
D.2 We rank/relative-risk provinces rather than declaring "X% are food-insecure." Is a
    **relative risk ranking** (who is most at risk *now vs. last quarter*) the right framing for
    a reference tool, versus an absolute prevalence number?
D.3 The system raises an **alert only on a sudden rise** in predicted risk (a quarter-on-quarter
    jump above a floor), not on steady/known-high levels. Does an **early-warning-on-deterioration**
    logic fit how you'd want to be notified? What would make an alert *actionable* vs. noise?
D.4 For scoring news, we use a **zero-shot** language model (no hand-labeled Filipino training
    data) so the pipeline is reproducible and can be re-run without retraining. Does the absence
    of manual labeling raise any concern for you, or is transparency/reproducibility the priority?
D.5 **Ground truth / validation:** we plan to have DOST 4A experts review a sample of the
    system's tagged articles (the annotation exercise) to measure accuracy. Is your office able
    to serve as that validator, and who would be the right reviewer(s)?
D.6 When a province/LGU has **very few news articles**, the system flags the output as
    `LIMITED_SIGNAL` rather than pretending confidence. Is that honest treatment of data-poor
    areas acceptable to you? How should low-signal LGUs be handled?
D.7 What would it take for your office to consider a risk indicator like this **credible enough
    to look at** (not act on) — accuracy level, transparency, source citation, expert validation?

> *Researcher note (defends: forecast horizon, ranking-vs-classification framing, alert logic,
> zero-shot justification, validation strategy, and honest handling of data sparsity.)*

---

## SECTION E — Output, usability & the product (built *from* this interview)

*Goal: capture concrete product requirements — these directly shape what gets built.*

E.1 If you opened an aiPHeed dashboard, **what would you want to see first** — a map, a ranked
    list of at-risk LGUs, a trend line, the driving triggers, the source articles?
E.2 For each risk score, how important is it that the tool **shows *why*** (which triggers/sources
    drove it) versus just the number?
E.3 What geographic view is most useful — **province, city/municipality, or barangay** heat map?
E.4 How should the tool present **uncertainty and its "reference-only" status** so it is never
    mistaken for an official classification?
E.5 What **outputs/exports** would be useful (a quarterly brief, a watchlist, downloadable data,
    the underlying article list for verification)?
E.6 Who in the regional ecosystem should be able to *view* such a tool, and what should it
    **never** be used for?
E.7 Is there a **single feature** that, if present, would make this genuinely worth looking at —
    and one that would make you distrust it?

> *Researcher note (defends: significance of the study & product-design requirements; every
> answer here becomes a concrete build item.)*

---

## SECTION F — Ethics, limitations & governance

F.1 What **risks or harms** could a food-insecurity risk indicator create if misread (stigma to
    an LGU, misallocation, political misuse)? How do we guard against them?
F.2 Any **data-privacy / data-sharing** constraints we must respect (news is public, but any
    official data)?
F.3 Are you comfortable with the positioning that DOST 4A is the **validator of the design**,
    with **no** implied endorsement of deployment or operational use?
F.4 What would you flag as the **main limitation** an examiner should know about this approach?

---

## Closing (5 min)

- Summarise back the 3–4 most important things you heard (validation checkpoint).
- Ask: *"Is there anything we didn't ask that we should have?"*
- Confirm: willingness to serve as annotation validator (Section D.5), preferred reviewer(s),
  and any documents/data they can share.
- Thank them; confirm how findings will be cited.

---
---

# APPENDIX 1 — Government / official data sources used in the thesis
*(Show this and ask the respondent to react — trust, gaps, additions. Ref: Section B.)*

### A. Structured "primary" data (feeds the LightGBM forecast model)

| # | Indicator (feature) | Source agency | What it captures |
|---|---|---|---|
| 1 | Food CPI & food inflation, YoY (`food_cpi`, `food_cpi_yoy`, lags) | **PSA** (Philippine Statistics Authority) | Cost/affordability of food |
| 2 | Food-minus-headline inflation gap (`food_minus_headline_yoy`) | **PSA** | Food stress relative to general prices |
| 3 | Commodity retail prices — livestock, leafy veg, fruit veg (`commodity_*`) | **PSA / DA Bantay Presyo** | Price of key food commodities |
| 4 | Rainfall anomaly % + acceleration (`rainfall_anomaly_pct`, lags) | **PAGASA** | Climate/weather shock to production |
| 5 | OFW remittances, YoY % (`ofw_remit_yoy_pct`, lag) | **BSP** (Bangko Sentral ng Pilipinas) | Household food-purchasing power |

### B. Label / ground-truth basis (defines what "food-insecure quarter" means)

| Signal | Source | Role |
|---|---|---|
| Self-rated **hunger incidence** | **SWS** (Social Weather Stations) | Component of the composite `label_stress` |
| **Food CPI deviation** | **PSA** | Component of the composite `label_stress` |
| **FIES / ENNS** nutrition & food-expenditure benchmarks | **PSA (FIES)** + **DOST-FNRI (ENNS)** | Anchors weak-supervision labels |

### C. Secondary data — the news corpus (scored by the NLP classifier)

- **News indexes / harvesters:** GDELT Project (V2/BigQuery), Event Registry, GNews,
  Common Crawl, Wayback Machine — used to collect CALABARZON food-security news, 2020–2026.
- **Geography:** re-geocoded to a **142-LGU PSGC gazetteer** (PSA PSGC) → province → city/
  municipality → barangay.

### D. Candidate sources to ask DOST 4A about adding (Section B.4)

DA / BFAR crop & fishery damage reports · DSWD DROMIC & 4Ps · DILG · PhilRice ·
DOST-PAGASA seasonal outlooks · LGU Nutrition Action Plans · RDRRMC bulletins ·
DOST-FNRI regional nutrition surveillance.

---

# APPENDIX 2 — Trigger compositions (validate in Section C)

### 2.1 — The six food-security dimensions (A–F)

| Code | Dimension | Typical CALABARZON story |
|---|---|---|
| **A** | Food **availability** (production & fisheries) | crop damage, low harvest, fish kill, ASF |
| **B** | Food **accessibility / affordability** | rice/vegetable price spikes, poverty limiting access |
| **C** | Food **utilization / nutrition** | malnutrition, stunting, feeding programs |
| **D** | Food **stability** (shocks) | typhoon/flood displacement, supply-chain disruption, unrest |
| **E** | **Hunger / food deprivation** (assistance) | relief/ayuda, food packs, hunger relief |
| **F** | **Livelihood** affecting food access | OFW remittance loss, job loss reducing food-buying power |

### 2.2 — The 10 detectable trigger hypotheses (zero-shot NLI) → dimension & event

| ID | Trigger hypothesis (what the classifier detects) | Dim. | Event type |
|---|---|---|---|
| **T1** | Food prices / supply problems / difficulty accessing food | B | food_price_change |
| **T2** | Hunger, malnutrition, nutrition & feeding programs | C | malnutrition_nutrition |
| **T3** | Government food assistance, rice subsidy, relief distribution | E | food_assistance |
| **T4** | Poverty, unemployment, economic hardship of families | B | poverty_hardship |
| **T5** | Roads / transport / storage problems affecting food supply | D | supply_disruption |
| **T6** | Farmland loss, crop damage, reduced harvests | A | crop_production_loss |
| **T7** | Evacuation / displacement of families due to disaster | D | disaster_displacement |
| **T8** | Strikes, protests, unrest disrupting food or livelihoods | D | unrest_disruption |
| **T1b** | Fish kills, fishing bans, aquaculture losses | A | fishery_loss |
| **T9** | OFWs / remittances supporting families | F | remittance_shock |

**Relevance gate:** an article counts as food-insecurity-relevant only if it entails a
**CORE** food trigger — **T1, T2, T3, T6, T1b** (prices/supply, hunger/nutrition, food aid,
crops, fish). The five indirect triggers (T4, T5, T7, T8, T9) are kept as *causal-driver
context* but cannot establish relevance on their own. Score = max entailment probability;
relevance threshold = 0.30.

### 2.3 — Severity triggers (High / Medium / Low)

- **High** — crisis, state of calamity, famine, starvation, deaths, thousands affected
- **Medium** — damage, losses, shortage, price surge, displacement, families "hit"/"affected"
- **Low** — mentioned as a risk or minor/localized issue, no major impact stated

### 2.4 — Forecast alert trigger (model output, not text)

An **alert fires only on a sudden rise**: predicted quarter-on-quarter risk increase
**≥ 0.15** **and** current risk **≥ 0.35** (floor). Steady or already-known-high risk is
informational, not an alert — the goal is to flag *deterioration* early. Data-poor LGUs are
flagged `LIMITED_SIGNAL` (< 5 articles) rather than given false confidence.

---

# APPENDIX 3 — One-page technical fact sheet (for defending the thesis)

- **Objective:** Quarterly, one-quarter-ahead **relative risk ranking** of food insecurity for
  the 5 CALABARZON provinces (extending to city/municipality/barangay).
- **Two data streams fused:** (1) structured official indicators (PSA, BSP, PAGASA) and
  (2) a news corpus scored by NLP → a Food-Security Sentiment/Stress signal (`FSSI`) + climate trigger.
- **Text classifier:** `joeddav/xlm-roberta-large-xnli`, **zero-shot** 10-hypothesis NLI —
  chosen for reproducibility (no Filipino food-insecurity training data needed; re-runnable
  without retraining). Precision-first core-hypothesis relevance gate.
- **Forecast model:** **LightGBM**, hyper-tuned with **Optuna**; top features by gain include
  food CPI (+lags/accel), commodity prices, rainfall anomaly, and OFW remittances.
- **Labels:** composite `label_stress` = SWS hunger + PSA food-CPI deviation, anchored to
  FIES/ENNS (weak supervision). **Leakage-controlled** (raw hunger excluded from features;
  time-ordered cross-validation vs. naive-persistence baseline).
- **Coverage:** 2020–2025, 5 provinces × 24 quarters; 142-LGU PSGC geocoding.
- **Positioning:** a **research/reference risk-indicator tool** — *not* an official
  classification; DOST 4A is the **expert validator**, not operator.

*Italic "Researcher note" lines and Appendix 3 are for your own preparation — remove them from
any copy handed to the respondent.*
