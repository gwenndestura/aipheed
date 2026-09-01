# CALABARZON Food-Insecurity News Dataset (audited)

**Rows:** 371 articles · **Period:** 2020-01-01 – 2026-08-21 · **Publishers:** 30
**Provinces:** Batangas 136 · Cavite 92 · Laguna 67 · Quezon 41 · Rizal 28
**Cities/municipalities named:** 52 of 142
**Scope mix:** 113 city/municipality · 251 province · 7 region-wide

> **Regenerating:** `build_final_dataset.py` now assembles the candidate pool and
> hands it to the audit, so running it reproduces this dataset exactly. Running
> the three audit stages directly does the same without re-collecting:
> `audit_stage1_repair.py` → `audit_stage2_rebuild.py` → `audit_stage3_validate.py`.
>
> This README is republished by stage 2 from `audit_review/README_audited.md` —
> edit that copy, not this one.
>
> **New articles are held out, not auto-added.** When collection turns up
> article_ids with no verdict in `audit_review/decisions.tsv`, they are excluded
> and listed in `audit_review/_awaiting_review.csv`. Review them, add verdicts,
> and re-run stage 2. Nothing enters the dataset without having been read.

## What this is

News articles that provide defensible evidence about food insecurity — or one of
its determinants — in Region IV-A (CALABARZON). Every row satisfies three
conditions, each verified by reading the article rather than by keyword match:

1. a substantive food-insecurity connection,
2. support for at least one of the thesis's 10 hypotheses, and
3. a location resolving to a CALABARZON province or city/municipality — or, for
   the 7 region-wide rows, an explicit CALABARZON-level subject.

## How the current version was produced

The 1,250-row pre-audit snapshot (`audit_review/dataset_pre_audit.csv`) was
rebuilt by a full audit. 879 rows were removed and every surviving row was read
and re-labelled by hand. Every removal is logged with its reason in
`calabarzon_dataset_dropped_audit.csv`.

### Stage 1 — structural repair (1,250 → 965)

The pool was dominated by Google-News RSS rows whose metadata had never been
normalised. Repaired in place:

* **`publication_date`** → ISO `yyyy-mm-dd`. 813 of 1,168 values had been
  unparseable: 554 were RFC-822 stamps and 210 were *truncated* to 10 characters
  (`"Sun, 27 Oc"`). Truncated stamps were recovered where weekday + day + month
  prefix identify exactly one date inside the study window; 102 that stayed
  ambiguous were dropped rather than guessed.
* **`news_source`** → the real publisher. 764 rows had recorded
  `news.google.com`; the outlet was recovered from the title suffix and the
  upstream corpus domain.
* **`title`** → publisher suffix stripped (hyphen, en-dash and em-dash forms),
  plus a text-hygiene pass: tokeniser artefacts (`"P3 . 06"` → `"P3.06"`,
  `"13 , 500"` → `"13,500"`, space before punctuation), aggregator section tails
  (`"| Photos | GMA News Online"`), trailing CMS story ids (`"… Typhoon Leon 2059"`,
  kept when the trailing number is a real date), `"Covid 19"` → `"COVID-19"`, and
  mojibake from UTF-8 bytes decoded as CP1252 (`"â€“"` → `"–"`).
* **`content_lead`** → 764 rows held the Google RSS HTML wrapper
  (`<a href=…>title</a>`) rather than article text; that markup was cleared. The
  surviving leads were de-mojibaked and stripped of CMS boilerplate, inline
  `READ:` cross-promos, sub-editor sign-off initials (`/jpv`) and embedded player
  URLs. Leads the fetchers cut mid-sentence are trimmed back to the last complete
  sentence, or marked with `…` when that would discard most of the text, so none
  reads as complete when it is not. Leads that merely restated their own headline
  were emptied.

Dropped at this stage: 121 out-of-window dates (down to 2005), 102 unrecoverable
dates, 60 Facebook posts, 2 rows with no CALABARZON province.

### Stage 2 — relevance review (965 → 371)

All 965 rows were read individually (title, lead, date, location). 554 were
removed as not genuinely about food insecurity and 40 as near-duplicates.

The retained/removed line follows one rule: **an article must report a
food-system condition, cause, or consequence.** A hazard on its own does not
qualify. So a typhoon story is kept when it reports crop or fishery damage,
evacuation or displacement, or relief distribution — and dropped when it reports
only casualties, debris, rainfall, class suspensions, or a bare state-of-calamity
declaration.

| Removed | Why |
|---|---|
| Disaster reporting with no food-system content | casualties, debris, weather bulletins, class suspensions, bare calamity declarations |
| National / non-CALABARZON stories | rice tariffs, WTO talks, NFA policy, Metro Manila and Visayas programmes |
| Geocoder false positives | see below |
| Food-adjacent but not food insecurity | recipes, farm-stay and travel features, festivals, food-safety advisories, research papers, corporate CSR and celebrity relief PR |
| Capacity-building programmes | irrigation, machinery, training, extension advisories — no insecurity condition evidenced |
| Opinion columns | national commentary with no CALABARZON subject |

### Geocoder false positives found and removed

The previous geography column was assigned lexically, which produced systematic
errors. Each was verified and corrected or dropped:

* **Surname collisions.** Agriculture Secretary Francisco Tiu **Laurel** tagged
  national policy stories to Laurel, Batangas; Trade Secretary Ramon **Lopez** to
  Lopez, Quezon; Senator **Pangilinan** to Pangil, Laguna; **Rodriguez** (an
  author, and E. Rodriguez Avenue in Quezon City) to Rodriguez, Rizal.
* **Foreign places.** Laguna Beach / Laguna Hills, California pulled in Orange
  County Register, Fox 5 San Diego, San Bernardino Sun, Forbes, Fox News and a
  Moldovan archaeology report — all tagged "Laguna". Nigeria's agricultural
  insurance corporation (**NAIC**, ₦1.014trn) was tagged to Naic, Cavite.
  Indonesian rice-export stories were tagged to Rizal.
* **Dateline vs subject.** The Inquirer's Lucena bureau datelines every
  CALABARZON story "LUCENA CITY", so stories about Taytay (Rizal), Lian
  (Batangas), Laguna and the region as a whole were all tagged Quezon/Lucena.
  These were re-pointed to the province the article is actually about.
* **Other-region subjects.** A Cebu relocation story (Talisay City exists in both
  Cebu and Batangas), San Juan City in NCR, Quezon City read as Quezon province,
  and a nationwide cash-aid story illustrated with Muntinlupa.

### Scope levels above the province

| scope | rows | `province` | `provinces_covered` |
|---|---|---|---|
| `city_municipality` | 113 | set | that province |
| `province` | 251 | set | that province |
| `region` | 7 | blank | the CALABARZON provinces the story covers |

Region-wide rows are retained deliberately: they are the regional DSWD tallies
whose subject is CALABARZON itself but which name no single province. Leaving
`province` blank on them keeps that column truthful — the silently wrong province
was the defect this audit set out to fix — while `provinces_covered` records what
each row actually speaks for. Filter to
`geographic_scope in ('city_municipality', 'province')` for province-level work.

**National-scope rows are excluded.** Nationwide drivers (rice tariffs, NFA
policy, food inflation, national feeding programmes) have no CALABARZON-specific
content, so admitting some and not others could only be arbitrary. 56 such rows
sit in the drop audit under national-scope reasons if a separate national-driver
table is ever wanted.

### Deduplication

One event reported by several outlets is collapsed to a single row. Two headlines
merge only when they share a province, an administrative scope, a hypothesis and
a date window, and then either read alike or quote the same figure:

* **Peso figures are compared with a 20% tolerance**, because one bulletin gets
  rounded differently by each outlet (`P577.39M`, `P577M`, `P578 million`). Where
  the figures genuinely differ the rows are kept: the January 2020 Taal damage
  story escalated P74.5M → P577M → P3.06B, and each milestone is real news.
* **A town-wide declaration never merges with the province-wide one.** Batangas
  ASF produced three distinct events in one week — Lobo (7 Aug), Calatagan
  (9 Aug) and the whole province (12 Aug) — which near-identical wording would
  otherwise have collapsed into one.

### Stage 3 — final validation

`audit_stage3_validate.py` re-checks the finished file independently across 49
assertions: hypothesis and category populated and mutually consistent, all
required columns non-null, dates ISO and in-window, provinces and province/city
pairs valid against the 142-LGU PSGC gazetteer, no duplicates by id / URL /
normalised title, no Google redirects, RSS markup, publisher suffixes, social
sources or replacement characters left, and a text-hygiene block that fails on
any of the artefact classes listed above. All checks pass.

## Columns

| Column | Notes |
|---|---|
| `title`, `publication_date`, `news_source`, `url` | publisher normalised; dates ISO |
| `date_provenance` | `iso` (116, direct fetch) / `rfc` (217, Google-feed stamp) / `recovered` (37, solved from a truncated stamp) / `corrected_from_article_text` (1, the stored stamp contradicted a date named in the article). Feed stamps can be crawl dates; ISO is the most reliable. |
| `content_lead` | present for 93 rows; see the lead-recovery note below |
| `lead_source` | `original_fetch` (86) / `publisher_url` (7, re-fetched from the publisher) / blank when no lead could be obtained |
| `title_provenance` | `headline`, or `slug_derived` where the headline was reconstructed from a URL slug, so its capitalisation is normalised rather than verbatim |
| `province`, `city_municipality`, `provinces_covered`, `geographic_scope`, `match_level` | see the scope table above; blanks are scope statements, not gaps |
| `hypothesis_topic`, `hypothesis_label` | the one hypothesis the article was verified to support |
| `food_security_dimension`, `food_insecurity_category`, `food_insecurity_topics`, `event_type` | derived from `hypothesis_topic` through the thesis's own CATEGORY map, so they are never blank and never disagree with each other |
| `relevance_tier` | `HIGH` = direct food evidence (T1, T1b, T2, T3, T6); `MEDIUM` = determinant (T4, T5, T7, T8) |
| `is_direct_food_insecurity` | boolean form of the same distinction |
| `review_basis` | why *this* article was kept, written during the review |
| `relevance_summary` | basis + hypothesis + location, composed from verified fields |
| `data_source`, `article_id` | provenance |

Removed from the previous schema: `author` (100% empty), `affected_commodity` and
`affected_population` (under 3% populated), `needs_review` (constant),
`relevance_score` (58% missing, produced by the classifier gates this audit
superseded), and `hypothesis_topics` (the regex multi-label — it tagged 693 rows
T6 and 393 T1b, largely off generic words like "yield", "planting" and "isda").

## Hypothesis distribution

| Code | Hypothesis | Rows |
|---|---|---|
| T1b | Fish kill or aquaculture collapse reducing fish food supply | 135 |
| T3 | Government food security programs ineffective or unavailable | 89 |
| T4 | Economic hardship reducing household income and food purchasing power | 45 |
| T7 | Civil displacement or evacuation reducing food access | 40 |
| T6 | Agricultural land loss or conversion reducing food production | 30 |
| T1 | Food supply disruption or food price increases | 13 |
| T2 | Health services or nutrition programs unavailable or unaffordable | 9 |
| T8 | Social unrest or conflict disrupting food systems | 8 |
| T5 | Infrastructure failures limiting food transport or storage | 2 |
| T9 | OFW remittance reduction | 0 |

T1b is the largest class because CALABARZON's two dominant food stories in this
period were the Taal Lake fish kills and the 2024 Batangas African swine fever
outbreak. T9 has no qualifying article: no CALABARZON-specific remittance story
in the pool met the relevance bar.

## Known limitations

* **Single-label.** Each article carries the one hypothesis it was verified to
  support. The previous multi-label column was a regex artefact and was not
  carried forward; a verified multi-label pass would need a second review.
* **Title-only rows.** 274 rows have no lead text, so their relevance was judged
  from the headline, date and location alone; headlines too vague to defend were
  dropped rather than kept. `audit_recover_leads.py` re-fetches leads from the
  publishers and recovered 7; the rest carry a Google-News redirect whose
  destination Google no longer encodes in the link, and every route to it is
  closed from this network (Google's resolver 503s, Cloudflare publishers 403,
  GMA and PhilStar render search in JavaScript, general search engines serve
  degraded pages to scripts). Re-running from a different network, or with a
  news-API key, would recover more. **Nothing was ever synthesised** — a lead is
  accepted only on a close headline match within a few days of the stored date.
* **Coverage.** 52 of 142 LGUs have a qualifying article
  (`calabarzon_lgu_coverage_matrix.csv`); 251 rows resolve to province level only.
  This is a news-availability limit, not a pipeline defect.
* **Feed dates.** 217 rows carry Google-feed stamps, which can reflect crawl
  rather than publication date; `date_provenance` flags them. Cross-checking
  cyclone names (which recur on PAGASA's 4-year cycle) and dates named inside the
  article text surfaced one wrong stamp, corrected, and one unverifiable row,
  dropped.
* **Later collection now folded in.** A subsequent `build_final_dataset.py` run
  produced 82 article_ids postdating the original audit input. All 82 were
  reviewed and merged into the pinned snapshot: 4 were added (a March 2026 Taal
  Lake fish kill, assistance to 12,000 distressed Batangas sugarcane farmers,
  House ayuda in Quezon, and a LANDBANK anti-hunger drive in Quezon); the rest
  were national-scope (43), foreign name collisions (Nigeria's NAIC, Texas's
  Laguna Madre, Laguna Beach California food pantries), pre-2020, or duplicates
  of rows already held. That build is preserved at
  `audit_review/prebuilt_1231_from_build_final_dataset.csv`.
